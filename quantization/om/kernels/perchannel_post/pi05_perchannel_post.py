"""310P single-pass asymmetric W8 / dynamic A8 postprocessing.

The integer GEMM is unchanged. Read its NZ INT32 tiles directly and write
FP16 ND without materializing an FP32 output or a zero-point outer product.
"""

from tbe import tik
from tbe.common.utils import para_check


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME,
)
def pi05_perchannel_post(
    c, row_sum, weight_zero, weight_scale, activation_scale, y, kernel_name="pi05_perchannel_post"
):
    rows, columns = map(int, y["shape"])
    if rows < 16 or columns % 128:
        raise ValueError("Initial fused-post kernel needs M>=16 and N divisible by 128")
    mblocks = (rows + 15) // 16
    width = next(size for size in (512, 256, 128) if columns % size == 0)
    elements = 16 * width
    ntiles = columns // width
    if tuple(c["shape"]) != (columns // 16, mblocks, 16, 16):
        raise ValueError("Expected INT32 FRACTAL_NZ [N/16,ceil(M/16),16,16], got " + str(c))
    inst = tik.Tik()
    cg = inst.Tensor("int32", (columns * mblocks * 16,), name="c_gm", scope=tik.scope_gm)
    rg = inst.Tensor("float32", (rows,), name="r_gm", scope=tik.scope_gm)
    zg = inst.Tensor("float32", weight_zero["shape"], name="z_gm", scope=tik.scope_gm)
    wg = inst.Tensor("float32", weight_scale["shape"], name="w_gm", scope=tik.scope_gm)
    ag = inst.Tensor("float32", (rows,), name="a_gm", scope=tik.scope_gm)
    yg = inst.Tensor("float16", (rows * columns,), name="y_gm", scope=tik.scope_gm)

    cores = min(10, mblocks * ntiles)
    with inst.for_range(0, cores, block_num=cores) as core:
        ci = inst.Tensor("int32", (elements,), name="c_ub", scope=tik.scope_ubuf)
        nzf = inst.Tensor("float32", (elements,), name="nz_float_ub", scope=tik.scope_ubuf)
        cf = inst.Tensor("float32", (elements,), name="cf_ub", scope=tik.scope_ubuf)
        corr = inst.Tensor("float32", (width,), name="corr_ub", scope=tik.scope_ubuf)
        out = inst.Tensor("float16", (elements,), name="out_ub", scope=tik.scope_ubuf)
        rr = inst.Tensor("float32", (16,), name="r_ub", scope=tik.scope_ubuf)
        aa = inst.Tensor("float32", (16,), name="a_ub", scope=tik.scope_ubuf)
        zz = inst.Tensor("float32", (width,), name="z_ub", scope=tik.scope_ubuf)
        ww = inst.Tensor("float32", (width,), name="w_ub", scope=tik.scope_ubuf)
        r_scalar = inst.Scalar("float32", name="r_scalar")
        a_scalar = inst.Scalar("float32", name="a_scalar")
        r_offset = inst.Scalar("int32", name="r_offset", init_value=0)
        for_core = (mblocks * ntiles + cores - 1) // cores
        with inst.for_range(0, for_core) as step:
            tile = step * cores + core
            with inst.if_scope(tile < mblocks * ntiles):
                mb = tile // ntiles
                nb = tile % ntiles
                r_offset.set_as(0)
                with inst.if_scope(mb * 16 + 16 <= rows):
                    inst.data_move(rr, rg[mb * 16], 0, 1, 2, 0, 0)
                    inst.data_move(aa, ag[mb * 16], 0, 1, 2, 0, 0)
                with inst.else_scope():
                    inst.data_move(rr, rg[rows - 16], 0, 1, 2, 0, 0)
                    inst.data_move(aa, ag[rows - 16], 0, 1, 2, 0, 0)
                    r_offset.set_as(16 - rows % 16)
                inst.data_move(zz, zg[nb * width], 0, 1, width // 8, 0, 0)
                inst.data_move(ww, wg[nb * width], 0, 1, width // 8, 0, 0)
                # Read each complete NZ microtile (1024 bytes), not sixteen
                # independent 64-byte GM bursts. Rearrange locally in UB.
                inst.data_move(
                    ci,
                    cg[(nb * (width // 16) * mblocks + mb) * 256],
                    0,
                    width // 16,
                    32,
                    (mblocks - 1) * 32,
                    0,
                )
                inst.vconv(64, "", nzf, ci, elements // 64, 1, 1, 8, 8)
                for block in range(width // 16):
                    inst.vmuls(16, cf[block * 16], nzf[block * 256], 1.0, 16, 1, 1, width // 8, 2)
                with inst.for_range(0, 16) as row:
                    with inst.if_scope(mb * 16 + row < rows):
                        r_scalar.set_as(rr[row + r_offset])
                        a_scalar.set_as(aa[row + r_offset])
                        inst.vmuls(64, corr, zz, r_scalar, width // 64, 1, 1, 8, 8)
                        inst.vsub(64, cf[row * width], cf[row * width], corr, width // 64, 1, 1, 1, 8, 8, 8)
                        inst.vmul(64, cf[row * width], cf[row * width], ww, width // 64, 1, 1, 1, 8, 8, 8)
                        inst.vmuls(64, cf[row * width], cf[row * width], a_scalar, width // 64, 1, 1, 8, 8)
                inst.vconv(64, "", out, cf, elements // 64, 1, 1, 4, 8)
                with inst.if_scope(mb * 16 + 16 <= rows):
                    inst.data_move(
                        yg[mb * 16 * columns + nb * width],
                        out,
                        0,
                        16,
                        width // 16,
                        0,
                        (columns - width) // 16,
                    )
                with inst.else_scope():
                    inst.data_move(
                        yg[mb * 16 * columns + nb * width],
                        out,
                        0,
                        rows % 16,
                        width // 16,
                        0,
                        (columns - width) // 16,
                    )
    inst.BuildCCE(kernel_name=kernel_name, inputs=[cg, rg, zg, wg, ag], outputs=[yg])
    return inst
