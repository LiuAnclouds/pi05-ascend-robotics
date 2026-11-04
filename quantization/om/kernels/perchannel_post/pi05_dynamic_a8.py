"""Parallelize the existing FP32 dynamic-A8 division/round/clip, unchanged."""

from tbe import tik
from tbe.common.utils import para_check


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME,
)
def pi05_dynamic_a8(x, bound, scale, clipped, kernel_name="pi05_dynamic_a8"):
    rows, columns = map(int, x["shape"])
    if rows % 8 or columns % 64 or columns > 8192:
        raise ValueError("Initial parallel A8 kernel needs M%8=0 and K%64=0, K<=8192")
    inst = tik.Tik()
    xg = inst.Tensor("float32", (rows * columns,), name="x_gm", scope=tik.scope_gm)
    bg = inst.Tensor("float32", (rows,), name="bound_gm", scope=tik.scope_gm)
    sg = inst.Tensor("float32", (rows,), name="scale_gm", scope=tik.scope_gm)
    qg = inst.Tensor("float32", (rows * columns,), name="q_gm", scope=tik.scope_gm)
    cores = min(10, rows // 8)
    with inst.for_range(0, cores, block_num=cores) as core:
        values = inst.Tensor("float32", (columns,), name="values", scope=tik.scope_ubuf)
        denom = inst.Tensor("float32", (columns,), name="denom", scope=tik.scope_ubuf)
        integer = inst.Tensor("int32", (columns,), name="integer", scope=tik.scope_ubuf)
        scales = inst.Tensor("float32", (8,), name="scales", scope=tik.scope_ubuf)
        scalar = inst.Scalar("float32", name="scale")
        with inst.for_range(0, (rows // 8 + cores - 1) // cores) as step:
            block = step * cores + core
            with inst.if_scope(block < rows // 8):
                inst.data_move(scales, bg[block * 8], 0, 1, 1, 0, 0)
                # Match the compiled ONNX Div(bound,127), which uses this FP32 reciprocal.
                inst.vmuls(8, scales, scales, 1.0 / 127.0, 1, 1, 1, 8, 8)
                inst.vmaxs(8, scales, scales, 1e-12, 1, 1, 1, 8, 8)
                inst.data_move(sg[block * 8], scales, 0, 1, 1, 0, 0)
                with inst.for_range(0, 8) as row:
                    scalar.set_as(scales[row])
                    inst.vector_dup(64, denom, scalar, columns // 64, 1, 8)
                    inst.data_move(values, xg[(block * 8 + row) * columns], 0, 1, columns // 8, 0, 0)
                    inst.vdiv(64, values, values, denom, columns // 64, 1, 1, 1, 8, 8, 8)
                    inst.vconv(64, "round", integer, values, columns // 64, 1, 1, 8, 8)
                    inst.vconv(64, "", values, integer, columns // 64, 1, 1, 8, 8)
                    inst.vmaxs(64, values, values, -127.0, columns // 64, 1, 1, 8, 8)
                    inst.vmins(64, values, values, 127.0, columns // 64, 1, 1, 8, 8)
                    inst.data_move(qg[(block * 8 + row) * columns], values, 0, 1, columns // 8, 0, 0)
    inst.BuildCCE(kernel_name=kernel_name, inputs=[xg, bg], outputs=[sg, qg])
    return inst
