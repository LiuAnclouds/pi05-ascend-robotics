"""W-stationary full-K integer GEMM with an on-chip asymmetric epilogue."""

from tbe import tik
from tbe.common.utils import para_check


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_INT,
    para_check.KERNEL_NAME,
)
def pi05_perchannel_linear(
    a,
    w,
    row_sum,
    weight_zero,
    weight_scale,
    activation_scale,
    bias,
    y,
    out_rows=0,
    out_columns=0,
    kernel_name="pi05_perchannel_linear",
):
    rows, output_columns = map(int, y["shape"])
    columns = int(weight_zero["shape"][0])
    physical_m = (rows + 15) // 16 * 16
    k = int(a["shape"][0]) * 32
    if columns % 32 or k % 32 or k > 16384 or output_columns % 16 or rows < 1:
        raise ValueError("Needs physical N%32=0, K%32=0, K<=16384, output N%16=0")
    if tuple(a["shape"]) != (k // 32, physical_m // 16, 16, 32):
        raise ValueError("Expected A in signed INT8 FRACTAL_NZ: " + str(a))
    if tuple(w["shape"]) != (k // 32, columns // 16, 16, 32):
        raise ValueError("Expected offline packed signed INT8 W [K/32,N/16,16,32]: " + str(w))
    # Keep both complete operands within the 310P's 1 MiB L1. This changes
    # storage tiling only: one W scale/zero still covers the whole K axis.
    # The complete W stripe remains stationary; for wide K, stream two A
    # storage tiles and accumulate both products in the same on-chip INT32 C.
    # This is not group quantization: there is still only one W/A scale.
    k_tile = min(k, 8192)
    height = min(physical_m, 64)
    width = next(
        v
        for v in (256, 128, 64, 32)
        if columns % v == 0
        and k * v + k_tile * height <= 1024 * 1024
        and columns // v >= min(8, columns // 32)
    )
    cache_c = k == 16384 and 64 <= physical_m <= 1024 and columns % 64 == 0
    if cache_c:
        width = 64
    ntiles = columns // width
    inst = tik.Tik()
    ag = inst.Tensor("int8", (physical_m * k,), name="a_gm", scope=tik.scope_gm)
    wg = inst.Tensor("int8", (k * columns,), name="w_gm", scope=tik.scope_gm)
    rg = inst.Tensor("float32", (physical_m,), name="r_gm", scope=tik.scope_gm)
    zg = inst.Tensor("float32", (columns,), name="z_gm", scope=tik.scope_gm)
    swg = inst.Tensor("float32", (columns,), name="sw_gm", scope=tik.scope_gm)
    sag = inst.Tensor("float32", (physical_m,), name="sa_gm", scope=tik.scope_gm)
    bg = inst.Tensor("float32", (columns,), name="bias_gm", scope=tik.scope_gm)
    yg = inst.Tensor("float16", (rows * output_columns,), name="y_gm", scope=tik.scope_gm)
    cores = min(10, ntiles)
    with inst.for_range(0, cores, block_num=cores) as core:
        wl = inst.Tensor("int8", ((k_tile if cache_c else k) * width,), name="w_l1", scope=tik.scope_cbuf)
        if cache_c:
            # 1024*64*INT32 fits exactly in 310P L0C. Keep all M tiles here
            # while each K half of the W stripe is reused across every M tile.
            all_c = inst.Tensor(
                "int32",
                (((rows + height - 1) // height) * height * width,),
                name="all_c_l0c",
                scope=tik.scope_cbuf_out,
            )
            stream_a = inst.Tensor("int8", (height * k_tile,), name="stream_a_l1", scope=tik.scope_cbuf)
        zz = inst.Tensor("float32", (width,), name="zero_ub", scope=tik.scope_ubuf)
        sw = inst.Tensor("float32", (width,), name="sw_ub", scope=tik.scope_ubuf)
        bb = inst.Tensor("float32", (width,), name="bias_ub", scope=tik.scope_ubuf)
        with inst.for_range(0, (ntiles + cores - 1) // cores) as step:
            nb = step * cores + core
            with inst.if_scope(nb < ntiles):
                # Entire K x width weight stripe stays in L1 across every M tile.
                if not cache_c:
                    inst.data_move(wl, wg[nb * width * 32], 0, k // 32, width, columns - width, 0)
                inst.data_move(zz, zg[nb * width], 0, 1, width // 8, 0, 0)
                inst.data_move(sw, swg[nb * width], 0, 1, width // 8, 0, 0)
                inst.data_move(bb, bg[nb * width], 0, 1, width // 8, 0, 0)

                def calculate(mstart, tile_rows, valid_rows, tail=False, cached_offset=None):
                    elements = tile_rows * width
                    if cached_offset is None:
                        al = inst.Tensor("int8", (tile_rows * k_tile,), name="a_l1", scope=tik.scope_cbuf)
                        cc = inst.Tensor("int32", (elements,), name="c_l0c", scope=tik.scope_cbuf_out)
                    else:
                        cc = all_c[cached_offset : cached_offset + elements]
                    ci = inst.Tensor("int32", (elements,), name="c_nz_ub", scope=tik.scope_ubuf)
                    nzf = inst.Tensor("float32", (elements,), name="nz_float_ub", scope=tik.scope_ubuf)
                    cf = inst.Tensor("float32", (tile_rows * 16,), name="zero_expanded", scope=tik.scope_ubuf)
                    we = inst.Tensor(
                        "float32", (tile_rows * 16,), name="weight_expanded", scope=tik.scope_ubuf
                    )
                    out = inst.Tensor("float16", (elements,), name="out_ub", scope=tik.scope_ubuf)
                    nd = inst.Tensor("float16", (elements,), name="nd_out_ub", scope=tik.scope_ubuf)
                    corr = inst.Tensor("float32", (tile_rows * 16,), name="corr_ub", scope=tik.scope_ubuf)
                    rr = inst.Tensor("float32", (tile_rows,), name="row_sum_ub", scope=tik.scope_ubuf)
                    sa = inst.Tensor("float32", (tile_rows,), name="sa_ub", scope=tik.scope_ubuf)
                    re = inst.Tensor("float32", (tile_rows * 16,), name="row_expanded", scope=tik.scope_ubuf)
                    se = inst.Tensor(
                        "float32", (tile_rows * 16,), name="scale_expanded", scope=tik.scope_ubuf
                    )
                    temp = inst.Tensor("float32", (256,), name="broadcast_temp", scope=tik.scope_ubuf)
                    inst.data_move(rr, rg[mstart], 0, 1, tile_rows // 8, 0, 0)
                    inst.data_move(sa, sag[mstart], 0, 1, tile_rows // 8, 0, 0)
                    if cached_offset is None:
                        for k_start in range(0, k, k_tile):
                            count = min(k_tile, k - k_start)
                            inst.data_move(
                                al,
                                ag[k_start * physical_m + mstart * 32],
                                0,
                                count // 32,
                                tile_rows,
                                physical_m - tile_rows,
                                0,
                            )
                            inst.matmul(
                                cc,
                                al,
                                wl[k_start * width : (k_start + count) * width],
                                tile_rows,
                                count,
                                width,
                                init_l1out=(k_start == 0),
                            )
                    # L0C -> UB, never materialize the full INT32 matrix in GM.
                    inst.data_move(ci, cc, 0, 1, elements // 256, 0, 0)
                    for start in range(0, elements, 8192):
                        count = min(8192, elements - start)
                        inst.vconv(64, "", nzf[start], ci[start], count // 64, 1, 1, 8, 8)
                    # Broadcast row scalars with vector transpose rather than
                    # synchronizing the scalar/vector pipelines for every row.
                    for group in range(tile_rows // 16):
                        inst.vmuls(16, temp, rr[group * 16], 1.0, 16, 1, 1, 2, 0)
                        inst.v4dtrans(True, re[group * 256], temp, 16, 16)
                        inst.vmuls(16, temp, sa[group * 16], 1.0, 16, 1, 1, 2, 0)
                        inst.v4dtrans(True, se[group * 256], temp, 16, 16)
                    for block in range(width // 16):
                        inst.vmuls(16, cf, zz[block * 16], 1.0, tile_rows, 1, 1, 2, 0)
                        inst.vmuls(16, we, sw[block * 16], 1.0, tile_rows, 1, 1, 2, 0)
                        repeats = tile_rows * 16 // 64
                        start = block * tile_rows * 16
                        inst.vmul(64, corr, cf, re, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vsub(64, nzf[start], nzf[start], corr, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmul(64, nzf[start], nzf[start], we, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmul(64, nzf[start], nzf[start], se, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmuls(16, cf, bb[block * 16], 1.0, tile_rows, 1, 1, 2, 0)
                        inst.vadd(64, nzf[start], nzf[start], cf, repeats, 1, 1, 1, 8, 8, 8)
                    for start in range(0, elements, 8192):
                        count = min(8192, elements - start)
                        inst.vconv(64, "", out[start], nzf[start], count // 64, 1, 1, 4, 8)
                    for block in range(width // 16):
                        inst.data_move(
                            nd[block * 16],
                            out[block * tile_rows * 16],
                            0,
                            tile_rows,
                            1,
                            0,
                            (width - 16) // 16,
                        )
                    with inst.if_scope(nb * width < output_columns):
                        with inst.if_scope((nb + 1) * width <= output_columns):
                            inst.data_move(
                                yg[mstart * output_columns + nb * width],
                                nd,
                                0,
                                valid_rows,
                                width // 16,
                                0,
                                (output_columns - width) // 16,
                            )
                        if output_columns % width:
                            with inst.else_scope():
                                valid_columns = output_columns % width
                                inst.data_move(
                                    yg[mstart * output_columns + nb * width],
                                    nd,
                                    0,
                                    valid_rows,
                                    valid_columns // 16,
                                    (width - valid_columns) // 16,
                                    (output_columns - valid_columns) // 16,
                                )

                # TIK builds the Python body even for a zero-trip loop. Do not
                # build an oversized A load for time/Norm projections with M=1.
                if cache_c:

                    def accumulate(mstart, tile_rows, c_offset, k_start):
                        inst.data_move(
                            stream_a,
                            ag[k_start * physical_m + mstart * 32],
                            0,
                            k_tile // 32,
                            tile_rows,
                            physical_m - tile_rows,
                            0,
                        )
                        inst.matmul(
                            all_c[c_offset : c_offset + tile_rows * width],
                            stream_a,
                            wl,
                            tile_rows,
                            k_tile,
                            width,
                            init_l1out=(k_start == 0),
                        )

                    for k_start in range(0, k, k_tile):
                        inst.data_move(
                            wl,
                            wg[k_start * columns + nb * width * 32],
                            0,
                            k_tile // 32,
                            width,
                            columns - width,
                            0,
                        )
                        with inst.for_range(0, rows // height) as mi:
                            accumulate(mi * height, height, mi * height * width, k_start)
                        if rows % height:
                            tail_rows = (rows % height + 15) // 16 * 16
                            accumulate(
                                rows // height * height, tail_rows, rows // height * height * width, k_start
                            )
                    with inst.for_range(0, rows // height) as mi:
                        calculate(mi * height, height, height, cached_offset=mi * height * width)
                    if rows % height:
                        tail_rows = (rows % height + 15) // 16 * 16
                        calculate(
                            rows // height * height,
                            tail_rows,
                            rows % height,
                            cached_offset=rows // height * height * width,
                        )
                else:
                    if rows >= height:
                        with inst.for_range(0, rows // height) as mi:
                            calculate(mi * height, height, height)
                    if rows % height:
                        tail_rows = (rows % height + 15) // 16 * 16
                        calculate(rows // height * height, tail_rows, rows % height, tail=True)
    inst.BuildCCE(kernel_name=kernel_name, inputs=[ag, wg, rg, zg, swg, sag, bg], outputs=[yg])
    return inst
