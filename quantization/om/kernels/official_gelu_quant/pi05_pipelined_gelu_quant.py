"""Same verified GELU/A8 arithmetic, with smaller UB tiles and row overlap.

This is an intra-kernel DMA/vector schedule, not asynchronous policy inference.
Keep all FP16/FP32 rounding boundaries and the full-K dynamic quantization max.
"""
from tbe import tik
from tbe.common.utils import para_check


@para_check.check_op_params(*([para_check.REQUIRED_INPUT] * 8 +
    [para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME]))
def pi05_pipelined_chunk_gelu_mul_dynamic_quant(g0, g1, g2, g3, u0, u1, u2, u3, a, scale,
        kernel_name="pi05_pipelined_chunk_gelu_mul_dynamic_quant"):
    return _build([g0, g1, g2, g3], [u0, u1, u2, u3], kernel_name)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                           para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def pi05_pipelined_vision_gelu_dynamic_quant(x, a, scale,
        kernel_name="pi05_pipelined_vision_gelu_dynamic_quant"):
    return _build([x], [], kernel_name)


def _build(gates, ups, kernel_name):
    rows, part_columns = map(int, gates[0]["shape"])
    vision = not ups
    output_columns = 4320 if vision else 16384
    columns = 4352 if vision else 16384
    chunk = min(columns, 8192)
    if (rows % 8 or any(tuple(v["shape"]) != (rows, part_columns) for v in gates + ups)
        or (vision and (len(gates) != 1 or part_columns != 4304))
        or (not vision and (len(gates) != 4 or len(ups) != 4 or part_columns != 4096))):
        raise ValueError("Only the already verified chunk/vision shapes are supported")
    inst = tik.Tik()
    gx = [inst.Tensor("float16", (rows * part_columns,), name="gate_gm_" + str(i), scope=tik.scope_gm)
          for i in range(len(gates))]
    gu = [inst.Tensor("float16", (rows * part_columns,), name="up_gm_" + str(i), scope=tik.scope_gm)
          for i in range(len(ups))]
    ga = inst.Tensor("int8", (rows * output_columns,), name="codes_gm", scope=tik.scope_gm)
    gs = inst.Tensor("float32", (rows,), name="scales_gm", scope=tik.scope_gm)
    cores = min(10, rows // 8)  # Installed Ascend310P1 CORE_NUM = 10.
    with inst.for_range(0, cores, block_num=cores) as core:
        scales = inst.Tensor("float32", (8,), name="block_scales", scope=tik.scope_ubuf)
        with inst.for_range(0, (rows // 8 + cores - 1) // cores) as loop:
            block = loop * cores + core
            with inst.if_scope(block < rows // 8):
                with inst.for_range(0, 8, thread_num=2) as local_row:
                    # Row-local allocations are double buffered by TIK. The
                    # 8K FP32 work tile keeps both live rows below 256 KiB UB.
                    half = inst.Tensor("float16", (columns,), name="half_values", scope=tik.scope_ubuf)
                    up = inst.Tensor("float16", (chunk,), name="up_values", scope=tik.scope_ubuf)
                    values = inst.Tensor("float32", (chunk,), name="float_values", scope=tik.scope_ubuf)
                    scratch = inst.Tensor("float32", (chunk,), name="scratch", scope=tik.scope_ubuf)
                    codes = scratch.reinterpret_cast_to("int8")
                    work = inst.Tensor("float32", (1024,), name="reduce_work", scope=tik.scope_ubuf)
                    result = inst.Tensor("float32", (8,), name="reduce_result", scope=tik.scope_ubuf)
                    denominator = inst.Tensor("float32", (8,), name="denominator", scope=tik.scope_ubuf)
                    numerator = inst.Tensor("float32", (8,), name="numerator", scope=tik.scope_ubuf)
                    inverse = inst.Tensor("float32", (8,), name="inverse", scope=tik.scope_ubuf)
                    bound = inst.Scalar("float32", name="bound")
                    part = inst.Scalar("float32", name="part")
                    inverse_half = inst.Scalar("float16", name="inverse_half")
                    row = block * 8 + local_row
                    if vision:
                        inst.vector_dup(128, half, 0., columns // 128, 1, 8)
                    for i, tensor in enumerate(gx):
                        inst.data_move(half[i * part_columns], tensor[row * part_columns], 0, 1, part_columns // 16, 0, 0)
                    for start in range(0, columns, 8192):
                        count = min(8192, columns - start)
                        repeats = count // 64
                        inst.vconv(64, "", values, half[start], repeats, 1, 1, 8, 4)
                        inst.vmul(64, scratch, values, values, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmul(64, scratch, scratch, values, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmuls(64, scratch, scratch, .044715, repeats, 1, 1, 8, 8)
                        inst.vadd(64, scratch, scratch, values, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmuls(64, scratch, scratch, -1.5957691, repeats, 1, 1, 8, 8)
                        inst.vexp(64, scratch, scratch, repeats, 1, 1, 8, 8)
                        inst.vadds(64, scratch, scratch, 1., repeats, 1, 1, 8, 8)
                        inst.vdiv(64, values, values, scratch, repeats, 1, 1, 1, 8, 8, 8)
                        inst.vconv(64, "", half[start], values, repeats, 1, 1, 4, 8)
                        if not vision:
                            for i, tensor in enumerate(gu):
                                left = max(start, i * part_columns)
                                right = min(start + count, (i + 1) * part_columns)
                                if left < right:
                                    inst.data_move(up[left - start], tensor[row * part_columns + left - i * part_columns],
                                                   0, 1, (right - left) // 16, 0, 0)
                            inst.vmul(128, half[start], half[start], up, count // 128, 1, 1, 1, 8, 8, 8)
                    bound.set_as(0.)
                    for start in range(0, columns, 8192):
                        repeats = min(8192, columns - start) // 64
                        inst.vconv(64, "", values, half[start], repeats, 1, 1, 8, 4)
                        inst.vabs(64, scratch, values, repeats, 1, 1, 8, 8)
                        inst.vec_reduce_max(64, result, scratch, work, repeats, 8)
                        part.set_as(result[0])
                        with inst.if_scope(part > bound):
                            bound.set_as(part)
                    denominator[0].set_as(bound)
                    inst.vmuls(1, result, denominator, 1. / 127., 1, 1, 1, 8, 8)
                    scales[local_row].set_as(result[0])
                    with inst.if_scope(bound == 0.):
                        inverse_half.set_as(0.)
                    with inst.else_scope():
                        numerator[0].set_as(127.)
                        inst.vdiv(1, inverse, numerator, denominator, 1, 1, 1, 1, 8, 8, 8)
                        inst.vconv(1, "", up, inverse, 1, 1, 1, 4, 8)
                        inverse_half.set_as(up[0])
                    inst.vmuls(128, half, half, inverse_half, columns // 128, 1, 1, 8, 8)
                    inst.vconv(128, "", codes, half, columns // 128, 1, 1, 4, 8)
                    inst.data_move(ga[row * output_columns], codes, 0, 1, output_columns // 32, 0, 0)
                inst.data_move(gs[block * 8], scales, 0, 1, 1, 0, 0)
    inst.BuildCCE(kernel_name=kernel_name, inputs=gx + gu, outputs=[ga, gs])
    return inst
