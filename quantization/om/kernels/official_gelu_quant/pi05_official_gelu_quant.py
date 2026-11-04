"""310P ND GELU/FP16-Mul/CANN-FP16-DynamicQuant fusion.

No group quantization or NZ reinterpretation. Preserve the FP16 GELU output,
FP16 product, FP16 quantization multiplier/product, and full-row A8 scale.
"""
from tbe import tik
from tbe.common.utils import para_check


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                           para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                           para_check.KERNEL_NAME)
def pi05_official_gelu_mul_dynamic_quant(x, other, a, scale,
                                        kernel_name="pi05_official_gelu_mul_dynamic_quant"):
    return _build([x], [other], a, scale, kernel_name)


@para_check.check_op_params(*([para_check.REQUIRED_INPUT] * 8 +
    [para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME]))
def pi05_official_chunk_gelu_mul_dynamic_quant(g0, g1, g2, g3, u0, u1, u2, u3, a, scale,
        kernel_name="pi05_official_chunk_gelu_mul_dynamic_quant"):
    return _build([g0, g1, g2, g3], [u0, u1, u2, u3], a, scale, kernel_name)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                           para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def pi05_official_vision_gelu_dynamic_quant(x, a, scale,
        kernel_name="pi05_official_vision_gelu_dynamic_quant"):
    return _build([x], [], a, scale, kernel_name)


def _build(gates, ups, a, scale, kernel_name):
    rows, part_columns = map(int, gates[0]["shape"])
    vision = len(ups) == 0
    output_columns = 4320 if vision else part_columns * len(gates)
    columns = 4352 if vision else output_columns
    if (any(tuple(v["shape"]) != (rows, part_columns) for v in gates + ups) or rows % 8 or
        (vision and (part_columns != 4304 or len(gates) != 1)) or
        (not vision and columns != 16384)):
        raise ValueError("Only matched [M divisible by 8,16384] FP16 inputs are supported")
    inst = tik.Tik()
    gx = [inst.Tensor("float16", (rows * part_columns,), name="gate_gm_" + str(i), scope=tik.scope_gm)
          for i in range(len(gates))]
    gu = [inst.Tensor("float16", (rows * part_columns,), name="up_gm_" + str(i), scope=tik.scope_gm)
          for i in range(len(ups))]
    ga = inst.Tensor("int8", (rows * output_columns,), name="codes_gm", scope=tik.scope_gm)
    gs = inst.Tensor("float32", (rows,), name="scales_gm", scope=tik.scope_gm)
    cores = min(10, rows // 8)
    with inst.for_range(0, cores, block_num=cores) as core:
        half = inst.Tensor("float16", (columns,), name="half_values", scope=tik.scope_ubuf)
        up = inst.Tensor("float16", (columns,), name="up_values", scope=tik.scope_ubuf)
        values = inst.Tensor("float32", (columns,), name="float_values", scope=tik.scope_ubuf)
        scratch = inst.Tensor("float32", (columns,), name="scratch", scope=tik.scope_ubuf)
        integers = scratch.reinterpret_cast_to("int32")
        codes = scratch.reinterpret_cast_to("int8")
        work = inst.Tensor("float32", (1024,), name="reduce_work", scope=tik.scope_ubuf)
        result = inst.Tensor("float32", (8,), name="reduce_result", scope=tik.scope_ubuf)
        denominator = inst.Tensor("float32", (8,), name="denominator", scope=tik.scope_ubuf)
        numerator = inst.Tensor("float32", (8,), name="numerator", scope=tik.scope_ubuf)
        inverse = inst.Tensor("float32", (8,), name="inverse", scope=tik.scope_ubuf)
        scales = inst.Tensor("float32", (8,), name="block_scales", scope=tik.scope_ubuf)
        bound = inst.Scalar("float32", name="bound")
        part = inst.Scalar("float32", name="part")
        inverse_half = inst.Scalar("float16", name="inverse_half")
        with inst.for_range(0, (rows // 8 + cores - 1) // cores) as loop:
            block = loop * cores + core
            with inst.if_scope(block < rows // 8):
                with inst.for_range(0, 8) as local_row:
                    row = block * 8 + local_row
                    if vision:
                        inst.vector_dup(128, half, 0.0, columns // 128, 1, 8)
                    for i, tensor in enumerate(gx):
                        inst.data_move(half[i * part_columns], tensor[row * part_columns], 0, 1, part_columns // 16, 0, 0)
                    for start in range(0, columns, 8192):
                        repeats = min(8192, columns - start) // 64
                        # Same high-precision tanh-GELU polynomial/logistic
                        # form as the existing verified 310P GELU fusion.
                        inst.vconv(64, "", values[start], half[start], repeats, 1, 1, 8, 4)
                        inst.vmul(64, scratch[start], values[start], values[start], repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmul(64, scratch[start], scratch[start], values[start], repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmuls(64, scratch[start], scratch[start], 0.044715, repeats, 1, 1, 8, 8)
                        inst.vadd(64, scratch[start], scratch[start], values[start], repeats, 1, 1, 1, 8, 8, 8)
                        inst.vmuls(64, scratch[start], scratch[start], -1.5957691, repeats, 1, 1, 8, 8)
                        inst.vexp(64, scratch[start], scratch[start], repeats, 1, 1, 8, 8)
                        inst.vadds(64, scratch[start], scratch[start], 1.0, repeats, 1, 1, 8, 8)
                        inst.vdiv(64, values[start], values[start], scratch[start], repeats, 1, 1, 1, 8, 8, 8)
                        inst.vconv(64, "", half[start], values[start], repeats, 1, 1, 4, 8)
                    for i, tensor in enumerate(gu):
                        inst.data_move(up[i * part_columns], tensor[row * part_columns], 0, 1, part_columns // 16, 0, 0)
                    if not vision:
                        inst.vmul(128, half, half, up, columns // 128, 1, 1, 1, 8, 8, 8)
                    bound.set_as(0.0)
                    for start in range(0, columns, 8192):
                        repeats = min(8192, columns - start) // 64
                        inst.vconv(64, "", values[start], half[start], repeats, 1, 1, 8, 4)
                        inst.vabs(64, scratch[start], values[start], repeats, 1, 1, 8, 8)
                        inst.vec_reduce_max(64, result, scratch[start], work, repeats, 8)
                        part.set_as(result[0])
                        with inst.if_scope(part > bound):
                            bound.set_as(part)
                    denominator[0].set_as(bound)
                    inst.vmuls(1, result, denominator, 1.0 / 127.0, 1, 1, 1, 8, 8)
                    scales[local_row].set_as(result[0])
                    # CANN dynamic_quant_align_310p.h FP16 path: FP32
                    # 127/max -> FP16 multiplier -> FP16 multiply -> ROUND.
                    with inst.if_scope(bound == 0.0):
                        inverse_half.set_as(0.0)
                    with inst.else_scope():
                        numerator[0].set_as(127.0)
                        inst.vdiv(1, inverse, numerator, denominator, 1, 1, 1, 1, 8, 8, 8)
                        inst.vconv(1, "", up, inverse, 1, 1, 1, 4, 8)
                        inverse_half.set_as(up[0])
                    inst.vmuls(128, half, half, inverse_half, columns // 128, 1, 1, 8, 8)
                    # Match the installed 310P Cast<half,int8_t> instruction,
                    # not half->INT32 RNE followed by a second cast. The latter
                    # changes half-integer ties (observed odd/even code drift).
                    # Full-row max normalization bounds the half values near
                    # +/-127, so an extra FP32 clamp is not part of this path.
                    inst.vconv(128, "", codes, half, columns // 128, 1, 1, 4, 8)
                    # Plain ND, contiguous row. Do NOT reinterpret as the
                    # old custom NZ layout; official GEMM still receives ND.
                    inst.data_move(ga[row * output_columns], codes, 0, 1, output_columns // 32, 0, 0)
                inst.data_move(gs[block * 8], scales, 0, 1, 1, 0, 0)
    inst.BuildCCE(kernel_name=kernel_name, inputs=gx + gu, outputs=[ga, gs])
    return inst
