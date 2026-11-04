"""FP16 -> exact dynamic A8, row sums and prepacked INT8, one AI Core task."""

from tbe import tik
from tbe.common.utils import para_check


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME,
)
def pi05_prepare_a8(x, a, row_sum, activation_scale, kernel_name="pi05_prepare_a8"):
    rows, columns = map(int, x["shape"])
    if columns % 16 or columns > 16384:
        raise ValueError("Input K must be a multiple of 16 and <=16384")
    if columns == 16384:
        return _prepare_wide(x, kernel_name)
    physical_m = (rows + 15) // 16 * 16
    physical_k = (columns + 31) // 32 * 32
    chunk = min(4096, max(128, (columns + 63) // 64 * 64))
    inst = tik.Tik()
    xg = inst.Tensor("float16", (rows * columns,), name="x_gm", scope=tik.scope_gm)
    ag = inst.Tensor("int8", (physical_m * physical_k,), name="a_gm", scope=tik.scope_gm)
    rg = inst.Tensor("float32", (physical_m,), name="row_sum_gm", scope=tik.scope_gm)
    sg = inst.Tensor("float32", (physical_m,), name="scale_gm", scope=tik.scope_gm)
    block_rows = 8
    cores = min(10, physical_m // block_rows)
    with inst.for_range(0, cores, block_num=cores) as core:
        hx = inst.Tensor("float16", (chunk,), name="half_input", scope=tik.scope_ubuf)
        fx = inst.Tensor("float32", (chunk,), name="float_input", scope=tik.scope_ubuf)
        temp = inst.Tensor("float32", (chunk,), name="temporary", scope=tik.scope_ubuf)
        integer = inst.Tensor("int32", (chunk,), name="integer", scope=tik.scope_ubuf)
        qhalf = inst.Tensor("float16", (chunk,), name="half_codes", scope=tik.scope_ubuf)
        qbyte = inst.Tensor("int8", (chunk,), name="byte_codes", scope=tik.scope_ubuf)
        packed_rows = (
            inst.Tensor("int8", (block_rows * physical_k,), name="packed_rows", scope=tik.scope_ubuf)
            if physical_k <= 8192
            else None
        )
        work = inst.Tensor("float32", (1024,), name="reduce_work", scope=tik.scope_ubuf)
        result = inst.Tensor("float32", (8,), name="reduce_result", scope=tik.scope_ubuf)
        scales = inst.Tensor("float32", (16,), name="scales", scope=tik.scope_ubuf)
        sums = inst.Tensor("float32", (16,), name="sums", scope=tik.scope_ubuf)
        bound = inst.Scalar("float32", name="bound")
        scale = inst.Scalar("float32", name="scale")
        total = inst.Scalar("float32", name="total")
        part = inst.Scalar("float32", name="part")
        with inst.for_range(0, (physical_m // block_rows + cores - 1) // cores) as step:
            block = step * cores + core
            with inst.if_scope(block < physical_m // block_rows):
                inst.vector_dup(16, scales, 1e-12, 1, 1, 8)
                inst.vector_dup(16, sums, 0.0, 1, 1, 8)
                with inst.for_range(0, block_rows) as local_row:
                    row = block * block_rows + local_row
                    with inst.if_scope(row < rows):
                        bound.set_as(0.0)
                        # Chunking is storage tiling, not group quantization:
                        # one absmax and scale still cover the entire K row.
                        for start in range(0, columns, chunk):
                            valid = min(chunk, columns - start)
                            padded = (valid + 63) // 64 * 64
                            inst.vector_dup(128, hx, 0.0, (padded + 127) // 128, 1, 8)
                            inst.data_move(hx, xg[row * columns + start], 0, 1, valid // 16, 0, 0)
                            inst.vconv(64, "", fx, hx, padded // 64, 1, 1, 8, 4)
                            inst.vabs(64, temp, fx, padded // 64, 1, 1, 8, 8)
                            inst.vec_reduce_max(64, result, temp, work, padded // 64, 8)
                            part.set_as(result[0])
                            with inst.if_scope(part > bound):
                                bound.set_as(part)
                        result[0].set_as(bound)
                        inst.vmuls(1, result, result, 1.0 / 127.0, 1, 1, 1, 8, 8)
                        inst.vmaxs(1, result, result, 1e-12, 1, 1, 1, 8, 8)
                        scale.set_as(result[0])
                        scales[local_row].set_as(scale)
                        total.set_as(0.0)
                        for start in range(0, columns, chunk):
                            valid = min(chunk, columns - start)
                            padded = (valid + 63) // 64 * 64
                            if columns > chunk:
                                inst.vector_dup(128, hx, 0.0, (padded + 127) // 128, 1, 8)
                                inst.data_move(hx, xg[row * columns + start], 0, 1, valid // 16, 0, 0)
                                inst.vconv(64, "", fx, hx, padded // 64, 1, 1, 8, 4)
                            inst.vector_dup(64, temp, scale, padded // 64, 1, 8)
                            inst.vdiv(64, fx, fx, temp, padded // 64, 1, 1, 1, 8, 8, 8)
                            inst.vconv(64, "round", integer, fx, padded // 64, 1, 1, 8, 8)
                            inst.vconv(64, "", fx, integer, padded // 64, 1, 1, 8, 8)
                            inst.vmaxs(64, fx, fx, -127.0, padded // 64, 1, 1, 8, 8)
                            inst.vmins(64, fx, fx, 127.0, padded // 64, 1, 1, 8, 8)
                            inst.vec_reduce_add(64, result, fx, work, padded // 64, 8)
                            part.set_as(result[0])
                            total.set_as(total + part)
                            inst.vconv(64, "", qhalf, fx, padded // 64, 1, 1, 4, 8)
                            inst.vconv(128, "", qbyte, qhalf, (padded + 127) // 128, 1, 1, 4, 8)
                            # Emit physical NZ directly; no separate TransData.
                            count = (valid + 31) // 32
                            if packed_rows is not None:
                                dest = ((start // 32) * block_rows + local_row) * 32
                                inst.data_move(packed_rows[dest], qbyte, 0, count, 1, 0, block_rows - 1)
                            else:
                                dest = ((start // 32) * physical_m + row) * 32
                                inst.data_move(ag[dest], qbyte, 0, count, 1, 0, physical_m - 1)
                        sums[local_row].set_as(total)
                    with inst.else_scope():
                        inst.vector_dup(128, qhalf, 0.0, chunk // 128, 1, 8)
                        inst.vconv(128, "", qbyte, qhalf, chunk // 128, 1, 1, 4, 8)
                        for start in range(0, physical_k, chunk):
                            count = min(chunk, physical_k - start) // 32
                            if packed_rows is not None:
                                dest = ((start // 32) * block_rows + local_row) * 32
                                inst.data_move(packed_rows[dest], qbyte, 0, count, 1, 0, block_rows - 1)
                            else:
                                dest = ((start // 32) * physical_m + row) * 32
                                inst.data_move(ag[dest], qbyte, 0, count, 1, 0, physical_m - 1)
                if packed_rows is not None:
                    inst.data_move(
                        ag[block * block_rows * 32],
                        packed_rows,
                        0,
                        physical_k // 32,
                        block_rows,
                        0,
                        physical_m - block_rows,
                    )
                inst.data_move(rg[block * block_rows], sums, 0, 1, block_rows // 8, 0, 0)
                inst.data_move(sg[block * block_rows], scales, 0, 1, block_rows // 8, 0, 0)
    inst.BuildCCE(kernel_name=kernel_name, inputs=[xg], outputs=[ag, rg, sg])
    return inst


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME,
)
def pi05_mul_prepare_a8(x, other, a, row_sum, activation_scale, kernel_name="pi05_mul_prepare_a8"):
    if tuple(x["shape"]) != tuple(other["shape"]) or len(x["shape"]) != 2 or x["shape"][1] != 16384:
        raise ValueError("Fused FFN multiply requires equal [M,16384] inputs")
    return _prepare_wide(x, kernel_name, other=other)


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME,
)
def pi05_gelu_mul_prepare_a8(x, other, a, row_sum, activation_scale, kernel_name="pi05_gelu_mul_prepare_a8"):
    if tuple(x["shape"]) != tuple(other["shape"]) or len(x["shape"]) != 2 or x["shape"][1] != 16384:
        raise ValueError("Fused GELU/FFN multiply requires equal [M,16384] inputs")
    return _prepare_wide(x, kernel_name, other=other, gelu=True)


def _prepare_wide(x, kernel_name, other=None, gelu=False):
    """Keep one complete FFN row in UB; reuse scratch storage across stages."""
    rows, columns = map(int, x["shape"])
    physical_m = (rows + 15) // 16 * 16
    inst = tik.Tik()
    xg = inst.Tensor("float16", (rows * columns,), name="x_gm", scope=tik.scope_gm)
    otherg = (
        inst.Tensor("float16", (rows * columns,), name="other_gm", scope=tik.scope_gm)
        if other is not None
        else None
    )
    ag = inst.Tensor("int8", (physical_m * columns,), name="a_gm", scope=tik.scope_gm)
    rg = inst.Tensor("float32", (physical_m,), name="row_sum_gm", scope=tik.scope_gm)
    sg = inst.Tensor("float32", (physical_m,), name="scale_gm", scope=tik.scope_gm)
    cores = min(10, physical_m // 8)
    with inst.for_range(0, cores, block_num=cores) as core:
        half = inst.Tensor("float16", (columns,), name="row_half", scope=tik.scope_ubuf)
        values = inst.Tensor("float32", (columns,), name="row_values", scope=tik.scope_ubuf)
        scratch = inst.Tensor("float32", (columns,), name="row_scratch", scope=tik.scope_ubuf)
        integer = scratch.reinterpret_cast_to("int32")
        byte = scratch.reinterpret_cast_to("int8")
        # Four quantized rows plus the full-row scratch still fit in 256 KiB
        # UB. Assemble adjacent NZ rows here instead of issuing 512 individual
        # 32-byte GM bursts for each row. This changes storage only, not A8.
        packed = inst.Tensor("int8", (4 * columns,), name="packed_four_rows", scope=tik.scope_ubuf)
        work = inst.Tensor("float32", (1024,), name="reduce_work", scope=tik.scope_ubuf)
        result = inst.Tensor("float32", (8,), name="reduce_result", scope=tik.scope_ubuf)
        scales = inst.Tensor("float32", (8,), name="scales", scope=tik.scope_ubuf)
        sums = inst.Tensor("float32", (8,), name="sums", scope=tik.scope_ubuf)
        bound = inst.Scalar("float32", name="bound")
        scale = inst.Scalar("float32", name="scale")
        total = inst.Scalar("float32", name="total")
        part = inst.Scalar("float32", name="part")
        with inst.for_range(0, (physical_m // 8 + cores - 1) // cores) as step:
            block = step * cores + core
            with inst.if_scope(block < physical_m // 8):
                inst.vector_dup(8, scales, 1e-12, 1, 1, 8)
                inst.vector_dup(8, sums, 0.0, 1, 1, 8)
                with inst.for_range(0, 2) as half_block:
                    with inst.for_range(0, 4) as local_row:
                        scalar_row = half_block * 4 + local_row
                        row = block * 8 + scalar_row
                        with inst.if_scope(row < rows):
                            # One GM load, not absmax pass plus quantization pass.
                            inst.data_move(half, xg[row * columns], 0, 1, columns // 16, 0, 0)
                            if gelu:
                                # Same high-precision tanh GELU used by CANN 8.5
                                # on Ascend310P (ops_legacy/dynamic/gelu.py).
                                # Keep the FP16 GELU boundary before FP16 Mul.
                                for start in range(0, columns, 8192):
                                    inst.vconv(64, "", values[start], half[start], 128, 1, 1, 8, 4)
                                    inst.vmul(
                                        64,
                                        scratch[start],
                                        values[start],
                                        values[start],
                                        128,
                                        1,
                                        1,
                                        1,
                                        8,
                                        8,
                                        8,
                                    )
                                    inst.vmul(
                                        64,
                                        scratch[start],
                                        scratch[start],
                                        values[start],
                                        128,
                                        1,
                                        1,
                                        1,
                                        8,
                                        8,
                                        8,
                                    )
                                    inst.vmuls(64, scratch[start], scratch[start], 0.044715, 128, 1, 1, 8, 8)
                                    inst.vadd(
                                        64,
                                        scratch[start],
                                        scratch[start],
                                        values[start],
                                        128,
                                        1,
                                        1,
                                        1,
                                        8,
                                        8,
                                        8,
                                    )
                                    inst.vmuls(
                                        64, scratch[start], scratch[start], -1.5957691, 128, 1, 1, 8, 8
                                    )
                                    inst.vexp(64, scratch[start], scratch[start], 128, 1, 1, 8, 8)
                                    inst.vadds(64, scratch[start], scratch[start], 1.0, 128, 1, 1, 8, 8)
                                    inst.vdiv(
                                        64,
                                        values[start],
                                        values[start],
                                        scratch[start],
                                        128,
                                        1,
                                        1,
                                        1,
                                        8,
                                        8,
                                        8,
                                    )
                                    inst.vconv(64, "", half[start], values[start], 128, 1, 1, 4, 8)
                            if otherg is not None:
                                # Preserve the original FP16 Mul rounding, then
                                # apply exactly the existing FP32 A8 arithmetic.
                                other_half = scratch.reinterpret_cast_to("float16")
                                inst.data_move(other_half, otherg[row * columns], 0, 1, columns // 16, 0, 0)
                                inst.vmul(128, half, half, other_half, columns // 128, 1, 1, 1, 8, 8, 8)
                            bound.set_as(0.0)
                            for start in range(0, columns, 8192):
                                inst.vconv(64, "", values[start], half[start], 128, 1, 1, 8, 4)
                                inst.vabs(64, scratch[start], values[start], 128, 1, 1, 8, 8)
                                inst.vec_reduce_max(64, result, scratch[start], work, 128, 8)
                                part.set_as(result[0])
                                with inst.if_scope(part > bound):
                                    bound.set_as(part)
                            result[0].set_as(bound)
                            inst.vmuls(1, result, result, 1.0 / 127.0, 1, 1, 1, 8, 8)
                            inst.vmaxs(1, result, result, 1e-12, 1, 1, 1, 8, 8)
                            scale.set_as(result[0])
                            scales[scalar_row].set_as(scale)
                            total.set_as(0.0)
                            for start in range(0, columns, 8192):
                                inst.vector_dup(64, scratch[start], scale, 128, 1, 8)
                                inst.vdiv(
                                    64, values[start], values[start], scratch[start], 128, 1, 1, 1, 8, 8, 8
                                )
                                inst.vconv(64, "round", integer[start], values[start], 128, 1, 1, 8, 8)
                                inst.vconv(64, "", values[start], integer[start], 128, 1, 1, 8, 8)
                                inst.vmaxs(64, values[start], values[start], -127.0, 128, 1, 1, 8, 8)
                                inst.vmins(64, values[start], values[start], 127.0, 128, 1, 1, 8, 8)
                                inst.vec_reduce_add(64, result, values[start], work, 128, 8)
                                part.set_as(result[0])
                                total.set_as(total + part)
                                inst.vconv(64, "", half[start], values[start], 128, 1, 1, 4, 8)
                            # Scratch no longer contains live denominators/codes.
                            inst.vconv(128, "", byte, half, columns // 128, 1, 1, 4, 8)
                            sums[scalar_row].set_as(total)
                        with inst.else_scope():
                            inst.vector_dup(128, half, 0.0, columns // 128, 1, 8)
                            inst.vconv(128, "", byte, half, columns // 128, 1, 1, 4, 8)
                        inst.data_move(packed[local_row * 32], byte, 0, columns // 32, 1, 0, 3)
                    inst.data_move(
                        ag[(block * 8 + half_block * 4) * 32], packed, 0, columns // 32, 4, 0, physical_m - 4
                    )
                inst.data_move(rg[block * 8], sums, 0, 1, 1, 0, 0)
                inst.data_move(sg[block * 8], scales, 0, 1, 1, 0, 0)
    inst.BuildCCE(
        kernel_name=kernel_name, inputs=[xg, otherg] if otherg is not None else [xg], outputs=[ag, rg, sg]
    )
    return inst
