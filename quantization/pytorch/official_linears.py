"""Native official-CANN W8/dynamic-A8 export; never replaces source weights.

Symmetric per-output-channel W8 is re-quantized from the original weights.
Dynamic A8 follows the installed FP16 DynamicQuant arithmetic. Norm/attention
boundaries and other floating operations are inherited, not called all-INT8.
"""

import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.onnx import symbolic_helper as sh

ACTION_LINEAR_MARKERS = (
    "gemma_expert.",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "time_mlp_in",
    "time_mlp_out",
    "state_proj",
)


def should_skip_linear(name: str, *, skip_action: bool) -> bool:
    """Decide whether a named source Linear stays outside the W8 conversion.

    Args:
        name: Full ``named_modules`` path from the OpenPI Pi0.5 model.
        skip_action: Preserve all action-expert and action-conditioning projections.
    Returns:
        True for unused language heads/terminal block and, when requested, all action layers.
    """
    return (
        name.endswith("lm_head")
        or ".language_model.layers.17.mlp." in name
        or (skip_action and any(marker in name for marker in ACTION_LINEAR_MARKERS))
    )


class OfficialLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, codes, scale, encoded, bias, frozen_shape, chunk_n):
        """Evaluate the existing host analytical reference used during tracing.

        Args: Autograd context, activation, INT8 weight codes, scales, bias,
            static shape and N-axis chunk size; encoded scales are for symbolic export.
        Returns: FP16 reference output; this function does not measure native INT8 speed.
        """
        x = value.half().reshape(-1, value.shape[-1])
        bound = x.abs().amax(-1, keepdim=True).float()
        inverse = torch.where(bound > 1e-12, 127 / bound, torch.zeros_like(bound)).half()
        q = (x * inverse).round().clamp(-128, 127).float()
        hi = (q / 16).round()
        lo = q - 16 * hi
        w = codes.float()
        accum = F.linear(lo, w) + F.linear(hi, w) * 16
        # Match the hardware contract: fused channel dequant emits FP16 first.
        channel = (accum * scale).half()
        y = channel * (bound / 127).half()
        y = y + bias
        return y.reshape(*frozen_shape[:-1], codes.shape[0])

    @staticmethod
    def symbolic(g, value, codes, scale, encoded, bias, frozen_shape, chunk_n):
        """Emit official CANN dynamic quantization and fused channel-scaled GEMM.

        Args: ONNX builder and the same activation/weight/bias/shape arguments.
        Returns: One output tensor; only wide N dimensions are partitioned.
        """
        shape = list(frozen_shape)
        n, k = sh._get_tensor_sizes(codes)
        m = 1
        for dim in shape[:-1]:
            m *= dim
        const = lambda values: g.op("Constant", value_t=torch.tensor(values, dtype=torch.int64))
        half = g.op("Cast", value, to_i=10)
        x = g.op("Reshape", half, const([m, k]))
        a, sa = g.op("NPUDynamicQuant", x, outputs=2, dst_type_i=2)
        a.setType(value.type().with_dtype(torch.int8).with_sizes([m, k]))
        sa.setType(value.type().with_dtype(torch.float32).with_sizes([m]))
        sa = g.op("Reshape", g.op("Cast", sa, to_i=10), const([m, 1]))
        outputs = []
        for start in range(0, n, chunk_n):
            end = min(start + chunk_n, n)
            if start == 0 and end == n:
                w, s, b = codes, encoded, bias
            else:

                def slice0(tensor):
                    """Slice output channels; input tensor, output matching N chunk."""
                    return g.op("Slice", tensor, const([start]), const([end]), const([0]), const([1]))

                w, s, b = slice0(codes), slice0(encoded), slice0(bias)
            # All three trailing optional ports are absent; omit them entirely.
            out = g.op("QuantBatchMatMul", a, w, s, dtype_i=1, transpose_x1_i=0, transpose_x2_i=1)
            out.setType(value.type().with_dtype(torch.float16).with_sizes([m, end - start]))
            out = g.op("Add", g.op("Mul", out, sa), b)
            outputs.append(out)
        y = outputs[0] if len(outputs) == 1 else g.op("Concat", *outputs, axis_i=1)
        result = g.op("Reshape", y, const([*shape[:-1], n]))
        result.setType(value.type().with_dtype(torch.float16).with_sizes([*shape[:-1], n]))
        return result


def install(engine, chunk_n=4096, *, skip_action=False):
    """Install the established official W8/dynamic-A8 Linear policy.

    Args:
        engine: Object exposing layers and original FP32 weight tensors.
        chunk_n: Internal output-channel chunk size, never a K-group split.
        skip_action: Keep original action-expert Linear weights/arithmetic.

    Returns:
        Installation metadata. Source checkpoint tensors are not modified.
    """
    installed, skipped = {}, {}
    with torch.no_grad():
        for name, layer in engine.layers.items():
            if should_skip_linear(name, skip_action=skip_action):
                skipped[name] = (
                    "preserved source action FP16"
                    if (skip_action and any(marker in name for marker in ACTION_LINEAR_MARKERS))
                    else "unused action-inference path"
                )
                continue
            original = engine.originals[name + ".weight"]
            n, k = original.shape
            scale = original.float().abs().amax(-1).div(127).clamp_min(2**-24).half().float()
            codes = (original.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            bits = scale.numpy().view(np.uint32).astype(np.uint64)
            encoded = torch.from_numpy(
                ((bits & np.uint64(0xFFFFE000)) | np.uint64(1 << 46)).view(np.int64).copy()
            )
            bias = torch.zeros(n, dtype=torch.float16) if layer.bias is None else layer.bias.detach().half()
            # Only wide outputs are partitioned, along N, never along K.
            chosen = chunk_n if n >= 8192 and n % chunk_n == 0 else n
            for key, tensor in (("codes", codes), ("scale", scale), ("encoded", encoded), ("bias", bias)):
                layer.register_buffer("_official_" + key, tensor, persistent=False)

            def forward(module, value, chunk=chosen):
                """Trace an installed Linear; input activation, output quantized-policy result."""
                return OfficialLinear.apply(
                    value,
                    module._official_codes,
                    module._official_scale,
                    module._official_encoded,
                    module._official_bias,
                    tuple(int(d) for d in value.shape),
                    chunk,
                )

            layer.forward = types.MethodType(forward, layer)
            installed[name] = dict(
                shape=[n, k], chunk_n=chosen, codes_bytes=codes.numel(), scale_bytes=encoded.numel() * 8
            )
            print("Official CANN Linear", name, n, k, chosen, flush=True)
    return dict(
        installed=installed,
        skipped=skipped,
        core="unmodified built-in QuantBatchMatmulV3 INT8xINT8 with fused per-channel dequant",
        input="unmodified built-in DynamicQuant FP16 per-token A8",
        epilogue="FP16 token scale and bias; official arithmetic, not old exact FP32 QDQ",
        granularity="symmetric per-output-channel W8 re-quantized from original source weights",
        floating="Norm/Softmax/RoPE/QK/PV/residual/Euler and inherited Conv/Embedding execution",
        note="CPU forward is an analytical reference approximation, not certified bit-exact NPU arithmetic",
    )
