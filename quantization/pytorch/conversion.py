"""PyTorch preparation/tracing; writes an explicit ONNX artifact for the next stage."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from quantization.include.definitions import (
    DEFAULT_CHANNEL_CHUNK,
    DEFAULT_ONNX_OPSET,
    DEFAULT_PART2_ONNX_OPSET,
    PART1_INPUT_NAMES,
    PART2_INPUT_NAMES,
)
from quantization.pytorch.graphs import DenoiseGraph, ManualPrefixGraph
from quantization.pytorch.model_utils import build_pi05_model
from quantization.pytorch.official_linears import install


def prepare_model(weights: Path, precision: str, action_precision: str) -> tuple[torch.nn.Module, dict]:
    """Load source weights and select the existing floating or official W8A8 policy.

    Args:
        weights: Directory containing the source model.safetensors.
        precision: ``fp16`` or symmetric per-output-channel ``w8a8`` Linear export.
        action_precision: ``fp16`` to retain original action weights, otherwise ``w8a8``.

    Returns:
        Evaluation model and quantization metadata. No checkpoint is written or trained.
    """
    if precision not in ("fp16", "w8a8") or action_precision not in ("fp16", "w8a8"):
        raise ValueError("Unsupported precision selection")
    if precision == "fp16" and action_precision != "fp16":
        raise ValueError("An FP16 export requires FP16 action precision")
    model = build_pi05_model(weights, "float32")
    metadata = {"precision": precision, "action_precision": action_precision, "training": False}
    if precision == "w8a8":
        layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)}
        originals = {name + ".weight": layer.weight.detach().float().cpu() for name, layer in layers.items()}
        engine = SimpleNamespace(layers=layers, originals=originals)
        metadata["linears"] = install(engine, DEFAULT_CHANNEL_CHUNK, skip_action=action_precision == "fp16")
    model.half().eval()
    return model, metadata


def export_part(
    weights: Path,
    inputs: Path,
    output: Path,
    part: int,
    precision: str = "fp16",
    action_precision: str = "fp16",
) -> dict:
    """Trace one prepared part into a new ONNX file and record the exact policy.

    Args:
        weights: Source checkpoint directory.
        inputs: Prepared .pt input from prepare_input or prepare_part2_input.
        output: New ONNX filename; existing graph/sidecar outputs are refused.
        part: 1 for vision/language prefix, 2 for one action denoising step.
        precision: FP16 or official per-channel W8 plus dynamic A8.
        action_precision: Original FP16 action or the same W8A8 policy.

    Returns:
        Export metadata. The graph is a conversion artifact, not proof of native device execution.
    """
    if part not in (1, 2):
        raise ValueError("part must be 1 or 2")
    if output.exists() or output.with_suffix(".export.json").exists():
        raise FileExistsError(f"Choose a new ONNX output: {output}")
    model, policy = prepare_model(weights, precision, action_precision)
    sample = torch.load(inputs, map_location="cpu", weights_only=False)
    if part == 1:
        names = PART1_INPUT_NAMES
        values = tuple(
            sample[name].to(
                torch.float16 if name.startswith("image") else torch.int64 if name == "tokens" else torch.bool
            )
            for name in names
        )
        graph, output_names, opset = (
            ManualPrefixGraph(model).eval(),
            ("past_kv_tensor", "prefix_pad_masks"),
            DEFAULT_ONNX_OPSET,
        )
    else:
        names = PART2_INPUT_NAMES
        values = (
            sample["past_kv"].half(),
            sample["prefix_pad_masks"].bool(),
            sample["noise"].half(),
            sample["timestep"].half(),
        )
        graph, output_names, opset = DenoiseGraph(model).eval(), ("velocity",), DEFAULT_PART2_ONNX_OPSET
    with torch.inference_mode():
        result = graph(*values)
        reference = result[0] if isinstance(result, tuple) else result
        if not torch.isfinite(reference).all():
            raise RuntimeError("Export reference is non-finite")
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        torch.onnx.export(
            graph,
            values,
            str(output),
            opset_version=opset,
            input_names=list(names),
            output_names=list(output_names),
            dynamo=False,
            do_constant_folding=True,
            external_data=True,
        )
    metadata = {
        "status": "exported",
        "part": part,
        "onnx": str(output),
        "weights": str(weights),
        "input": str(inputs),
        "policy": policy,
        "opset": opset,
        "reference_shape": list(reference.shape),
        "reference_finite": True,
        "export_ms": (time.perf_counter() - started) * 1000,
        "native_integer_execution_verified": False,
    }
    output.with_suffix(".export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
