#!/usr/bin/env python3
"""Export one official OpenPI Pi0.5 action-expert denoising step to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from include.export_defs import (
    DEFAULT_PART2_ONNX_OPSET,
    DEFAULT_PART2_ONNX_DIR,
    DEFAULT_PART2_INPUT,
    DEFAULT_WEIGHTS,
)
def parse_args() -> argparse.Namespace:
    """Parse the small public Part2 ONNX export interface.

    Returns:
        Parsed checkpoint, input, output, and precision options.
    """
    parser = argparse.ArgumentParser(
        description="Export one official Pi0.5 Part2 denoising step to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Checkpoint directory containing model.safetensors.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PART2_INPUT,
                        help="Prepared Part2 tensor file (.pt).")
    parser.add_argument("--output", type=Path, default=DEFAULT_PART2_ONNX_DIR / "2.onnx",
                        help="Destination ONNX file.")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16",
                        help="Arithmetic dtype used while tracing the graph.")
    return parser.parse_args()


def main() -> None:
    """Export one Part2 denoising step and write export metrics.

    Returns:
        None. ONNX output and a JSON metrics file are created.
    """
    args = parse_args()
    import numpy as np
    import torch
    from export.graphs import DenoiseGraph
    from export.model_utils import build_pi05_model
    model = build_pi05_model(args.weights, args.dtype)
    sample = torch.load(args.input, map_location="cpu", weights_only=False)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    values = (sample["past_kv"].to(dtype), sample["prefix_pad_masks"].bool(),
              sample["noise"].to(dtype), sample["timestep"].to(dtype))
    graph = DenoiseGraph(model).eval()
    with torch.inference_mode():
        reference = graph(*values)
    output = reference.float().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("official PyTorch Part2 output is non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.onnx.export(
        graph, values, str(args.output), opset_version=DEFAULT_PART2_ONNX_OPSET,
        input_names=["past_kv_tensor", "prefix_pad_masks", "noise", "timestep"],
        output_names=["velocity"], dynamo=False, do_constant_folding=True, external_data=True,
    )
    metrics = {"framework": "official_openpi", "model": "pi05", "checkpoint": str(args.weights),
               "status": "ok", "dtype": args.dtype, "onnx": str(args.output), "opset": DEFAULT_PART2_ONNX_OPSET,
               "export_ms": (time.perf_counter() - started) * 1000.0, "output_shape": list(output.shape)}
    args.output.with_suffix(".export.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
