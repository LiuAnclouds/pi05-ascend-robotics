#!/usr/bin/env python3
"""Export the official OpenPI Pi0.5 prefix/VLM graph to ONNX."""

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
    DEFAULT_ONNX_OPSET,
    DEFAULT_PART1_ONNX_DIR,
    DEFAULT_PART1_INPUT,
    DEFAULT_REPORT_DIR,
    DEFAULT_WEIGHTS,
)
def parse_args() -> argparse.Namespace:
    """Parse the small public Part1 ONNX export interface.

    Returns:
        Parsed checkpoint, input, output, metrics, and precision options.
    """
    parser = argparse.ArgumentParser(
        description="Export the official Pi0.5 Part1 prefix graph to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Checkpoint directory containing model.safetensors.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PART1_INPUT,
                        help="Prepared Part1 tensor file (.pt).")
    parser.add_argument("--output", type=Path, default=DEFAULT_PART1_ONNX_DIR / "prefix_part1.onnx",
                        help="Destination ONNX file.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_REPORT_DIR / "official_part1_export.json",
                        help="JSON file for export metadata and finiteness checks.")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32",
                        help="Arithmetic dtype used while tracing the graph.")
    return parser.parse_args()


def main() -> None:
    """Export the official Part1 prefix graph and write export metrics.

    Returns:
        None. ONNX output and a JSON metrics file are created.
    """
    args = parse_args()
    import numpy as np
    import torch
    from export.graphs import PrefixGraph
    from export.model_utils import build_pi05_model
    model = build_pi05_model(args.weights, args.dtype)
    sample = torch.load(args.input, map_location="cpu", weights_only=False)
    export_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    values = (
        sample["image0"].to(export_dtype), sample["image1"].to(export_dtype), sample["image2"].to(export_dtype),
        sample["mask0"].bool(), sample["mask1"].bool(), sample["mask2"].bool(),
        sample["tokens"].long(), sample["token_mask"].bool(),
    )
    graph = PrefixGraph(model).eval()
    with torch.inference_mode():
        reference = graph(*values)
    flat_ref = reference[0].float().numpy()
    if not np.isfinite(flat_ref).all():
        raise RuntimeError("official PyTorch Part1 output is non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.onnx.export(
        graph, values, str(args.output), opset_version=DEFAULT_ONNX_OPSET,
        input_names=["image0", "image1", "image2", "mask0", "mask1", "mask2", "tokens", "token_mask"],
        output_names=["past_kv_tensor", "prefix_pad_masks"], dynamo=False,
        do_constant_folding=True, external_data=True,
    )
    metrics = {"framework": "official_openpi", "model": "pi05", "checkpoint": str(args.weights),
               "onnx": str(args.output), "opset": DEFAULT_ONNX_OPSET, "dtype": args.dtype,
               "exporter": "legacy",
               "export_ms": (time.perf_counter() - started) * 1000.0,
               "pytorch_finite": True, "pytorch_output_shape": list(flat_ref.shape)}
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
