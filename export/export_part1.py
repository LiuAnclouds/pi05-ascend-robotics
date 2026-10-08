#!/usr/bin/env python3
"""Export the verified explicit-layer Part1 graph."""

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
    DEFAULT_WEIGHTS,
)
def main() -> None:
    """Export the verified Part1 graph and save metrics beside its ONNX file.

    Returns:
        None. ONNX output and a JSON metrics file are created.
    """
    parser = argparse.ArgumentParser(
        description="Export the verified explicit-layer Part1 graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Checkpoint directory containing model.safetensors.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PART1_INPUT,
                        help="Prepared Part1 tensor file (.pt).")
    parser.add_argument("--output", type=Path, default=DEFAULT_PART1_ONNX_DIR / "1.onnx",
                        help="Destination ONNX file.")
    args = parser.parse_args()
    import numpy as np
    import torch
    from export.graphs import ManualPrefixGraph
    from export.model_utils import build_pi05_model
    model = build_pi05_model(args.weights, "float16")
    sample = torch.load(args.input, map_location="cpu", weights_only=False)
    values = (sample["image0"].half(), sample["image1"].half(), sample["image2"].half(),
              sample["mask0"].bool(), sample["mask1"].bool(), sample["mask2"].bool(),
              sample["tokens"].long(), sample["token_mask"].bool())
    graph = ManualPrefixGraph(model).eval()
    with torch.inference_mode():
        reference = graph(*values)[0].float().numpy()
    if not np.isfinite(reference).all():
        raise RuntimeError("official PyTorch Part1 output is non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.onnx.export(graph, values, str(args.output), opset_version=DEFAULT_ONNX_OPSET,
                      input_names=["image0", "image1", "image2", "mask0", "mask1", "mask2", "tokens", "token_mask"],
                      output_names=["past_kv_tensor", "prefix_pad_masks"], dynamo=False,
                      do_constant_folding=True, external_data=True)
    metrics = {"status": "ok", "onnx": str(args.output), "exporter": "legacy_manual_layers",
               "opset": DEFAULT_ONNX_OPSET, "export_ms": (time.perf_counter() - started) * 1000.0,
               "reference_shape": list(reference.shape), "reference_finite": True}
    args.output.with_suffix(".export.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
