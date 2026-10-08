#!/usr/bin/env python3
"""Compare the official PyTorch Part2 reference with one Ascend OM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from include.project_paths import DEFAULT_PART2_INPUT, DEFAULT_WEIGHTS
from include.runtime_defs import DEFAULT_PART2_OM


def main() -> None:
    """Compare PyTorch Part2 output with OM output and record metrics.

    Returns:
        None. Cosine similarity, errors, finiteness, and latency are written.
    """
    parser = argparse.ArgumentParser(
        description="Compare the PyTorch Part2 reference with one Ascend OM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Checkpoint directory containing model.safetensors.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PART2_INPUT,
                        help="Prepared Part2 tensor file (.pt).")
    parser.add_argument("--om", type=Path, default=DEFAULT_PART2_OM,
                        help="Part2 OM file to validate.")
    parser.add_argument("--output", type=Path,
                        help="JSON report; defaults to <OM name>.validation.json beside the OM.")
    parser.add_argument("--runs", type=int, default=10,
                        help="Measured OM executions after one warm-up run.")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.om.with_suffix(".validation.json")
    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "openpi" / "src"))
    sys.path.insert(0, str(root / "runtime" / "acllite"))
    sys.path.insert(0, str(root / "runtime" / "acllite" / "acllite"))
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_model
    from export.graphs import DenoiseGraph
    from export.model_utils import patch_ascend_rotary
    from acllite.acllite_resource import AclLiteResource
    from acllite.acllite_model import AclLiteModel

    model = PI0Pytorch(Pi0Config(pi05=True, action_dim=32, action_horizon=50, dtype="float32", pytorch_compile_mode=None))
    load_model(model, str(args.weights / "model.safetensors"))
    model.half()
    patch_ascend_rotary(model)
    model.eval()

    sample = torch.load(args.input, map_location="cpu", weights_only=False)
    past_kv = sample["past_kv"].to(torch.float16)
    prefix_pad_masks = sample["prefix_pad_masks"].bool()
    noise = sample["noise"].to(torch.float16)
    timestep = sample["timestep"].to(torch.float16)
    graph = DenoiseGraph(model).eval()
    with torch.inference_mode():
        reference = graph(past_kv, prefix_pad_masks, noise, timestep).float().cpu().numpy()

    resource = AclLiteResource()
    resource.init()
    runtime = AclLiteModel(str(args.om))
    inputs = [
        np.ascontiguousarray(past_kv.numpy()),
        np.ascontiguousarray(prefix_pad_masks.numpy()),
        np.ascontiguousarray(noise.numpy()),
        np.ascontiguousarray(timestep.numpy()),
    ]
    runtime.execute(inputs)  # warmup
    timings = []
    output = None
    for _ in range(max(1, args.runs)):
        started = time.perf_counter()
        result = runtime.execute(inputs)
        timings.append((time.perf_counter() - started) * 1000.0)
        output = np.asarray(result[0]).astype(np.float32, copy=False)

    flat_ref = reference.reshape(-1)
    flat_out = output.reshape(-1)
    cosine = float(flat_ref @ flat_out / (np.linalg.norm(flat_ref) * np.linalg.norm(flat_out) + 1e-12))
    diff = np.abs(reference - output)
    metrics = {
        "status": "ok" if np.isfinite(output).all() else "non_finite",
        "weights": str(args.weights),
        "input": str(args.input),
        "om": str(args.om),
        "reference_dtype": "float32_from_fp16_model",
        "om_output_dtype": str(result[0].dtype),
        "shape": list(output.shape),
        "finite": bool(np.isfinite(output).all()),
        "cosine": cosine,
        "rmse": float(np.sqrt(np.mean((reference - output) ** 2))),
        "max_abs_error": float(diff.max()),
        "latency_ms": {
            "mean": float(np.mean(timings)),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "runs": len(timings),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
