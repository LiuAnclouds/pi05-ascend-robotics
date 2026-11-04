"""Export prepared Pi0.5 tensors through PyTorch tracing and ONNX adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from include.project_paths import DEFAULT_WEIGHTS

from quantization.onnx.precision import add_serial_euler_state, finalize_export
from quantization.pytorch.conversion import export_part


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return user-facing model, prepared-input, precision and output options.

    Args:
        argv: Explicit CLI tokens, or None for the process command line.
    Returns:
        Validated options; internal operator tuning stays in include/definitions.py.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--part",
        type=int,
        choices=(1, 2),
        required=True,
        help="1: vision/language prefix; 2: one action denoising step",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Checkpoint directory containing model.safetensors",
    )
    parser.add_argument("--input", type=Path, required=True, help="Prepared .pt tensors for this part")
    parser.add_argument("--output", type=Path, required=True, help="New ONNX filename")
    parser.add_argument(
        "--precision",
        choices=("fp16", "w8a8"),
        required=True,
        help="Vision/language Linear precision; action precision is selected separately",
    )
    parser.add_argument(
        "--action-precision",
        choices=("fp16", "w8a8"),
        default="fp16",
        help="Action Linear precision, including Part2; w8a8 must be requested explicitly",
    )
    parser.add_argument(
        "--serial-euler-state",
        action="store_true",
        help="Also export a Part2 graph whose FP16 Euler output stays on the device between steps",
    )
    args = parser.parse_args(argv)
    if args.precision == "fp16" and args.action_precision != "fp16":
        parser.error("--precision fp16 requires --action-precision fp16")
    if args.serial_euler_state and args.part != 2:
        parser.error("--serial-euler-state is available for Part2 only")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run only the requested conversion and explicit ONNX normalization fixes.

    Args:
        argv: Explicit CLI tokens, or None for the process command line.
    Returns:
        None. A new graph, external tensors and a conversion sidecar are written.
    """
    args = parse_args(argv)
    selection = {
        "vision_language_linears": args.precision,
        "action_linears": args.action_precision,
        "exported_part_linears": args.precision if args.part == 1 else args.action_precision,
    }
    labels = {"fp16": "W16A16 (FP16)", "w8a8": "W8 + dynamic A8, per-output-channel weights"}
    print(f"Vision/language Linear: {labels[args.precision]}", flush=True)
    print(f"Action Linear: {labels[args.action_precision]}", flush=True)
    print(f"Exporting Part{args.part}: {labels[selection['exported_part_linears']]}", flush=True)
    print("Other operators retain their exported dtypes; this is not an all-INT8 graph.", flush=True)
    metadata = export_part(
        args.weights, args.input, args.output, args.part, args.precision, args.action_precision
    )
    metadata["precision_selection"] = selection
    # This is the ONNX stage. The PyTorch stage above does not import ONNX repairs.
    metadata["onnx_adaptation"] = finalize_export(args.output)
    if args.serial_euler_state:
        serial_path = args.output.with_name(args.output.stem + "_serial_euler.onnx")
        metadata["serial_euler_graph"] = add_serial_euler_state(args.output, serial_path)
    args.output.with_suffix(".export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
