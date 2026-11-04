"""Adapt compiled RTC endpoint errors to executable reference intervals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnx
from onnx import helper


def adapt_reference_bounds(source: Path, output: Path) -> dict:
    """Replace the exact-target residual with distance to a reference interval.

    Args:
        source: Full RTC ONNX or a partition containing `prefix - endpoint`.
        output: New graph header; external weights remain in their source files.
    Returns:
        Changed node and input names. Writes a graph with FP32 prefix_lower and
        prefix_upper in place of prefix; all other graph contracts stay intact.

    Runtime supplies equal bounds for exact targets. Gripper references at the
    actuator boundaries become half-infinite intervals. The residual is the
    negative derivative of half the squared distance to that interval, so the
    existing endpoint VJP and its shared forward need no derivative changes.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    model = onnx.load(str(source), load_external_data=False)
    residuals = [node for node in model.graph.node if node.op_type == "Sub" and node.input[0] == "prefix"]
    if len(residuals) != 1:
        raise ValueError("Expected exactly one RTC prefix-minus-endpoint residual")
    if output.exists():
        raise FileExistsError(output)
    residual = residuals[0]
    endpoint = residual.input[1]
    replacements = [
        helper.make_node("Max", [endpoint, "prefix_lower"], ["rtc_reference_above_lower"], name="RTC_ReferenceLower", domain=residual.domain),
        helper.make_node("Min", ["rtc_reference_above_lower", "prefix_upper"], ["rtc_executable_reference"], name="RTC_ReferenceUpper", domain=residual.domain),
        helper.make_node("Sub", ["rtc_executable_reference", endpoint], list(residual.output), name=residual.name, domain=residual.domain),
    ]
    nodes = []
    for node in model.graph.node:
        nodes.extend(replacements if node == residual else [node])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    inputs = []
    for value in model.graph.input:
        if value.name == "prefix":
            for name in ("prefix_lower", "prefix_upper"):
                replacement = onnx.ValueInfoProto()
                replacement.CopyFrom(value)
                replacement.name = name
                inputs.append(replacement)
        else:
            inputs.append(value)
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = os.path.relpath(source.parent / entry.value, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output))
    return dict(source=str(source), output=str(output), added_nodes=2, reference="executable endpoint interval")


def main(argv: list[str] | None = None) -> None:
    """Read explicit ONNX paths and write one adapted RTC graph header.

    Args: Optional CLI arguments, otherwise process arguments.
    Returns: None; prints the adaptation result without compiling or moving OMs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(adapt_reference_bounds(args.input, args.output))


if __name__ == "__main__":
    main()
