"""Compile an explicit ONNX graph to OM with the installed CANN compiler."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

try:
    import onnx
except ModuleNotFoundError:
    onnx = None

from quantization.include.definitions import DEFAULT_SOC
from quantization.om.environment import cann_environment


def static_input_shapes(source: Path) -> str:
    """Read static ONNX runtime-input shapes without loading external weights.

    Args:
        source: ONNX graph header.
    Returns:
        ATC input_shape string. Dynamic or empty shapes raise ValueError.
    """
    if onnx is None:
        raise RuntimeError("Install quantization/requirements.txt before OM compilation")
    graph = onnx.load(str(source), load_external_data=False).graph
    constants = {tensor.name for tensor in graph.initializer}
    shapes = []
    for value in graph.input:
        if value.name in constants:
            continue
        dimensions = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if not dimensions or any(size <= 0 for size in dimensions):
            raise ValueError(f"Expected static dimensions for {value.name}")
        shapes.append(f"{value.name}:{','.join(map(str, dimensions))}")
    if not shapes:
        raise ValueError("ONNX graph has no runtime inputs")
    return ";".join(shapes)


def compile_model(source: Path, output: Path, *, part: int, soc: str = DEFAULT_SOC) -> dict:
    """Compile one explicit ONNX graph to a new OM, retaining logs on failure.

    Args:
        source: ONNX model; its external tensor files stay beside the graph.
        output: New OM filename. Existing model/log/metadata are refused.
        part: 1 for prefix or 2 for one action denoising step.
        soc: Target SoC supported by the installed CANN.
    Returns:
        ATC command and result metadata; original inputs are never changed.
    """
    source, output = source.resolve(), output.resolve()
    if part not in (1, 2):
        raise ValueError("part must be 1 or 2")
    if not source.is_file():
        raise FileNotFoundError(source)
    prefix = output.with_suffix("") if output.suffix == ".om" else output
    produced, log = Path(f"{prefix}.om"), Path(f"{prefix}.compile.log")
    metadata_path = Path(f"{prefix}.compile.json")
    for artifact in (produced, log, metadata_path):
        if artifact.exists():
            raise FileExistsError(f"Choose a new OM output: {artifact}")
    command = [
        "atc",
        "--framework=5",
        f"--model={source}",
        f"--output={prefix}",
        f"--soc_version={soc}",
        "--input_format=ND",
        f"--input_shape={static_input_shapes(source)}",
        "--precision_mode_v2=origin",
        "--log=info",
    ]
    if part == 1:
        command.append("--op_select_implmode=high_performance_for_all")
    environment = None if shutil.which("atc") else cann_environment()
    produced.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "part": part,
        "command": command,
        "output": str(produced),
        "log": str(log),
        "status": "running",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(shlex.join(command), flush=True)
    print(f"Compiler log: {log}", flush=True)
    try:
        with log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        metadata["returncode"] = completed.returncode
        completed.check_returncode()
        if not produced.exists():
            variants = list(prefix.parent.glob(prefix.name + "*linux_aarch64.om"))
            if len(variants) != 1:
                raise RuntimeError(f"ATC succeeded but OM output is missing or ambiguous: {produced}")
            variants[0].rename(produced)
        metadata["status"] = "ok"
    except BaseException as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ONNX input, OM output, partition, and target SoC options.

    Args:
        argv: Explicit CLI tokens, or None to read the process command line.
    Returns:
        Validated compiler options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Explicit ONNX filename")
    parser.add_argument("--output", type=Path, required=True, help="New OM filename")
    parser.add_argument("--soc", default=DEFAULT_SOC)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Compile the requested ONNX graph and print the result metadata.

    Args:
        argv: Optional explicit CLI tokens; None reads the process command line.
    Returns:
        None; writes the OM and compiler sidecars.
    """
    args = parse_args(argv)
    metadata = compile_model(args.input, args.output, part=args.part, soc=args.soc)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
