#!/usr/bin/env python3
"""Compile either verified static ONNX part with CANN ATC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from include.project_paths import DEFAULT_PART1_ONNX_DIR, DEFAULT_PART2_ONNX_DIR


def main() -> None:
    """Read part/input/output CLI options; produce an OM and compiler log.

    Input shapes come from the static ONNX graph. Precision remains origin;
    the Part1 implementation preference follows the measured fast candidate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', type=int, choices=(1, 2), required=True,
                        help='1: vision/language prefix; 2: one denoising step.')
    parser.add_argument('--input', type=Path, help='ONNX file; defaults to the exported part.')
    parser.add_argument('--output', type=Path, help='OM file; defaults to outputs/om/<part>.om.')
    parser.add_argument('--soc', default='Ascend310P1', help='Target SoC (default: Ascend310P1).')
    args = parser.parse_args()
    source = args.input or (DEFAULT_PART1_ONNX_DIR / '1.onnx'
                           if args.part == 1 else DEFAULT_PART2_ONNX_DIR / '2.onnx')
    output = args.output or ROOT / f'outputs/om/{args.part}.om'
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix('') if output.suffix == '.om' else output
    produced = Path(f'{prefix}.om')
    if produced.exists():
        parser.error(f'Output already exists: {produced}. Choose --output for a candidate.')
    import onnx
    graph = onnx.load(str(source), load_external_data=False).graph
    shapes = []
    for value in graph.input:
        dimensions = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if not dimensions or any(d <= 0 for d in dimensions):
            parser.error(f'Expected static dimensions for {value.name}')
        shapes.append(f"{value.name}:{','.join(map(str, dimensions))}")
    command = ['atc', '--framework=5', f'--model={source}', f'--output={prefix}',
               f'--soc_version={args.soc}', '--input_format=ND',
               f"--input_shape={';'.join(shapes)}", '--precision_mode_v2=origin', '--log=info']
    if args.part == 1:
        command.append('--op_select_implmode=high_performance_for_all')
    log = Path(f'{prefix}.compile.log')
    print(f'======== Model Module | Part {args.part} ========', flush=True)
    print('Precision: preserve ONNX dtypes (origin)', flush=True)
    print(shlex.join(command), flush=True)
    print(f'Compiler log: {log}', flush=True)
    metadata_path = Path(f'{prefix}.compile.json')
    metadata = {'part': args.part, 'command': command, 'output': str(produced),
                'log': str(log), 'status': 'running'}
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        with log.open('w') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        metadata['returncode'] = completed.returncode
        completed.check_returncode()
        # Some CANN releases suffix cross-compilation output with the target host.
        if not produced.exists():
            variants = list(prefix.parent.glob(prefix.name + '*linux_aarch64.om'))
            if len(variants) != 1:
                raise RuntimeError(f'ATC returned success but expected OM was not found: {produced}')
            variants[0].rename(produced)
        metadata['status'] = 'ok'
    except Exception as error:
        metadata.update(status='failed', error=str(error))
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'OM ready: {produced}', flush=True)


if __name__ == '__main__':
    main()
