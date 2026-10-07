"""Export-side definitions: official source path and graph dimensions."""

from __future__ import annotations

import sys
from include.project_paths import (
    DEFAULT_PART1_INPUT,
    DEFAULT_PART2_INPUT,
    DEFAULT_REPORT_DIR,
    DEFAULT_WEIGHTS,
    DEFAULT_ONNX_DIR,
    DEFAULT_PART1_ONNX_DIR,
    DEFAULT_PART1_MANUAL_ONNX_DIR,
    DEFAULT_PART2_ONNX_DIR,
    PROJECT_ROOT,
)

OPENPI_SOURCE = PROJECT_ROOT / "openpi" / "src"
DEFAULT_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 50
DEFAULT_ONNX_OPSET = 17
DEFAULT_PART2_ONNX_OPSET = 14


def add_official_source() -> None:
    """Add the untouched official OpenPI source directory to ``sys.path``.

    Args:
        None.

    Returns:
        None. The process import path is updated in place.
    """
    source = str(OPENPI_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
