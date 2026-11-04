"""Quantization/export definitions; model dimensions are not user options."""

from __future__ import annotations

import sys

from include.project_paths import OPENPI_SOURCE

DEFAULT_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 50
DEFAULT_ONNX_OPSET = 17
DEFAULT_PART2_ONNX_OPSET = 14
DEFAULT_SOC = "Ascend310P1"
DEFAULT_WEIGHT_BITS = 8
DEFAULT_CHANNEL_CHUNK = 4096
DEFAULT_RTC_MAX_GUIDANCE = 5.0
PART1_INPUT_NAMES = (
    "image0",
    "image1",
    "image2",
    "mask0",
    "mask1",
    "mask2",
    "tokens",
    "token_mask",
)
PART2_INPUT_NAMES = ("past_kv_tensor", "prefix_pad_masks", "noise", "timestep")


def add_official_source() -> None:
    """Add the retained Ascend-patched OpenPI source directory to ``sys.path``.

    Args:
        None.

    Returns:
        None. The process import path is updated in place.
    """
    source = str(OPENPI_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
