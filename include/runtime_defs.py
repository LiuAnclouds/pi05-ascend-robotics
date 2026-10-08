"""Runtime-side definitions: paths, camera defaults, and stable dimensions."""

from __future__ import annotations

from include.project_paths import (
    DEFAULT_STATS,
    DEFAULT_TOKENIZER,
    OUTPUT_ROOT,
)

DEFAULT_PART1_OM = OUTPUT_ROOT / "om/part1.om"
DEFAULT_PART2_OM = OUTPUT_ROOT / "om/part2.om"
CAMERA_A_DEFAULT = "/dev/video0"
CAMERA_B_DEFAULT = "/dev/video2"
PART1_INPUT_COUNT = 8
PIPER_STATE_DIM = 7
