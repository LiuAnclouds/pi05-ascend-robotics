"""Project-relative paths shared by export and runtime entry points.

Keeping these paths in one small module prevents command scripts from
depending on the caller's working directory or on one another's internals.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "config"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_WEIGHTS = MODELS_ROOT / "weights/instrction_9.14_float32"
DEFAULT_TOKENIZER = MODELS_ROOT / "paligemma-3b-pt-224/tokenizer.model"
DEFAULT_STATS = CONFIG_ROOT / "norm_stats.json"
DEFAULT_PART1_INPUT = DATA_ROOT / "part1.pt"
DEFAULT_PART2_INPUT = DATA_ROOT / "part2.pt"
DEFAULT_ONNX_DIR = OUTPUT_ROOT / "onnx"
DEFAULT_PART1_ONNX_DIR = DEFAULT_ONNX_DIR / "1"
DEFAULT_PART2_ONNX_DIR = DEFAULT_ONNX_DIR / "2"
DEFAULT_RUN_DIR = OUTPUT_ROOT / "runs"
