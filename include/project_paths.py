"""Project-relative defaults; users select runtime paths in YAML and export paths in CLI."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "config"
MODELS_ROOT = CONFIG_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_WEIGHTS = MODELS_ROOT / "weights"
DEFAULT_TOKENIZER = MODELS_ROOT / "tokenizer.model"
DEFAULT_STATS = CONFIG_ROOT / "norm_stats.json"
DEFAULT_PART1_INPUT = DATA_ROOT / "part1.pt"
DEFAULT_PART2_INPUT = DATA_ROOT / "part2.pt"
OPENPI_SOURCE = PROJECT_ROOT / "third_party/openpi/src"
