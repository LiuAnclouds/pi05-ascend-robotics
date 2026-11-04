#!/usr/bin/env bash
# Install only the quantization/export environment; ROS2 uses its system Python.
set -e
pi05_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pi05_conda_env="${PI05_CONDA_ENV:-pi05}"
if [ -n "${PI05_CONDA_ROOT:-}" ]; then
    source "$PI05_CONDA_ROOT/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    printf 'Conda not found. Activate Conda or set PI05_CONDA_ROOT.\n' >&2
    exit 1
fi
if ! conda activate "$pi05_conda_env" 2>/dev/null; then
  conda create -y -n "$pi05_conda_env" python=3.10 pip
  conda activate "$pi05_conda_env"
fi
PYTHONPATH= python -m pip install -r "$pi05_project_dir/requirements.txt"
# OpenPI requires its Gemma/SigLIP replacements. Apply only inside this env.
python -m pip install --no-deps -e "$pi05_project_dir"
python "$pi05_project_dir/quantization/pytorch/install_transformers.py"
# Check this Conda environment separately from CANN's external package metadata.
PYTHONPATH= python -m pip check
python - <<'PY'
import cv2, numpy, onnx, torch, transformers
print('Core imports OK:', cv2.__version__, numpy.__version__, onnx.__version__, torch.__version__, transformers.__version__)
PY
printf '\nEnvironment ready: %s\n' "$pi05_conda_env"
