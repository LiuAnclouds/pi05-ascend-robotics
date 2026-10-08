#!/usr/bin/env bash
# Install export, compiler dependencies and runtime into a single Conda env.
set -e
pi05_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pi05_conda_root="${PI05_CONDA_ROOT:-$HOME/miniconda3}"
source "$pi05_conda_root/etc/profile.d/conda.sh"
if ! conda activate Pi05_ascend 2>/dev/null; then
  conda create -y -n Pi05_ascend python=3.10 pip
  conda activate Pi05_ascend
fi
PYTHONPATH= python -m pip install -r "$pi05_project_dir/requirements.txt"
# OpenPI requires its Gemma/SigLIP replacements. Apply only inside this env.
python "$pi05_project_dir/include/install_transformers.py"
# Check this Conda environment separately from CANN's external package metadata.
PYTHONPATH= python -m pip check
python - <<'PY'
import cv2, numpy, onnx, torch, transformers
print('Core imports OK:', cv2.__version__, numpy.__version__, onnx.__version__, torch.__version__, transformers.__version__)
PY
printf '\nEnvironment ready: Pi05_ascend\n'
