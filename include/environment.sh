#!/usr/bin/env bash
# Source from a launcher or interactive shell to use the unified environment.
pi05_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pi05_cann_root="${PI05_CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
pi05_conda_root="${PI05_CONDA_ROOT:-$HOME/miniconda3}"
# CANN/Conda initialization references optional variables; preserve the caller's
# nounset setting while allowing these vendor scripts to load.
pi05_restore_nounset=0
case $- in *u*) pi05_restore_nounset=1; set +u ;; esac
source "$pi05_cann_root/set_env.sh"
source "$pi05_conda_root/etc/profile.d/conda.sh"
conda activate Pi05_ascend
export PYTHONPATH="$pi05_project_dir/runtime/acllite:$pi05_project_dir/runtime/acllite/acllite:$pi05_cann_root/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
if [ "$pi05_restore_nounset" = 1 ]; then set -u; fi
unset pi05_restore_nounset
