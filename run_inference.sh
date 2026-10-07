#!/usr/bin/env bash
# Load the board runtime, then forward all arguments to the Python entry.
# Usage: bash run_inference.sh --task "Task instruction" [--send-motion]
# Ctrl+C reaches Python directly so its quick-stop and report cleanup can run.
set -e

pi05_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$pi05_project_dir/include/environment.sh"
cd "$pi05_project_dir"

# Keep required task validation and every inference default in Python.
exec python runtime/realtime_inference.py "$@"
