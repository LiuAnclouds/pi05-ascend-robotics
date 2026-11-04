#!/usr/bin/env bash
# Configure host SocketCAN only; no ROS, CANN, motor enable or motion commands.
set -euo pipefail

pi05_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    printf 'Usage: sudo bash activate_can.sh [name] [--source existing_interface]\n'
    printf 'Default: can0 at 1 Mbps. Run on the host before starting Docker.\n'
    printf 'A unique CAN device may be renamed; ambiguous selection requires --source.\n'
    printf 'Example: sudo bash activate_can.sh can0 --source can1\n'
    printf 'An already configured interface is left unchanged. No motor or motion commands.\n'
    exit 0
fi
if [ "$(uname -s)" != Linux ]; then
    printf 'SocketCAN activation requires the Linux deployment host.\n' >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    printf 'python3 is required; no ROS2, CANN or Piper SDK is needed for CAN setup.\n' >&2
    exit 1
fi
exec python3 "$pi05_project_dir/inference/core/motion/can_interface.py" "$@"
