#!/usr/bin/env bash
# Run the project image with board NPU, cameras/CAN, and persistent project assets.
# Inference: bash run_docker.sh --task "Task instruction" [--send-motion]
# Export:    bash run_docker.sh python export/compile_om.py --part 1
set -e
pi05_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pi05_cann_root="${PI05_CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
pi05_image="${PI05_IMAGE:-pi05-ascend-robotics:cann8.5}"

if ! docker image inspect "$pi05_image" >/dev/null 2>&1; then
    printf 'Project image unavailable: %s\nRun bash setup_docker.sh first.\n' "$pi05_image" >&2
    exit 1
fi
if [ ! -f "$pi05_cann_root/set_env.sh" ]; then
    printf 'CANN not found: %s\nSet PI05_CANN_ROOT to the installed CANN 8.5 directory.\n' "$pi05_cann_root" >&2
    exit 1
fi
if [ ! -d /usr/local/Ascend/driver ]; then
    printf 'Ascend driver not found. Install the board driver as described in README.md.\n' >&2
    exit 1
fi
# Only the model installer writes assets. Inference/export keep them read-only.
pi05_asset_access=,readonly
if [ "${1:-}" = python ] && [ "${2:-}" = export/download_models.py ]; then
    pi05_asset_access=
fi
mkdir -p "$pi05_project_dir/models" "$pi05_project_dir/data" "$pi05_project_dir/outputs"
pi05_options=(--rm -i --network host --cap-add NET_ADMIN --shm-size 1g --stop-timeout 30
    --mount "type=bind,src=$pi05_cann_root,dst=/usr/local/Ascend/cann-8.5.0,readonly"
    --mount "type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly"
    --mount "type=bind,src=$pi05_project_dir/models,dst=/app/models$pi05_asset_access"
    --mount "type=bind,src=$pi05_project_dir/config,dst=/app/config$pi05_asset_access"
    --mount "type=bind,src=$pi05_project_dir/data,dst=/app/data"
    --mount "type=bind,src=$pi05_project_dir/outputs,dst=/app/outputs")
if [ -t 0 ] && [ -t 1 ]; then pi05_options+=(-t); fi
# OrangePi's SoC driver also uses /dev/upgrade to verify the AICPU kernel package.
for pi05_device in /dev/davinci0 /dev/davinci_manager /dev/devmm_svm /dev/svm0 /dev/hisi_hdc /dev/upgrade /dev/video*; do
    if [ -c "$pi05_device" ]; then pi05_options+=(--device "$pi05_device"); fi
done
for pi05_resource in /etc/ascend_install.info /etc/hdcBasic.cfg /etc/slog.conf /usr/slog /usr/local/dcmi /usr/local/bin/npu-smi; do
    if [ -e "$pi05_resource" ]; then
        pi05_options+=(--mount "type=bind,src=$pi05_resource,dst=$pi05_resource,readonly")
    fi
done
# Option arguments go to inference; an explicit command allows export/compile/CAN setup.
if [ "$#" -eq 0 ]; then set -- --help; fi
case "$1" in -*) set -- python runtime/realtime_inference.py "$@" ;; esac
exec docker run "${pi05_options[@]}" "$pi05_image" "$@"
