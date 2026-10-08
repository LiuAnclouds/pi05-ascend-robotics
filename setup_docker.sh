#!/usr/bin/env bash
# Build the ARM64 image locally; Docker and CANN are installed on the host first.
set -e
pi05_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pi05_image="${PI05_IMAGE:-pi05-ascend-robotics:cann8.5}"
pi05_base_image="${PI05_BASE_IMAGE:-python:3.10-slim-bookworm}"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    printf 'Usage: bash setup_docker.sh\n\nBuild the project image on the ARM64 board.\nPI05_BASE_IMAGE: Python 3.10 ARM64 base image or registry mirror.\nPI05_IMAGE: output tag (default: pi05-ascend-robotics:cann8.5).\n'
    exit 0
fi
if [ "$#" -ne 0 ]; then printf 'Use bash setup_docker.sh --help\n' >&2; exit 2; fi
case "$(uname -m)" in
    aarch64|arm64) ;;
    *) printf 'Build on the ARM64 Ascend board, not an x86/Windows host.\n' >&2; exit 1 ;;
esac
if ! command -v docker >/dev/null 2>&1; then
    printf 'Docker is not installed. Follow the Docker installation step in README.md.\n' >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    printf 'Cannot access Docker. Start the Docker service and run with Docker permissions.\n' >&2
    exit 1
fi
printf '\n======== Build Pi0.5 Docker image ========\nImage: %s\nBase: %s\n' "$pi05_image" "$pi05_base_image"
docker build --network host --build-arg "BASE_IMAGE=$pi05_base_image" -t "$pi05_image" "$pi05_project_dir"
printf '\n======== Image ready ========\n'
docker image inspect "$pi05_image" --format '{{.RepoTags}} | {{.Architecture}} | {{.Id}}'
printf 'Next: prepare the model files in README.md, then run bash run_docker.sh --task "..."\n'
