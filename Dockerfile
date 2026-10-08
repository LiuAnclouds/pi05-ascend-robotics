# Build on the ARM64 deployment host. CANN and the driver are mounted at runtime.
ARG BASE_IMAGE=python:3.10-slim-bookworm
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PI05_CONTAINER=1 \
    PI05_CANN_ROOT=/usr/local/Ascend/cann-8.5.0 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gcc g++ make iproute2 procps tini \
        libgomp1 libglib2.0-0 libnuma1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY . .
RUN python include/install_transformers.py \
    && python -c "from export.model_utils import add_official_source; add_official_source(); from openpi.models.pi0_config import Pi0Config; from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; import cv2, onnxruntime, piper_sdk; print('Export and runtime imports OK')"

ENV ASCEND_GLOBAL_LOG_LEVEL=3
LABEL org.opencontainers.image.title="pi05-ascend-robotics" \
      org.opencontainers.image.source="https://github.com/LiuAnclouds/pi05-ascend-robotics" \
      org.opencontainers.image.description="OpenPI pi05 export and Piper inference on Ascend310P1" \
      org.opencontainers.image.licenses="Apache-2.0"
STOPSIGNAL SIGINT
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "bash", "/app/include/docker_entrypoint.sh"]
CMD ["python", "runtime/realtime_inference.py", "--help"]
