ARG WORKER_PROFILE=video

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS gpu-base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub \
    TORCH_HOME=/runpod-volume/torch \
    COMFYUI_ROOT=/opt/ComfyUI \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    MUSETALK_PATH=/opt/MuseTalk \
    MUSETALK_MODELS_DIR=/runpod-volume/musetalk/models
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

FROM python:3.10-slim-bookworm AS cpu-base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub \
    TORCH_HOME=/runpod-volume/torch \
    COMFYUI_ROOT=/opt/ComfyUI \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    MUSETALK_PATH=/opt/MuseTalk \
    MUSETALK_MODELS_DIR=/runpod-volume/musetalk/models
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m pip install --upgrade pip setuptools wheel

FROM gpu-base AS video
ENV WORKER_PROFILE=video
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
COPY . .
CMD ["python3", "handler.py"]

FROM gpu-base AS comfyui
ENV WORKER_PROFILE=comfyui
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt && \
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    python3 -m pip install -r /opt/ComfyUI/requirements.txt
COPY . .
CMD ["python3", "handler.py"]

FROM cpu-base AS enhancement
ENV WORKER_PROFILE=enhancement
RUN python -m pip install \
    "torch==2.2.2+cpu" "torchvision==0.17.2+cpu" --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install \
    runpod requests pillow opencv-python-headless "numpy>=1.26,<2" \
    realesrgan gfpgan basicsr facexlib rembg onnxruntime && \
    python -c "from pathlib import Path; p=Path('/usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py'); s=p.read_text(); s=s.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale'); p.write_text(s)"
COPY . .
CMD ["python", "handler.py"]

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS avatar
ARG MUSETALK_COMMIT=0a89dec45a0192b824e3cf4daf96c239440c5ed8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    TORCH_HOME=/runpod-volume/torch \
    WORKER_PROFILE=avatar \
    MUSETALK_PATH=/opt/MuseTalk \
    MUSETALK_MODELS_DIR=/runpod-volume/musetalk/models
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git ffmpeg curl ca-certificates build-essential \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    git clone https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk && \
    cd /opt/MuseTalk && git checkout "${MUSETALK_COMMIT}" && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install "runpod>=1.7.12" "requests>=2.32.5" openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0" && \
    sed -i '/export HF_ENDPOINT=/d' /opt/MuseTalk/download_weights.sh
WORKDIR /app
COPY . .
CMD ["python3", "handler.py"]

FROM ${WORKER_PROFILE} AS final
