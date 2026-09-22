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
    WAV2LIP_PATH=/opt/Wav2Lip
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
    WAV2LIP_PATH=/opt/Wav2Lip
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
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install \
    runpod requests pillow opencv-python-headless "numpy>=1.26,<2.3" \
    realesrgan gfpgan basicsr facexlib rembg onnxruntime
COPY . .
CMD ["python", "handler.py"]

FROM gpu-base AS avatar
ENV WORKER_PROFILE=avatar
RUN python3 -m pip install runpod requests pillow opencv-python-headless "numpy>=1.26,<2.3" && \
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /opt/Wav2Lip && \
    python3 -m pip install \
      "librosa>=0.10.2,<0.12" \
      "numpy>=1.26,<2.3" \
      "scipy>=1.10" \
      "numba>=0.57" \
      "opencv-python-headless>=4.10" \
      "tqdm>=4.66" \
      "soundfile>=0.12" \
      "audioread>=3.0"
COPY . .
CMD ["python3", "handler.py"]

FROM ${WORKER_PROFILE} AS final
