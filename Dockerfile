FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

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

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    python3 -m pip install -r /opt/ComfyUI/requirements.txt

RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /opt/Wav2Lip && \
    python3 -m pip install -r /opt/Wav2Lip/requirements.txt

COPY . .

CMD ["python3", "handler.py"]
