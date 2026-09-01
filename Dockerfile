FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/huggingface \
    TRANSFORMERS_CACHE=/workspace/huggingface \
    TORCH_HOME=/workspace/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    python3 -m pip install -r requirements.txt

COPY . .

CMD ["python3", "handler.py"]
