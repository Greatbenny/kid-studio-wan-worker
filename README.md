# Kid Studio Wan-Animate-2 Worker

RunPod Serverless worker used to benchmark Wan-Animate-2 for Kid Studio character motion.

## Phase 1 gate

Before wiring the full generation call into Kid Studio, the worker verifies:

1. RunPod Serverless container starts.
2. CUDA/PyTorch runtime is compatible.
3. Wan-Animate-2 Diffusers model can be loaded on the selected GPU/offload configuration.
4. Reference image and driving video can be downloaded into the job workspace.

This deliberately separates infrastructure/model compatibility from animation-quality testing.

## Environment

- `WAN_MODEL_ID` — defaults to `Wan-AI/Wan2.2-Animate-2-14B-Diffusers`
- `HF_HOME` — defaults in Docker to `/workspace/huggingface`

Attach a RunPod network volume at `/workspace` so Hugging Face model files survive worker replacement/cold starts where supported.

## Health check input

```json
{
  "input": {
    "healthcheck": true
  }
}
```

## Model-load benchmark input

The two URLs must be directly downloadable by the worker.

```json
{
  "input": {
    "reference_image": "https://example.com/character.png",
    "driving_video": "https://example.com/walk.mp4",
    "prompt": "A cartoon woman walking naturally while preserving her identity and clothing.",
    "seed": 12345
  }
}
```

The current phase returns `stage: model_loaded` after the Animate-2 pipeline loads. Once this gate passes on the selected Serverless GPU, the exact Animate-2 generation invocation and output upload are enabled in `wan_engine.py`.
