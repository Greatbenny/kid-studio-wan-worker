# Unified RunPod GPU Worker

This branch extends the existing Kid Studio WAN worker so the same RunPod Serverless image can also expose the GPU-backed operations needed by StoryMind, without modifying StoryMind upstream.

## Existing Kid Studio tasks

- Wan Animate-2 character animation
- Wan TI2V performance generation

## StoryMind-compatible tasks

Send one of these values in `input.task`:

- `storymind_comfyui_image`
- `storymind_comfyui_video`
- `storymind_upscale`
- `storymind_face_restore`
- `storymind_bg_remove`
- `storymind_lip_sync`

Health/capabilities:

```json
{
  "input": {
    "task": "capabilities"
  }
}
```

The response reports GPU availability, FFmpeg availability, whether `COMFYUI_SERVER_URL` is configured, and the supported StoryMind task names.

## ComfyUI compatibility

StoryMind's official ComfyUI tools submit API-format workflow JSON and an `output_node`. The unified worker accepts those same two concepts and returns generated artifacts as base64.

Set `COMFYUI_SERVER_URL` if using an external/sidecar ComfyUI server.

The image also bundles ComfyUI at `/opt/ComfyUI` for future local-process wiring, but the worker does not silently start or substitute it. Provider/runtime choices remain explicit.

## Persistent storage

Attach a RunPod network volume at `/runpod-volume`.

The worker stores Hugging Face, Torch, and temporary cache data there so large model downloads can persist across worker replacement where RunPod supports it.

## Models/checkpoints still required at runtime

The container includes software dependencies, not all large model weights.

- WAN weights are downloaded through Hugging Face and cached on the attached volume.
- Real-ESRGAN / GFPGAN weights are fetched on first use and can be cached by the runtime.
- Wav2Lip requires a checkpoint under `/opt/Wav2Lip/checkpoints` or a future persistent-volume bootstrap step.
- Official StoryMind bundled ComfyUI workflows require their documented model stacks on the ComfyUI server.

## Cost and routing

The worker does not choose or hide RunPod GPU pricing. Codex Web remains responsible for:

- endpoint GPU priority/fallback configuration,
- user-visible cost estimation,
- approval before paid execution,
- actual-cost reconciliation,
- job ownership/status/cancellation.

That preserves StoryMind's governance contract while keeping shared RunPod credentials out of user runtimes.
