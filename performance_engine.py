import base64
import os
import tempfile
from pathlib import Path


class WanPerformanceEngine:
    """Generate reusable motion-reference drafts from Director activity text.

    This is deliberately separate from Wan-Animate-2. The output is a generic
    performance clip that can later drive Animate-2; character identity remains
    owned by Kid Studio's approved canonical assets.
    """

    def __init__(self):
        self.model_id = os.getenv(
            "WAN_PERFORMANCE_MODEL_ID",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        )
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from diffusers import AutoencoderKLWan, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe = WanPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

        self._pipeline = pipe
        return pipe

    def close(self):
        self._pipeline = None

    def generate(self, data):
        import torch
        from diffusers.utils import export_to_video

        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Missing required input: prompt")

        negative_prompt = str(
            data.get("negative_prompt")
            or (
                "static pose, still image, frozen body, blurry, low quality, "
                "distorted anatomy, deformed limbs, extra limbs, fused hands, "
                "camera cuts, subtitles, text, watermark"
            )
        ).strip()

        seed = int(data.get("seed", 12345))
        fps = int(data.get("fps", 24))
        height = int(data.get("height", 704))
        width = int(data.get("width", 1280))
        num_frames = int(data.get("num_frames", 121))
        num_inference_steps = int(data.get("num_inference_steps", 50))
        guidance_scale = float(data.get("guidance_scale", 5.0))

        # Wan video frame counts should follow 4*k+1.
        if num_frames < 5 or (num_frames - 1) % 4 != 0:
            raise ValueError("num_frames must follow 4*k+1 and be at least 5")

        pipe = self._load_pipeline()
        generator = torch.Generator(device="cuda").manual_seed(seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]

        with tempfile.TemporaryDirectory(prefix="wan-performance-") as tmp:
            output_path = Path(tmp) / "performance.mp4"
            export_to_video(output, str(output_path), fps=fps)
            payload = base64.b64encode(output_path.read_bytes()).decode("ascii")

            return {
                "ok": True,
                "stage": "generated",
                "task": "performance_generate",
                "model_id": self.model_id,
                "pipeline_class": type(pipe).__name__,
                "seed": seed,
                "fps": fps,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "output_bytes": output_path.stat().st_size,
                "video_mime": "video/mp4",
                "video_name": "wan-performance-reference.mp4",
                "video_base64": payload,
            }
