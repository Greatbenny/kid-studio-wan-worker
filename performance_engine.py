import base64
import gc
import io
import os
import tempfile
from pathlib import Path

import requests
from PIL import Image


class WanPerformanceEngine:
    """Generate reusable motion-reference drafts from Director activity data.

    Text-only generation uses WanPipeline. When an approved reference image is
    supplied, image-to-video generation uses WanImageToVideoPipeline. The
    resulting video can later drive Wan-Animate-2; canonical character identity
    remains owned by Kid Studio assets.
    """

    def __init__(self):
        self.model_id = os.getenv(
            "WAN_PERFORMANCE_MODEL_ID",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        )
        self._pipeline = None
        self._pipeline_mode = None

    def _release_pipeline(self):
        self._pipeline = None
        self._pipeline_mode = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_pipeline(self, mode="text"):
        if self._pipeline is not None and self._pipeline_mode == mode:
            return self._pipeline

        if self._pipeline is not None:
            self._release_pipeline()

        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        pipeline_cls = WanImageToVideoPipeline if mode == "image" else WanPipeline
        pipe = pipeline_cls.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

        self._pipeline = pipe
        self._pipeline_mode = mode
        return pipe

    def close(self):
        self._release_pipeline()

    def _load_reference_image(self, data):
        image_b64 = str(data.get("reference_image_base64") or "").strip()
        image_url = str(data.get("reference_image") or "").strip()

        if image_b64:
            try:
                payload = base64.b64decode(image_b64, validate=True)
            except Exception as exc:
                raise ValueError("Invalid reference_image_base64") from exc

            try:
                return Image.open(io.BytesIO(payload)).convert("RGB")
            except Exception as exc:
                raise ValueError("reference_image_base64 is not a readable image") from exc

        if image_url:
            try:
                response = requests.get(image_url, timeout=60)
                response.raise_for_status()
            except Exception as exc:
                raise ValueError("Could not download reference_image") from exc

            try:
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception as exc:
                raise ValueError("reference_image URL did not return a readable image") from exc

        return None

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

        reference_image = self._load_reference_image(data)
        mode = "image" if reference_image is not None else "text"
        pipe = self._load_pipeline(mode)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        if reference_image is not None:
            kwargs["image"] = reference_image

        output = pipe(**kwargs).frames[0]

        with tempfile.TemporaryDirectory(prefix="wan-performance-") as tmp:
            output_path = Path(tmp) / "performance.mp4"
            export_to_video(output, str(output_path), fps=fps)
            payload = base64.b64encode(output_path.read_bytes()).decode("ascii")

            return {
                "ok": True,
                "stage": "generated",
                "task": "performance_generate",
                "mode": "image-to-video" if reference_image is not None else "text-to-video",
                "model_id": self.model_id,
                "pipeline_class": type(pipe).__name__,
                "seed": seed,
                "fps": fps,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "reference_image_used": reference_image is not None,
                "output_bytes": output_path.stat().st_size,
                "video_mime": "video/mp4",
                "video_name": "wan-performance-reference.mp4",
                "video_base64": payload,
            }
