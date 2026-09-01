import base64
import os
import tempfile
from pathlib import Path

import requests


class WanAnimate2Engine:
    """Thin adapter around Wan-Animate-2.

    RunPod transport is kept separate from model execution so Kid Studio can
    later swap checkpoints/runtimes without changing its public API.
    """

    def __init__(self):
        self.model_id = os.getenv(
            "WAN_MODEL_ID",
            "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
        )
        self._pipeline = None

    @staticmethod
    def _download(url: str, destination: Path):
        with requests.get(url, stream=True, timeout=180) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from diffusers import ModularPipeline

        pipe = ModularPipeline.from_pretrained(self.model_id)
        pipe.load_components(dtype=torch.bfloat16)

        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            use_stream=True,
        )
        pipe.text_encoder.to("cuda")
        pipe.image_encoder.to("cuda")
        pipe.vae.to("cuda")

        # Required by the current Animate-2 Diffusers implementation because
        # its in-context attention runs on the flex-attention backend.
        pipe.transformer.compile_repeated_blocks(fullgraph=False)

        self._pipeline = pipe
        return pipe

    def generate(self, data):
        import torch
        from diffusers.utils import export_to_video, load_image, load_video

        reference_url = data["reference_image"]
        driving_url = data["driving_video"]
        prompt = data.get("prompt", "")
        seed = int(data.get("seed", 12345))
        fps = int(data.get("fps", 24))
        height = int(data.get("height", 800))
        width = int(data.get("width", 640))

        with tempfile.TemporaryDirectory(prefix="wan-a2-") as tmp:
            tmpdir = Path(tmp)
            reference_path = tmpdir / "reference.png"
            driving_path = tmpdir / "driving.mp4"
            output_path = tmpdir / "output.mp4"

            self._download(reference_url, reference_path)
            self._download(driving_url, driving_path)

            pipe = self._load_pipeline()

            image = load_image(str(reference_path))
            driving_video, driving_video_fps = load_video(
                str(driving_path), return_fps=True
            )

            generator = torch.Generator(device="cuda").manual_seed(seed)

            videos = pipe(
                image=image,
                driving_video=driving_video,
                driving_video_fps=driving_video_fps,
                prompt=prompt,
                height=height,
                width=width,
                fps=fps,
                generator=generator,
                output="videos",
            )

            export_to_video(videos[0], str(output_path), fps=fps)
            payload = base64.b64encode(output_path.read_bytes()).decode("ascii")

            return {
                "ok": True,
                "stage": "generated",
                "model_id": self.model_id,
                "pipeline_class": type(pipe).__name__,
                "seed": seed,
                "fps": fps,
                "reference_bytes": reference_path.stat().st_size,
                "driving_bytes": driving_path.stat().st_size,
                "output_bytes": output_path.stat().st_size,
                "video_mime": "video/mp4",
                "video_base64": payload,
            }
