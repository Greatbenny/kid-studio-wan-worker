import os
import tempfile
from pathlib import Path

import requests


class WanAnimate2Engine:
    """Thin adapter around Wan-Animate-2.

    The RunPod transport is intentionally separated from model execution so
    Kid Studio can later swap model/checkpoint/runtime without changing its API.
    """

    def __init__(self):
        self.model_id = os.getenv(
            "WAN_MODEL_ID",
            "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
        )
        self._pipeline = None

    @staticmethod
    def _download(url: str, destination: Path):
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        # Import lazily so a cheap healthcheck can validate the Serverless
        # container before loading a multi-billion-parameter model.
        import torch
        from diffusers import ModularPipeline

        pipe = ModularPipeline.from_pretrained(self.model_id)
        pipe.load_components(dtype=torch.bfloat16)

        transformer = getattr(pipe, "transformer", None)
        if transformer is not None and hasattr(transformer, "enable_group_offload"):
            transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                use_stream=True,
            )

        # Keep small/latency-sensitive components resident when possible.
        for component_name in ("text_encoder", "image_encoder", "vae"):
            component = getattr(pipe, component_name, None)
            if component is not None and hasattr(component, "to"):
                component.to("cuda")

        self._pipeline = pipe
        return pipe

    def generate(self, data):
        """First benchmark implementation.

        Input contract is frozen here before Kid Studio integration. We first
        prove that the selected Animate-2 runtime can load successfully on the
        chosen RunPod GPU. The exact Diffusers Animate-2 call contract is kept
        isolated here so it can be adjusted without touching handler.py.
        """
        reference_url = data["reference_image"]
        driving_url = data["driving_video"]
        prompt = data.get("prompt", "")
        seed = int(data.get("seed", 12345))

        with tempfile.TemporaryDirectory(prefix="wan-a2-") as tmp:
            tmpdir = Path(tmp)
            reference_path = tmpdir / "reference.png"
            driving_path = tmpdir / "driving.mp4"
            self._download(reference_url, reference_path)
            self._download(driving_url, driving_path)

            pipe = self._load_pipeline()

            return {
                "ok": True,
                "stage": "model_loaded",
                "model_id": self.model_id,
                "reference_bytes": reference_path.stat().st_size,
                "driving_bytes": driving_path.stat().st_size,
                "prompt": prompt,
                "seed": seed,
                "message": (
                    "Wan-Animate-2 loaded successfully. The next benchmark "
                    "step enables the exact generation call after this "
                    "container/GPU compatibility gate passes."
                ),
                "pipeline_class": type(pipe).__name__,
            }
