import gc
import os
import shutil
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = Path(
    os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
)
CACHE_PATHS = (
    VOLUME_ROOT / "huggingface",
    VOLUME_ROOT / "huggingface" / "hub",
    VOLUME_ROOT / "torch",
    VOLUME_ROOT / "tmp",
)

for cache_path in CACHE_PATHS:
    cache_path.mkdir(parents=True, exist_ok=True)

# The network volume exists only when the worker container starts.
# Set TMPDIR after creating its runtime directory, never during image build.
os.environ["TMPDIR"] = str(VOLUME_ROOT / "tmp")

from performance_engine import WanPerformanceEngine
from wan_engine import WanAnimate2Engine

WORKER_BUILD = "volume-cache-v3"

_engine = None
_engine_kind = None


def _storage_status():
    usage = shutil.disk_usage(VOLUME_ROOT)
    return {
        "root": str(VOLUME_ROOT),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "hf_home": os.getenv("HF_HOME", ""),
        "hf_hub_cache": os.getenv("HF_HUB_CACHE", ""),
        "torch_home": os.getenv("TORCH_HOME", ""),
        "tmpdir": os.getenv("TMPDIR", ""),
    }


def _release_engine():
    global _engine, _engine_kind

    if _engine is not None:
        close = getattr(_engine, "close", None)
        if callable(close):
            close()

    _engine = None
    _engine_kind = None
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_engine(kind="animate"):
    global _engine, _engine_kind

    if _engine is not None and _engine_kind == kind:
        return _engine

    if _engine is not None:
        _release_engine()

    if kind == "performance":
        _engine = WanPerformanceEngine()
    else:
        _engine = WanAnimate2Engine()

    _engine_kind = kind
    return _engine


def handler(job):
    data = job.get("input") or {}
    task = str(data.get("task") or "animate").strip().lower()

    if data.get("healthcheck"):
        return {
            "ok": True,
            "service": "kid-studio-wan-worker",
            "worker_build": WORKER_BUILD,
            "storage": _storage_status(),
            "engines": {
                "animate": os.getenv(
                    "WAN_MODEL_ID",
                    "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
                ),
                "performance": os.getenv(
                    "WAN_PERFORMANCE_MODEL_ID",
                    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                ),
            },
        }

    try:
        if task in {"performance", "performance_generate"}:
            if not str(data.get("prompt") or "").strip():
                return {
                    "ok": False,
                    "error": "Missing required input: prompt",
                }
            return get_engine("performance").generate(data)

        if not data.get("reference_image") and not data.get("reference_image_base64"):
            return {
                "ok": False,
                "error": "Missing required input: reference_image or reference_image_base64",
            }

        if not data.get("driving_video") and not data.get("driving_video_base64"):
            return {
                "ok": False,
                "error": "Missing required input: driving_video or driving_video_base64",
            }

        return get_engine("animate").generate(data)
    except Exception as exc:
        trace = "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
            )
        )[-8000:]
        print(trace, flush=True)
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc) or repr(exc),
            "traceback": trace,
            "worker_build": WORKER_BUILD,
            "storage": _storage_status(),
        }


runpod.serverless.start({"handler": handler})
