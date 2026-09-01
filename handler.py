import gc
import os

import runpod

from performance_engine import WanPerformanceEngine
from wan_engine import WanAnimate2Engine

_engine = None
_engine_kind = None


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
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
        }


runpod.serverless.start({"handler": handler})
