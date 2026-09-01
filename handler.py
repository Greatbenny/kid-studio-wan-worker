import os
import runpod

from wan_engine import WanAnimate2Engine

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = WanAnimate2Engine()
    return _engine


def handler(job):
    data = job.get("input") or {}

    if data.get("healthcheck"):
        return {
            "ok": True,
            "service": "kid-studio-wan-worker",
            "engine": "wan-animate-2",
            "model_id": os.getenv(
                "WAN_MODEL_ID",
                "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
            ),
        }

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

    try:
        return get_engine().generate(data)
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
        }


runpod.serverless.start({"handler": handler})
