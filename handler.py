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

    required = ["reference_image", "driving_video"]
    missing = [name for name in required if not data.get(name)]
    if missing:
        return {"ok": False, "error": f"Missing required input: {', '.join(missing)}"}

    try:
        return get_engine().generate(data)
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
        }


runpod.serverless.start({"handler": handler})
