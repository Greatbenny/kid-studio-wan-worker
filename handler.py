import gc
import os
import shutil
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
CACHE_PATHS = (
    VOLUME_ROOT / "huggingface",
    VOLUME_ROOT / "huggingface" / "hub",
    VOLUME_ROOT / "torch",
    VOLUME_ROOT / "tmp",
)

for cache_path in CACHE_PATHS:
    cache_path.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(VOLUME_ROOT / "tmp")

from performance_engine import WanPerformanceEngine
from wan_engine import WanAnimate2Engine
from storymind_worker import StoryMindWorker

WORKER_BUILD = "storymind-profiled-v6-storage-inventory"
WORKER_PROFILE = os.getenv("WORKER_PROFILE", "video").strip().lower()

_engine = None
_engine_kind = None
_storymind = None


def _dir_size(path: Path):
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total

def _largest_entries(path: Path, limit=25):
    if not path.exists() or not path.is_dir():
        return []
    entries = []
    try:
        children = list(path.iterdir())
    except OSError:
        return []
    for child in children:
        try:
            size = _dir_size(child) if child.is_dir() else child.stat().st_size
            entries.append({
                "name": child.name,
                "path": str(child),
                "type": "dir" if child.is_dir() else "file",
                "bytes": size,
            })
        except OSError:
            continue
    entries.sort(key=lambda item: item["bytes"], reverse=True)
    return entries[:limit]


def _partial_downloads(path: Path, limit=50):
    if not path.exists():
        return []
    matches = []
    for root, _, files in os.walk(path):
        for name in files:
            lower = name.lower()
            if not (
                lower.endswith(".partial")
                or lower.endswith(".incomplete")
                or lower.endswith(".tmp")
                or ".incomplete" in lower
            ):
                continue
            candidate = Path(root) / name
            try:
                matches.append({
                    "path": str(candidate),
                    "bytes": candidate.stat().st_size,
                })
            except OSError:
                pass
    matches.sort(key=lambda item: item["bytes"], reverse=True)
    return matches[:limit]


def _disk_status(path: Path):
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": str(path),
            "exists": path.exists(),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except OSError as exc:
        return {
            "path": str(path),
            "exists": path.exists(),
            "error": str(exc),
        }


def _storage_status():
    usage = shutil.disk_usage(VOLUME_ROOT)
    comfy_models = VOLUME_ROOT / "comfyui" / "models"
    hf_root = Path(os.getenv("HF_HOME", str(VOLUME_ROOT / "huggingface")))
    hf_cache = Path(os.getenv("HF_HUB_CACHE", str(hf_root / "hub")))
    return {
        "root": str(VOLUME_ROOT),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "hf_home": str(hf_root),
        "hf_hub_cache": str(hf_cache),
        "torch_home": os.getenv("TORCH_HOME", ""),
        "tmpdir": os.getenv("TMPDIR", ""),
        "mount_compare": {
            "runpod_volume": _disk_status(Path("/runpod-volume")),
            "workspace": _disk_status(Path("/workspace")),
        },
        "paths": {
            "comfy_models": {
                "path": str(comfy_models),
                "exists": comfy_models.exists(),
                "bytes": _dir_size(comfy_models),
            },
            "hf_home": {
                "path": str(hf_root),
                "exists": hf_root.exists(),
                "bytes": _dir_size(hf_root),
            },
            "hf_hub_cache": {
                "path": str(hf_cache),
                "exists": hf_cache.exists(),
                "bytes": _dir_size(hf_cache),
            },
            "tmp": {
                "path": str(VOLUME_ROOT / "tmp"),
                "exists": (VOLUME_ROOT / "tmp").exists(),
                "bytes": _dir_size(VOLUME_ROOT / "tmp"),
            },
        },
        "largest": {
            "volume_root": _largest_entries(VOLUME_ROOT),
            "huggingface": _largest_entries(hf_root),
            "huggingface_hub": _largest_entries(hf_cache),
            "comfyui": _largest_entries(VOLUME_ROOT / "comfyui"),
            "comfyui_models": _largest_entries(comfy_models),
        },
        "partial_downloads": _partial_downloads(VOLUME_ROOT),
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
    _engine = WanPerformanceEngine() if kind == "performance" else WanAnimate2Engine()
    _engine_kind = kind
    return _engine


def get_storymind():
    global _storymind
    if _storymind is None:
        _storymind = StoryMindWorker(VOLUME_ROOT)
    return _storymind


def handler(job):
    data = job.get("input") or {}
    task = str(data.get("task") or "animate").strip().lower()

    if data.get("healthcheck") or task in {"health", "healthcheck", "capabilities"}:
        return {
            "ok": True,
            "worker_profile": WORKER_PROFILE,
            "service": "kid-studio-wan-worker",
            "worker_build": WORKER_BUILD,
            "storage": _storage_status(),
            "engines": {
                "animate": os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"),
                "performance": os.getenv("WAN_PERFORMANCE_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
            },
            "storymind": get_storymind().capabilities(),
        }

    try:
        if task.startswith("storymind_"):
            allowed = {
                "comfyui": {
                    "storymind_comfyui_image",
                    "storymind_comfyui_video",
                    "storymind_comfyui_provision",
                },
                "enhancement": {"storymind_upscale", "storymind_face_restore", "storymind_bg_remove"},
                "avatar": {"storymind_lip_sync"},
                "video": {
                    "storymind_text_to_video",
                    "storymind_image_to_video",
                },
            }.get(WORKER_PROFILE, set())
            if task not in allowed:
                return {"ok": False, "error": f"Task {task} is not enabled for worker profile {WORKER_PROFILE}"}
            return get_storymind().run(task, data)

        if WORKER_PROFILE != "video":
            return {"ok": False, "error": f"Task {task} is not enabled for worker profile {WORKER_PROFILE}"}

        if task in {"performance", "performance_generate"}:
            if not str(data.get("prompt") or "").strip():
                return {"ok": False, "error": "Missing required input: prompt"}
            return get_engine("performance").generate(data)

        if not data.get("reference_image") and not data.get("reference_image_base64"):
            return {"ok": False, "error": "Missing required input: reference_image or reference_image_base64"}
        if not data.get("driving_video") and not data.get("driving_video_base64"):
            return {"ok": False, "error": "Missing required input: driving_video or driving_video_base64"}
        return get_engine("animate").generate(data)
    except Exception as exc:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]
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
