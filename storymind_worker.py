import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests


def _decode_or_download(data, url_key, b64_key, dest: Path):
    raw = str(data.get(b64_key) or "").strip()
    if raw:
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        dest.write_bytes(base64.b64decode(raw))
        return
    url = str(data.get(url_key) or "").strip()
    if not url:
        raise ValueError(f"Missing {url_key} or {b64_key}")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _encode_file(path: Path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


class StoryMindWorker:
    def __init__(self, volume_root: Path):
        self.volume_root = volume_root
        self.comfy_timeout = int(os.getenv("COMFYUI_TIMEOUT_SECONDS", "900"))
        configured = os.getenv("COMFYUI_SERVER_URL", "").strip().rstrip("/")
        self.comfy_process = None
        if configured:
            self.comfy_url = configured
            self.comfy_mode = "external"
        else:
            host = os.getenv("COMFYUI_HOST", "127.0.0.1")
            port = int(os.getenv("COMFYUI_PORT", "8188"))
            self.comfy_url = f"http://{host}:{port}"
            self.comfy_mode = "bundled"

    def capabilities(self):
        return {
            "tasks": [
                "storymind_comfyui_image",
                "storymind_comfyui_video",
                "storymind_upscale",
                "storymind_face_restore",
                "storymind_bg_remove",
                "storymind_lip_sync",
            ],
            "comfyui_server_url_configured": bool(self.comfy_url),
            "comfyui_mode": self.comfy_mode,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "gpu": self._gpu_info(),
        }

    @staticmethod
    def _gpu_info():
        try:
            import torch
            if not torch.cuda.is_available():
                return {"available": False}
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        except Exception:
            return {"available": False}

    def run(self, task, data):
        method = getattr(self, task, None)
        if method is None:
            return {"ok": False, "error": f"Unsupported StoryMind task: {task}"}
        return method(data)

    def _ensure_comfyui(self):
        try:
            requests.get(f"{self.comfy_url}/system_stats", timeout=3).raise_for_status()
            return
        except Exception:
            pass

        if self.comfy_mode != "bundled":
            raise RuntimeError("Configured COMFYUI_SERVER_URL is unavailable")

        root = Path(os.getenv("COMFYUI_ROOT", "/opt/ComfyUI"))
        main = root / "main.py"
        if not main.exists():
            raise RuntimeError(f"Bundled ComfyUI not found at {root}")

        if self.comfy_process is None or self.comfy_process.poll() is not None:
            host = os.getenv("COMFYUI_HOST", "127.0.0.1")
            port = str(os.getenv("COMFYUI_PORT", "8188"))
            self.comfy_process = subprocess.Popen(
                ["python3", str(main), "--listen", host, "--port", port],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                requests.get(f"{self.comfy_url}/system_stats", timeout=3).raise_for_status()
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("Bundled ComfyUI failed to start within 90 seconds")

    def _comfy_prompt(self, workflow, output_node):
        self._ensure_comfyui()
        prompt = requests.post(
            f"{self.comfy_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        prompt.raise_for_status()
        prompt_id = prompt.json()["prompt_id"]

        deadline = time.time() + self.comfy_timeout
        while time.time() < deadline:
            history = requests.get(
                f"{self.comfy_url}/history/{prompt_id}",
                timeout=30,
            )
            history.raise_for_status()
            item = history.json().get(prompt_id)
            if item:
                outputs = item.get("outputs", {}).get(str(output_node), {})
                files = outputs.get("images") or outputs.get("gifs") or []
                if files:
                    artifacts = []
                    for file_info in files:
                        params = {
                            "filename": file_info["filename"],
                            "subfolder": file_info.get("subfolder", ""),
                            "type": file_info.get("type", "output"),
                        }
                        response = requests.get(
                            f"{self.comfy_url}/view",
                            params=params,
                            timeout=120,
                        )
                        response.raise_for_status()
                        artifacts.append({
                            "filename": file_info["filename"],
                            "content_type": response.headers.get("content-type", "application/octet-stream"),
                            "base64": base64.b64encode(response.content).decode("ascii"),
                        })
                    return artifacts
            time.sleep(1)
        raise TimeoutError("ComfyUI job timed out")

    def _workflow(self, data):
        raw = data.get("workflow_json")
        if raw:
            return json.loads(raw) if isinstance(raw, str) else raw
        raise ValueError("workflow_json is required for StoryMind ComfyUI serverless tasks")

    def storymind_comfyui_image(self, data):
        output_node = str(data.get("output_node") or "")
        if not output_node:
            raise ValueError("output_node is required")
        artifacts = self._comfy_prompt(self._workflow(data), output_node)
        return {"ok": True, "task": "storymind_comfyui_image", "artifacts": artifacts}

    def storymind_comfyui_video(self, data):
        output_node = str(data.get("output_node") or "")
        if not output_node:
            raise ValueError("output_node is required")
        artifacts = self._comfy_prompt(self._workflow(data), output_node)
        return {"ok": True, "task": "storymind_comfyui_video", "artifacts": artifacts}

    def storymind_upscale(self, data):
        try:
            import cv2
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise RuntimeError("Real-ESRGAN dependencies are not installed in this worker image") from exc

        with tempfile.TemporaryDirectory(prefix="sm-upscale-") as tmp:
            root = Path(tmp)
            src = root / "input.png"
            out = root / "output.png"
            _decode_or_download(data, "input_url", "input_base64", src)
            scale = int(data.get("scale", 4))
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth",
                model=model,
                tile=int(data.get("tile", 0)),
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
            )
            image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Could not read input image")
            result, _ = upsampler.enhance(image, outscale=scale)
            cv2.imwrite(str(out), result)
            return {"ok": True, "task": "storymind_upscale", "mime": "image/png", "base64": _encode_file(out)}

    def storymind_face_restore(self, data):
        try:
            import cv2
            from gfpgan import GFPGANer
        except ImportError as exc:
            raise RuntimeError("GFPGAN dependencies are not installed in this worker image") from exc

        with tempfile.TemporaryDirectory(prefix="sm-face-") as tmp:
            root = Path(tmp)
            src = root / "input.png"
            out = root / "output.png"
            _decode_or_download(data, "input_url", "input_base64", src)
            restorer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=int(data.get("upscale", 2)),
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not read input image")
            _, _, restored = restorer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
            if restored is None:
                raise RuntimeError("Face restoration produced no output")
            cv2.imwrite(str(out), restored)
            return {"ok": True, "task": "storymind_face_restore", "mime": "image/png", "base64": _encode_file(out)}

    def storymind_bg_remove(self, data):
        try:
            from rembg import remove
        except ImportError as exc:
            raise RuntimeError("rembg is not installed in this worker image") from exc
        with tempfile.TemporaryDirectory(prefix="sm-bg-") as tmp:
            root = Path(tmp)
            src = root / "input.png"
            out = root / "output.png"
            _decode_or_download(data, "input_url", "input_base64", src)
            out.write_bytes(remove(src.read_bytes()))
            return {"ok": True, "task": "storymind_bg_remove", "mime": "image/png", "base64": _encode_file(out)}

    def storymind_lip_sync(self, data):
        wav2lip_path = Path(os.getenv("WAV2LIP_PATH", "/opt/Wav2Lip"))
        checkpoint_dir = Path(os.getenv("WAV2LIP_CHECKPOINT_DIR", str(wav2lip_path / "checkpoints")))
        checkpoint = checkpoint_dir / str(data.get("checkpoint", "wav2lip_gan.pth"))
        inference = wav2lip_path / "inference.py"
        if not inference.exists() or not checkpoint.exists():
            raise RuntimeError(f"Wav2Lip runtime/checkpoint is unavailable: {checkpoint}")

        with tempfile.TemporaryDirectory(prefix="sm-lipsync-") as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            audio = root / "audio.wav"
            out = root / "output.mp4"
            _decode_or_download(data, "video_url", "video_base64", video)
            _decode_or_download(data, "audio_url", "audio_base64", audio)
            pads = data.get("face_padding", [0, 10, 0, 0])
            cmd = [
                "python3", str(inference),
                "--checkpoint_path", str(checkpoint),
                "--face", str(video),
                "--audio", str(audio),
                "--outfile", str(out),
                "--pads", *[str(x) for x in pads],
                "--resize_factor", str(int(data.get("resize_factor", 1))),
            ]
            subprocess.run(cmd, cwd=wav2lip_path, check=True, timeout=900)
            return {"ok": True, "task": "storymind_lip_sync", "mime": "video/mp4", "base64": _encode_file(out)}
