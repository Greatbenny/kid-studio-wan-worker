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
        if task == "storymind_comfyui_provision":
            return self._provision_comfyui_models()
        method = getattr(self, task, None)
        if method is None:
            return {"ok": False, "error": f"Unsupported StoryMind task: {task}"}
        return method(data)

    def _ensure_comfyui_models(self):
        if self.comfy_mode != "bundled":
            return

        persistent = self.volume_root / "comfyui" / "models"
        model_specs = [
            {
                "repo_id": "black-forest-labs/FLUX.2-dev-NVFP4",
                "filename": "flux2-dev-nvfp4.safetensors",
                "subdir": "diffusion_models",
            },
            {
                "repo_id": "Comfy-Org/flux2-dev",
                "filename": "split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors",
                "subdir": "text_encoders",
                "target_name": "mistral_3_small_flux2_fp4_mixed.safetensors",
            },
            {
                "repo_id": "Comfy-Org/flux2-dev",
                "filename": "split_files/vae/flux2-vae.safetensors",
                "subdir": "vae",
                "target_name": "flux2-vae.safetensors",
            },
        ]

        for subdir in ("diffusion_models", "text_encoders", "vae"):
            (persistent / subdir).mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to provision ComfyUI models") from exc

        for spec in model_specs:
            target_name = spec.get("target_name") or Path(spec["filename"]).name
            target = persistent / spec["subdir"] / target_name
            if target.exists() and target.stat().st_size > 0:
                continue
            downloaded = Path(hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                cache_dir=str(self.volume_root / "huggingface" / "hub"),
            ))
            tmp = target.with_suffix(target.suffix + ".partial")
            if tmp.exists():
                tmp.unlink()
            try:
                os.link(downloaded, tmp)
            except OSError:
                shutil.copy2(downloaded, tmp)
            tmp.replace(target)

    def _comfyui_extra_model_paths(self):
        persistent = self.volume_root / "comfyui" / "models"
        config = self.volume_root / "comfyui" / "extra_model_paths.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            "storymind_persistent:\n"
            f"  base_path: {persistent}\n"
            "  diffusion_models: diffusion_models\n"
            "  text_encoders: text_encoders\n"
            "  vae: vae\n",
            encoding="utf-8",
        )
        return config

    def _provision_comfyui_models(self):
        self._ensure_comfyui_models()
        persistent = self.volume_root / "comfyui" / "models"
        return {
            "ok": True,
            "task": "storymind_comfyui_provision",
            "models": {
                "diffusion_models": str(persistent / "diffusion_models"),
                "text_encoders": str(persistent / "text_encoders"),
                "vae": str(persistent / "vae"),
            },
        }

    def _ensure_comfyui(self):
        self._ensure_comfyui_models()
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
            extra_paths = self._comfyui_extra_model_paths()
            self.comfy_process = subprocess.Popen(
                [
                    "python3",
                    str(main),
                    "--listen",
                    host,
                    "--port",
                    port,
                    "--extra-model-paths-config",
                    str(extra_paths),
                ],
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
        if not prompt.ok:
            detail = prompt.text.strip()
            if len(detail) > 4000:
                detail = detail[:4000] + "..."
            raise RuntimeError(
                f"ComfyUI /prompt failed ({prompt.status_code}): "
                f"{detail or '<empty response body>'}"
            )
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
            return json.loads(raw.lstrip("\ufeff")) if isinstance(raw, str) else raw
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

    def storymind_text_to_video(self, data):
        from performance_engine import WanPerformanceEngine

        payload = dict(data)
        payload.pop("task", None)
        payload.pop("input_base64", None)

        return WanPerformanceEngine().generate(payload)

    def storymind_image_to_video(self, data):
        from performance_engine import WanPerformanceEngine

        payload = dict(data)
        payload.pop("task", None)

        input_b64 = payload.pop("input_base64", None)
        if input_b64 and not payload.get("reference_image_base64"):
            payload["reference_image_base64"] = input_b64

        return WanPerformanceEngine().generate(payload)

    def storymind_lip_sync(self, data):
        musetalk_path = Path(os.getenv("MUSETALK_PATH", "/opt/MuseTalk"))
        models_dir = Path(os.getenv("MUSETALK_MODELS_DIR", str(self.volume_root / "musetalk" / "models")))
        inference = musetalk_path / "scripts" / "inference.py"
        if not inference.exists():
            raise RuntimeError(f"MuseTalk runtime is unavailable: {inference}")

        required = [
            models_dir / "musetalkV15" / "unet.pth",
            models_dir / "musetalkV15" / "musetalk.json",
            models_dir / "sd-vae" / "diffusion_pytorch_model.bin",
            models_dir / "whisper" / "pytorch_model.bin",
            models_dir / "dwpose" / "dw-ll_ucoco_384.pth",
            models_dir / "face-parse-bisent" / "79999_iter.pth",
            models_dir / "face-parse-bisent" / "resnet18-5c106cde.pth",
        ]
        if any(not path.exists() for path in required):
            models_dir.parent.mkdir(parents=True, exist_ok=True)
            link = musetalk_path / "models"
            if link.is_symlink() or link.exists():
                if link.is_symlink():
                    link.unlink()
                elif link.resolve() != models_dir.resolve():
                    shutil.rmtree(link)
            if not link.exists():
                link.symlink_to(models_dir, target_is_directory=True)
            subprocess.run(
                ["bash", str(musetalk_path / "download_weights.sh")],
                cwd=musetalk_path,
                check=True,
                timeout=1800,
            )

        link = musetalk_path / "models"
        if not link.exists():
            link.symlink_to(models_dir, target_is_directory=True)

        with tempfile.TemporaryDirectory(prefix="sm-musetalk-") as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            audio = root / "audio.wav"
            result_dir = root / "results"
            config = root / "inference.yaml"
            _decode_or_download(data, "video_url", "video_base64", video)
            _decode_or_download(data, "audio_url", "audio_base64", audio)

            bbox_shift = int(data.get("bbox_shift", 0))
            config.write_text(
                "task_0:\n"
                f" video_path: {json.dumps(str(video))}\n"
                f" audio_path: {json.dumps(str(audio))}\n"
                f" bbox_shift: {bbox_shift}\n"
            )

            cmd = [
                "python3", "-m", "scripts.inference",
                "--inference_config", str(config),
                "--result_dir", str(result_dir),
                "--unet_model_path", str(models_dir / "musetalkV15" / "unet.pth"),
                "--unet_config", str(models_dir / "musetalkV15" / "musetalk.json"),
                "--whisper_dir", str(models_dir / "whisper"),
                "--version", "v15",
                "--ffmpeg_path", str(Path(shutil.which("ffmpeg") or "/usr/bin/ffmpeg").parent),
                "--batch_size", str(int(data.get("batch_size", 8))),
                "--use_float16",
            ]
            subprocess.run(cmd, cwd=musetalk_path, check=True, timeout=1800)

            outputs = sorted(result_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime)
            if not outputs:
                raise RuntimeError("MuseTalk 1.5 produced no MP4 output")
            out = outputs[-1]
            return {
                "ok": True,
                "task": "storymind_lip_sync",
                "engine": "musetalk",
                "engine_version": "1.5",
                "mime": "video/mp4",
                "base64": _encode_file(out),
            }
