"""VideoSeal Backend for Watermark Freedom.

Uses Facebook Research's VideoSeal model for watermarking.
- 256-bit capacity
- Flexible resolution (no fixed size requirement)
- MIT licensed

Provides two APIs:
  - embed(img_source, msg) / verify(img_source)    — direct, requires Python 3.12
  - embed_bytes(bytes, msg) / verify_bytes(bytes)  — subprocess bridge, works from any Python
"""

import os
import sys
import io
import json
import re
import base64
import subprocess
import tempfile
import torch
from PIL import Image
from typing import Optional, Dict, Any, Tuple, Union
import torchvision.transforms as T

PYTHON312 = "/usr/bin/python3.12"
PYTHON312_SITE = "/home/h/.local/lib/python3.12/site-packages"

_EMBED_SCRIPT = """
import sys, os, io, base64, json
sys.path.insert(0, '{site}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{site}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend
from viewframe import draw_corner_brackets
import cv2, numpy as np

img = Image.open('{img_path}').convert('RGB')
wm = VideoSealBackend()
result, binary, _ = wm.embed(img, '{message}')

img_np = np.array(result)
if len(img_np.shape) == 2:
    img_np = np.stack([img_np]*3, axis=-1)
elif img_np.shape[2] == 4:
    img_np = img_np[:,:,:3]
iw, ih = img_np.shape[1], img_np.shape[0]
m = int(min(iw, ih) * {margin})
x, y, w, h = m, m, min(iw, ih) - 2*m, min(iw, ih) - 2*m

img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
cl = int(min(w, h) * 0.15)
lt = max(2, int(min(w, h) * 0.012))
draw_corner_brackets(img_bgr, x, y, w, h, cl, lt, method='distinctive')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
out = Image.fromarray(img_rgb)
buf = io.BytesIO()
out.save(buf, format='PNG')

print(json.dumps({{
    'image': base64.b64encode(buf.getvalue()).decode(),
    'binary': binary[:32],
    'coords': {{'x': x, 'y': y, 'width': w, 'height': h,
                'x_percent': x/iw, 'y_percent': y/ih,
                'width_percent': w/iw, 'height_percent': h/ih,
                'viewframe_size': min(w, h), 'backend': 'videoseal'}}
}}))
"""

_VERIFY_SCRIPT = """
import sys, os, json
sys.path.insert(0, '{site}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{site}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend

img = Image.open('{img_path}').convert('RGB')
wm = VideoSealBackend()
result = wm.verify(img, {original_message})

print(json.dumps({{
    'readable': result['readable_message'][:32],
    'accuracy': result.get('bit_accuracy'),
    'viewframe': result.get('viewframe')
}}))
"""


def _find_json_in_output(stdout: str) -> dict:
    import json, re

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stdout, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise RuntimeError(f"No JSON in subprocess output: {stdout[:500]}")


class VideoSealBackend:
    def __init__(
        self, device: Optional[torch.device] = None, margin_percent: float = 0.15
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None
        self._transform = T.ToTensor()
        self._n_bits = 256
        self._margin_percent = margin_percent

    def _get_model(self):
        if self._model is None:
            sys.path.insert(0, PYTHON312_SITE)
            os.chdir(PYTHON312_SITE)
            import videoseal

            self._model = videoseal.load("videoseal_1.0")
            self._model.eval()
            self._model.to(self.device)
        return self._model

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.cpu()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).byte()
        return Image.fromarray(tensor.numpy())

    def _pil_to_tensor(self, img: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(img, str):
            img = Image.open(img)
        tensor = self._transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def _message_to_bits(self, message: str, n_bits: int) -> list:
        bits = []
        for char in message:
            val = ord(char) if isinstance(char, str) else char
            for i in range(8):
                bits.append((val >> (7 - i)) & 1)
        while len(bits) < n_bits:
            bits.append(0)
        return bits[:n_bits]

    def _bits_to_message(self, bits: list, bytes_needed: int = 32) -> str:
        message = []
        for i in range(0, min(len(bits), bytes_needed * 8), 8):
            byte_bits = bits[i : i + 8]
            if len(byte_bits) < 8:
                break
            val = sum(b << (7 - j) for j, b in enumerate(byte_bits))
            if 32 <= val <= 126:
                message.append(chr(val))
            else:
                message.append("?")
        return "".join(message)

    def _inner_square_coords(
        self, iw: int, ih: int
    ) -> Tuple[int, int, int, int, float, float, float, float]:
        m = int(min(iw, ih) * self._margin_percent)
        x = y = m
        w = h = min(iw, ih) - 2 * m
        return x, y, w, h, x / iw, y / ih, w / iw, h / ih

    def embed_bytes(
        self, image_bytes: bytes, message: str
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            margin = self._margin_percent
            script = _EMBED_SCRIPT.format(
                site=PYTHON312_SITE,
                img_path=img_path,
                message=message,
                margin=margin,
            )
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(script)
                script_path = f.name

            result = subprocess.run(
                [PYTHON312, script_path], capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout)
            data = _find_json_in_output(result.stdout)
            return (
                base64.b64decode(data["image"]),
                data["binary"],
                data["coords"],
            )
        finally:
            if img_path:
                os.unlink(img_path)
            if script_path:
                os.unlink(script_path)

    def verify_bytes(
        self, image_bytes: bytes, original_message: Optional[str] = None
    ) -> Dict[str, Any]:
        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            script = _VERIFY_SCRIPT.format(
                site=PYTHON312_SITE,
                img_path=img_path,
                original_message=repr(original_message),
            )
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(script)
                script_path = f.name

            proc = subprocess.run(
                [PYTHON312, script_path], capture_output=True, text=True, timeout=60
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
            return _find_json_in_output(proc.stdout)
        finally:
            if img_path:
                os.unlink(img_path)
            if script_path:
                os.unlink(script_path)

    def embed(
        self,
        image_source: Union[str, Image.Image],
        message: str,
    ) -> Tuple[Image.Image, str, Dict[str, Any]]:
        model = self._get_model()
        img_tensor = self._pil_to_tensor(image_source)
        if img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3, :, :]
        h, w = img_tensor.shape[2:]

        msg_bits = self._message_to_bits(message, self._n_bits)
        msg_tensor = torch.tensor([msg_bits], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = model.embed(img_tensor, msg_tensor)
            watermarked = outputs["imgs_w"]

        result_img = self._tensor_to_pil(watermarked)
        binary_str = "".join(str(b) for b in msg_bits)

        coords = {
            "x": 0,
            "y": 0,
            "width": w,
            "height": h,
            "x_percent": 0.0,
            "y_percent": 0.0,
            "width_percent": 1.0,
            "height_percent": 1.0,
            "viewframe_size": min(h, w),
            "backend": "videoseal",
        }
        return result_img, binary_str, coords

    def verify(
        self,
        image_source: Union[str, Image.Image],
        original_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = self._get_model()
        img_tensor = self._pil_to_tensor(image_source)
        if img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3, :, :]
        h, w = img_tensor.shape[2:]

        with torch.no_grad():
            detected = model.detect(img_tensor)
            preds = detected["preds"]

        msg_bits = (preds[0, 1:] > 0.5).long().cpu().tolist()
        binary_str = "".join(str(b) for b in msg_bits)
        readable = self._bits_to_message(msg_bits)

        result = {
            "binary_message": binary_str,
            "readable_message": readable,
            "bit_error_rate_percent": None,
            "corrected_bitflips": None,
            "ecc_valid": None,
            "bit_accuracy": None,
            "viewframe": {
                "x": 0,
                "y": 0,
                "width": w,
                "height": h,
                "x_percent": 0.0,
                "y_percent": 0.0,
                "width_percent": 1.0,
                "height_percent": 1.0,
                "ratio": 1.0,
                "size": min(h, w),
            },
        }

        if original_message:
            orig_bits = self._message_to_bits(original_message, self._n_bits)
            accuracy = sum(1 for p, o in zip(msg_bits, orig_bits) if p == o) / len(
                orig_bits
            )
            result["bit_accuracy"] = accuracy

        return result
