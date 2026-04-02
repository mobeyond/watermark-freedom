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
from viewframe import get_default_viewframe_coords, draw_viewframe, crop_to_centered_square
import cv2, numpy as np

# 1. Load and crop to centered square
img = Image.open('{img_path}').convert('RGB')
img_np = np.array(img)
img_square = crop_to_centered_square(img_np)

# 2. Get viewframe coordinates (15% margin)
coords = get_default_viewframe_coords(img_square.shape[:2], margin_pct={margin})

# 3. Extract viewframe region
x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
viewframe_region = img_square[y:y+h, x:x+w]

# 4. Embed watermark ONLY in viewframe region (VideoSeal accepts any size)
viewframe_pil = Image.fromarray(viewframe_region)
wm = VideoSealBackend()
watermarked_viewframe, binary, _ = wm.embed(viewframe_pil, '{message}')

# 5. Place watermarked viewframe back into cropped image
watermarked_np = np.array(watermarked_viewframe)
result_np = img_square.copy()
result_np[y:y+h, x:x+w] = watermarked_np

# 6. Draw corner brackets on output
img_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
draw_viewframe(img_bgr, x, y, w, h, method='distinctive')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

out = Image.fromarray(img_rgb)
buf = io.BytesIO()
out.save(buf, format='PNG')

print(json.dumps({{
    'image': base64.b64encode(buf.getvalue()).decode(),
    'binary': binary,
    'coords': {{'x': coords['x'], 'y': coords['y'], 'width': coords['width'], 'height': coords['height'],
                'x_percent': coords['x_percent'], 'y_percent': coords['y_percent'],
                'width_percent': coords['width_percent'], 'height_percent': coords['height_percent'],
                'viewframe_size': min(coords['width'], coords['height']), 'backend': 'videoseal'}}
}}))
"""

_VERIFY_SCRIPT = """
import sys, os, json
sys.path.insert(0, '{site}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{site}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend
from viewframe import get_default_viewframe_coords, crop_to_centered_square, detect_viewframe
import numpy as np
import cv2

# 1. Load and crop to centered square (same as embed)
img = Image.open('{img_path}').convert('RGB')
img_np = np.array(img)
img_square = crop_to_centered_square(img_np)

# 2. Try to detect actual viewframe from corner brackets
img_bgr = cv2.cvtColor(img_square, cv2.COLOR_RGB2BGR)
h_img, w_img = img_bgr.shape[:2]
detected = detect_viewframe(img_bgr, method='diagonal')

# 3. Use detected if valid, otherwise fallback to default margin
if detected:
    det_x = int(detected['x'])
    det_y = int(detected['y'])
    det_w = int(detected['width'])
    det_h = int(detected['height'])
    min_d = min(h_img, w_img)
    margin_val = det_x / min_d if min_d > 0 else 0.15
    coords = {{
        'x': det_x, 'y': det_y, 'width': det_w, 'height': det_h,
        'x_percent': det_x / w_img if w_img > 0 else 0,
        'y_percent': det_y / h_img if h_img > 0 else 0,
        'width_percent': det_w / w_img if w_img > 0 else 0,
        'height_percent': det_h / h_img if h_img > 0 else 0,
        'margin_pct': margin_val,
        'detected': True
    }}
else:
    coords = get_default_viewframe_coords((h_img, w_img), margin_pct={margin})
    coords['detected'] = False

# 4. Extract viewframe region (same as embed)
cx = coords['x']
cy = coords['y']
cw = coords['width']
ch = coords['height']
viewframe_region = img_square[cy:cy+ch, cx:cx+cw]

# 5. Verify ONLY on viewframe region
viewframe_pil = Image.fromarray(viewframe_region)
wm = VideoSealBackend()
result = wm.verify(viewframe_pil, {original_message})

print(json.dumps({{
    'readable': result.get('readable_message', ''),
    'ecc_valid': result.get('ecc_valid'),
    'corrected_bitflips': result.get('corrected_bitflips'),
    'bit_accuracy': result.get('bit_accuracy'),
    'binary_message': result.get('binary_message', ''),
    'viewframe': {{'x': coords['x'], 'y': coords['y'], 'width': coords['width'], 'height': coords['height'],
                  'x_percent': coords['x_percent'], 'y_percent': coords['y_percent'],
                  'width_percent': coords['width_percent'], 'height_percent': coords['height_percent'],
                  'detected': coords.get('detected', False)}}
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
        if len(message) <= 3:
            from roco_core import encode_to_bits
            from roco_ecc import encode_with_ecc

            data_bits = encode_to_bits(message)
            payload_bytes = data_bits.to_bytes(2, "big")
            codeword_bytes = encode_with_ecc(payload_bytes)
            codeword_str = "".join(f"{b:08b}" for b in codeword_bytes)
            repeated = ("0" + codeword_str * ((n_bits // (len(codeword_str) + 1)) + 1))[
                :n_bits
            ]
            return [int(b) for b in repeated]
        else:
            bits = [1]
            for char in message[: n_bits // 8 - 1]:
                bits.extend([int(b) for b in format(ord(char), "08b")])
            if len(bits) < n_bits:
                bits.extend([0] * (n_bits - len(bits)))
            return bits[:n_bits]

    def _bits_to_message(self, bits: list) -> Tuple[str, bool, int, float]:
        if not bits:
            return "", False, 0, 0.0

        if bits[0] == 1:
            decoded = ""
            for i in range(1, min(len(bits), 256), 8):
                byte_bits = bits[i : i + 8]
                if len(byte_bits) == 8:
                    char_code = int("".join(str(b) for b in byte_bits), 2)
                    if 32 <= char_code <= 126:
                        decoded += chr(char_code)
            return decoded, False, 0, 1.0
        else:
            from roco_core import decode_from_bits
            from roco_ecc import decode_with_ecc

            bits_str = "".join(str(b) for b in bits)
            codeword_str = bits_str[1:33]
            codeword_bytes = int(codeword_str, 2).to_bytes(4, "big")
            corrected_data, ecc_valid, bitflips = decode_with_ecc(codeword_bytes)

            if corrected_data:
                data_bits = int.from_bytes(corrected_data, "big") & 0xFFFF
                decoded = decode_from_bits(data_bits)
            else:
                decoded = "DECODE_FAIL"

            accuracy = (
                sum(1 for i in range(32) if bits_str[i + 1] == codeword_str[i]) / 32.0
            )
            return decoded, ecc_valid, bitflips, accuracy

    def _inner_square_coords(
        self, iw: int, ih: int
    ) -> Tuple[int, int, int, int, float, float, float, float]:
        m = int(min(iw, ih) * self._margin_percent)
        x = y = m
        w = h = min(iw, ih) - 2 * m
        return x, y, w, h, x / iw, y / ih, w / iw, h / ih

    def embed_bytes(
        self, image_bytes: bytes, message: str, margin_pct: Optional[float] = None
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            margin = margin_pct if margin_pct is not None else self._margin_percent
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
                [PYTHON312, script_path],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
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
                margin=self._margin_percent,
            )
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(script)
                script_path = f.name

            proc = subprocess.run(
                [PYTHON312, script_path],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
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
        readable, ecc_valid, bitflips, accuracy = self._bits_to_message(msg_bits)

        result = {
            "binary_message": binary_str,
            "readable_message": readable,
            "ecc_valid": ecc_valid,
            "corrected_bitflips": bitflips,
            "bit_accuracy": accuracy,
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

        return result
