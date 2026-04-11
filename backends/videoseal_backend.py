"""VideoSeal Backend for Watermark Freedom.

Uses Facebook Research's VideoSeal model for watermarking.
- 32-bit capacity with roco ECC (3 chars encoded) - same as WAM
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
import uuid
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, Dict, Any, Tuple, Union
import torchvision.transforms as T
from roco32 import encode as roco32_encode, decode as roco32_decode

# Import base backend for shared procedures
from backends.base_backend import (
    BaseWatermarkBackend,
    VIEWFRAME_PADDING,
    CORNER_LENGTH_RATIO,
)

PYTHON312 = "/usr/bin/python3.12"
PYTHON312_SITE = "/home/h/.local/lib/python3.12/site-packages"
LOG_DIR = "/tmp/videoseal_logging"  # Directory for logged images

# Note: The custom checkpoint (checkpoint.pth) uses a different architecture
# (unet_base_yuv_quant) that isn't compatible with the standard Videoseal model.
# We use the official Videoseal model for maximum compatibility.

_EMBED_SCRIPT = """
import sys, os, io, base64, json
sys.path.insert(0, '{site}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{site}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend
from viewframe import get_default_viewframe_coords, draw_viewframe, crop_to_centered_square, calculate_line_thickness
import cv2, numpy as np

# Debug logging
LOG_DIR = '/tmp/videoseal_logging/embed'
os.makedirs(LOG_DIR, exist_ok=True)

# 1. Load and crop to centered square
img = Image.open('{img_path}').convert('RGB')
img.save(LOG_DIR + '/01_input.png')
img_np = np.array(img)
img_square = crop_to_centered_square(img_np)
Image.fromarray(img_square).save(LOG_DIR + '/02_cropped_square.png')

# 2. Get viewframe coordinates (15% margin)
coords = get_default_viewframe_coords(img_square.shape[:2], margin_pct={margin})

# 3. Extract viewframe region with padding to exclude bracket arms
x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
# Calculate line_thickness based on image size
# cv2.line draws thicker due to antialiasing: thickness=1 → 1px, thickness=2 → 3px
min_dim = min(img_square.shape[0], img_square.shape[1])
line_thickness = calculate_line_thickness(min_dim)
x_draw = x - line_thickness // 2
y_draw = y - line_thickness // 2
# Padding should match actual bracket thickness
padding = line_thickness
x_padded, y_padded = x + padding, y + padding
w_padded, h_padded = w - 2*padding, h - 2*padding
viewframe_region = img_square[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]

# 4. Embed watermark ONLY in viewframe region (VideoSeal accepts any size)
viewframe_pil = Image.fromarray(viewframe_region)
wm = VideoSealBackend()
watermarked_viewframe, binary, _ = wm.embed(viewframe_pil, '{message}')

# 5. Place watermarked viewframe back into cropped image (use padded region)
watermarked_np = np.array(watermarked_viewframe)
result_np = img_square.copy()
result_np[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded] = watermarked_np

# 6. Draw corner brackets on output (adjusted for line thickness)
img_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
draw_viewframe(img_bgr, x_draw, y_draw, w, h, method='distinctive', line_thickness=line_thickness)
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
                'margin_pct': {margin}, 'viewframe_size': min(coords['width'], coords['height']), 'backend': 'videoseal'}}
}}))
"""

_EMBED_SCRIPT_ROCO = """
import sys, os, io, base64, json
sys.path.insert(0, '{site}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{site}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend
from viewframe import get_default_viewframe_coords, draw_viewframe, crop_to_centered_square, calculate_line_thickness
import cv2, numpy as np
from watermark_utils import roco_encode_to_binary_tensor
import torch

# Setup logging directory
LOG_DIR = '/tmp/videoseal_logging/embed'
os.makedirs(LOG_DIR, exist_ok=True)
SESSION_ID = '{session_id}'

def log_img(img, name):
    path = os.path.join(LOG_DIR, SESSION_ID + '_' + name + '.png')
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(path, format='PNG')
    print('[LOG] Embedded: ' + name + ' -> ' + path, file=sys.stderr)

# 1. Load original image
img = Image.open('{img_path}').convert('RGB')
log_img(img, '01_original')

# 2. Crop to centered square
img_np = np.array(img)
img_square = crop_to_centered_square(img_np)
log_img(img_square, '02_cropped_square')

# 3. Get viewframe coordinates (15% margin)
coords = get_default_viewframe_coords(img_square.shape[:2], margin_pct={margin})

# 4. Extract viewframe region with padding to exclude bracket arms
x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
# Calculate line_thickness based on image size
# cv2.line draws thicker due to antialiasing: thickness=1 → 1px, thickness=2 → 3px
min_dim = min(img_square.shape[0], img_square.shape[1])
line_thickness = calculate_line_thickness(min_dim)
x_draw = x - line_thickness // 2
y_draw = y - line_thickness // 2
# Padding should match actual bracket thickness
padding = line_thickness
x_padded, y_padded = x + padding, y + padding
w_padded, h_padded = w - 2*padding, h - 2*padding
viewframe_region = img_square[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
log_img(viewframe_region, '03_viewframe_before_embed')

# 5. Encode message to 32 bits using ROCO
message = '{message}'
binary_tensor = roco_encode_to_binary_tensor(message)
roco_bits = [int(b) for b in binary_tensor.tolist()]
roco_str = ''.join(str(b) for b in roco_bits)
print('[LOG] ROCO bits for %s: %s' % (message, roco_str), file=sys.stderr)

# 6. Embed 32 bits directly using VideoSeal
viewframe_pil = Image.fromarray(viewframe_region)
wm = VideoSealBackend()
model = wm._get_model()
img_tensor = wm._pil_to_tensor(viewframe_pil)
if img_tensor.shape[1] == 4:
    img_tensor = img_tensor[:, :3, :, :]

# Extend 32 bits to 256 bits (repeat 8 times) for TorchScript model
roco_bits_256 = roco_bits * 8
msg_tensor = torch.tensor([roco_bits_256], dtype=torch.float32, device=wm.device)

# Embed with training mode enabled
model.train()
try:
    result = model.embed(img_tensor, msg_tensor)
    # TorchScript returns tensor directly, not dict
    if isinstance(result, torch.Tensor):
        watermarked = result
    elif isinstance(result, tuple):
        watermarked = result[0]
    else:
        watermarked = result['imgs_w']
finally:
    model.eval()

watermarked_pil = wm._tensor_to_pil(watermarked)
watermarked_np = np.array(watermarked_pil)
log_img(watermarked_np, '04_viewframe_after_embed')

# 7. Place watermarked viewframe back into cropped image (with padding)
result_np = img_square.copy()
result_np[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded] = watermarked_np
log_img(result_np, '05_combined_with_watermark')

# 8. Draw corner brackets on output
img_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
draw_viewframe(img_bgr, x_draw, y_draw, w, h, method='distinctive', line_thickness=line_thickness)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
log_img(img_rgb, '06_final_with_brackets')

# 9. Save final output
out = Image.fromarray(img_rgb)
buf = io.BytesIO()
out.save(buf, format='PNG')

binary_str = roco_str

output_dict = {{
    'image': base64.b64encode(buf.getvalue()).decode(),
    'binary': binary_str,
    'coords': {{'x': coords['x'], 'y': coords['y'], 'width': coords['width'], 'height': coords['height'],
                'x_percent': coords['x_percent'], 'y_percent': coords['y_percent'],
                'width_percent': coords['width_percent'], 'height_percent': coords['height_percent'],
                'margin_pct': {margin}, 'viewframe_size': min(coords['width'], coords['height']), 'backend': 'videoseal-roco'}}
}}

print(json.dumps(output_dict))
"""

_VERIFY_SCRIPT = """
import sys, os, uuid, json
sys.path.insert(0, '{site}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{site}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend
from viewframe import get_default_viewframe_coords, crop_to_centered_square, detect_viewframe, calculate_line_thickness
import numpy as np
import cv2

LOG_DIR = '/tmp/videoseal_logging/verify'
SESSION_ID = '{session_id}'
os.makedirs(LOG_DIR, exist_ok=True)

# 1. Load and crop to centered square (same as embed)
img = Image.open('{img_path}').convert('RGB')
img.save(LOG_DIR + '/01_watermarked.png')
img_np = np.array(img)
img_square = crop_to_centered_square(img_np)
Image.fromarray(img_square).save(LOG_DIR + '/02_cropped_square.png')

# 2. Try to detect actual viewframe from corner brackets
img_bgr = cv2.cvtColor(img_square, cv2.COLOR_RGB2BGR)
h_img, w_img = img_bgr.shape[:2]
detected = detect_viewframe(img_bgr, method='diagonal')

# 3. Use DETECTED viewframe (auto-detection), fall back to default margin
if detected:
    coords = detected
    coords['detected'] = True
else:
    coords = get_default_viewframe_coords((h_img, w_img), margin_pct={margin})
    coords['detected'] = False

# 4. Log DETECTED viewframe region (EXACT position, no padding)
detected_viewframe = img_square[coords['y']:coords['y']+coords['height'],
                                 coords['x']:coords['x']+coords['width']]
Image.fromarray(detected_viewframe).save(LOG_DIR + '/' + SESSION_ID + '_02_detected_viewframe.png')

# 5. Extract inner region - padding should match bracket thickness
# Use detected line thickness if available, otherwise calculate
if detected and 'detected_line_thickness' in coords:
    padding = coords['detected_line_thickness']
else:
    min_dim = min(img_square.shape[0], img_square.shape[1])
    padding = calculate_line_thickness(min_dim)
cx = coords['x'] + padding
cy = coords['y'] + padding
cw = max(1, coords['width'] - 2 * padding)
ch = max(1, coords['height'] - 2 * padding)
viewframe_region = img_square[cy:cy+ch, cx:cx+cw]
Image.fromarray(viewframe_region).save(LOG_DIR + '/' + SESSION_ID + '_03_extracted_viewframe.png')

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
                  'detected': coords.get('detected', False),
                  'detected_line_thickness': coords.get('detected_line_thickness', 1)}}
}}))
"""

_VERIFY_SCRIPT_ROCO = """
import sys, os, json, torch
sys.path.insert(0, '{site}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{site}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend
from viewframe import get_default_viewframe_coords, crop_to_centered_square, detect_viewframe, calculate_line_thickness
import numpy as np
import cv2
from watermark_utils import roco_decode_from_binary_tensor

# Setup logging directory
LOG_DIR = '/tmp/videoseal_logging/verify'
os.makedirs(LOG_DIR, exist_ok=True)
SESSION_ID = '{session_id}'

def log_img(img, name):
    path = os.path.join(LOG_DIR, SESSION_ID + '_' + name + '.png')
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(path, format='PNG')
    print('[LOG] Verified: ' + name + ' -> ' + path, file=sys.stderr)

# 1. Load watermarked image
img = Image.open('{img_path}').convert('RGB')
log_img(img, '01_watermarked_image')

# 2. Crop to centered square (same as embed)
img_np = np.array(img)
img_square = crop_to_centered_square(img_np)
log_img(img_square, '02_cropped_square')

# 3. Try to detect actual viewframe from corner brackets
img_bgr = cv2.cvtColor(img_square, cv2.COLOR_RGB2BGR)
h_img, w_img = img_bgr.shape[:2]
detected = detect_viewframe(img_bgr, method='diagonal')

# 4. Use DETECTED viewframe (auto-detection), fall back to default margin
if detected:
    coords = detected
    coords['detected'] = True
else:
    coords = get_default_viewframe_coords((h_img, w_img), margin_pct={margin})
    coords['detected'] = False

# 5. Log DETECTED viewframe region (EXACT position, no padding)
detected_viewframe = img_square[coords['y']:coords['y']+coords['height'],
                                 coords['x']:coords['x']+coords['width']]
log_img(detected_viewframe, '02_detected_viewframe')

# 6. Extract viewframe region with padding (same as embed)
cx = coords['x']
cy = coords['y']
cw = coords['width']
ch = coords['height']
# Add padding to exclude bracket line from the extracted region
# Use detected line thickness if available, otherwise calculate
if detected and 'detected_line_thickness' in coords:
    padding = coords['detected_line_thickness']
else:
    min_dim = min(img_square.shape[0], img_square.shape[1])
    padding = calculate_line_thickness(min_dim)
cx_padded, cy_padded = cx + padding, cy + padding
cw_padded, ch_padded = cw - 2*padding, ch - 2*padding
viewframe_region = img_square[cy_padded:cy_padded+ch_padded, cx_padded:cx_padded+cw_padded]
log_img(viewframe_region, '03_extracted_viewframe')

# 5. Detect watermark using VideoSeal
viewframe_pil = Image.fromarray(viewframe_region)
wm = VideoSealBackend()
model = wm._get_model()
img_tensor = wm._pil_to_tensor(viewframe_pil)
if img_tensor.shape[1] == 4:
    img_tensor = img_tensor[:, :3, :, :]

model.eval()
with torch.no_grad():
    detected = model.detect(img_tensor)
    # TorchScript returns tensor directly, not dict
    if isinstance(detected, torch.Tensor):
        preds = detected
    else:
        preds = detected['preds']

# Get confidence and predictions
confidence = float(preds[0, 0].item())
print('[LOG] Detection confidence: %.4f' % confidence, file=sys.stderr)
print('[LOG] First 10 pred values: ' + str(preds[0, :11].tolist()), file=sys.stderr)

# Extract all 256 bits (after confidence at index 0) and use majority voting for 32-bit
all_bits = (preds[0, 1:] > 0.0).long().cpu().tolist()

# Majority voting across 8 blocks of 32 bits each
blocks = [all_bits[i*32:(i+1)*32] for i in range(8)]
msg_bits = []
for bit_idx in range(32):
    votes = sum(block[bit_idx] for block in blocks)
    msg_bits.append(1 if votes >= 4 else 0)

detected_str = ''.join(str(b) for b in msg_bits)
print('[LOG] Detected 32 bits (majority vote): ' + detected_str, file=sys.stderr)

# Decode using ROCO - use majority voted bits
binary_tensor = torch.tensor(msg_bits, dtype=torch.float32)
readable, ecc_valid, bitflips = roco_decode_from_binary_tensor(binary_tensor)

print('[LOG] Decoded: ' + readable + ', ECC valid: ' + str(ecc_valid) + ', bitflips: ' + str(bitflips), file=sys.stderr)

output_dict = {{
    'readable': readable,
    'ecc_valid': ecc_valid,
    'corrected_bitflips': bitflips,
    'bit_accuracy': 1.0 if ecc_valid else 0.0,
    'binary_message': detected_str,
    'confidence': confidence,
    'viewframe': {{'x': coords['x'], 'y': coords['y'], 'width': coords['width'], 'height': coords['height'],
                  'x_percent': coords['x_percent'], 'y_percent': coords['y_percent'],
                  'width_percent': coords['width_percent'], 'height_percent': coords['height_percent'],
                  'detected': False,
                  'detected_line_thickness': coords.get('detected_line_thickness', 1)}}
}}

print(json.dumps(output_dict))
"""


def _find_json_in_output(stdout: str) -> dict:
    import json, re

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stdout, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise RuntimeError(f"No JSON in subprocess output: {stdout[:500]}")


def _log_image(img_or_bytes, filename: str, step_name: str = ""):
    """Log an image to disk for debugging.

    Args:
        img_or_bytes: PIL Image or bytes
        filename: Filename to save as
        step_name: Optional step name prefix
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        full_path = os.path.join(
            LOG_DIR, f"{step_name}_{filename}" if step_name else filename
        )

        if isinstance(img_or_bytes, bytes):
            img = Image.open(io.BytesIO(img_or_bytes))
        else:
            img = img_or_bytes

        img.save(full_path, format="PNG")
        print(f"[LOG] Saved: {full_path}", file=sys.stderr)
    except Exception as e:
        print(f"[LOG] Error saving {filename}: {e}", file=sys.stderr)


class VideoSealBackend(BaseWatermarkBackend):
    """VideoSeal watermark backend.

    Inherits shared procedures from BaseWatermarkBackend:
    - Viewframe coordinate calculation
    - Padding-aware region extraction
    - Format conversion (PIL ↔ numpy ↔ tensor)

    Implements VideoSeal-specific:
    - embed_watermark(): Embed using VideoSeal model
    - detect_watermark(): Detect using VideoSeal model
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        margin_percent: float = 0.10,
        custom_model_path: Optional[str] = None,
        force_cpu: bool = False,  # New parameter to allow GPU
    ):
        # Call parent init for shared setup
        super().__init__(device)

        # VideoSeal-specific initialization
        # Force CPU for TorchScript to avoid CUDA version mismatch
        # Set force_cpu=False to enable GPU acceleration
        self._force_cpu = force_cpu
        if self._force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self._model = None
        self._transform = T.ToTensor()
        self._n_bits = 32  # ROCO uses 32-bit codeword (was 256)
        self._margin_percent = margin_percent
        self._custom_model_path = custom_model_path  # Optional custom model path
        self._use_torchscript = True  # Use TorchScript for faster loading
        self._jit_model_path = (
            "/home/h/FLY/videoseal/ckpts/y_256b_img.jit"  # TorchScript model path
        )

    def _get_model(self):
        if self._model is None:
            if self._use_torchscript:
                # Use TorchScript model - no dependency issues, faster loading
                import torch

                # Force CPU for TorchScript to avoid CUDA version mismatch
                model_device = torch.device("cpu") if self._force_cpu else self.device
                print(
                    f"[VideoSeal] Loading TorchScript model from: {self._jit_model_path} on {model_device}",
                    file=sys.stderr,
                )
                self._model = torch.jit.load(
                    self._jit_model_path, map_location=str(model_device)
                )
                self._model.eval()
                print(
                    f"[VideoSeal] TorchScript model loaded successfully!",
                    file=sys.stderr,
                )
            elif self._custom_model_path and os.path.exists(self._custom_model_path):
                # Load custom model from checkpoint file (only if path is provided and exists)
                sys.path.insert(0, PYTHON312_SITE)
                os.chdir(PYTHON312_SITE)
                import videoseal

                print(
                    f"[VideoSeal] Loading custom model from: {self._custom_model_path}",
                    file=sys.stderr,
                )

                # Load the model architecture first to get the correct structure
                model_card = videoseal.load("videoseal_1.0")

                # Load the custom weights
                checkpoint = torch.load(
                    self._custom_model_path, map_location="cpu", weights_only=False
                )

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    # Assume checkpoint is the state_dict directly
                    state_dict = checkpoint

                # Load weights into the model (non-strict to handle architecture differences)
                model_card.load_state_dict(state_dict, strict=False)
                self._model = model_card

                self._model.eval()
                self._model.to(self.device)

                print(f"[VideoSeal] Custom model loaded successfully!", file=sys.stderr)
            else:
                # Use official VideoSeal model via full package
                sys.path.insert(0, PYTHON312_SITE)
                os.chdir(PYTHON312_SITE)
                import videoseal

                print(
                    f"[VideoSeal] Loading official VideoSeal model...", file=sys.stderr
                )
                self._model = videoseal.load("videoseal_1.0")
                self._model.to(self.device)
                print(f"[VideoSeal] Official VideoSeal model loaded!", file=sys.stderr)
        return self._model

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array (RGB)."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.cpu()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).byte()
        return tensor.numpy()

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
        """Convert message to bit array using ROCO or ROCO32 encoding.

        For 3-character messages: Uses ROCO to encode into 32-bit codeword.
        For 4-character messages: Uses ROCO32 to encode into 256-bit codeword.

        Args:
            message: Message string (3 characters for ROCO, 4 for ROCO32)
            n_bits: Target bit length (32 for ROCO, 256 for ROCO32)

        Returns:
            List of n_bits bits (integers 0 or 1)
        """
        msg_len = len(message)

        if msg_len == 4:
            # Use ROCO32 encoding (256-bit codeword)
            from roco32 import encode as roco32_encode

            bits = roco32_encode(message)
        elif msg_len <= 3:
            # Use ROCO encoding (32-bit codeword)
            from watermark_utils import roco_encode_to_binary_tensor
            import torch

            # Pad message to 3 characters if needed
            padded_message = message.ljust(3, ".")[:3]

            # Use ROCO encoding
            binary_tensor = roco_encode_to_binary_tensor(padded_message)
            bits = [int(b) for b in binary_tensor.tolist()]
        else:
            raise ValueError(
                f"Message must be 1-4 characters, got {msg_len}: '{message}'"
            )

        # Pad/truncate to target length if needed
        if len(bits) < n_bits:
            bits = bits + [0] * (n_bits - len(bits))
        elif len(bits) > n_bits:
            bits = bits[:n_bits]

        return bits

    def _bits_to_message(
        self, bits: list, n_bits: int = 32
    ) -> Tuple[str, bool, int, float]:
        """Decode bits to message using ROCO or ROCO32 decoding.

        Decodes 32-bit (ROCO) or 256-bit (ROCO32) codeword from VideoSeal back to message.
        Uses ECC verification and error correction.

        Args:
            bits: List of bits (integers 0 or 1)
            n_bits: Number of bits (32 for ROCO, 256 for ROCO32)

        Returns:
            Tuple of (decoded_message, ecc_valid, bitflips_corrected, accuracy)
        """
        from watermark_utils import roco_decode_from_binary_tensor
        from roco32 import decode as roco32_decode
        import torch

        if not bits:
            return "", False, 0, 0.0

        # Determine which decoding to use based on bit length
        if n_bits == 256 or len(bits) == 256:
            # Use ROCO32 decoding (256-bit codeword -> 4 chars)
            decoded, is_valid, errors = roco32_decode(bits)
        else:
            # Use ROCO decoding (32-bit codeword -> 3 chars)
            # Ensure we have exactly 32 bits
            if len(bits) < 32:
                bits = bits + [0] * (32 - len(bits))
            elif len(bits) > 32:
                bits = bits[:32]

            binary_tensor = torch.tensor(bits, dtype=torch.float32)
            decoded, is_valid, errors = roco_decode_from_binary_tensor(binary_tensor)

        # Calculate accuracy based on ECC validation
        accuracy = 1.0 if is_valid else 0.0

        return decoded, is_valid, errors, accuracy

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
            session_id = uuid.uuid4().hex[:8]  # Short unique ID for logging
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
        self,
        image_bytes: bytes,
        original_message: Optional[str] = None,
        margin_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            margin = margin_pct if margin_pct is not None else self._margin_percent
            session_id = uuid.uuid4().hex[:8]  # Short unique ID for logging
            script = _VERIFY_SCRIPT.format(
                site=PYTHON312_SITE,
                img_path=img_path,
                original_message=repr(original_message),
                margin=margin,
                session_id=session_id,
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

    def embed_bytes_roco(
        self,
        image_bytes: bytes,
        message: str,
        margin_pct: float = 0.10,
        padding: int = 4,
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """Embed watermark using WAM-compatible ROCO 3-char encoding via subprocess.

        This method uses the same ROCO encoding as the WAM algorithm (3 characters, 32 bits).
        It uses VideoSealBackendROCO which shares the same encoding/decoding as WAM.

        Args:
            image_bytes: Raw image bytes
            message: 3-character message (uses WAM-compatible ROCO encoding)
            margin_pct: Margin percentage for viewframe (default: 0.10 = 10%)
            padding: Pixels to pad from each edge to exclude bracket arms (default: 4)

        Returns:
            (watermarked_bytes, binary_string, coords)
        """
        if len(message) != 3:
            raise ValueError(
                f"ROCO requires exactly 3 chars, got {len(message)}: '{message}'"
            )

        # Use the same subprocess approach but with VideoSealBackendROCO
        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            session_id = uuid.uuid4().hex[:8]  # Short unique ID for logging
            script = _EMBED_SCRIPT_ROCO.format(
                site=PYTHON312_SITE,
                img_path=img_path,
                message=message,
                margin=margin_pct,
                padding=padding,
                session_id=session_id,
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

    def verify_bytes_roco(
        self,
        image_bytes: bytes,
        original_message: Optional[str] = None,
        margin_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Verify watermark using WAM-compatible ROCO 3-char decoding via subprocess.

        This method uses the same ROCO decoding as the WAM algorithm (3 characters, 32 bits).
        It uses VideoSealBackendROCO which shares the same encoding/decoding as WAM.

        Uses AUTO-DETECTION: Viewframe is detected from corner brackets automatically.
        Falls back to 10% margin if detection fails.

        Args:
            image_bytes: Raw image bytes
            original_message: Optional original message for comparison

        Returns:
            Dict with decoded message and verification info
        """
        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            margin = margin_pct if margin_pct is not None else self._margin_percent
            session_id = uuid.uuid4().hex[:8]  # Short unique ID for logging
            script = _VERIFY_SCRIPT_ROCO.format(
                site=PYTHON312_SITE,
                img_path=img_path,
                original_message=repr(original_message),
                margin=margin,
                session_id=session_id,
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
        margin_pct: Optional[float] = None,
    ) -> Tuple[Image.Image, str, Dict[str, Any]]:
        """Embed watermark using VideoSeal with viewframe support.

        Uses base class methods for:
        - Cropping to centered square
        - Calculating viewframe coordinates
        - Applying padding to exclude bracket arms

        Args:
            image_source: PIL Image or path to image
            message: Message to embed
            margin_pct: Margin percentage for viewframe (overrides default)

        Returns:
            (watermarked_image, binary_string, coords_dict)
        """
        margin = margin_pct if margin_pct is not None else self._margin_percent

        # Use base class methods for shared procedures
        img_pil = (
            Image.open(image_source) if isinstance(image_source, str) else image_source
        )
        img_np = self._pil_to_numpy(img_pil.convert("RGB"))
        img_square = self._crop_to_centered_square(img_np)

        # Get viewframe and embed region coordinates
        viewframe_coords = self._get_viewframe_coords(
            img_square.shape[:2], margin_pct=margin
        )
        # Calculate dynamic padding based on image size
        min_dim = min(img_square.shape[0], img_square.shape[1])
        dynamic_padding = 1 if min_dim <= 150 else 2
        x, y, width, height = self._get_embed_region_coords(
            viewframe_coords, padding=dynamic_padding
        )

        # Extract embed region
        embed_region = self._crop_to_embed_region_numpy(img_square, x, y, width, height)
        embed_pil = self._numpy_to_pil(embed_region)

        # Embed watermark (VideoSeal-specific)
        model = self._get_model()
        embed_tensor = self._pil_to_tensor(embed_pil)
        if embed_tensor.shape[1] == 4:
            embed_tensor = embed_tensor[:, :3, :, :]

        msg_bits = self._message_to_bits(message, self._n_bits)

        # Extend 32 bits to 256 bits for TorchScript model
        if self._n_bits == 32:
            msg_bits_256 = msg_bits * 8
            msg_tensor = torch.tensor(
                [msg_bits_256], dtype=torch.float32, device=self.device
            )
        else:
            msg_tensor = torch.tensor(
                [msg_bits], dtype=torch.float32, device=self.device
            )

        was_training = model.training
        model.train()
        try:
            result = model.embed(embed_tensor, msg_tensor)
            if isinstance(result, torch.Tensor):
                watermarked = result
            elif isinstance(result, tuple):
                watermarked = result[0]
            else:
                watermarked = result["imgs_w"]
        finally:
            if not was_training:
                model.eval()

        # Convert back and place into image
        watermarked_np = self._tensor_to_numpy(watermarked)
        result_np = self._place_back_numpy(img_square, watermarked_np, x, y)

        # Draw corner brackets (using viewframe coords, not embed region)
        from viewframe import draw_viewframe

        img_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        vf_x, vf_y, vf_w, vf_h = (
            viewframe_coords["x"],
            viewframe_coords["y"],
            viewframe_coords["width"],
            viewframe_coords["height"],
        )
        draw_viewframe(
            img_bgr, vf_x, vf_y, vf_w, vf_h, method="distinctive"  # line_thickness auto-calculated
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        result_img = self._numpy_to_pil(img_rgb)
        binary_str = "".join(str(b) for b in msg_bits)

        # Return coords with backend info
        coords = {
            "x": viewframe_coords["x"],
            "y": viewframe_coords["y"],
            "width": viewframe_coords["width"],
            "height": viewframe_coords["height"],
            "x_percent": viewframe_coords["x_percent"],
            "y_percent": viewframe_coords["y_percent"],
            "width_percent": viewframe_coords["width_percent"],
            "height_percent": viewframe_coords["height_percent"],
            "viewframe_size": min(
                viewframe_coords["width"], viewframe_coords["height"]
            ),
            "backend": "videoseal",
        }
        return result_img, binary_str, coords

    def verify(
        self,
        image_source: Union[str, Image.Image],
        original_message: Optional[str] = None,
        margin_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Verify watermark using VideoSeal with viewframe support.

        Uses base class methods for:
        - Cropping to centered square
        - Calculating viewframe coordinates
        - Applying padding to exclude bracket arms

        Args:
            image_source: PIL Image or path to image
            original_message: Optional original message for comparison
            margin_pct: Margin percentage for viewframe (overrides default)

        Returns:
            Detection result dict
        """
        margin = margin_pct if margin_pct is not None else self._margin_percent

        # Use base class methods for shared procedures
        img_pil = (
            Image.open(image_source) if isinstance(image_source, str) else image_source
        )
        img_np = self._pil_to_numpy(img_pil.convert("RGB"))
        img_square = self._crop_to_centered_square(img_np)

        # Get viewframe and embed region coordinates
        viewframe_coords = self._get_viewframe_coords(
            img_square.shape[:2], margin_pct=margin
        )
        # Calculate dynamic padding based on image size
        min_dim = min(img_square.shape[0], img_square.shape[1])
        dynamic_padding = 1 if min_dim <= 150 else 2
        x, y, width, height = self._get_embed_region_coords(
            viewframe_coords, padding=dynamic_padding
        )

        # Extract embed region for detection
        embed_region = self._crop_to_embed_region_numpy(img_square, x, y, width, height)
        embed_pil = self._numpy_to_pil(embed_region)

        # Detect watermark (VideoSeal-specific)
        model = self._get_model()
        embed_tensor = self._pil_to_tensor(embed_pil)
        if embed_tensor.shape[1] == 4:
            embed_tensor = embed_tensor[:, :3, :, :]

        with torch.no_grad():
            detected = model.detect(embed_tensor)
            if isinstance(detected, torch.Tensor):
                preds = detected
            else:
                preds = detected["preds"]

        # Extract bits
        all_bits = (preds[0, 1:] > 0.0).long().cpu().tolist()

        # For 32-bit mode with 8x repetition: take consensus
        if self._n_bits == 32 and len(all_bits) >= 256:
            blocks = [all_bits[i * 32 : (i + 1) * 32] for i in range(8)]
            msg_bits = []
            for bit_idx in range(32):
                votes = sum(block[bit_idx] for block in blocks)
                msg_bits.append(1 if votes >= 4 else 0)
        else:
            msg_bits = all_bits[: self._n_bits]

        binary_str = "".join(str(b) for b in msg_bits)
        readable, ecc_valid, bitflips, accuracy = self._bits_to_message(msg_bits)

        # Calculate bit accuracy if original_message provided
        bit_accuracy = accuracy
        if original_message:
            original_bits = self._message_to_bits(original_message, self._n_bits)
            bit_accuracy = sum(
                b1 == b2 for b1, b2 in zip(msg_bits, original_bits)
            ) / len(original_bits)

        result = {
            "binary_message": binary_str,
            "readable_message": readable,
            "ecc_valid": ecc_valid,
            "corrected_bitflips": bitflips,
            "bit_accuracy": bit_accuracy,
            "viewframe": {
                "x": viewframe_coords["x"],
                "y": viewframe_coords["y"],
                "width": viewframe_coords["width"],
                "height": viewframe_coords["height"],
                "x_percent": viewframe_coords["x_percent"],
                "y_percent": viewframe_coords["y_percent"],
                "width_percent": viewframe_coords["width_percent"],
                "height_percent": viewframe_coords["height_percent"],
                "ratio": viewframe_coords["width_percent"]
                * viewframe_coords["height_percent"],
                "size": min(viewframe_coords["width"], viewframe_coords["height"]),
            },
        }

        return result


# ============================================================================
# ROCO Support - 3-char encoding with ECC (same as WAM)
# ============================================================================


class VideoSealBackendROCO(VideoSealBackend):
    """VideoSeal backend with ROCO 3-char encoding (32 bits).

    Uses the same ROCO encoding as the WAM algorithm (3 characters, 32 bits).
    Embeds directly on 32 bits without interleaving for simpler encoding/decoding.

    Usage:
        wm = VideoSealBackendROCO()
        img_w, binary, coords = wm.embed(img, "ABC")
        result = wm.verify(img_w)
        print(result['readable_message'])  # "ABC"
    """

    def __init__(self, device=None, margin_percent: float = 0.10):
        """Initialize VideoSealBackendROCO.

        Args:
            device: Device to run model on (default: cuda if available)
            margin_percent: Margin percentage for viewframe (default: 0.10 = 10%)
        """
        super().__init__(device, margin_percent)
        self._n_bits = 32  # ROCO uses 32 bits

    def embed_bytes_roco(
        self,
        image_bytes: bytes,
        message: str,
        margin_pct: float = 0.10,
        padding: int = 4,
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """Embed watermark using ROCO 3-char encoding via subprocess.

        Args:
            image_bytes: Raw image bytes
            message: 3-character message (uses ROCO encoding)
            margin_pct: Margin percentage for viewframe (default: 0.10 = 10%)
            padding: Pixels to pad from each edge to exclude bracket arms (default: 4)

        Returns:
            (watermarked_bytes, binary_string, coords)
        """
        if len(message) != 3:
            raise ValueError(f"ROCO requires exactly 3 chars, got {len(message)}")

        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            session_id = uuid.uuid4().hex[:8]
            script = _EMBED_SCRIPT_ROCO.format(
                site=PYTHON312_SITE,
                img_path=img_path,
                message=message,
                margin=margin_pct,
                padding=padding,
                session_id=session_id,
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

    def verify_bytes_roco(
        self,
        image_bytes: bytes,
        original_message: Optional[str] = None,
        margin_pct: Optional[float] = None,
        padding: int = 4,
    ) -> Dict[str, Any]:
        """Verify watermark using ROCO 3-char decoding via subprocess.

        Args:
            image_bytes: Raw image bytes
            original_message: Optional original message for comparison
            margin_pct: Margin percentage for viewframe (default: 0.10 = 10%)
            padding: Pixels to pad from each edge to exclude bracket arms (default: 4)

        Returns:
            Dict with decoded message and verification info
        """
        img_path = None
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, mode="wb"
            ) as f:
                f.write(image_bytes)
                img_path = f.name

            session_id = uuid.uuid4().hex[:8]
            margin = margin_pct if margin_pct is not None else self._margin_percent
            script = _VERIFY_SCRIPT_ROCO.format(
                site=PYTHON312_SITE,
                img_path=img_path,
                original_message=repr(original_message),
                margin=margin,
                session_id=session_id,
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
