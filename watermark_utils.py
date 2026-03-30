import os
import numpy as np
from PIL import Image
import torch
import cv2
from flask import jsonify
from typing import Tuple

from roco_core import encode_to_bits, decode_from_bits
from roco_ecc import encode_with_ecc, decode_with_ecc


def load_image(image_file):
    """Load an image file and convert to RGB format."""
    img = Image.open(image_file).convert("RGB")
    return img


def crop_to_centered_square(image):
    """Returns the largest centered square crop from the input image.
    Works with both PIL Image and numpy arrays (cv2 format)."""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2

    if isinstance(image, np.ndarray):
        return image[top:top+min_dim, left:left+min_dim]
    else:
        return image.crop((left, top, left + min_dim, top + min_dim))


def pil_to_cv2(img):
    """Convert PIL Image to cv2 format (BGR)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    """Convert cv2 format image (BGR) to PIL Image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def create_mask_from_coords(img_tensor, x, y, width, height):
    """Create a mask using pixel coordinates."""
    batch_size, channels, img_height, img_width = img_tensor.shape

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Pixel coordinates must be non-negative and dimensions must be positive")
    if x + width > img_width or y + height > img_height:
        raise ValueError("Mask region exceeds image dimensions")

    mask = torch.zeros((batch_size, 1, img_height, img_width), device=img_tensor.device)
    mask[:, :, y:y+height, x:x+width] = 1.0
    return mask


def create_mask_from_percentages(img_tensor, x_percent, y_percent, width_percent, height_percent):
    """Create a mask using percentage coordinates (0-1)."""
    h, w = img_tensor.shape[2:]
    x = int(w * x_percent)
    y = int(h * y_percent)
    width = int(w * width_percent)
    height = int(h * height_percent)
    return create_mask_from_coords(img_tensor, x, y, width, height)


def validate_pixel_coords(w, h, x, y, width, height):
    """Validate pixel coordinates and dimensions."""
    if any(val < 0 for val in [x, y, width, height]):
        return False, "Pixel values must be non-negative integers"
    if x + width > w or y + height > h:
        return False, f"Watermark region exceeds image dimensions (image: {w}x{h}, mask: {x}+{width}x{y}+{height})"
    return True, None


def validate_percentage_coords(x_percent, y_percent, width_percent, height_percent):
    """Validate percentage coordinates (0-1)."""
    if not all(0 <= val <= 1 for val in [x_percent, y_percent, width_percent, height_percent]):
        return False, "Percentage values must be between 0 and 1"
    return True, None


def create_error_response(message, status_code=400, additional_info=None):
    """Create a standardized error response."""
    response = {"error": message}
    if additional_info:
        response.update(additional_info)
    return jsonify(response), status_code


def roco_encode_to_binary_tensor(payload: str) -> torch.Tensor:
    """Encodes a string payload using ROCO into a 32-bit binary tensor.
    Payload is limited to 3 chars from the ROCO alphabet."""
    data_bits = encode_to_bits(payload)
    payload_bytes = data_bits.to_bytes(2, 'big')
    codeword_bytes = encode_with_ecc(payload_bytes)
    binary_str = ''.join(format(byte, '08b') for byte in codeword_bytes)
    binary_tensor = torch.tensor([int(b) for b in binary_str], dtype=torch.float32)
    return binary_tensor


def roco_decode_from_binary_tensor(binary_tensor: torch.Tensor) -> Tuple[str, bool, int]:
    """Decodes a 32-bit binary tensor using ROCO.
    Returns: (decoded_payload, is_valid, bitflips_corrected)"""
    binary_str = "".join([str(int(b.item())) for b in binary_tensor])
    
    if len(binary_str) != 32:
        return "INVALID_LENGTH", False, -1
    codeword_bytes = int(binary_str, 2).to_bytes(4, 'big')
    
    corrected_data, is_valid, bitflips = decode_with_ecc(codeword_bytes)
    
    if not corrected_data:
        return "DECODE_FAIL", False, bitflips
        
    data_bits = int.from_bytes(corrected_data, 'big')
    decoded_payload = decode_from_bits(data_bits)
    
    return decoded_payload, is_valid, bitflips
