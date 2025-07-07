import os
import numpy as np
from PIL import Image
import torch
import cv2
from flask import jsonify

def init_model(device=None):
    """Initialize the watermark model with default paths"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = "checkpoints"
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')

    from notebooks.inference_utils import load_model_from_checkpoint
    model = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

    return model

def load_image(image_file):
    """Load an image file and convert to RGB format"""
    img = Image.open(image_file).convert("RGB")
    return img

def crop_to_centered_square(image):
    """
    Returns the largest centered square crop from the input image.
    Works with both PIL Image and numpy arrays (cv2 format).
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:  # PIL.Image
        w, h = image.size

    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2

    if isinstance(image, np.ndarray):
        return image[top:top+min_dim, left:left+min_dim]
    else:
        return image.crop((left, top, left + min_dim, top + min_dim))

def pil_to_cv2(img):
    """Convert PIL Image to cv2 format"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    """Convert cv2 format image to PIL Image"""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def create_mask_from_coords(img_tensor, x, y, width, height):
    """Create a mask using pixel coordinates"""
    batch_size, channels, img_height, img_width = img_tensor.shape

    # Validate pixel coordinates
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Pixel coordinates must be non-negative and dimensions must be positive")
    if x + width > img_width or y + height > img_height:
        raise ValueError("Mask region exceeds image dimensions")

    # Create mask
    mask = torch.zeros((batch_size, 1, img_height, img_width), device=img_tensor.device)
    mask[:, :, y:y+height, x:x+width] = 1.0
    return mask

def create_mask_from_percentages(img_tensor, x_percent, y_percent, width_percent, height_percent):
    """Create a mask using percentage coordinates (0-1)"""
    h, w = img_tensor.shape[2:]

    # Convert percentages to pixels
    x = int(w * x_percent)
    y = int(h * y_percent)
    width = int(w * width_percent)
    height = int(h * height_percent)

    return create_mask_from_coords(img_tensor, x, y, width, height)

def calculate_checksum(binary_str):
    """Calculate a simple checksum for error detection"""
    checksum = 0
    for bit in binary_str:
        checksum ^= int(bit)
    return str(checksum)

def robust_str_to_binary(msg_str, nbits=32):
    """
    Convert a string to a binary tensor with error correction.
    Uses a portion of the bits for checksum while preserving message bits.
    Format: [message bits][checksum bits]
    """
    # Calculate how many bits we can use for the actual message
    # Reserve 4 bits for checksum and error correction
    message_bits = nbits - 4

    # Convert message to binary
    binary_str = ''.join(format(ord(c), '08b') for c in msg_str)

    # Truncate or pad to fit available message bits
    if len(binary_str) > message_bits:
        print(f"Warning: Message '{msg_str}' is too long for {message_bits} bits. Truncating...")
        binary_str = binary_str[:message_bits]
    elif len(binary_str) < message_bits:
        binary_str = binary_str.ljust(message_bits, '0')

    # Calculate checksum
    checksum = calculate_checksum(binary_str)

    # Combine message and checksum
    full_binary = binary_str + checksum

    # Convert to tensor
    binary_tensor = torch.tensor([int(b) for b in full_binary], dtype=torch.float32)

    return binary_tensor

def validate_pixel_coords(w, h, x, y, width, height):
    """Validate pixel coordinates and dimensions"""
    if any(val < 0 for val in [x, y, width, height]):
        return False, "Pixel values must be non-negative integers"

    if x + width > w or y + height > h:
        return False, f"Watermark region exceeds image dimensions (image: {w}x{h}, mask: {x}+{width}x{y}+{height})"

    return True, None

def validate_percentage_coords(x_percent, y_percent, width_percent, height_percent):
    """Validate percentage coordinates (0-1)"""
    if not all(0 <= val <= 1 for val in [x_percent, y_percent, width_percent, height_percent]):
        return False, "Percentage values must be between 0 and 1"

    return True, None

def create_error_response(message, status_code=400, additional_info=None):
    """Create a standardized error response"""
    response = {"error": message}

    if additional_info:
        response.update(additional_info)

    return jsonify(response), status_code
