import os
import numpy as np
from PIL import Image
import torch
import cv2
from flask import jsonify
from typing import Tuple

# ROCO imports
from roco_core import encode_to_bits, decode_from_bits
from roco_ecc import encode_with_ecc, decode_with_ecc


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

def robust_binary_to_str(binary_tensor, nbits=32):
    """
    Convert a binary tensor back to a string, with checksum verification.
    """
    binary_str = "".join([str(int(b)) for b in binary_tensor])
    
    # Separate message and checksum
    message_bits = nbits - 4
    msg_part = binary_str[:message_bits]
    checksum_part = binary_str[message_bits:]
    
    # Verify checksum
    calculated_checksum = calculate_checksum(msg_part)
    checksum_ok = (calculated_checksum == checksum_part)
    
    # Convert binary to string
    readable_message = ""
    for i in range(0, len(msg_part), 8):
        byte = msg_part[i:i+8]
        if '1' not in byte:  # Stop if we hit padding
            break
        char_code = int(byte, 2)
        if char_code > 0:
            readable_message += chr(char_code)
            
    return readable_message, checksum_ok

def roco_encode_to_binary_tensor(payload: str) -> torch.Tensor:
    """
    Encodes a string payload using ROCO into a 32-bit binary tensor.
    Payload is limited to 3 chars from the ROCO alphabet.
    """
    # 1. Encode payload string to 16-bit integer
    data_bits = encode_to_bits(payload)
    
    # 2. Convert to 2-byte payload
    payload_bytes = data_bits.to_bytes(2, 'big')
    
    # 3. Encode with ECC to get 4-byte (32-bit) codeword
    codeword_bytes = encode_with_ecc(payload_bytes)
    
    # 4. Convert codeword to a binary string
    binary_str = ''.join(format(byte, '08b') for byte in codeword_bytes)
    
    # 5. Convert binary string to a PyTorch tensor of floats
    binary_tensor = torch.tensor([int(b) for b in binary_str], dtype=torch.float32)
    
    return binary_tensor

def roco_decode_from_binary_tensor(binary_tensor: torch.Tensor) -> Tuple[str, bool, int]:
    """
    Decodes a 32-bit binary tensor using ROCO.
    Returns the decoded payload, a validity flag, and the number of bitflips corrected.
    """
    # 1. Convert tensor back to a binary string
    binary_str = "".join([str(int(b.item())) for b in binary_tensor])
    
    # 2. Convert 32-bit binary string to 4 bytes
    if len(binary_str) != 32:
        return "INVALID_LENGTH", False, -1
    codeword_bytes = int(binary_str, 2).to_bytes(4, 'big')
    
    # 3. Decode with ECC
    corrected_data, is_valid, bitflips = decode_with_ecc(codeword_bytes)
    
    if not corrected_data:
        return "DECODE_FAIL", False, bitflips
        
    # 4. Convert corrected 2-byte payload back to 16-bit integer
    data_bits = int.from_bytes(corrected_data, 'big')
    
    # 5. Decode bits to get the final string
    decoded_payload = decode_from_bits(data_bits)
    
    return decoded_payload, is_valid, bitflips