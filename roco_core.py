#!/usr/bin/env python3
"""
ROCO (Resilient Optimized Code Operation) Core Encoder/Decoder
Handles the transformation of payload characters into bits and back.
"""

from roco_ecc import encode_with_ecc, decode_with_ecc, ALPHABET, ALPHABET_MAP
from typing import Tuple

PAYLOAD_BITS = 16
PAYLOAD_BYTES = 2

# Define allowed characters and their 5-bit encoding (0-31)
# Note: Padding (.) is assigned to 0 for better clarity
ALLOWED_CHARS = (
    '.'                           # 0 (padding)
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 1-26
    '4679#'                       # 27-31
)


def encode_to_bits(payload: str) -> int:
    """Encode up to 3 alphanumeric characters to 16-bit integer.
    
    First bit is version/notation label (0 by default).
    Next 15 bits encode up to 3 characters from the allowed set:
    - A-Z (26 letters)
    - 4,6,7,9 (digits)
    - . (period, used for padding)
    - # (sharp symbol)
    """
    # Create mapping from character to 5-bit value
    CHAR_TO_BITS = {c: i for i, c in enumerate(ALLOWED_CHARS)}
    
    if len(payload) > 3:
        raise ValueError("Payload cannot exceed 3 characters")
    
    # Pad with '.' if needed
    payload = payload.ljust(3, '.')
    
    # Start with version bit 0 (leftmost bit)
    result = 0
    
    # Encode each character (3 characters * 5 bits = 15 bits)
    for char in payload:
        if char not in CHAR_TO_BITS:
            raise ValueError(f"Invalid character '{char}'. Allowed: A-Z, 4,6,7,9, ., #")
        result = (result << 5) | CHAR_TO_BITS[char]
    
    # Ensure we only use the 15 bits for payload (version bit will be added)
    result &= 0x7FFF  # Clear any bits beyond 15
    
    # Version bit is 0 (left shift result by 1)
    return result << 1

def decode_from_bits(bits: int) -> str:
    """Decode 16-bit integer to original payload (1-3 characters).
    
    First bit is version/notation label (should be 0).
    Next 15 bits encode up to 3 characters from the allowed set.
    Returns the original payload with padding removed.
    """
    # Extract the 15-bit payload (ignore version bit for now)
    payload_bits = (bits >> 1) & 0x7FFF
    
    # Extract characters in reverse order (LSB first)
    chars = []
    for _ in range(3):
        char_code = payload_bits & 0x1F
        chars.append(ALLOWED_CHARS[char_code])
        payload_bits >>= 5
    
    # The characters were stored with the first character in the most significant bits
    # So we need to reverse them to get the original order
    chars = chars[::-1]
    
    # Remove trailing padding and return
    return ''.join(chars).rstrip('.')

def encode_string(payload: str) -> str:
    """Encode payload to hex string representation."""
    data_bits = encode_to_bits(payload)
    data_bytes = data_bits.to_bytes(PAYLOAD_BYTES, 'big')
    codeword_bytes = encode_with_ecc(data_bytes)
    return codeword_bytes.hex().upper()

def decode_string(hex_string: str) -> Tuple[str, bool, int]:
    """Decode hex string to payload with error correction status."""
    try:
        codeword_bytes = bytes.fromhex(hex_string)
        corrected_data, valid, bitflips = decode_with_ecc(codeword_bytes)
        if corrected_data:
            data_bits = int.from_bytes(corrected_data, 'big') & 0xFFFF
            decoded_payload = decode_from_bits(data_bits)
            return decoded_payload, valid, bitflips
        else:
            return "DECODE_FAIL", False, -1
    except (ValueError, TypeError):
        return "DECODE_FAIL", False, -1
