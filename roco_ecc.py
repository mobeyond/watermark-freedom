#!/usr/bin/env python3
"""
ROCO (Resilient Optimized Code Operation) ECC
Handles error correction code (ECC) generation and verification.
"""

import bchlib
from typing import Tuple

# BCH object for m=6, t=2 (n=63, but we will shorten to 32 bits)
# New bchlib 2.x API: BCH(polynomial, t)
# Primitive polynomial for GF(2^6): x^6 + x + 1 = 0x43
BCH = bchlib.BCH(0x43, 2)

ALPHABET = "234679ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET_MAP = {char: i for i, char in enumerate(ALPHABET)}

# 16 bits payload, 16 bits ECC, 32 bits total (4 bytes)
def encode_with_ecc(payload_bytes: bytes) -> bytes:
    """Encode 16-bit payload (2 bytes) with BCH ECC (2 bytes)."""
    if len(payload_bytes) != 2:
        raise ValueError("Payload must be exactly 2 bytes (16 bits)")
    ecc = BCH.encode(payload_bytes)  # 2 bytes ECC
    codeword = payload_bytes + ecc  # 4 bytes
    return codeword

def decode_with_ecc(codeword: bytes) -> Tuple[bytes, bool, int]:
    """Decode 32-bit codeword (4 bytes) with BCH ECC."""
    if len(codeword) != 4:
        return b"", False, -1
    data = bytearray(codeword[:2])
    ecc = bytearray(codeword[2:])

    # New bchlib 2.x API: decode() returns (bitflips, corrected_data, corrected_ecc)
    # Old API: decode() returns just bitflips, then need correct()
    try:
        result = BCH.decode(data, ecc)
        if isinstance(result, tuple):
            bitflips, data, ecc = result
        else:
            # Old API: need to call correct()
            bitflips = result
            if bitflips > 0 and hasattr(BCH, 'correct'):
                BCH.correct(data, ecc)
    except Exception:
        return bytes(data), False, -1
    
    # Check if properly corrected
    is_valid = (bitflips >= 0 and bitflips <= BCH.t)
    return bytes(data), is_valid, bitflips
