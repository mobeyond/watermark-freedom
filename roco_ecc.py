#!/usr/bin/env python3
"""
ROCO (Resilient Optimized Code Operation) ECC
Handles error correction code (ECC) generation and verification.
"""

import bchlib
from typing import Tuple

# BCH object for m=6, t=2 (n=63, but we will shorten to 32 bits)
BCH = bchlib.BCH(t=2, m=6)

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
    # 1. DECODE: Find bit error locations
    try:
        bitflips = BCH.decode(data, ecc)
    except Exception:
        return data, False, -1
    # 2. CORRECT: Fix the errors in-place
    if bitflips > 0:
        try:
            BCH.correct(data, ecc)
        except Exception:
            return data, False, -1
    # Final check
    try:
        final_flips = BCH.decode(data, ecc)
    except Exception:
        return data, False, -1
    is_valid = (final_flips == 0)
    return bytes(data), is_valid, bitflips
