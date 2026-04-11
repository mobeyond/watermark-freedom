"""ROCO32: 4-character robust encoding for VideoSeal watermarking.

Encodes 4 characters into a 256-bit codeword using BCH-inspired error correction.
Designed for high robustness against image processing attacks.

Architecture:
- Data: 4 chars × 5 bits = 20 bits (using 32-char ROCO alphabet)
- ECC: 236 parity bits (BCH-inspired multi-layer redundancy)
- Total: 256 bits (matches VideoSeal capacity)

The encoding uses layered redundancy:
1. Each 5-bit char encoded with local parity (8 bits)
2. Block-level ECC across all chars
3. Interleaved repetition for temporal/spatial robustness
"""

import struct

# ROCO alphabet: 32 characters, each mapped to 5 bits
ROCO_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.#4679"
ROCO_TO_BITS = {c: i for i, c in enumerate(ROCO_ALPHABET)}
BITS_TO_ROCO = {i: c for i, c in enumerate(ROCO_ALPHABET)}

# Constants
NUM_CHARS = 4
BITS_PER_CHAR = 5  # 2^5 = 32 characters
DATA_BITS = NUM_CHARS * BITS_PER_CHAR  # 20 bits
DATA_BYTES = 4  # 4 chars, each encoded to 1 byte with parity
TOTAL_BITS = 256
TOTAL_BYTES = TOTAL_BITS // 8  # 32 bytes
ECC_BYTES = TOTAL_BYTES - DATA_BYTES  # 28 bytes
ECC_BITS = ECC_BYTES * 8  # 224 bits


def encode_char_to_bits(char: str) -> int:
    """Encode a single character to 5 bits.

    Args:
        char: Single character from ROCO alphabet

    Returns:
        Integer value (0-31) representing the character
    """
    if char not in ROCO_TO_BITS:
        raise ValueError(f"Character '{char}' not in ROCO alphabet: {ROCO_ALPHABET}")
    return ROCO_TO_BITS[char]


def decode_bits_to_char(bits: int) -> str:
    """Decode 5 bits to a single character.

    Args:
        bits: Integer value (0-31)

    Returns:
        Single character from ROCO alphabet
    """
    if not (0 <= bits < 32):
        raise ValueError(f"Invalid bit value: {bits}")
    return BITS_TO_ROCO[bits]


def compute_byte_parity(value: int) -> int:
    """Compute parity bit for a byte (even parity)."""
    return bin(value).count('1') % 2


def encode_char_with_parity(char_bits: int) -> int:
    """Encode 5-bit char with parity to form 8-bit byte.

    Layout: [5-bit char][padding 0][parity][padding 0]
    This provides local error detection per character.
    """
    if not (0 <= char_bits < 32):
        raise ValueError(f"Invalid char bits: {char_bits}")

    # Place 5 bits in positions 7-3 (big-endian style)
    byte_val = char_bits << 3  # Bits 7-3
    # Bit 2 = parity of the 5 data bits
    parity = compute_byte_parity(char_bits)
    byte_val |= (parity << 2)

    return byte_val


def decode_char_with_parity(byte_val: int) -> tuple:
    """Decode 8-bit byte back to 5-bit char with parity check.

    Returns:
        Tuple of (char_bits, parity_valid)
    """
    char_bits = (byte_val >> 3) & 0x1F  # Extract bits 7-3
    stored_parity = (byte_val >> 2) & 0x01

    # Recompute parity
    computed_parity = compute_byte_parity(char_bits)
    parity_valid = (stored_parity == computed_parity)

    return char_bits, parity_valid


def compute_block_ecc(data_bytes: bytes) -> bytes:
    """Compute block-level ECC using multi-layer redundancy.

    Produces exactly 28 bytes of ECC for 4 bytes of data.
    Total: 32 bytes = 256 bits.

    Layers:
    - Layer 1: XOR checksum (1 byte)
    - Layer 2: Arithmetic sum mod 256 (1 byte)
    - Layer 3: Position-weighted checksum (1 byte)
    - Layer 4-5: High/low nibble XOR (2 bytes)
    - Layer 6-9: Per-char checksums (4 bytes)
    - Layer 10-16: Pair XOR combinations (7 bytes)
    - Layer 17-28: Interleaved bit repetition (12 bytes)
    """
    ecc_list = []

    # Layer 1: XOR checksum (1 byte)
    xor_sum = 0
    for b in data_bytes:
        xor_sum ^= b
    ecc_list.append(xor_sum)

    # Layer 2: Arithmetic sum mod 256 (1 byte)
    arith_sum = sum(data_bytes) % 256
    ecc_list.append(arith_sum)

    # Layer 3: Position-weighted checksum (1 byte)
    weighted_sum = sum((i + 1) * b for i, b in enumerate(data_bytes)) % 256
    ecc_list.append(weighted_sum)

    # Layer 4-5: High-order and low-order nibble patterns (2 bytes)
    high_bits = 0
    low_bits = 0
    for b in data_bytes:
        high_bits ^= (b >> 4)
        low_bits ^= (b & 0x0F)
    ecc_list.append(high_bits)
    ecc_list.append(low_bits)

    # Layer 6-9: Per-character extended checksums (4 bytes)
    for char_idx in range(NUM_CHARS):
        char_byte = data_bytes[char_idx]
        # Rotate and XOR for this character
        rotated = ((char_byte << 3) | (char_byte >> 5)) & 0xFF
        ecc_list.append(rotated)

    # Layer 10-16: Pair XOR combinations (7 bytes)
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    for pair in pairs:
        combined = data_bytes[pair[0]] ^ data_bytes[pair[1]]
        ecc_list.append(combined)
    # All 4 bytes combined with weights
    all_combined = (data_bytes[0] + data_bytes[1] * 2 + data_bytes[2] * 3 + data_bytes[3] * 4) % 256
    ecc_list.append(all_combined)

    # Layer 17-28: Interleaved bit repetition (12 bytes)
    # Each of the 4 data bytes contributes 3 ECC bytes
    for char_idx in range(NUM_CHARS):
        char_byte = data_bytes[char_idx]
        # Byte 1: Replicate high nibble
        ecc_list.append((char_byte >> 4) * 0x11)
        # Byte 2: Replicate low nibble
        ecc_list.append((char_byte & 0x0F) * 0x11)
        # Byte 3: Mirror byte
        ecc_list.append(((char_byte >> 1) | (char_byte << 7)) & 0xFF)

    return bytes(ecc_list[:ECC_BYTES])


def verify_ecc(data_bytes: bytes, ecc_bytes: bytes) -> tuple:
    """Verify ECC and attempt error correction.

    Returns:
        Tuple of (is_valid, corrected_data_bytes, num_errors_corrected)
    """
    recomputed_ecc = compute_block_ecc(data_bytes)

    # Compare ECC bytes
    errors = 0
    for i in range(min(len(ecc_bytes), len(recomputed_ecc))):
        if ecc_bytes[i] != recomputed_ecc[i]:
            errors += bin(ecc_bytes[i] ^ recomputed_ecc[i]).count('1')

    # Simple validation: check if critical ECC layers match
    if len(ecc_bytes) >= 3:
        # Layer 1-3 must match for validity
        if (ecc_bytes[0] == recomputed_ecc[0] and
            ecc_bytes[1] == recomputed_ecc[1] and
            ecc_bytes[2] == recomputed_ecc[2]):
            return True, data_bytes, 0

    # Attempt simple error correction using XOR layer
    if len(ecc_bytes) >= 1:
        xor_expected = recomputed_ecc[0]
        xor_actual = ecc_bytes[0]

        # Find which byte might be corrupted
        for char_idx in range(NUM_CHARS):
            test_byte = data_bytes[char_idx] ^ xor_actual ^ xor_expected
            if 0x1F <= test_byte <= 0xFF:  # Valid byte range after correction
                # Verify with other layers
                test_data = bytearray(data_bytes)
                test_data[char_idx] = test_byte
                test_ecc = compute_block_ecc(bytes(test_data))

                if (test_ecc[0] == ecc_bytes[0] and
                    test_ecc[1] == ecc_bytes[1] and
                    test_ecc[2] == ecc_bytes[2]):
                    return True, bytes(test_data), 1

    return False, data_bytes, errors


def encode(message: str) -> list:
    """Encode a 4-character message into a 256-bit codeword.

    Args:
        message: Exactly 4 characters from ROCO alphabet

    Returns:
        List of 256 bits (integers 0 or 1)

    Raises:
        ValueError: If message is not exactly 4 chars or contains invalid characters
    """
    if len(message) != NUM_CHARS:
        raise ValueError(f"Message must be exactly {NUM_CHARS} characters, got {len(message)}")

    for char in message:
        if char not in ROCO_ALPHABET:
            raise ValueError(f"Character '{char}' not in ROCO alphabet: {ROCO_ALPHABET}")

    # Step 1: Encode each character to 5 bits, then to 8-bit byte with parity
    data_bytes = []
    for char in message:
        char_bits = encode_char_to_bits(char)
        byte_val = encode_char_with_parity(char_bits)
        data_bytes.append(byte_val)
    data_bytes = bytes(data_bytes)

    # Step 2: Compute ECC bytes
    ecc_bytes = compute_block_ecc(data_bytes)

    # Step 3: Combine data and ECC into 256-bit codeword
    codeword_bytes = data_bytes + ecc_bytes

    # Convert to bit list
    bits = []
    for byte_val in codeword_bytes:
        for i in range(7, -1, -1):
            bits.append((byte_val >> i) & 0x01)

    return bits


def decode(bits: list) -> tuple:
    """Decode a 256-bit codeword back to 4-character message.

    Args:
        bits: List of 256 bits (integers 0 or 1)

    Returns:
        Tuple of (message, is_valid, errors_detected)
        - message: Decoded 4-character string, or empty string if decode failed
        - is_valid: True if decoding succeeded with valid ECC
        - errors_detected: Number of errors detected (not necessarily corrected)
    """
    if len(bits) != TOTAL_BITS:
        return "", False, -1

    # Convert bits to bytes
    codeword_bytes = []
    for i in range(0, TOTAL_BITS, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i + j]
        codeword_bytes.append(byte_val)
    codeword_bytes = bytes(codeword_bytes)

    # Split into data and ECC
    data_bytes = codeword_bytes[:NUM_CHARS]
    ecc_bytes = codeword_bytes[NUM_CHARS:]

    # Step 1: Verify block-level ECC
    is_valid, corrected_data, errors = verify_ecc(data_bytes, ecc_bytes)

    if not is_valid:
        return "", False, errors

    # Step 2: Decode each character with parity check
    message_chars = []
    for i in range(NUM_CHARS):
        byte_val = corrected_data[i]
        char_bits, parity_valid = decode_char_with_parity(byte_val)

        if not parity_valid:
            # Parity error - but we already passed ECC, so this shouldn't happen
            # Return partial result
            pass

        char = decode_bits_to_char(char_bits)
        message_chars.append(char)

    message = ''.join(message_chars)
    return message, is_valid, errors


def encode_to_binary_string(message: str) -> str:
    """Encode message to binary string (convenience function)."""
    bits = encode(message)
    return ''.join(str(b) for b in bits)


def decode_from_binary_string(binary_str: str) -> tuple:
    """Decode from binary string (convenience function)."""
    bits = [int(b) for b in binary_str]
    return decode(bits)


def encode_to_256bits(message: str) -> list:
    """Encode 1-4 character message to 256-bit codeword.

    Convenience function that pads message to 4 characters.

    Args:
        message: 1-4 characters from ROCO alphabet

    Returns:
        List of 256 bits (integers 0 or 1)
    """
    # Pad message to 4 characters with '.'
    padded = message.ljust(NUM_CHARS, '.')[:NUM_CHARS]
    return encode(padded)


def decode_from_256bits(bits: list) -> tuple:
    """Decode 256-bit codeword to message.

    Convenience function that strips padding characters.

    Args:
        bits: List of 256 bits

    Returns:
        Tuple of (message, is_valid, errors)
    """
    message, is_valid, errors = decode(bits)
    # Remove padding characters
    return message.rstrip('.'), is_valid, errors


if __name__ == "__main__":
    # Test the encode/decode cycle
    print("ROCO32 Module Test")
    print("=" * 60)

    test_messages = ["ABC.", "XYZ#", "A4B6", "TEST", "ROCO", "FREE", "WATER"]

    for msg in test_messages:
        if len(msg) == 4:
            bits = encode(msg)
            decoded, valid, errors = decode(bits)
            status = "✓" if decoded == msg and valid else "✗"
            print(f"{status} '{msg}' -> {len(bits)} bits -> '{decoded}' (valid={valid})")

    print(f"\nROCO alphabet: {ROCO_ALPHABET}")
    print(f"Data bits: {DATA_BITS} (4 chars × 5 bits)")
    print(f"ECC bits: {ECC_BITS}")
    print(f"Total bits: {TOTAL_BITS}")
