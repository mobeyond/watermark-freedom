#!/usr/bin/env python3
"""
Quick backward compatibility test for the watermark system.
Ensures existing functionality still works after v2.0 changes.
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from core import WatermarkManager
from watermark_utils import roco_encode_to_binary_tensor, roco_decode_from_binary_tensor
from notebooks.inference_utils import unnormalize_img
from torchvision.utils import save_image


def test_embed_and_verify():
    """Test basic embed and verify cycle."""
    print("Test 1: Basic embed and verify cycle")

    # Create test image
    test_size = 512
    test_img = np.random.randint(0, 255, (test_size, test_size, 3), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(test_img).save(temp_path)

    try:
        manager = WatermarkManager()

        # Embed watermark
        original_msg = "ABC"
        img_tensor, binary_msg, coords = manager.embed(temp_path, original_msg, 'corners')

        # Save watermarked image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            watermarked_path = f.name
            save_image(unnormalize_img(img_tensor), watermarked_path)

        try:
            # Verify watermark
            result = manager.verify(watermarked_path, original_msg)

            assert result['ecc_valid'], "ECC validation failed"
            assert result['viewframe']['width'] > 0, "Viewframe width is 0"
            assert result['viewframe']['height'] > 0, "Viewframe height is 0"

            print(f"  ✓ Decoded: '{result['readable_message']}'")
            print(f"  ✓ ECC valid: {result['ecc_valid']}")
            print(f"  ✓ Bit flips corrected: {result['corrected_bitflips']}")

        finally:
            os.unlink(watermarked_path)

    finally:
        os.unlink(temp_path)

    print("  PASSED\n")


def test_roco_encoding():
    """Test ROCO encoding/decoding."""
    print("Test 2: ROCO encoding/decoding")

    # Allowed characters: A-Z, 4, 6, 7, 9, ., #
    test_messages = ["ABC", "XYZ", "QWE", "467", "A9#", "A.B"]

    for msg in test_messages:
        encoded = roco_encode_to_binary_tensor(msg)
        decoded, valid, flips = roco_decode_from_binary_tensor(encoded)

        assert decoded == msg, f"Message mismatch: '{msg}' -> '{decoded}'"
        assert valid, f"Message not valid: '{msg}'"

    print(f"  ✓ All {len(test_messages)} test messages encoded/decoded correctly")
    print("  PASSED\n")


def test_mask_modes():
    """Test different mask modes."""
    print("Test 3: Different mask modes")

    test_size = 512
    test_img = np.random.randint(0, 255, (test_size, test_size, 3), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(test_img).save(temp_path)

    try:
        manager = WatermarkManager()

        # Test corners mode
        img_tensor, _, coords = manager.embed(temp_path, "XYZ", 'corners')
        assert coords['x_percent'] > 0, "Corners mode: x_percent invalid"
        print(f"  ✓ Corners mode: x={coords['x']}, y={coords['y']}")

        # Test percentage mode
        params = {'x_percent': 0.2, 'y_percent': 0.2, 'width_percent': 0.3, 'height_percent': 0.3}
        img_tensor, _, coords = manager.embed(temp_path, "XYZ", 'percentage', params)
        assert abs(coords['x_percent'] - 0.2) < 0.01, "Percentage mode: x_percent invalid"
        print(f"  ✓ Percentage mode: x%={coords['x_percent']:.2f}, y%={coords['y_percent']:.2f}")

        # Test pixels mode
        params = {'x': 100, 'y': 100, 'width': 100, 'height': 100}
        img_tensor, _, coords = manager.embed(temp_path, "XYZ", 'pixels', params)
        assert coords['x'] == 100, "Pixels mode: x invalid"
        print(f"  ✓ Pixels mode: x={coords['x']}, y={coords['y']}")

    finally:
        os.unlink(temp_path)

    print("  PASSED\n")


def test_verify_tensor():
    """Test verify_tensor method (used by optimizer)."""
    print("Test 4: verify_tensor method")

    test_size = 512
    test_img = np.random.randint(0, 255, (test_size, test_size, 3), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(test_img).save(temp_path)

    try:
        manager = WatermarkManager()

        # Embed watermark (max 3 characters)
        original_msg = "XYZ"
        img_tensor, _, _ = manager.embed(temp_path, original_msg, 'corners')

        # Use verify_tensor
        result = manager.verify_tensor(img_tensor, original_msg)

        assert 'correct_bits' in result, "correct_bits not in result"
        assert 'ecc_valid' in result, "ecc_valid not in result"

        print(f"  ✓ correct_bits: {result['correct_bits']}")
        print(f"  ✓ ecc_valid: {result['ecc_valid']}")

    finally:
        os.unlink(temp_path)

    print("  PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("BACKWARD COMPATIBILITY TESTS")
    print("=" * 60 + "\n")

    try:
        test_embed_and_verify()
        test_roco_encoding()
        test_mask_modes()
        test_verify_tensor()

        print("=" * 60)
        print("ALL BACKWARD COMPATIBILITY TESTS PASSED")
        print("=" * 60)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
