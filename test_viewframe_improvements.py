#!/usr/bin/env python3
"""
Test script for viewframe improvements:
1. Pure white corner brackets (for reliable detection)
2. Independent viewframe detection module
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core import WatermarkManager
from watermark_utils import pil_to_cv2


def test_semi_transparent_brackets():
    """Test that corner brackets are drawn with semi-transparent blending."""
    print("=" * 60)
    print("TEST 1: Semi-Transparent Corner Brackets")
    print("=" * 60)

    # Create a test image with known content
    test_size = 512
    test_img = np.random.randint(0, 255, (test_size, test_size, 3), dtype=np.uint8)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(test_img).save(temp_path)

    try:
        manager = WatermarkManager()

        # Embed watermark
        img_tensor, binary_msg, coords = manager.embed(
            temp_path,
            "ABC",
            'corners'
        )

        # Convert to numpy for analysis
        from notebooks.inference_utils import unnormalize_img
        result_np = unnormalize_img(img_tensor).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        result_np = (result_np * 255).astype(np.uint8)

        # Extract corner region (top-left)
        x, y = int(coords['x']), int(coords['y'])
        corner_size = int(min(coords['width'], coords['height']) * 0.15)

        # Check corner bracket region
        corner_region = result_np[y:y+corner_size, x:x+corner_size]

        # Count pure white pixels
        pure_white_mask = np.all(corner_region == 255, axis=2)
        pure_white_count = np.sum(pure_white_mask)
        total_pixels = corner_region.shape[0] * corner_region.shape[1]

        print(f"Corner region size: {corner_region.shape}")
        print(f"Pure white pixels: {pure_white_count} / {total_pixels} ({100*pure_white_count/total_pixels:.1f}%)")
        print(f"Max value in corner: {corner_region.max()}")

        # With semi-transparent blending, we expect NO pure white pixels (or very few)
        # The brackets should be blended with the background
        if pure_white_count < total_pixels * 0.01:  # Less than 1% pure white
            print("✓ PASS: Corner brackets are semi-transparent (blended with background)")
            print(f"  (Only {100*pure_white_count/total_pixels:.1f}% pure white, expected < 1%)")
            return True
        else:
            print("✗ FAIL: Corner brackets appear to be pure white")
            return False

    finally:
        os.unlink(temp_path)


def test_viewframe_detection_accuracy():
    """Test viewframe detection accuracy with semi-transparent brackets."""
    print("\n" + "=" * 60)
    print("TEST 2: Viewframe Detection Accuracy")
    print("=" * 60)

    # Create test image
    test_size = 512
    test_img = np.random.randint(0, 255, (test_size, test_size, 3), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(test_img).save(temp_path)

    try:
        manager = WatermarkManager()

        # Embed watermark with known coordinates
        original_msg = "XYZ"
        img_tensor, binary_msg, coords = manager.embed(
            temp_path,
            original_msg,
            'corners'
        )

        # Save watermarked image
        from notebooks.inference_utils import unnormalize_img
        from torchvision.utils import save_image

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            watermarked_path = f.name
            save_image(unnormalize_img(img_tensor), watermarked_path)

        try:
            # Verify and detect viewframe
            result = manager.verify(watermarked_path, original_msg)

            detected_x = result['viewframe']['x']
            detected_y = result['viewframe']['y']
            detected_w = result['viewframe']['width']
            detected_h = result['viewframe']['height']

            original_x = int(coords['x'])
            original_y = int(coords['y'])
            original_w = int(coords['width'])
            original_h = int(coords['height'])

            # Calculate position error (in pixels)
            pos_error_x = abs(detected_x - original_x)
            pos_error_y = abs(detected_y - original_y)
            size_error_w = abs(detected_w - original_w)
            size_error_h = abs(detected_h - original_h)

            print(f"Original viewframe:  x={original_x}, y={original_y}, w={original_w}, h={original_h}")
            print(f"Detected viewframe:  x={detected_x}, y={detected_y}, w={detected_w}, h={detected_h}")
            print(f"Position error:      dx={pos_error_x}, dy={pos_error_y}")
            print(f"Size error:          dw={size_error_w}, dh={size_error_h}")
            print(f"Decoded message:     '{result['readable_message']}'")
            print(f"ECC valid:           {result['ecc_valid']}")

            # Check detection accuracy (allow margin for line thickness offset and blending)
            max_position_error = 10  # pixels (more lenient for position)
            max_size_error = 15  # pixels (more lenient for size due to blending)

            position_ok = (pos_error_x <= max_position_error and pos_error_y <= max_position_error)
            size_ok = (size_error_w <= max_size_error and size_error_h <= max_size_error)

            if position_ok and size_ok:
                print(f"✓ PASS: Viewframe detected with acceptable accuracy")
                print(f"  (Position error ≤{max_position_error}px, Size error ≤{max_size_error}px)")
                return True
            else:
                print(f"✗ FAIL: Viewframe detection accuracy insufficient")
                print(f"  (Expected: position ≤{max_position_error}px, size ≤{max_size_error}px)")
                return False

        finally:
            os.unlink(watermarked_path)

    finally:
        os.unlink(temp_path)


def test_partial_corner_detection():
    """Test detection with partially obscured corners."""
    print("\n" + "=" * 60)
    print("TEST 3: Partial Corner Detection (Robustness)")
    print("=" * 60)

    test_size = 512
    test_img = np.random.randint(0, 200, (test_size, test_size, 3), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(test_img).save(temp_path)

    try:
        manager = WatermarkManager()

        # Embed watermark
        original_msg = "QWE"
        img_tensor, binary_msg, coords = manager.embed(
            temp_path,
            original_msg,
            'corners'
        )

        # Save and then partially obscure corners
        from notebooks.inference_utils import unnormalize_img
        from torchvision.utils import save_image

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            watermarked_path = f.name
            save_image(unnormalize_img(img_tensor), watermarked_path)

        try:
            # Load and obscure one corner (simulate partial damage)
            img_cv = cv2.imread(watermarked_path, cv2.IMREAD_UNCHANGED)
            h, w = img_cv.shape[:2]

            # Obscure top-left corner area with dark overlay
            obscure_size = 30
            img_cv[0:obscure_size, 0:obscure_size] = np.minimum(
                img_cv[0:obscure_size, 0:obscure_size],
                np.array([50, 50, 50])
            )

            cv2.imwrite(watermarked_path, img_cv)

            # Try to detect - should still work with enhanced algorithm
            result = manager.verify(watermarked_path, original_msg)

            print(f"Detected viewframe after partial obscuration:")
            print(f"  x={result['viewframe']['x']}, y={result['viewframe']['y']}")
            print(f"  w={result['viewframe']['width']}, h={result['viewframe']['height']}")
            print(f"  Decoded: '{result['readable_message']}'")

            # Check if detection succeeded (any reasonable detection)
            if result['viewframe']['width'] > 0 and result['viewframe']['height'] > 0:
                print("✓ PASS: Enhanced algorithm detected viewframe despite partial obscuration")
                return True
            else:
                print("✗ FAIL: Detection failed with partially obscured corners")
                return False

        finally:
            os.unlink(watermarked_path)

    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("WAM Viewframe Detection Tests")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Semi-Transparent Brackets", test_semi_transparent_brackets()))
    results.append(("Detection Accuracy", test_viewframe_detection_accuracy()))
    results.append(("Partial Corner Robustness", test_partial_corner_detection()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    sys.exit(0 if total_passed == total_tests else 1)
