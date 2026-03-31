#!/usr/bin/env python3
"""
Test VideoSeal watermarking on small images (< 512x512) to verify
embed_bytes and verify_bytes work across resolutions.
Tests VideoSealBackend().embed_bytes() and verify_bytes() APIs.
"""

import sys
import os
import io
import warnings

# Suppress torch/ffmpeg warnings during import
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, "/home/h/FLY/watermark-freedom")

import numpy as np
from PIL import Image

from backends.videoseal_backend import VideoSealBackend


def create_test_image(width, height, color=None):
    """Create a test image with specified dimensions."""
    if color is not None:
        img_array = np.ones((height, width, 3), dtype=np.uint8) * color
    else:
        np.random.seed(42)  # Reproducible
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_small_images():
    """Test VideoSeal embed_bytes and verify_bytes on various small image sizes."""
    print("Initializing VideoSealBackend...")
    wm = VideoSealBackend()

    # Same sizes as reference file, plus non-square variants
    test_cases = [
        (64, 64, "64x64 - very small"),
        (128, 128, "128x128 - small"),
        (200, 200, "200x200 - below threshold"),
        (256, 256, "256x256 - threshold"),
        (512, 512, "512x512 - above threshold"),
        (130, 139, "130x139 - non-square tall"),
        (80, 200, "80x200 - non-square tall narrow"),
        (200, 80, "200x80 - non-square wide"),
    ]

    test_message = "HI"  # 2 chars for VideoSeal

    print("\n" + "=" * 70)
    print("Testing VideoSeal embed_bytes / verify_bytes on small images")
    print("=" * 70)

    success_count = 0
    total_count = len(test_cases)

    for width, height, description in test_cases:
        print(f"\nTest: {description}")
        print("-" * 40)

        img = create_test_image(width, height)
        img_path = f"/tmp/vs_test_{width}x{height}.png"
        img.save(img_path)

        try:
            print(f"  Embedding via embed_bytes...")
            img_bytes = open(img_path, "rb").read()
            img_out, binary, coords = wm.embed_bytes(img_bytes, test_message)

            print(f"  Original size: {width}x{height}")
            print(f"  Output size: {coords['width']}x{coords['height']}")
            print(
                f"  Viewframe: x={coords['x']}, y={coords['y']}, "
                f"w={coords['width']}, h={coords['height']}"
            )

            # Save watermarked output for verification
            watermarked_path = f"/tmp/vs_test_{width}x{height}_wm.png"
            Image.open(io.BytesIO(img_out)).save(watermarked_path)
            print(f"  Saved watermarked image to: {watermarked_path}")

            print(f"  Verifying via verify_bytes...")
            result = wm.verify_bytes(img_out, test_message)

            accuracy = result.get("accuracy")
            readable = result.get("readable", "")
            acc_str = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"

            print(f"  Readable message: {readable}")
            print(f"  Bit accuracy vs original: {acc_str}")

            # Check and report pass/warn/error
            if accuracy is not None and accuracy >= 0.93:
                print(f"  ✓ PASS: {acc_str} accuracy (>= 93%)")
                success_count += 1
            elif accuracy is not None and accuracy >= 0.80:
                print(f"  ~ WARN: {acc_str} accuracy (80-92%)")
            else:
                print(f"  ✗ ERROR: {acc_str if accuracy else 'N/A'} accuracy (< 80%)")

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Testing complete! {success_count}/{total_count} tests passed")
    print("=" * 70)

    return success_count == total_count


if __name__ == "__main__":
    test_small_images()
