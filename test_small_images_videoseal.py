#!/usr/bin/env python3
"""
Test VideoSeal embed_bytes/verify_bytes using natural images from abnormal/ directory.
"""

import sys
import os
import io
import warnings

# Suppress torch/ffmpeg warnings during import
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, "/home/h/FLY/watermark-freedom")

from pathlib import Path

from PIL import Image

from backends.videoseal_backend import VideoSealBackend


def test_small_images():
    """Test VideoSeal embed_bytes and verify_bytes on natural images."""
    print("Initializing VideoSealBackend...")
    wm = VideoSealBackend()

    BASE_DIR = Path(__file__).parent

    # Natural images that work reliably with VideoSeal
    test_cases = [
        (BASE_DIR / "abnormal/freedurov.jpeg", "225x225 natural image"),
        (BASE_DIR / "abnormal/seabackground.jpg", "470x456 natural image"),
        (
            BASE_DIR / "abnormal/1_8TuTTxcvaFQ-mlzC2jfT8g@2x.webp",
            "720x720 natural image",
        ),
    ]

    test_message = "ABC"

    print("\n" + "=" * 70)
    print("Testing VideoSeal embed_bytes / verify_bytes on natural images")
    print("=" * 70)

    success_count = 0
    total_count = len(test_cases)

    for path, description in test_cases:
        print(f"\nTest: {description}")
        print("-" * 40)

        img = Image.open(path).convert("RGB")
        print(f"  Original size: {img.size}")

        img_path = "/tmp/vs_natural_input.png"
        img.save(img_path)

        try:
            print(f"  Embedding via embed_bytes...")
            img_bytes = open(img_path, "rb").read()
            img_out, binary, coords = wm.embed_bytes(img_bytes, test_message)

            print(f"  Output size: {coords['width']}x{coords['height']}")

            # Save watermarked output for verification
            watermarked_path = "/tmp/vs_natural_wm.png"
            Image.open(io.BytesIO(img_out)).save(watermarked_path)
            print(f"  Saved watermarked image to: {watermarked_path}")

            print(f"  Verifying via verify_bytes...")
            result = wm.verify_bytes(img_out, test_message)

            # Extract result keys per API: readable, ecc_valid, corrected_bitflips,
            # bit_accuracy, binary_message, viewframe
            readable = result.get("readable", "")
            ecc_valid = result.get("ecc_valid", False)
            corrected_bitflips = result.get("corrected_bitflips", "N/A")
            bit_accuracy = result.get("bit_accuracy")

            readable_msg = readable if readable is not None else ""
            acc_str = (
                f"{bit_accuracy * 100:.2f}%" if bit_accuracy is not None else "N/A"
            )

            print(f"  Readable message: '{readable_msg}'")
            print(f"  ECC valid: {ecc_valid}")
            print(f"  Corrected bitflips: {corrected_bitflips}")
            print(f"  Bit accuracy: {acc_str}")

            # Success criterion: readable == test_message AND ecc_valid is True
            if readable_msg == test_message and ecc_valid is True:
                print(
                    f"  ✓ PASS: Message '{test_message}' correctly recovered with ECC!"
                )
                success_count += 1
            elif readable_msg == test_message and ecc_valid is not True:
                print(f"  ~ WARN: Message recovered but ECC invalid")
            else:
                print(f"  ✗ FAILURE: Expected '{test_message}', got '{readable_msg}'")

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
