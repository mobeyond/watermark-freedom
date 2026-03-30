#!/usr/bin/env python3
"""
Verify Watermark CLI Tool
=========================
Verifies watermark in all images under a given directory.
Output format matches the app.py "Verify Watermark" module.

Usage:
    python verify_cli.py <directory_path> [--original_message <msg>] [--show-image]
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from core import WatermarkManager

# Suppress small viewframe warnings for cleaner output
warnings.filterwarnings(
    "ignore", message=".*Viewframe size.*smaller than recommended.*"
)


def enable_warnings():
    """Re-enable suppressed warnings."""
    warnings.filterwarnings(
        "default", message=".*Viewframe size.*smaller than recommended.*"
    )


def format_bit_accuracy(accuracy):
    """Format bit accuracy percentage."""
    if accuracy is None:
        return "N/A"
    return f"{accuracy * 100:.1f}%"


def format_percentage(value, decimals=2):
    """Format a value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_viewframe(viewframe: dict) -> str:
    """Format viewframe info with position and size in percentages."""
    if not viewframe:
        return ""

    x = viewframe.get("x", 0)
    y = viewframe.get("y", 0)
    width = viewframe.get("width", 0)
    height = viewframe.get("height", 0)

    # Use percentage values if available, otherwise convert
    x_pct = viewframe.get("x_percent", x)
    y_pct = viewframe.get("y_percent", y)

    return (
        f"  Viewframe: ({int(x)}, {int(y)}) pixels, "
        f"size {int(width)}x{int(height)}, "
        f"at ({format_percentage(x_pct, 1)}, {format_percentage(y_pct, 1)}) of image, "
        f"Area: {format_percentage(viewframe.get('ratio', 0), 1)} of image"
    )


def verify_image(image_path, manager: WatermarkManager, original_message=None) -> dict:
    """
    Verify watermark in a single image.

    Returns dict matching app.py verification output format.
    """
    result = manager.verify(str(image_path), original_message)

    # Format viewframe info
    vf_info = format_viewframe(result.get("viewframe", {}))

    # Format bit accuracy
    bit_accuracy = result.get("bit_accuracy")
    acc_str = format_bit_accuracy(bit_accuracy)

    return {
        "filename": os.path.basename(image_path),
        "readable_message": result.get("readable_message", "N/A"),
        "bit_error_rate": result.get("bit_error_rate_percent", "N/A"),
        "bitflips": result.get("corrected_bitflips", "N/A"),
        "ecc_valid": result.get("ecc_valid", False),
        "bit_accuracy": acc_str,
        "viewframe": vf_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify watermark in images under a directory."
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing images to verify"
    )
    parser.add_argument(
        "--original-message",
        "-m",
        type=str,
        default=None,
        help='Original message for comparison (e.g., "ABC")',
    )
    parser.add_argument(
        "--show-image",
        "-i",
        action="store_true",
        help="Show image after verification (opens with default viewer)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.9,
        help="ECC validity threshold (0-1, default: 0.9)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--warnings", "-w", action="store_true", help="Show warning messages"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.warnings:
        enable_warnings()

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        sys.exit(1)

    # Initialize manager
    manager = WatermarkManager()

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"}
    image_files = sorted(
        [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not image_files:
        print(f"No images found in: {directory}")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"VERIFY WATERMARK CLI")
    print(f"{'=' * 60}")
    print(f"Directory: {directory}")
    print(f"Files to verify: {len(image_files)}")
    if args.original_message:
        print(f"Original message: {args.original_message}")
    print(f"ECC threshold: {args.threshold}")
    print(f"{'=' * 60}\n")

    results = []
    success_count = 0
    fail_count = 0

    for img_path in image_files:
        try:
            result = verify_image(img_path, manager, args.original_message)
            results.append({"filename": result["filename"], **result})

            if result["ecc_valid"] and result["ecc_valid"] >= args.threshold:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"  ✗ {img_path.name}: ERROR - {e}")
            results.append(
                {
                    "filename": img_path.name,
                    "error": str(e),
                }
            )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Total:      {success_count + fail_count}")

    # Print individual results
    for r in results:
        print(f"\n  [{r['filename']}]")
        print(f"    Message:       {r['readable_message']}")
        print(f"    Bit Error Rate: {r['bit_error_rate']}")
        print(f"    ECC Valid:     {r['ecc_valid']}")
        print(f"    Bit Accuracy:  {r['bit_accuracy']}")
        if r.get("viewframe"):
            print(f"    {r['viewframe']}")
        if r.get("error"):
            print(f"    Error: {r['error']}")

    # JSON output
    if args.json:
        import json

        print(f"\n{'=' * 60}")
        print(f"JSON OUTPUT")
        print(f"{'=' * 60}")
        print(json.dumps(results, indent=2))

    sys.exit(0 if success_count >= fail_count else 1)


if __name__ == "__main__":
    main()
