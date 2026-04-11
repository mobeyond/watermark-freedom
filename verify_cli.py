#!/usr/bin/env python3
"""
Verify Watermark CLI Tool
=========================
Verifies watermark in all images under a given directory.

Supports two backends:
- VideoSeal: Facebook's VideoSeal model (supports ROCO 3-char and ROCO32 4-char)
- Custom (WAM): WAM-based watermarking (ROCO 3-char only)

Usage:
    python verify_cli.py <directory_path> [--backend videoseal|custom] [--encoding roco|roco32]
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from backends.videoseal_backend import VideoSealBackend
from backends.custom_model_backend import CustomModelBackend

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
    pct = value * 100
    if pct < 0.01:
        return f"<0.01%"
    return f"{pct:.{decimals}f}%"


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
        f"Viewframe: ({int(x)}, {int(y)}) px, "
        f"size {int(width)}x{int(height)}, "
        f"at ({format_percentage(x_pct, 1)}, {format_percentage(y_pct, 1)}), "
        f"Area: {format_percentage(viewframe.get('ratio', 0), 1)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Verify watermark in images under a directory."
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing images to verify"
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="videoseal",
        choices=["videoseal", "custom"],
        help='Backend to use: "videoseal" (default) or "custom" (WAM)',
    )
    parser.add_argument(
        "--encoding",
        "-e",
        type=str,
        default="roco",
        choices=["roco", "roco32"],
        help='Encoding: "roco" (3-char, 32-bit) or "roco32" (4-char, 256-bit)',
    )
    parser.add_argument(
        "--original-message",
        "-m",
        type=str,
        default=None,
        help='Original message for comparison (e.g., "ABC" or "ABCD")',
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

    # Validate encoding-backend combination
    if args.backend == "custom" and args.encoding == "roco32":
        print("Error: ROCO32 encoding is only available with VideoSeal backend")
        print("Use --backend videoseal --encoding roco32 for 4-char messages")
        sys.exit(1)

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

    # Initialize backends
    vs_watermarker = VideoSealBackend()
    custom_watermarker = CustomModelBackend()

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
    print(f"Backend: {args.backend}")
    print(f"Encoding: {args.encoding}")
    if args.original_message:
        print(f"Original message: {args.original_message}")
    print(f"ECC threshold: {args.threshold}")
    print(f"{'=' * 60}\n")

    results = []
    success_count = 0
    fail_count = 0

    for img_path in image_files:
        try:
            # Read image as bytes for backend verification
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            if args.backend == "videoseal":
                if args.encoding == "roco32":
                    result = vs_watermarker.verify_bytes(image_bytes, args.original_message)
                else:
                    result = vs_watermarker.verify_bytes_roco(image_bytes, args.original_message)
            else:  # custom backend
                result = custom_watermarker.verify_bytes_roco(image_bytes, args.original_message)

            # Format result for output
            result_dict = {
                "filename": img_path.name,
                "readable_message": result.get("readable", "N/A"),
                "bit_error_rate": result.get("bit_accuracy", "N/A"),
                "bitflips": result.get("corrected_bitflips", "N/A"),
                "ecc_valid": result.get("ecc_valid", False),
                "bit_accuracy": format_bit_accuracy(result.get("bit_accuracy")),
                "viewframe": format_viewframe(result.get("viewframe", {})),
            }
            results.append(result_dict)

            if result.get("ecc_valid"):
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
