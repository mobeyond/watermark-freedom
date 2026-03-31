#!/usr/bin/env python3
"""
VideoSeal Verify Watermark CLI Tool
====================================
Verifies watermark in all images under a given directory using VideoSeal.
Output format matches verify_cli.py.

Usage:
    python verify_cli_videoseal.py <directory_path> [--original_message <msg>] [--json] [--quiet]
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

# Suppress torch and ffmpeg warnings during import
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Tokenizers.*")
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*ffmpeg.*")
warnings.filterwarnings("ignore", message=".*lib.*")

sys.path.insert(0, str(Path(__file__).parent))

from backends.videoseal_backend import VideoSealBackend


def format_accuracy(accuracy):
    if accuracy is None:
        return "N/A"
    return f"{accuracy * 100:.1f}%"


def format_viewframe(viewframe: dict) -> str:
    if not viewframe:
        return ""

    x = viewframe.get("x", 0)
    y = viewframe.get("y", 0)
    width = viewframe.get("width", 0)
    height = viewframe.get("height", 0)
    x_pct = viewframe.get("x_percent", 0)
    y_pct = viewframe.get("y_percent", 0)
    ratio = viewframe.get("ratio", 0)

    return (
        f"  Viewframe: ({int(x)}, {int(y)}) pixels, "
        f"size {int(width)}x{int(height)}, "
        f"at ({format_percentage(x_pct, 1)}, {format_percentage(y_pct, 1)}) of image, "
        f"Area: {format_percentage(ratio, 1)} of image"
    )


def format_percentage(value, decimals=2):
    return f"{value * 100:.{decimals}f}%"


def verify_image(image_path, backend: VideoSealBackend, original_message=None) -> dict:
    img_bytes = open(image_path, "rb").read()
    result = backend.verify_bytes(img_bytes, original_message)

    vf_info = format_viewframe(result.get("viewframe", {}))
    ecc_valid = result.get("ecc_valid")
    bitflips = result.get("corrected_bitflips")
    accuracy = result.get("bit_accuracy")
    acc_str = format_accuracy(accuracy)
    readable_msg = result.get("readable", "")
    passed = (
        ecc_valid is True
        and bool(readable_msg)
        and (original_message is None or readable_msg == original_message)
    )

    return {
        "filename": os.path.basename(image_path),
        "readable_message": result.get("readable", "N/A"),
        "ecc_valid": ecc_valid,
        "corrected_bitflips": bitflips,
        "bit_accuracy": acc_str,
        "raw_binary": result.get("binary_message", "")[:64],
        "viewframe": vf_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify watermark in images under a directory using VideoSeal."
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing images to verify"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="videoseal",
        choices=["videoseal"],
        help="Backend to use (default: videoseal)",
    )
    parser.add_argument(
        "--original-message",
        "-m",
        type=str,
        default=None,
        help='Original message for comparison (e.g., "ABC")',
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.9,
        help="Accuracy threshold (0-1, default: 0.9)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        sys.exit(1)

    backend = VideoSealBackend()

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

    if not args.quiet and not args.json:
        print(f"\n{'=' * 60}")
        print(f"VERIFY WATERMARK CLI (VideoSeal)")
        print(f"{'=' * 60}")
        print(f"Directory: {directory}")
        print(f"Files to verify: {len(image_files)}")
        if args.original_message:
            print(f"Original message: {args.original_message}")
        print(f"Accuracy threshold: {args.threshold}")
        print(f"{'=' * 60}\n")

    results = []
    success_count = 0
    fail_count = 0

    for img_path in image_files:
        try:
            result = verify_image(img_path, backend, args.original_message)
            results.append({"filename": result["filename"], **result})

            if result.get("ecc_valid") is True and bool(
                result.get("readable_message", "")
            ):
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            if not args.quiet and not args.json:
                print(f"  ✗ {img_path.name}: ERROR - {e}")
            results.append(
                {
                    "filename": img_path.name,
                    "error": str(e),
                }
            )
            fail_count += 1

    if not args.quiet and not args.json:
        print(f"\n{'=' * 60}")
        print(f"SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Successful: {success_count}")
        print(f"  Failed:     {fail_count}")
        print(f"  Total:      {success_count + fail_count}")

        for r in results:
            print(f"\n  [{r['filename']}]")
            if r.get("error"):
                print(f"    Error: {r['error']}")
                continue
            print(f"    Message:       {r.get('readable_message', 'N/A')}")
            print(f"    Bit Error Rate: {r.get('bit_error_rate', 'N/A')}")
            print(f"    ECC Valid:     {r.get('ecc_valid', 'N/A')}")
            print(f"    Bit Accuracy:  {r.get('bit_accuracy', 'N/A')}")
            if r.get("viewframe"):
                print(f"    {r['viewframe']}")

    if args.json:
        import json

        print(json.dumps(results, indent=2))

    sys.exit(0 if success_count >= fail_count else 1)


if __name__ == "__main__":
    main()
