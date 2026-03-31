import os
import argparse
from core import WatermarkManager
from backends.videoseal_backend import VideoSealBackend

SUPPORTED_BACKENDS = ["wam", "videoseal"]
DEFAULT_BACKEND = "wam"


def verify_image_and_print(img_path, original_message=None, backend=DEFAULT_BACKEND):
    if backend == "videoseal":
        wm = VideoSealBackend()
        img_bytes = open(img_path, "rb").read()
        result = wm.verify_bytes(img_bytes, original_message)
        print(f"\n--- VideoSeal Verification: {os.path.basename(img_path)} ---")
        print(f"Decoded Message: '{result.get('readable_message', '')}'")
        print(f"ECC Valid: {result.get('ecc_valid')}")
        print(f"Corrected Bitflips: {result.get('corrected_bitflips')}")
        if result.get("bit_accuracy") is not None:
            print(f"Codeword Accuracy: {result['bit_accuracy'] * 100:.2f}%")
        print(f"Raw Binary: {result.get('binary_message', '')[:64]}...")
        print("--- End of Report ---")
    else:
        watermarker = WatermarkManager()
        results = watermarker.verify(img_path, original_message)
        print(f"\n--- WAM Verification: {os.path.basename(img_path)} ---")
        print(f"Decoded Message: '{results['readable_message']}'")
        print(f"ECC Valid: {results['ecc_valid']}")
        print(f"Corrected Bitflips: {results['corrected_bitflips']}")
        if results["bit_error_rate_percent"] >= 0:
            print(f"Bit Error Rate: {results['bit_error_rate_percent']:.2f}%")
        else:
            print("Bit Error Rate: N/A (Decoding failed)")
        if results.get("bit_accuracy") is not None:
            print(f"Bit Accuracy vs. Original: {results['bit_accuracy'] * 100:.2f}%")
        print(f"Raw Binary Message: {results['binary_message']}")
        print("--- End of Report ---")


def main():
    parser = argparse.ArgumentParser(description="Verify watermark in an image")
    parser.add_argument(
        "--watermarked", type=str, required=True, help="Path to watermarked image"
    )
    parser.add_argument(
        "--original_message",
        type=str,
        help="Original message to calculate bit accuracy",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=SUPPORTED_BACKENDS,
        help=f"Watermarking backend (default: {DEFAULT_BACKEND})",
    )

    args = parser.parse_args()
    try:
        verify_image_and_print(args.watermarked, args.original_message, args.backend)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
