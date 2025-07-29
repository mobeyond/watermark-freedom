import os
import argparse
from core import WatermarkManager

def verify_image_and_print(img_path, original_message=None):
    """
    Loads a watermarked image, verifies the watermark, and prints the results.
    """
    watermarker = WatermarkManager()
    results = watermarker.verify(img_path, original_message)

    # Print results
    print(f"\n--- Verification Results for {os.path.basename(img_path)} ---")
    print(f"Decoded Message: '{results['readable_message']}'")
    print(f"ECC Valid: {results['ecc_valid']}")
    print(f"Corrected Bitflips: {results['corrected_bitflips']}")
    
    if results['bit_error_rate_percent'] >= 0:
        print(f"Bit Error Rate: {results['bit_error_rate_percent']:.2f}%")
    else:
        print("Bit Error Rate: N/A (Decoding failed)")

    if results.get('bit_accuracy') is not None:
        print(f"Bit Accuracy vs. Original: {results['bit_accuracy'] * 100:.2f}%")
    
    print(f"Raw Binary Message: {results['binary_message']}")
    print("--- End of Report ---")

def main():
    parser = argparse.ArgumentParser(description='Verify watermark in an image using ROCO ECC')
    parser.add_argument('--watermarked', type=str, required=True, help='Path to watermarked image')
    parser.add_argument('--original_message', type=str, help='Original message (e.g., "ABC") to calculate bit accuracy')
    
    args = parser.parse_args()

    try:
        verify_image_and_print(args.watermarked, args.original_message)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()