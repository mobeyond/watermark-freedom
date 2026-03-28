"""
Test that small images (< 256x256) are processed correctly with adaptive resizing.
"""
import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '/home/h/FLY/watermark-freedom')

from core import WatermarkManager

def create_test_image(width, height, color=None):
    """Create a test image with specified dimensions."""
    if color is not None:
        img_array = np.ones((height, width, 3), dtype=np.uint8) * color
    else:
        # Random colored image
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def test_small_image_watermarking():
    """Test watermarking on various small image sizes."""
    print("Initializing WatermarkManager...")
    manager = WatermarkManager()

    test_cases = [
        (64, 64, "64x64 - very small"),
        (128, 128, "128x128 - small"),
        (200, 200, "200x200 - below threshold"),
        (256, 256, "256x256 - threshold"),
        (512, 512, "512x512 - above threshold"),
    ]

    test_message = "ABC"  # Max 3 characters

    print("\n" + "="*70)
    print("Testing watermark embedding and verification on small images")
    print("="*70)

    for width, height, description in test_cases:
        print(f"\nTest: {description}")
        print("-" * 40)

        # Create test image
        img = create_test_image(width, height)
        img_path = f"/tmp/test_{width}x{height}.png"
        img.save(img_path)

        try:
            # Embed watermark
            print(f"  Embedding watermark in {width}x{height} image...")
            watermarked_img, binary_msg, coords = manager.embed(
                img_path,
                test_message,
                mask_mode='corners',
                margin_percent=0.15
            )

            # Calculate what target size would be used (always 256x256 for WAM)
            print(f"  Original size: {width}x{height}")
            print(f"  WAM processing size: 256x256")
            print(f"  Viewframe coords: x={coords['x']}, y={coords['y']}, "
                  f"w={coords['width']}, h={coords['height']}")

            # Convert watermarked tensor back to image for verification
            from notebooks.inference_utils import unnormalize_img
            img_np = unnormalize_img(watermarked_img).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            watermarked_pil = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
            watermarked_path = f"/tmp/test_{width}x{height}_wm.png"
            watermarked_pil.save(watermarked_path)

            # Verify watermark
            print(f"  Verifying watermark...")
            result = manager.verify(watermarked_path)

            print(f"  Readable message: {result['readable_message']}")
            print(f"  ECC valid: {result['ecc_valid']}")
            print(f"  Bit error rate: {result['bit_error_rate_percent']:.2f}%")

            if result['readable_message'] == test_message:
                print(f"  ✓ SUCCESS: Message correctly recovered!")
            else:
                print(f"  ✗ FAILURE: Expected '{test_message}', got '{result['readable_message']}'")

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)

if __name__ == "__main__":
    import cv2
    test_small_image_watermarking()
