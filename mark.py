import os
import torch
import argparse
from torchvision.utils import save_image
from watermark_utils import init_model
from notebooks.inference_utils import unnormalize_img
from core import (
    preprocess_image, create_watermark_mask, embed_watermark
)

def watermark_image_and_save(img_path, message, output_path=None, mask_mode='corners', mask_params=None):
    """
    Loads an image, embeds a watermark, and saves the result.
    """
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wam = init_model().to(device).eval()

    # Preprocess the image
    img_pt, cv_img = preprocess_image(img_path)

    # Create the watermark mask
    mask, coords = create_watermark_mask(img_pt, cv_img, mask_mode, mask_params)
    if mask is None:
        raise ValueError("Could not create a valid mask for watermarking.")

    # Embed the watermark
    img_w, _ = embed_watermark(wam, img_pt, cv_img, message, mask)

    # Un-normalize the image tensor before saving
    img_w_to_save = unnormalize_img(img_w)

    # Determine output path
    if not output_path:
        filename_base, file_ext = os.path.splitext(img_path)
        output_path = f"{filename_base}_watermarked{file_ext}"

    # Save the watermarked image
    save_image(img_w_to_save, output_path)

    print(f"\nWatermark embedded successfully.")
    print(f"Mask coordinates: {coords}")
    print(f"Saved watermarked image to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Watermark an image')
    parser.add_argument('--cover', type=str, required=True, help='Path to cover image')
    parser.add_argument('--message', type=str, default='SIR', help='Message to embed')
    parser.add_argument('--output', type=str, help='Path to save watermarked image')

    # Mask type group
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument('--use_corners', action='store_true', help='Use inner frame corners for the mask (default)')
    mask_group.add_argument('--use_pixels', action='store_true', help='Use pixel coordinates for mask')
    mask_group.add_argument('--use_percentage', action='store_true', help='Use percentage coordinates for mask')

    # Pixel coordinates
    parser.add_argument('--x', type=int, help='X coordinate in pixels')
    parser.add_argument('--y', type=int, help='Y coordinate in pixels')
    parser.add_argument('--width', type=int, help='Width in pixels')
    parser.add_argument('--height', type=int, help='Height in pixels')

    # Percentage coordinates
    parser.add_argument('--x_percent', type=float, help='X coordinate as percentage (0-1)')
    parser.add_argument('--y_percent', type=float, help='Y coordinate as percentage (0-1)')
    parser.add_argument('--width_percent', type=float, help='Width as percentage (0-1)')
    parser.add_argument('--height_percent', type=float, help='Height as percentage (0-1)')
    
    parser.set_defaults(use_corners=True) # Default to using corners

    args = parser.parse_args()

    # Determine mask type and parameters
    mask_mode = 'corners'
    mask_params = None

    if args.use_pixels:
        if not all(v is not None for v in [args.x, args.y, args.width, args.height]):
            parser.error("--use_pixels requires --x, --y, --width, and --height")
        mask_mode = 'pixels'
        mask_params = {'x': args.x, 'y': args.y, 'width': args.width, 'height': args.height}
    elif args.use_percentage:
        if not all(v is not None for v in [args.x_percent, args.y_percent, args.width_percent, args.height_percent]):
            parser.error("--use_percentage requires --x_percent, --y_percent, --width_percent, and --height_percent")
        mask_mode = 'percentage'
        mask_params = {
            'x_percent': args.x_percent, 'y_percent': args.y_percent,
            'width_percent': args.width_percent, 'height_percent': args.height_percent
        }

    # Process image
    try:
        watermark_image_and_save(
            args.cover,
            args.message,
            args.output,
            mask_mode,
            mask_params
        )
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
