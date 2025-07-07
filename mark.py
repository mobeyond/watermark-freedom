import os
import numpy as np
from PIL import Image
import torch
import argparse
import cv2

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import default_transform, unnormalize_img
from viewframe import get_inner_square_region, draw_viewframe_overlay
from watermark_utils import (
    init_model, load_image, crop_to_centered_square,
    pil_to_cv2, cv2_to_pil, robust_str_to_binary,
    create_mask_from_coords, validate_pixel_coords, validate_percentage_coords
)
from torchvision.utils import save_image

def process_image(img_path, message, mask_type=None, mask_params=None, output_path=None):
    """Process image with watermarking based on mask type"""
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wam = init_model().to(device).eval()

    # Load and process image
    img = load_image(img_path)
    cv_img = pil_to_cv2(img)

    # Crop to centered square if needed
    cv_img = crop_to_centered_square(cv_img)
    img = cv2_to_pil(cv_img)
    img_pt = default_transform(img).unsqueeze(0).to(device)

    # Create watermark message with error correction
    wm_msg = robust_str_to_binary(message).unsqueeze(0).to(device)

    # Embed watermark
    outputs = wam.embed(img_pt, wm_msg)

    # Get image dimensions for coordinate calculations
    h, w = img_pt.shape[2:]

    # Create mask based on mask type
    if mask_type == 'pixels':
        x, y, width, height = mask_params

        # Validate pixel coordinates
        is_valid, error_msg = validate_pixel_coords(w, h, x, y, width, height)
        if not is_valid:
            raise ValueError(f"Invalid pixel coordinates: {error_msg}")

        mask = create_mask_from_coords(img_pt, x, y, width, height)

        # Calculate percentages for debugging
        x_percent = x / w
        y_percent = y / h
        width_percent = width / w
        height_percent = height / h

    elif mask_type == 'percentage':
        x_percent, y_percent, width_percent, height_percent = mask_params

        # Validate percentage coordinates
        if not all(0 <= p <= 1 for p in [x_percent, y_percent, width_percent, height_percent]):
            raise ValueError("Percentage values must be between 0 and 1")

        # Convert percentages to pixels
        x = int(w * x_percent)
        y = int(h * y_percent)
        width = int(w * width_percent)
        height = int(h * height_percent)

        mask = create_mask_from_coords(img_pt, x, y, width, height)

    else:  # None - use frame corners
        # Get inner square region coordinates
        crop_top, crop_left, crop_bottom, crop_right = get_inner_square_region(cv_img)
        x = crop_left
        y = crop_top
        width = crop_right - crop_left
        height = crop_bottom - crop_top

        mask = create_mask_from_coords(img_pt, x, y, width, height)

        # Calculate percentages for debugging
        x_percent = x / w
        y_percent = y / h
        width_percent = width / w
        height_percent = height / h

    # Now draw the viewframe overlay using the mask boundaries
    overlay = draw_viewframe_overlay(cv_img)
    overlay_pil = cv2_to_pil(overlay)
    overlay_pt = default_transform(overlay_pil).unsqueeze(0).to(device)

    # Debug: Export mask region
    mask_np = mask.squeeze().cpu().numpy()
    mask_img = (mask_np * 255).astype(np.uint8)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # Draw rectangle on mask image for visualization
    cv2.rectangle(mask_img, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Save mask image for debugging
    filename_base, _ = os.path.splitext(img_path)
    mask_output_path = f"{filename_base}_mask_debug.jpg"
    cv2.imwrite(mask_output_path, mask_img)

    # Print coordinates for reference
    print("\nMask Region Coordinates:")
    print(f"Pixel coordinates: x={x}, y={y}, width={width}, height={height}")
    print(f"Percentage coordinates: x={x_percent:.3f}, y={y_percent:.3f}, width={width_percent:.3f}, height={height_percent:.3f}")
    print(f"Mask debug image saved to: {mask_output_path}\n")

    # Apply watermark using mask
    img_w = outputs['imgs_w'] * mask + overlay_pt * (1 - mask)

    # Save results
    if output_path:
        save_image(unnormalize_img(img_w), output_path)
    else:
        filename_base, file_ext = os.path.splitext(img_path)
        output_path = f"{filename_base}_watermarked{file_ext}"
        save_image(unnormalize_img(img_w), output_path)

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Watermark an image')
    parser.add_argument('--cover', type=str, required=True, help='Path to cover image')
    parser.add_argument('--message', type=str, default='Hello World!', help='Message to embed')
    parser.add_argument('--output', type=str, help='Path to save watermarked image')

    # Mask type group
    mask_group = parser.add_mutually_exclusive_group()
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

    args = parser.parse_args()

    # Determine mask type and parameters
    mask_type = None
    mask_params = None

    if args.use_pixels:
        if not all(v is not None for v in [args.x, args.y, args.width, args.height]):
            parser.error("--use_pixels requires --x, --y, --width, and --height")
        mask_type = 'pixels'
        mask_params = (args.x, args.y, args.width, args.height)
    elif args.use_percentage:
        if not all(v is not None for v in [args.x_percent, args.y_percent, args.width_percent, args.height_percent]):
            parser.error("--use_percentage requires --x_percent, --y_percent, --width_percent, and --height_percent")
        mask_type = 'percentage'
        mask_params = (args.x_percent, args.y_percent, args.width_percent, args.height_percent)

    # Process image
    output_path = process_image(
        args.cover,
        args.message,
        mask_type,
        mask_params,
        args.output
    )

    print(f"Saved watermarked image to {output_path}")

if __name__ == '__main__':
    main()
