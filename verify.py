import os
import torch
from PIL import Image
import argparse
from torchvision.utils import save_image

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str
)

def main():
    parser = argparse.ArgumentParser(description='Verify watermark in an image')
    parser.add_argument('--watermarked', type=str, required=True, help='Path to watermarked image')
    parser.add_argument('--use_frame_corners', action='store_true', help='Image was watermarked using frame corners')
    args = parser.parse_args()

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = "checkpoints"
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')
    wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

    # Load and process image
    img = Image.open(args.watermarked).convert("RGB")
    
    if args.use_frame_corners:
        # For frame-corner watermarked images, we need to crop to centered square
        width, height = img.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        img = img.crop((left, top, right, bottom))
    
    # Convert to tensor
    img_pt = default_transform(img).unsqueeze(0).to(device)
    
    # Detect watermark
    preds = wam.detect(img_pt)["preds"]
    mask_preds = torch.sigmoid(preds[:, 0, :, :])
    bit_preds = preds[:, 1:, :, :]
    
    # Predict message
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
    binary_str = msg2str(pred_message[0].numpy())
    
    # Convert binary to readable string
    readable_message = ''
    for i in range(0, len(binary_str), 8):
        if i + 8 <= len(binary_str):
            byte = binary_str[i:i+8]
            char = chr(int(byte, 2))
            if char.isprintable():  # Only include printable characters
                readable_message += char
    
    # Calculate confidence
    mask_confidence = mask_preds.mean().item()
    
    # Print results
    print(f"\nVerification Results for {args.watermarked}:")
    print(f"Binary Message: {binary_str}")
    print(f"Readable Message: {readable_message}")
    print(f"Mask Confidence: {mask_confidence:.4f}")
    
    # Save detection mask
    filename_base, file_ext = os.path.splitext(args.watermarked)
    mask_output_path = f"{filename_base}_detection_mask{file_ext}"
    save_image(mask_preds, mask_output_path)
    print(f"Saved detection mask to {mask_output_path}")

if __name__ == '__main__':
    main() 
