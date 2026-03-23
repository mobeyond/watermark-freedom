import numpy as np
import cv2
import os
import sys
from contextlib import contextmanager
import torch
from torchvision.utils import save_image
from notebooks.inference_utils import unnormalize_img, msg2str, default_transform, load_model_from_checkpoint
from watermark_anything.data.metrics import msg_predict_inference
from watermark_utils import (
    load_image, crop_to_centered_square, pil_to_cv2, cv2_to_pil,
    create_mask_from_coords, create_mask_from_percentages,
    validate_pixel_coords, validate_percentage_coords,
    roco_encode_to_binary_tensor, roco_decode_from_binary_tensor
)
from viewframe import get_inner_square_region, draw_viewframe_overlay

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class WatermarkManager:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        exp_dir = "checkpoints"
        json_path = os.path.join(exp_dir, "params.json")
        ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')
        
        with suppress_stdout():
            self.wam = load_model_from_checkpoint(json_path, ckpt_path).to(self.device).eval()
        print(f"Model loaded successfully from {ckpt_path}")

    def _preprocess_image(self, source):
        img = load_image(source)
        cv_img = pil_to_cv2(img)
        cv_img = crop_to_centered_square(cv_img)
        img = cv2_to_pil(cv_img)
        img_pt = default_transform(img).unsqueeze(0).to(self.device)
        return img_pt, cv_img

    def _create_watermark_mask(self, img_pt, cv_img, mode, params=None):
        h, w = img_pt.shape[2:]
        mask = None
        coords = {}

        if mode == 'corners':
            crop_top, crop_left, crop_bottom, crop_right = get_inner_square_region(cv_img)
            x, y, width, height = crop_left, crop_top, crop_right - crop_left, crop_bottom - crop_top
            mask = create_mask_from_coords(img_pt, x, y, width, height)
            coords = {'x': x, 'y': y, 'width': width, 'height': height}
        elif mode == 'pixels':
            x, y, width, height = params['x'], params['y'], params['width'], params['height']
            is_valid, error_msg = validate_pixel_coords(w, h, x, y, width, height)
            if not is_valid:
                raise ValueError(error_msg)
            mask = create_mask_from_coords(img_pt, x, y, width, height)
            coords = {'x': x, 'y': y, 'width': width, 'height': height}
        elif mode == 'percentage':
            x_p, y_p, w_p, h_p = params['x_percent'], params['y_percent'], params['width_percent'], params['height_percent']
            is_valid, error_msg = validate_percentage_coords(x_p, y_p, w_p, h_p)
            if not is_valid:
                raise ValueError(error_msg)
            mask = create_mask_from_percentages(img_pt, x_p, y_p, w_p, h_p)
            x, y, width, height = int(w * x_p), int(h * y_p), int(w * w_p), int(h * h_p)
            coords = {'x': x, 'y': y, 'width': width, 'height': height}
            
        if mask is not None:
            coords.update({
                'x_percent': coords['x'] / w, 'y_percent': coords['y'] / h,
                'width_percent': coords['width'] / w, 'height_percent': coords['height'] / h
            })

        return mask, coords

    def embed(self, image_source, message, mask_mode, mask_params=None):
        img_pt, cv_img = self._preprocess_image(image_source)
        h, w = img_pt.shape[2:]
        
        # Get viewframe region based on mode
        if mask_mode == 'pixels' and mask_params:
            # Pixel mode: use provided coordinates
            x, y = mask_params['x'], mask_params['y']
            width, height = mask_params['width'], mask_params['height']
        elif mask_mode == 'percentage' and mask_params:
            # Percentage mode: convert to pixels
            x = int(mask_params['x_percent'] * w)
            y = int(mask_params['y_percent'] * h)
            width = int(mask_params['width_percent'] * w)
            height = int(mask_params['height_percent'] * h)
        else:
            # Corners mode (default): centered square
            min_dim = min(h, w)
            center = (w // 2, h // 2)
            square_size = int(min_dim * 0.7)
            x = center[0] - square_size // 2
            y = center[1] - square_size // 2
            width = height = square_size
        
        coords = {
            'x': x, 'y': y, 'width': width, 'height': height,
            'x_percent': x / w, 'y_percent': y / h,
            'width_percent': width / w, 'height_percent': height / h
        }
        
        # 1. CROP to region (no border yet)
        cropped = img_pt[:, :, y:y+height, x:x+width]
        
        # 2. RESIZE to 256x256 for WAM
        cropped_256 = torch.nn.functional.interpolate(cropped, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 3. EMBED watermark
        wm_msg_tensor = roco_encode_to_binary_tensor(message)
        wm_msg = wm_msg_tensor.unsqueeze(0).to(self.device)
        outputs = self.wam.embed(cropped_256, wm_msg)
        
        # 4. RESIZE back
        watermarked_crop = torch.nn.functional.interpolate(outputs['imgs_w'], size=(height, width), mode='bilinear', align_corners=False)
        
        # 5. PLACE back into original image
        img_w = img_pt.clone()
        img_w[:, :, y:y+height, x:x+width] = watermarked_crop
        
        # 6. Draw corner brackets (AFTER embedding, visible marker)
        img_np = unnormalize_img(img_w).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Draw 4 corner brackets (more visible)
        corner_length = int(min(width, height) * 0.15)  # 15% of region size
        line_thickness = 3  # Thicker for visibility
        color = (255, 255, 255)  # White, full opacity
        
        # Top-left corner
        cv2.line(img_bgr, (x, y), (x + corner_length, y), color, line_thickness)
        cv2.line(img_bgr, (x, y), (x, y + corner_length), color, line_thickness)
        
        # Top-right corner
        cv2.line(img_bgr, (x + width, y), (x + width - corner_length, y), color, line_thickness)
        cv2.line(img_bgr, (x + width, y), (x + width, y + corner_length), color, line_thickness)
        
        # Bottom-left corner
        cv2.line(img_bgr, (x, y + height), (x + corner_length, y + height), color, line_thickness)
        cv2.line(img_bgr, (x, y + height), (x, y + height - corner_length), color, line_thickness)
        
        # Bottom-right corner
        cv2.line(img_bgr, (x + width, y + height), (x + width - corner_length, y + height), color, line_thickness)
        cv2.line(img_bgr, (x + width, y + height), (x + width, y + height - corner_length), color, line_thickness)
        
        # Convert back to tensor
        overlay_pil = cv2_to_pil(img_bgr)
        img_w = default_transform(overlay_pil).unsqueeze(0).to(self.device)
        
        binary_message_str = "".join(map(str, wm_msg_tensor.int().tolist()))
        
        return img_w, binary_message_str, coords

    def verify(self, image_source, original_message=None, viewframe_coords=None):
        img_pt, cv_img = self._preprocess_image(image_source)
        h, w = img_pt.shape[2:]
        
        # Get viewframe region based on params
        if viewframe_coords:
            # Use provided coordinates
            x = int(viewframe_coords.get('x_percent', 0) * w)
            y = int(viewframe_coords.get('y_percent', 0) * h)
            width = int(viewframe_coords.get('width_percent', 0) * w)
            height = int(viewframe_coords.get('height_percent', 0) * h)
        else:
            # Default: centered square (same as embed)
            min_dim = min(h, w)
            center = (w // 2, h // 2)
            square_size = int(min_dim * 0.7)
            x = center[0] - square_size // 2
            y = center[1] - square_size // 2
            width = height = square_size
        
        # Crop to region
        cropped = img_pt[:, :, y:y+height, x:x+width]
        cropped_256 = torch.nn.functional.interpolate(cropped, size=(256, 256), mode='bilinear', align_corners=False)
        
        preds = self.wam.detect(cropped_256)["preds"]
        
        mask_preds = torch.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        pred_message_tensor = msg_predict_inference(bit_preds, mask_preds).cpu().float()
        
        readable_message, is_valid, bitflips = roco_decode_from_binary_tensor(pred_message_tensor[0])
        
        total_bits = 32
        bit_error_rate = (bitflips / total_bits) * 100 if bitflips >= 0 else -1

        binary_str = msg2str(pred_message_tensor[0].numpy())
        
        results = {
            'binary_message': binary_str,
            'readable_message': readable_message,
            'bit_error_rate_percent': bit_error_rate,
            'corrected_bitflips': bitflips,
            'ecc_valid': is_valid,
            'bit_accuracy': None,
        }
        
        if original_message:
            original_binary_tensor = roco_encode_to_binary_tensor(original_message)
            results['bit_accuracy'] = (pred_message_tensor[0] == original_binary_tensor).float().mean().item()
            
        return results