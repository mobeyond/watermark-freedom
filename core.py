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
        mask, coords = self._create_watermark_mask(img_pt, cv_img, mask_mode, mask_params)

        wm_msg_tensor = roco_encode_to_binary_tensor(message)
        wm_msg = wm_msg_tensor.unsqueeze(0).to(self.device)
        outputs = self.wam.embed(img_pt, wm_msg)
        
        overlay = draw_viewframe_overlay(cv_img)
        overlay_pil = cv2_to_pil(overlay)
        overlay_pt = default_transform(overlay_pil).unsqueeze(0).to(self.device)
        
        img_w = outputs['imgs_w'] * mask + overlay_pt * (1 - mask)
        
        binary_message_str = "".join(map(str, wm_msg_tensor.int().tolist()))
        
        return img_w, binary_message_str, coords

    def verify(self, image_source, original_message=None):
        img_pt, _ = self._preprocess_image(image_source)
        
        preds = self.wam.detect(img_pt)["preds"]
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