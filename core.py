import os
import io
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import save_image

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    default_transform, unnormalize_img, msg2str
)
from viewframe import get_inner_square_region, draw_viewframe_overlay
from watermark_utils import (
    init_model, load_image, crop_to_centered_square, pil_to_cv2, cv2_to_pil,
    create_mask_from_coords, create_mask_from_percentages,
    validate_pixel_coords, validate_percentage_coords,
    roco_encode_to_binary_tensor, roco_decode_from_binary_tensor
)

def get_device():
    """Gets the available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(source):
    """Loads and preprocesses an image from a source (file path or file-like object)."""
    img = load_image(source)
    cv_img = pil_to_cv2(img)
    cv_img = crop_to_centered_square(cv_img)
    img = cv2_to_pil(cv_img)
    
    device = get_device()
    img_pt = default_transform(img).unsqueeze(0).to(device)
    
    return img_pt, cv_img

def create_watermark_mask(img_pt, cv_img, mode, params=None):
    """Creates a watermark mask based on the specified mode and parameters."""
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
            'x_percent': coords['x'] / w,
            'y_percent': coords['y'] / h,
            'width_percent': coords['width'] / w,
            'height_percent': coords['height'] / h
        })

    return mask, coords

def embed_watermark(wam, img_pt, cv_img, message, mask):
    """Embeds the watermark into the image."""
    wm_msg_tensor = roco_encode_to_binary_tensor(message)
    wm_msg = wm_msg_tensor.unsqueeze(0).to(img_pt.device)
    outputs = wam.embed(img_pt, wm_msg)
    
    overlay = draw_viewframe_overlay(cv_img)
    overlay_pil = cv2_to_pil(overlay)
    overlay_pt = default_transform(overlay_pil).unsqueeze(0).to(img_pt.device)
    
    img_w = outputs['imgs_w'] * mask + overlay_pt * (1 - mask)
    
    # Convert the binary tensor to a string for display
    binary_message_str = "".join(map(str, wm_msg_tensor.int().tolist()))
    
    return img_w, binary_message_str

def verify_watermark(wam, img_pt, original_message=None):
    """Verifies the watermark in an image using ROCO ECC."""
    preds = wam.detect(img_pt)["preds"]
    mask_preds = torch.sigmoid(preds[:, 0, :, :])
    bit_preds = preds[:, 1:, :, :]
    
    pred_message_tensor = msg_predict_inference(bit_preds, mask_preds).cpu().float()
    
    # Use the new ROCO decoder
    readable_message, is_valid, bitflips = roco_decode_from_binary_tensor(pred_message_tensor[0])
    
    # Calculate the bit error rate based on corrected bitflips
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
        # Re-encode original message to compare for bit accuracy
        original_binary_tensor = roco_encode_to_binary_tensor(original_message)
        results['bit_accuracy'] = (pred_message_tensor[0] == original_binary_tensor).float().mean().item()
        
    return results