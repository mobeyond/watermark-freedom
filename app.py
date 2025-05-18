import os
import io
from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from werkzeug.utils import secure_filename
import numpy as np
import cv2

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str
)
from viewframe import get_inner_square_region, draw_viewframe_overlay

app = Flask(__name__)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_dir = "checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

def str_to_binary(msg_str, nbits=32):
    """Convert a string to a binary tensor of length nbits"""
    binary_str = ''.join(format(ord(c), '08b') for c in msg_str)
    if len(binary_str) > nbits:
        print(f"Warning: Message '{msg_str}' is too long for {nbits} bits. Truncating...")
        binary_str = binary_str[:nbits]
    elif len(binary_str) < nbits:
        binary_str = binary_str.ljust(nbits, '0')
    binary_tensor = torch.tensor([int(b) for b in binary_str], dtype=torch.float32)
    return binary_tensor

def crop_to_centered_square(image):
    """
    Returns the largest centered square crop from the input image.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return image[top:top+min_dim, left:left+min_dim]

def create_mask_from_pixels(img_tensor, x, y, width, height):
    """Create a mask using pixel coordinates"""
    batch_size, channels, img_height, img_width = img_tensor.shape
    
    # Validate pixel coordinates
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Pixel coordinates must be non-negative and dimensions must be positive")
    if x + width > img_width or y + height > img_height:
        raise ValueError("Mask region exceeds image dimensions")
    
    # Create mask
    mask = torch.zeros((batch_size, 1, img_height, img_width), device=img_tensor.device)
    mask[:, :, y:y+height, x:x+width] = 1.0
    return mask

def create_fixed_mask(img_tensor, x_percent=0.3, y_percent=0.3, width_percent=0.4, height_percent=0.4):
    """Create a fixed mask for watermarking based on percentage coordinates"""
    batch_size, channels, height, width = img_tensor.shape
    x_start = int(width * x_percent)
    y_start = int(height * y_percent)
    x_end = int(width * (x_percent + width_percent))
    y_end = int(height * (y_percent + height_percent))
    mask = torch.zeros((batch_size, 1, height, width), device=img_tensor.device)
    mask[:, :, y_start:y_end, x_start:x_end] = 1.0
    return mask

def get_viewframe_overlay_and_inner_square(img):
    """Get viewframe overlay and inner square coordinates using template function"""
    height, width = img.shape[:2]
    # Use a fixed frame width based on image size
    frame_width = int(min(width, height) * 0.1)  # 10% of the smaller dimension
    
    # Create frame overlay with transparent background
    overlay = np.zeros_like(img)
    
    # Only draw the frame lines (not corners)
    # Top frame (excluding corners)
    overlay[frame_width:frame_width*2, frame_width*2:width-frame_width*2] = (255, 255, 255)
    # Bottom frame (excluding corners)
    overlay[height-frame_width*2:height-frame_width, frame_width*2:width-frame_width*2] = (255, 255, 255)
    # Left frame (excluding corners)
    overlay[frame_width*2:height-frame_width*2, frame_width:frame_width*2] = (255, 255, 255)
    # Right frame (excluding corners)
    overlay[frame_width*2:height-frame_width*2, width-frame_width*2:width-frame_width] = (255, 255, 255)
    
    # Define frame corner regions using template function
    crop_top = frame_width
    crop_left = frame_width
    crop_bottom = height - frame_width
    crop_right = width - frame_width
    
    return overlay, (crop_top, crop_left, crop_bottom, crop_right)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/watermark', methods=['POST'])
def watermark_image():
    try:
        # Get parameters from request
        if 'cover' not in request.files:
            return jsonify({'error': 'No cover image provided'}), 400
        
        cover_file = request.files['cover']
        if cover_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Get original filename and create watermarked filename
        original_filename = secure_filename(cover_file.filename)
        filename_base, file_ext = os.path.splitext(original_filename)
        watermarked_filename = f"{filename_base}_watermarked{file_ext}"
        
        # Get watermark parameters
        message = request.form.get('message', 'Hello World!')
        
        # Load and process image
        img = Image.open(cover_file).convert("RGB")
        
        # Convert PIL Image to cv2 format for cropping
        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # Always crop to centered square if image is not square
        cv_img = crop_to_centered_square(cv_img)
        
        # Convert back to PIL Image for processing
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        img_pt = default_transform(img).unsqueeze(0).to(device)
        
        # Create watermark message
        wm_msg = str_to_binary(message).unsqueeze(0).to(device)
        
        # Embed watermark
        outputs = wam.embed(img_pt, wm_msg)
        
        # Get image dimensions for coordinate calculations
        h, w = img_pt.shape[2:]
        
        # Check if using frame corners - default to true if not specified
        use_frame_corners = request.form.get('use_frame_corners', 'true').lower() == 'true'
        
        if use_frame_corners:
            # Get inner square region coordinates
            crop_top, crop_left, crop_bottom, crop_right = get_inner_square_region(cv_img)
            x = crop_left
            y = crop_top
            width = crop_right - crop_left
            height = crop_bottom - crop_top
            mask = create_mask_from_pixels(img_pt, x, y, width, height)
        else:
            # Get mask parameters - check if using pixels or percentages
            use_pixels = request.form.get('use_pixels', 'false').lower() == 'true'
            
            if use_pixels:
                # Get pixel values and validate they are positive integers
                try:
                    x = int(request.form['x_pixels'])
                    y = int(request.form['y_pixels'])
                    width = int(request.form['width_pixels'])
                    height = int(request.form['height_pixels'])
                    
                    if any(val < 0 for val in [x, y, width, height]):
                        return jsonify({'error': 'Pixel values must be non-negative integers'}), 400
                    
                    if x + width > w or y + height > h:
                        return jsonify({
                            'error': 'Watermark region exceeds image dimensions',
                            'image_size': {'width': w, 'height': h},
                            'watermark_region': {'x': x, 'y': y, 'width': width, 'height': height}
                        }), 400
                    
                except ValueError:
                    return jsonify({'error': 'Pixel values must be valid integers'}), 400
            else:
                # Get percentage values and validate they are between 0 and 1
                try:
                    x_percent = float(request.form['x_percent'])
                    y_percent = float(request.form['y_percent'])
                    width_percent = float(request.form['width_percent'])
                    height_percent = float(request.form['height_percent'])
                    
                    if not all(0 <= val <= 1 for val in [x_percent, y_percent, width_percent, height_percent]):
                        return jsonify({'error': 'Percentage values must be between 0 and 1'}), 400
                    
                    # Convert percentages to pixels
                    x = int(w * x_percent)
                    y = int(h * y_percent)
                    width = int(w * width_percent)
                    height = int(h * height_percent)
                    
                except ValueError:
                    return jsonify({'error': 'Percentage values must be valid numbers'}), 400
            
            mask = create_mask_from_pixels(img_pt, x, y, width, height)
        
        # Now draw the viewframe overlay using the mask boundaries
        overlay = draw_viewframe_overlay(cv_img)
        
        # Convert overlay back to tensor
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        overlay_pt = default_transform(overlay_pil).unsqueeze(0).to(device)
        
        # Apply watermark using mask
        img_w = outputs['imgs_w'] * mask + overlay_pt * (1 - mask)
        
        # Convert final image to PIL
        img_w_pil = unnormalize_img(img_w).squeeze(0).cpu()
        img_w_pil = Image.fromarray((img_w_pil.detach().numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
        
        # Save to memory buffer for web response
        img_buffer = io.BytesIO()
        img_w_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=watermarked_filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_watermark():
    try:
        # Validate input
        if 'watermarked' not in request.files:
            return jsonify({'error': 'No watermarked image provided'}), 400
        
        watermarked_file = request.files['watermarked']
        if watermarked_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Load and process image
        img = Image.open(watermarked_file).convert("RGB")
        
        # Convert PIL Image to cv2 format for cropping
        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # Always crop to centered square if image is not square
        cv_img = crop_to_centered_square(cv_img)
        
        # Convert back to PIL Image for processing
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        img_pt = default_transform(img).unsqueeze(0).to(device)
        
        # Get image dimensions
        h, w = img_pt.shape[2:]
        
        # Check if using frame corners - default to true if not specified
        use_frame_corners = request.form.get('use_frame_corners', 'true').lower() == 'true'
        
        if use_frame_corners:
            # Get inner square region coordinates
            crop_top, crop_left, crop_bottom, crop_right = get_inner_square_region(cv_img)
            x = crop_left
            y = crop_top
            width = crop_right - crop_left
            height = crop_bottom - crop_top
            mask = create_mask_from_pixels(img_pt, x, y, width, height)
        else:
            # Get mask parameters - check if using pixels or percentages
            use_pixels = request.form.get('use_pixels', 'false').lower() == 'true'
            
            if use_pixels:
                # Get pixel values and validate they are positive integers
                try:
                    x = int(request.form['x_pixels'])
                    y = int(request.form['y_pixels'])
                    width = int(request.form['width_pixels'])
                    height = int(request.form['height_pixels'])
                    
                    if any(val < 0 for val in [x, y, width, height]):
                        return jsonify({'error': 'Pixel values must be non-negative integers'}), 400
                    
                    if x + width > w or y + height > h:
                        return jsonify({
                            'error': 'Watermark region exceeds image dimensions',
                            'image_size': {'width': w, 'height': h},
                            'watermark_region': {'x': x, 'y': y, 'width': width, 'height': height}
                        }), 400
                    
                except ValueError:
                    return jsonify({'error': 'Pixel values must be valid integers'}), 400
            else:
                # Get percentage values and validate they are between 0 and 1
                try:
                    x_percent = float(request.form['x_percent'])
                    y_percent = float(request.form['y_percent'])
                    width_percent = float(request.form['width_percent'])
                    height_percent = float(request.form['height_percent'])
                    
                    if not all(0 <= val <= 1 for val in [x_percent, y_percent, width_percent, height_percent]):
                        return jsonify({'error': 'Percentage values must be between 0 and 1'}), 400
                    
                    # Convert percentages to pixels
                    x = int(w * x_percent)
                    y = int(h * y_percent)
                    width = int(w * width_percent)
                    height = int(h * height_percent)
                    
                except ValueError:
                    return jsonify({'error': 'Percentage values must be valid numbers'}), 400
            
            mask = create_mask_from_pixels(img_pt, x, y, width, height)
        
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
        
        # Calculate bit accuracy if original message is provided
        bit_accuracy = None
        if 'original_message' in request.form:
            original_message = request.form['original_message']
            original_binary = str_to_binary(original_message)
            bit_accuracy = (pred_message[0] == original_binary).float().mean().item()
        
        return jsonify({
            'filename': secure_filename(watermarked_file.filename),
            'binary_message': binary_str,
            'readable_message': readable_message,
            'mask_confidence': mask_confidence,
            'bit_accuracy': bit_accuracy,
            'mask_region': {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'x_percent': x / w,
                'y_percent': y / h,
                'width_percent': width / w,
                'height_percent': height / h
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True) 