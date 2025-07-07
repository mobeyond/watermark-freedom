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
    default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str
)
from viewframe import get_inner_square_region, draw_viewframe_overlay
from watermark_utils import (
    init_model, load_image, crop_to_centered_square, pil_to_cv2, cv2_to_pil,
    robust_str_to_binary, create_mask_from_coords, create_mask_from_percentages,
    validate_pixel_coords, validate_percentage_coords, create_error_response
)

app = Flask(__name__)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wam = init_model(device)

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

        # Load and process image using utilities
        img = load_image(cover_file)
        cv_img = pil_to_cv2(img)
        cv_img = crop_to_centered_square(cv_img)

        # Convert back to PIL Image for processing
        img = cv2_to_pil(cv_img)
        img_pt = default_transform(img).unsqueeze(0).to(device)

        # Create watermark message with error correction
        wm_msg = robust_str_to_binary(message).unsqueeze(0).to(device)

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
            mask = create_mask_from_coords(img_pt, x, y, width, height)
        else:
            # Get mask parameters - check if using pixels or percentages
            use_pixels = request.form.get('use_pixels', 'false').lower() == 'true'

            if use_pixels:
                try:
                    x = int(request.form['x_pixels'])
                    y = int(request.form['y_pixels'])
                    width = int(request.form['width_pixels'])
                    height = int(request.form['height_pixels'])

                    # Validate pixel coordinates
                    is_valid, error_msg = validate_pixel_coords(w, h, x, y, width, height)
                    if not is_valid:
                        return create_error_response(error_msg, 400)

                    mask = create_mask_from_coords(img_pt, x, y, width, height)
                except ValueError:
                    return create_error_response('Pixel values must be valid integers', 400)
            else:
                try:
                    x_percent = float(request.form['x_percent'])
                    y_percent = float(request.form['y_percent'])
                    width_percent = float(request.form['width_percent'])
                    height_percent = float(request.form['height_percent'])

                    # Validate percentage coordinates
                    is_valid, error_msg = validate_percentage_coords(x_percent, y_percent, width_percent, height_percent)
                    if not is_valid:
                        return create_error_response(error_msg, 400)

                    mask = create_mask_from_percentages(img_pt, x_percent, y_percent, width_percent, height_percent)
                except ValueError:
                    return create_error_response('Percentage values must be valid numbers', 400)

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

        # Load and process image using utilities
        img = load_image(watermarked_file)
        cv_img = pil_to_cv2(img)
        cv_img = crop_to_centered_square(cv_img)

        # Convert back to PIL Image for processing
        img = cv2_to_pil(cv_img)
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
            mask = create_mask_from_coords(img_pt, x, y, width, height)
        else:
            # Get mask parameters - check if using pixels or percentages
            use_pixels = request.form.get('use_pixels', 'false').lower() == 'true'

            if use_pixels:
                try:
                    x = int(request.form['x_pixels'])
                    y = int(request.form['y_pixels'])
                    width = int(request.form['width_pixels'])
                    height = int(request.form['height_pixels'])

                    # Validate pixel coordinates
                    is_valid, error_msg = validate_pixel_coords(w, h, x, y, width, height)
                    if not is_valid:
                        return create_error_response(error_msg, 400)

                    mask = create_mask_from_coords(img_pt, x, y, width, height)
                except ValueError:
                    return create_error_response('Pixel values must be valid integers', 400)
            else:
                try:
                    x_percent = float(request.form['x_percent'])
                    y_percent = float(request.form['y_percent'])
                    width_percent = float(request.form['width_percent'])
                    height_percent = float(request.form['height_percent'])

                    # Validate percentage coordinates
                    is_valid, error_msg = validate_percentage_coords(x_percent, y_percent, width_percent, height_percent)
                    if not is_valid:
                        return create_error_response(error_msg, 400)

                    mask = create_mask_from_percentages(img_pt, x_percent, y_percent, width_percent, height_percent)
                except ValueError:
                    return create_error_response('Percentage values must be valid numbers', 400)

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
            original_binary = robust_str_to_binary(original_message)
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
