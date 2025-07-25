import os
import io
from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image
import torch
import numpy as np
from werkzeug.utils import secure_filename

from watermark_utils import init_model, create_error_response
from notebooks.inference_utils import unnormalize_img
from core import (
    preprocess_image, create_watermark_mask, embed_watermark, verify_watermark
)

app = Flask(__name__)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wam = init_model(device)

@app.route('/')
def index():
    return render_template('index.html')

def get_mask_params_from_request(req):
    """Extracts mask mode and parameters from a Flask request."""
    use_frame_corners = req.form.get('use_frame_corners', 'true').lower() == 'true'
    if use_frame_corners:
        return 'corners', None

    use_pixels = req.form.get('use_pixels', 'false').lower() == 'true'
    if use_pixels:
        try:
            params = {
                'x': int(req.form['x_pixels']),
                'y': int(req.form['y_pixels']),
                'width': int(req.form['width_pixels']),
                'height': int(req.form['height_pixels'])
            }
            return 'pixels', params
        except (ValueError, KeyError):
            raise ValueError('Pixel values must be provided as valid integers.')
    else:
        try:
            params = {
                'x_percent': float(req.form['x_percent']),
                'y_percent': float(req.form['y_percent']),
                'width_percent': float(req.form['width_percent']),
                'height_percent': float(req.form['height_percent'])
            }
            return 'percentage', params
        except (ValueError, KeyError):
            raise ValueError('Percentage values must be provided as valid numbers.')

import base64

from torchvision.utils import save_image

@app.route('/watermark', methods=['POST'])
def watermark_image_route():
    try:
        if 'cover' not in request.files:
            return create_error_response('No cover image provided', 400)
        
        cover_file = request.files['cover']
        if cover_file.filename == '':
            return create_error_response('No selected file', 400)

        original_filename = secure_filename(cover_file.filename)
        filename_base, file_ext = os.path.splitext(original_filename)
        watermarked_filename = f"{filename_base}_watermarked{file_ext}"

        message = request.form.get('message', 'Hello World!')

        img_pt, cv_img = preprocess_image(cover_file)
        
        mask_mode, mask_params = get_mask_params_from_request(request)
        mask, _ = create_watermark_mask(img_pt, cv_img, mask_mode, mask_params)

        img_w, binary_message = embed_watermark(wam, img_pt, cv_img, message, mask)

        # Un-normalize the image tensor before saving
        img_w_to_save = unnormalize_img(img_w)

        # Use a buffer and save_image to ensure consistency with mark.py
        img_buffer = io.BytesIO()
        save_image(img_w_to_save, img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Encode image to base64
        encoded_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'image': encoded_img,
            'filename': watermarked_filename,
            'binary_message': binary_message
        })

    except (ValueError, KeyError) as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f'An unexpected error occurred: {e}', 500)

@app.route('/verify', methods=['POST'])
def verify_watermark_route():
    try:
        if 'watermarked' not in request.files:
            return create_error_response('No watermarked image provided', 400)

        watermarked_file = request.files['watermarked']
        if watermarked_file.filename == '':
            return create_error_response('No selected file', 400)

        img_pt, cv_img = preprocess_image(watermarked_file)
        
        # Although mask is not strictly needed for verification, we get it for response consistency
        mask_mode, mask_params = get_mask_params_from_request(request)
        _, coords = create_watermark_mask(img_pt, cv_img, mask_mode, mask_params)

        original_message = request.form.get('original_message')
        results = verify_watermark(wam, img_pt, original_message)
        
        # Format the results for a user-friendly JSON response
        final_response = {
            'filename': secure_filename(watermarked_file.filename),
            'readable_message': results['readable_message'],
            'bit_error_rate_percent': f"{results['bit_error_rate_percent']:.2f}%" if results['bit_error_rate_percent'] >= 0 else "N/A",
            'corrected_bitflips': results['corrected_bitflips'],
            'is_valid_codeword': results['ecc_valid'],
            'mask_region': coords,
            'raw_binary_message': results['binary_message'],
        }
        
        if results['bit_accuracy'] is not None:
            final_response['bit_accuracy_vs_original'] = f"{results['bit_accuracy'] * 100:.2f}%"

        return jsonify(final_response)

    except (ValueError, KeyError) as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f'An unexpected error occurred: {e}', 500)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

