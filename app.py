import os
import io
import base64
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from torchvision.utils import save_image
from notebooks.inference_utils import unnormalize_img
from core import WatermarkManager
from watermark_utils import create_error_response

app = Flask(__name__)
watermarker = WatermarkManager()

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
            return 'pixels', {
                'x': int(req.form['x_pixels']), 'y': int(req.form['y_pixels']),
                'width': int(req.form['width_pixels']), 'height': int(req.form['height_pixels'])
            }
        except (ValueError, KeyError):
            raise ValueError('Pixel values must be provided as valid integers.')
    else:
        try:
            return 'percentage', {
                'x_percent': float(req.form['x_percent']), 'y_percent': float(req.form['y_percent']),
                'width_percent': float(req.form['width_percent']), 'height_percent': float(req.form['height_percent'])
            }
        except (ValueError, KeyError):
            raise ValueError('Percentage values must be provided as valid numbers.')

@app.route('/watermark', methods=['POST'])
def watermark_image_route():
    try:
        if 'cover' not in request.files:
            return create_error_response('No cover image provided', 400)
        
        cover_file = request.files['cover']
        if cover_file.filename == '':
            return create_error_response('No selected file', 400)

        message = request.form.get('message', 'Hello World!')
        mask_mode, mask_params = get_mask_params_from_request(request)

        img_w, binary_message, coords = watermarker.embed(cover_file, message, mask_mode, mask_params)
        
        img_w_to_save = unnormalize_img(img_w)
        img_buffer = io.BytesIO()
        save_image(img_w_to_save, img_buffer, format='PNG')
        img_buffer.seek(0)
        
        encoded_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        original_filename = secure_filename(cover_file.filename)
        filename_base, file_ext = os.path.splitext(original_filename)
        watermarked_filename = f"{filename_base}_watermarked{file_ext}"

        # Calculate viewframe coordinates for frontend overlay
        viewframe = {
            'x': coords.get('x_percent', 0),
            'y': coords.get('y_percent', 0),
            'width': coords.get('width_percent', 0),
            'height': coords.get('height_percent', 0)
        }
        
        return jsonify({
            'image': encoded_img,
            'filename': watermarked_filename,
            'binary_message': binary_message,
            'viewframe': viewframe
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

        original_message = request.form.get('original_message')
        
        # Get viewframe coordinates if provided
        viewframe_coords = None
        if 'viewframe_x' in request.form:
            viewframe_coords = {
                'x_percent': float(request.form.get('viewframe_x', 0)),
                'y_percent': float(request.form.get('viewframe_y', 0)),
                'width_percent': float(request.form.get('viewframe_width', 0)),
                'height_percent': float(request.form.get('viewframe_height', 0))
            }
        
        results = watermarker.verify(watermarked_file, original_message, viewframe_coords)
        
        # Format the results for a user-friendly JSON response, matching frontend keys
        final_response = {
            'filename': secure_filename(watermarked_file.filename),
            'readable_message': results['readable_message'],
            'bit_error_rate_percent': f"{results['bit_error_rate_percent']:.2f}%" if results['bit_error_rate_percent'] >= 0 else "N/A",
            'corrected_bitflips': results['corrected_bitflips'],
            'is_valid_codeword': results['ecc_valid'],
            'raw_binary_message': results['binary_message'],
        }
        
        if results.get('bit_accuracy') is not None:
            final_response['bit_accuracy_vs_original'] = f"{results['bit_accuracy'] * 100:.2f}%"

        return jsonify(final_response)

    except (ValueError, KeyError) as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f'An unexpected error occurred: {e}', 500)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

