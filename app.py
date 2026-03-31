import os
import io
import base64
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from torchvision.utils import save_image
from typing import Optional, Tuple

from notebooks.inference_utils import unnormalize_img
from core import WatermarkManager
from watermark_utils import create_error_response

app = Flask(__name__)
wam_watermarker = WatermarkManager()

PYTHON312 = "/usr/bin/python3.12"
PYTHON312_SITE = "/home/h/.local/lib/python3.12/site-packages"


# ---------------------------------------------------------------------------
# VideoSeal subprocess helpers
# ---------------------------------------------------------------------------


def _find_json_in_output(stdout: str) -> dict:
    import json
    import re

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stdout, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise RuntimeError(f"No JSON found in output: {stdout[:500]}")


def videoseal_embed(image_bytes, message):
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode="wb") as f:
        f.write(image_bytes)
        img_path = f.name

    script = f"""
import sys
import os
import io
import base64
import json
sys.path.insert(0, '{PYTHON312_SITE}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{PYTHON312_SITE}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend
from viewframe import draw_corner_brackets
import cv2
import numpy as np

img = Image.open('{img_path}').convert('RGB')
wm = VideoSealBackend()
result, binary, coords = wm.embed(img, '{message}')

img_np = np.array(result)
if len(img_np.shape) == 2:
    img_np = np.stack([img_np]*3, axis=-1)
elif img_np.shape[2] == 4:
    img_np = img_np[:,:,:3]
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
x, y = int(coords.get('x', 0)), int(coords.get('y', 0))
w, h = int(coords.get('width', img_np.shape[1])), int(coords.get('height', img_np.shape[0]))
corner_length = int(min(w, h) * 0.15)
line_thickness = max(2, int(min(w, h) * 0.012))
draw_corner_brackets(img_bgr, x, y, w, h, corner_length, line_thickness, method='distinctive')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
result_img = Image.fromarray(img_rgb)

buf = io.BytesIO()
result_img.save(buf, format='PNG')
img_b64 = base64.b64encode(buf.getvalue()).decode()

print(json.dumps({{'image': img_b64, 'binary': binary[:32], 'coords': coords}}))
"""

    try:
        result = subprocess.run(
            [PYTHON312, "-c", script], capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        return _find_json_in_output(result.stdout)
    finally:
        os.unlink(img_path)


def videoseal_verify(image_bytes, original_message=None):
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode="wb") as f:
        f.write(image_bytes)
        img_path = f.name

    script = f"""
import sys
import os
import json
sys.path.insert(0, '{PYTHON312_SITE}')
sys.path.insert(0, '/home/h/FLY/watermark-freedom')
os.chdir('{PYTHON312_SITE}')
from PIL import Image
from backends.videoseal_backend import VideoSealBackend

img = Image.open('{img_path}').convert('RGB')
wm = VideoSealBackend()
result = wm.verify(img, {repr(original_message)})

print(json.dumps({{
    'readable': result['readable_message'][:32],
    'accuracy': result.get('bit_accuracy'),
    'viewframe': result.get('viewframe')
}}))
"""

    try:
        proc = subprocess.run(
            [PYTHON312, "-c", script], capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        return _find_json_in_output(proc.stdout)
    finally:
        os.unlink(img_path)


# ---------------------------------------------------------------------------
# Mask / viewframe helpers (WAM)
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH_WAM = 3
MAX_MESSAGE_LENGTH_VIDEO = 32


def get_mask_params_from_request(req) -> Tuple[str, Optional[dict]]:
    use_frame_corners = req.form.get("use_frame_corners", "true").lower() == "true"
    if use_frame_corners:
        return "corners", None

    use_pixels = req.form.get("use_pixels", "false").lower() == "true"
    if use_pixels:
        try:
            return "pixels", {
                "x": int(req.form["x_pixels"]),
                "y": int(req.form["y_pixels"]),
                "width": int(req.form["width_pixels"]),
                "height": int(req.form["height_pixels"]),
            }
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid pixel parameters: {e}")
    else:
        try:
            return "percentage", {
                "x_percent": float(req.form["x_percent"]),
                "y_percent": float(req.form["y_percent"]),
                "width_percent": float(req.form["width_percent"]),
                "height_percent": float(req.form["height_percent"]),
            }
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid percentage parameters: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/watermark", methods=["POST"])
def watermark_image_route():
    try:
        if "cover" not in request.files:
            return create_error_response("No cover image provided", 400)

        cover_file = request.files["cover"]
        if cover_file.filename == "":
            return create_error_response("No selected file", 400)

        backend = request.form.get("backend", "wam").lower()
        message = request.form.get("message", "ABC")

        if backend == "videoseal":
            return handle_videoseal_watermark(cover_file, message)
        else:
            return handle_wam_watermark(cover_file, message)

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f"An unexpected error occurred: {e}", 500)


@app.route("/verify", methods=["POST"])
def verify_watermark_route():
    try:
        if "watermarked" not in request.files:
            return create_error_response("No watermarked image provided", 400)

        watermarked_file = request.files["watermarked"]
        if watermarked_file.filename == "":
            return create_error_response("No selected file", 400)

        backend = request.form.get("backend", "wam").lower()
        original_message = request.form.get("original_message")

        if backend == "videoseal":
            return handle_videoseal_verify(watermarked_file, original_message)
        else:
            return handle_wam_verify(watermarked_file, original_message)

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f"An unexpected error occurred: {e}", 500)


# ---------------------------------------------------------------------------
# WAM handlers
# ---------------------------------------------------------------------------


def handle_wam_watermark(cover_file, message):
    if len(message) > MAX_MESSAGE_LENGTH_WAM:
        return create_error_response(
            f"Message too long. Maximum {MAX_MESSAGE_LENGTH_WAM} characters for WAM.",
            400,
        )

    mask_mode, mask_params = get_mask_params_from_request(request)

    img_w, binary_message, coords = wam_watermarker.embed(
        cover_file, message, mask_mode, mask_params
    )

    img_w_to_save = unnormalize_img(img_w)
    img_buffer = io.BytesIO()
    save_image(img_w_to_save, img_buffer, format="PNG")
    img_buffer.seek(0)

    encoded_img = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    original_filename = secure_filename(cover_file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    watermarked_filename = f"{filename_base}_watermarked{file_ext}"

    viewframe = {
        "x": float(coords.get("x_percent", 0)),
        "y": float(coords.get("y_percent", 0)),
        "width": float(coords.get("width_percent", 0)),
        "height": float(coords.get("height_percent", 0)),
    }

    return jsonify(
        {
            "image": encoded_img,
            "filename": watermarked_filename,
            "binary_message": binary_message,
            "viewframe": viewframe,
            "backend": "wam",
        }
    )


def handle_wam_verify(watermarked_file, original_message):
    results = wam_watermarker.verify(watermarked_file, original_message)

    final_response = {
        "filename": secure_filename(watermarked_file.filename),
        "readable_message": results["readable_message"],
        "bit_error_rate_percent": f"{results['bit_error_rate_percent']:.2f}%"
        if results["bit_error_rate_percent"] >= 0
        else "N/A",
        "corrected_bitflips": results["corrected_bitflips"],
        "is_valid_codeword": results["ecc_valid"],
        "raw_binary_message": results["binary_message"],
        "backend": "wam",
    }

    if "viewframe" in results:
        final_response["viewframe"] = results["viewframe"]

    if results.get("bit_accuracy") is not None:
        final_response["bit_accuracy_vs_original"] = (
            f"{results['bit_accuracy'] * 100:.2f}%"
        )

    return jsonify(final_response)


# ---------------------------------------------------------------------------
# VideoSeal handlers
# ---------------------------------------------------------------------------


def handle_videoseal_watermark(cover_file, message):
    if len(message) > MAX_MESSAGE_LENGTH_VIDEO:
        return create_error_response(
            f"Message too long. Maximum {MAX_MESSAGE_LENGTH_VIDEO} characters for VideoSeal.",
            400,
        )

    image_bytes = cover_file.read()
    result = videoseal_embed(image_bytes, message)

    original_filename = secure_filename(cover_file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    watermarked_filename = f"{filename_base}_watermarked{file_ext}"

    viewframe = {
        "x": float(result["coords"].get("x_percent", 0)),
        "y": float(result["coords"].get("y_percent", 0)),
        "width": float(result["coords"].get("width_percent", 1.0)),
        "height": float(result["coords"].get("height_percent", 1.0)),
    }

    return jsonify(
        {
            "image": result["image"],
            "filename": watermarked_filename,
            "binary_message": result["binary"],
            "viewframe": viewframe,
            "backend": "videoseal",
        }
    )


def handle_videoseal_verify(watermarked_file, original_message):
    image_bytes = watermarked_file.read()
    result = videoseal_verify(image_bytes, original_message)

    final_response = {
        "filename": secure_filename(watermarked_file.filename),
        "readable_message": result.get("readable", ""),
        "bit_accuracy": result.get("accuracy"),
        "viewframe": result.get("viewframe"),
        "backend": "videoseal",
    }

    if original_message and result.get("accuracy") is not None:
        final_response["bit_accuracy_vs_original"] = f"{result['accuracy'] * 100:.2f}%"

    return jsonify(final_response)


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
