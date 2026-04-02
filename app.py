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
from backends.videoseal_backend import VideoSealBackend

app = Flask(__name__)
wam_watermarker = WatermarkManager()
vs_watermarker = VideoSealBackend()

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

        backend = request.form.get("backend", "videoseal").lower()
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

        backend = request.form.get("backend", "videoseal").lower()
        original_message = request.form.get("original_message")

        if backend == "videoseal":
            return handle_videoseal_verify(watermarked_file, original_message)
        else:
            return handle_wam_verify(watermarked_file, original_message)

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f"An unexpected error occurred: {e}", 500)


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


def handle_videoseal_watermark(cover_file, message):
    if len(message) > MAX_MESSAGE_LENGTH_VIDEO:
        return create_error_response(
            f"Message too long. Maximum {MAX_MESSAGE_LENGTH_VIDEO} characters.",
            400,
        )

    # Get viewframe parameters from request (same as WAM)
    try:
        x_percent = float(request.form.get("x_percent", 0.15))
        y_percent = float(request.form.get("y_percent", 0.15))
    except (ValueError, TypeError):
        x_percent = y_percent = 0.15

    # Use user's X,Y position for margin calculation (symmetric: margin = x = y)
    margin_pct = x_percent  # Both x and y should be same for centered square

    image_bytes = cover_file.read()
    img_bytes, binary, coords = vs_watermarker.embed_bytes(
        image_bytes, message, margin_pct
    )

    original_filename = secure_filename(cover_file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    watermarked_filename = f"{filename_base}_watermarked{file_ext}"

    return jsonify(
        {
            "image": base64.b64encode(img_bytes).decode(),
            "filename": watermarked_filename,
            "binary_message": binary,
            "viewframe": {
                "x": float(coords.get("x_percent", 0)),
                "y": float(coords.get("y_percent", 0)),
                "width": float(coords.get("width_percent", 1.0)),
                "height": float(coords.get("height_percent", 1.0)),
            },
            "backend": "videoseal",
        }
    )


def handle_videoseal_verify(watermarked_file, original_message):
    image_bytes = watermarked_file.read()
    result = vs_watermarker.verify_bytes(image_bytes, original_message)

    final_response = {
        "filename": secure_filename(watermarked_file.filename),
        "readable_message": result.get("readable", ""),
        "ecc_valid": result.get("ecc_valid"),
        "corrected_bitflips": result.get("corrected_bitflips"),
        "bit_accuracy": result.get("bit_accuracy"),
        "raw_binary_message": result.get("binary_message", ""),
        "viewframe": result.get("viewframe"),
        "backend": "videoseal",
    }

    if result.get("bit_accuracy") is not None:
        final_response["bit_accuracy_vs_original"] = (
            f"{result['bit_accuracy'] * 100:.2f}%"
        )

    return jsonify(final_response)


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
