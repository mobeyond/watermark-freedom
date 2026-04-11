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
from backends.custom_model_backend import CustomModelBackend

app = Flask(__name__)
wam_watermarker = WatermarkManager()
vs_watermarker = VideoSealBackend()
custom_watermarker = CustomModelBackend()

MAX_MESSAGE_LENGTH_WAM = 3
MAX_MESSAGE_LENGTH_VIDEO_ROCO = 3  # ROCO 3-char (default)
MAX_MESSAGE_LENGTH_VIDEO_ROCO32 = 4  # ROCO32 4-char (advanced)
MAX_MESSAGE_LENGTH_CUSTOM_ROCO = 3  # ROCO 3-char for custom model
MAX_MESSAGE_LENGTH_CUSTOM_ROCO32 = 4  # ROCO32 4-char for custom model


def log_viewframe_corners(filename: str, operation: str, coords: dict, backend: str = "") -> None:
    """Log viewframe corner coordinates in a compact format.

    Args:
        filename: Name of the processed image file
        operation: Type of operation ("EMBED" or "VERIFY")
        coords: Dictionary with viewframe coordinates (x, y, width, height in pixels)
        backend: Backend name for logging purposes
    """
    x = int(coords.get("x", 0))
    y = int(coords.get("y", 0))
    width = int(coords.get("width", 0))
    height = int(coords.get("height", 0))

    # Calculate 4 corner positions
    tl = f"({x},{y})"  # Top-Left
    tr = f"({x+width},{y})"  # Top-Right
    br = f"({x+width},{y+height})"  # Bottom-Right
    bl = f"({x},{y+height})"  # Bottom-Left

    backend_str = f"[{backend.upper()}]" if backend else ""
    print(f"[{operation}] {filename} {backend_str} viewframe: TL={tl} TR={tr} BR={br} BL={bl} [{width}x{height}]")


def get_mask_params_from_request(req) -> Tuple[str, Optional[dict]]:
    use_frame_corners = req.form.get("use_frame_corners", "true").lower() == "true"
    if use_frame_corners:
        return "corners", None

    use_pixels = req.form.get("use_pixels", "false").lower() == "true"
    if use_pixels:
        try:
            # Check for new single input or old separate inputs
            if "margin_pixels" in req.form:
                margin = int(req.form.get("margin_pixels", 0))
                x = y = margin
            else:
                x = int(req.form.get("x_pixels", 0))
                y = int(req.form.get("y_pixels", 0))
            # Calculate width/height automatically for centered square
            # Assuming 256x256 image size
            img_size = 256
            width = max(0, img_size - 2 * max(x, y))
            height = width
            return "pixels", {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid pixel parameters: {e}")
    else:
        try:
            # Check for new single input or old separate inputs
            if "margin_percent" in req.form:
                margin = float(req.form.get("margin_percent", 0.10))
                x_percent = y_percent = margin
            else:
                x_percent = float(req.form.get("x_percent", 0.10))
                y_percent = float(req.form.get("y_percent", 0.10))
            # Calculate width/height automatically for centered square: 1 - 2*margin
            margin_val = max(x_percent, y_percent)
            width_percent = max(0, 1 - 2 * margin_val)
            height_percent = width_percent
            return "percentage", {
                "x_percent": x_percent,
                "y_percent": y_percent,
                "width_percent": width_percent,
                "height_percent": height_percent,
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
        elif backend == "custom":
            return handle_custom_model_watermark(cover_file, message)
        else:
            return handle_wam_watermark(cover_file, message)

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

    # Log embedding info
    original_filename = secure_filename(cover_file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    watermarked_filename = f"{filename_base}_watermarked{file_ext}"
    log_viewframe_corners(watermarked_filename, "EMBED", coords, "WAM")

    img_w_to_save = unnormalize_img(img_w)
    img_buffer = io.BytesIO()
    save_image(img_w_to_save, img_buffer, format="PNG")
    img_buffer.seek(0)

    encoded_img = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

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

    # Log verification info
    filename = secure_filename(watermarked_file.filename)
    vf = results.get("viewframe", {})
    if vf:
        # Convert percentage coords to pixel coords for logging (assuming 256x256)
        pixel_coords = {
            "x": int(vf.get("x_percent", 0) * 256),
            "y": int(vf.get("y_percent", 0) * 256),
            "width": int(vf.get("width_percent", 1) * 256),
            "height": int(vf.get("height_percent", 1) * 256),
        }
        log_viewframe_corners(filename, "VERIFY", pixel_coords, "WAM")

    final_response = {
        "filename": filename,
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
    # Check encoding type - default to roco (3-char) for robustness
    encoding = request.form.get("encoding", "roco").lower()

    if encoding == "roco32":
        max_len = MAX_MESSAGE_LENGTH_VIDEO_ROCO32
    else:  # roco (default)
        max_len = MAX_MESSAGE_LENGTH_VIDEO_ROCO

    if len(message) > max_len:
        return create_error_response(
            f"Message too long. Maximum {max_len} characters for {encoding} encoding.",
            89,
        )

    # Get viewframe parameters from request
    # Frontend sends "margin_percent" for region-based mode
    try:
        margin_pct = float(request.form.get("margin_percent", 0.10))
    except (ValueError, TypeError):
        margin_pct = 0.10

    image_bytes = cover_file.read()

    if encoding == "roco32":
        # ROCO32: 4-char encoding with 256 unique bits
        img_bytes, binary, coords = vs_watermarker.embed_bytes(
            image_bytes, message, margin_pct
        )
    else:
        # ROCO: 3-char encoding with bit-level repetition (more robust)
        img_bytes, binary, coords = vs_watermarker.embed_bytes_roco(
            image_bytes, message, margin_pct
        )

    original_filename = secure_filename(cover_file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    watermarked_filename = f"{filename_base}_watermarked{file_ext}"

    # Log embedding info with pixel coordinates
    log_viewframe_corners(watermarked_filename, "EMBED", coords, "VIDEOSEAL")

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
            "encoding": encoding,
        }
    )


def handle_videoseal_verify(watermarked_file, original_message):
    encoding = request.form.get("encoding", "roco").lower()
    image_bytes = watermarked_file.read()

    # Verification uses AUTO-DETECTION - no margin parameter needed
    # The backend detects viewframe from bracket corners automatically

    if encoding == "roco32":
        result = vs_watermarker.verify_bytes(image_bytes, original_message)
    else:
        result = vs_watermarker.verify_bytes_roco(image_bytes, original_message)

    # Log verification info
    filename = secure_filename(watermarked_file.filename)
    vf = result.get("viewframe", {})
    if vf:
        log_viewframe_corners(filename, "VERIFY", vf, "VIDEOSEAL")

    final_response = {
        "filename": filename,
        "readable_message": result.get("readable", ""),
        "ecc_valid": result.get("ecc_valid"),
        "corrected_bitflips": result.get("corrected_bitflips"),
        "bit_accuracy": result.get("bit_accuracy"),
        "raw_binary_message": result.get("binary_message", ""),
        "viewframe": vf,
        "backend": "videoseal",
        "encoding": encoding,
    }

    if result.get("bit_accuracy") is not None:
        final_response["bit_accuracy_vs_original"] = (
            f"{result['bit_accuracy'] * 100:.2f}%"
        )

    return jsonify(final_response)


def handle_custom_model_watermark(cover_file, message):
    # Check encoding type - default to roco (3-char) for robustness
    encoding = request.form.get("encoding", "roco").lower()

    if encoding == "roco32":
        max_len = MAX_MESSAGE_LENGTH_CUSTOM_ROCO32
    else:  # roco (default)
        max_len = MAX_MESSAGE_LENGTH_CUSTOM_ROCO

    if len(message) > max_len:
        return create_error_response(
            f"Message too long. Maximum {max_len} characters for {encoding} encoding.",
            400,
        )

    # Get viewframe parameters from request
    # Frontend sends "margin_percent" for region-based mode
    try:
        margin_pct = float(request.form.get("margin_percent", 0.10))
    except (ValueError, TypeError):
        margin_pct = 0.10

    image_bytes = cover_file.read()

    if encoding == "roco32":
        # ROCO32: 4-char encoding with 256 unique bits
        # For custom model, we use the same embed_bytes_roco but accept 4 chars
        img_bytes, binary, coords = custom_watermarker.embed_bytes_roco(
            image_bytes, message, margin_pct
        )
    else:
        # ROCO: 3-char encoding with bit-level repetition (more robust)
        img_bytes, binary, coords = custom_watermarker.embed_bytes_roco(
            image_bytes, message, margin_pct
        )

    original_filename = secure_filename(cover_file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    watermarked_filename = f"{filename_base}_watermarked{file_ext}"

    # Log embedding info with pixel coordinates
    log_viewframe_corners(watermarked_filename, "EMBED", coords, "CUSTOM")

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
            "backend": "custom",
            "encoding": encoding,
        }
    )


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
