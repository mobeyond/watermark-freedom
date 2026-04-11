#!/usr/bin/env python3
"""
Verify Watermark Service - Port 5001
Serves only the watermark verification functionality.
"""

import os
import base64
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from watermark_utils import create_error_response
from backends.videoseal_backend import VideoSealBackend
from backends.custom_model_backend import CustomModelBackend

app = Flask(__name__)
vs_watermarker = VideoSealBackend()
custom_watermarker = CustomModelBackend()

MAX_MESSAGE_LENGTH_VIDEO_ROCO = 3
MAX_MESSAGE_LENGTH_VIDEO_ROCO32 = 4
MAX_MESSAGE_LENGTH_CUSTOM_ROCO = 3
MAX_MESSAGE_LENGTH_CUSTOM_ROCO32 = 4


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


@app.route("/")
def index():
    return render_template("verify.html")


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
        elif backend == "custom":
            return handle_custom_model_verify(watermarked_file, original_message)
        else:
            return create_error_response(f"Unsupported backend: {backend}", 400)

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f"An unexpected error occurred: {e}", 500)


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


def handle_custom_model_verify(watermarked_file, original_message):
    encoding = request.form.get("encoding", "roco").lower()
    image_bytes = watermarked_file.read()

    # Custom model uses roco encoding (3-char max with 32 bits)
    result = custom_watermarker.verify_bytes_roco(image_bytes, original_message)

    # Log verification info
    filename = secure_filename(watermarked_file.filename)
    vf = result.get("viewframe", {})
    if vf:
        log_viewframe_corners(filename, "VERIFY", vf, "CUSTOM")

    final_response = {
        "filename": filename,
        "readable_message": result.get("readable", ""),
        "ecc_valid": result.get("ecc_valid"),
        "corrected_bitflips": result.get("corrected_bitflips"),
        "bit_accuracy": result.get("bit_accuracy"),
        "raw_binary_message": result.get("binary_message", ""),
        "viewframe": vf,
        "backend": "custom",
        "encoding": encoding,
    }

    if result.get("bit_accuracy") is not None:
        final_response["bit_accuracy_vs_original"] = (
            f"{result['bit_accuracy'] * 100:.2f}%"
        )

    return jsonify(final_response)


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5001, debug=False)
