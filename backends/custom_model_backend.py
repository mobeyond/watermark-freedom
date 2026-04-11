"""Custom Model Backend for Watermark Freedom.

This backend provides ROCO32 (4-character) watermarking support using VideoSeal.

NOTE: The original custom checkpoint (checkpoint.pth) uses a U-Net based
architecture (unet_base_yuv_quant) that isn't defined in the standard
build_embedder function. The architecture would need to be recreated to
use that specific checkpoint.

For now, this backend uses VideoSeal which natively supports 256-bit
(ROCO32) encoding. VideoSeal provides:
- 256-bit capacity (perfect for ROCO32 4-char encoding)
- Flexible resolution (no fixed size requirement)
- MIT licensed
- High robustness
"""

import os
import sys
import io
import json
import base64
import numpy as np
import torch
from PIL import Image
from typing import Optional, Dict, Any, Tuple, Union

# Use VideoSealBackend which natively supports 256-bit/ROCO32
from backends.videoseal_backend import VideoSealBackend
from watermark_utils import (
    load_image,
    crop_to_centered_square,
    roco_encode_to_binary_tensor,
    roco_decode_from_binary_tensor,
)
from viewframe import (
    get_default_viewframe_coords,
    draw_viewframe,
    BRACKET_METHOD_DISTINCTIVE,
    calculate_line_thickness,
)
from viewframe_detector import ViewframeDetector
from viewframe_config import viewframe_config

# ROCO32: 4-character encoding with 256 unique bits
ROCO32_MAX_CHARS = 4
ROCO_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.#4679"


class CustomModelBackend:
    """Custom WAM model backend for 32-bit watermarking with ROCO encoding.

    This backend uses VideoSeal which natively supports 256-bit capacity,
    making it perfect for ROCO32 (4-character) encoding.

    The original custom checkpoint (checkpoint.pth) uses a U-Net based
    architecture that would require additional setup to load.

    Args:
        device: torch.device to run the model on (default: cuda if available)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading custom model backend (using VideoSeal for ROCO32 support)...")

        # Use VideoSealBackend which natively supports 256-bit/ROCO32
        self.videoseal_backend = VideoSealBackend()
        print("Custom model backend loaded successfully (VideoSeal for ROCO32)")

        # Initialize viewframe detector for verification
        self.viewframe_detector = ViewframeDetector(
            line_thickness=3, brightness_threshold=200, min_region_area=100
        )

    def embed(
        self,
        img_source: Union[str, Image.Image, np.ndarray],
        message: str,
        margin_pct: float = 0.10,
        padding: int = 4,
    ) -> Tuple[Image.Image, str, Dict[str, Any]]:
        """Embed watermark into image using VideoSeal with ROCO/ROCO32 encoding.

        Args:
            img_source: Input image (PIL Image, numpy array, or file path)
            message: 1-4 character message from ROCO alphabet
                    - 1-3 chars: Uses ROCO encoding (32 bits) - more robust
                    - 4 chars: Uses ROCO32 encoding (256 bits) - higher capacity
            margin_pct: Margin percentage for viewframe (default: 0.10)
            padding: Pixels to pad from each edge to exclude bracket arms (default: 4)

        Returns:
            Tuple of (watermarked_image, binary_string, coords_dict)
        """
        # Validate message
        if len(message) > 4:
            raise ValueError(f"Message must be 1-4 characters, got {len(message)}")
        for char in message:
            if char not in ROCO_ALPHABET:
                raise ValueError(
                    f"Character '{char}' not in ROCO alphabet: {ROCO_ALPHABET}"
                )

        # Load and preprocess image - handle different input types
        if isinstance(img_source, Image.Image):
            img = img_source.convert("RGB")
        elif isinstance(img_source, np.ndarray):
            img = Image.fromarray(img_source, mode="RGB")
        else:
            img = load_image(img_source)

        img_np = np.array(img)

        # Crop to centered square
        img_square = crop_to_centered_square(img_np)

        # Get viewframe coordinates
        coords = get_default_viewframe_coords(
            img_square.shape[:2], margin_pct=margin_pct
        )
        x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]

        # Extract viewframe region with padding to exclude bracket arms
        x_padded, y_padded = x + padding, y + padding
        w_padded, h_padded = w - 2 * padding, h - 2 * padding
        viewframe_region = img_square[
            y_padded : y_padded + h_padded, x_padded : x_padded + w_padded
        ]
        viewframe_pil = Image.fromarray(viewframe_region, mode="RGB")

        # Convert viewframe to bytes
        viewframe_bytes = io.BytesIO()
        viewframe_pil.save(viewframe_bytes, format="PNG")
        viewframe_bytes.seek(0)

        # Use subprocess-based embedding (runs in Python 3.12)
        # For 3-char messages, use embed_bytes_roco
        # For 4-char messages, we need to use embed_bytes with ROCO32 encoding
        if len(message) <= 3:
            watermarked_viewframe_bytes, binary_str, _ = (
                self.videoseal_backend.embed_bytes_roco(
                    viewframe_bytes.getvalue(),
                    message,
                    padding=0,  # Already padded
                )
            )
        else:
            # For 4-char messages, use embed_bytes with the message directly
            # The embed_bytes method uses the _EMBED_SCRIPT which calls wm.embed()
            watermarked_viewframe_bytes, binary_str, _ = (
                self.videoseal_backend.embed_bytes(viewframe_bytes.getvalue(), message)
            )

        # Convert back to PIL and resize to padded viewframe size
        watermarked_viewframe = Image.open(
            io.BytesIO(watermarked_viewframe_bytes)
        ).resize((w_padded, h_padded), Image.LANCZOS)

        # Place watermarked viewframe back into image (with padding)
        result_np = img_square.copy()
        result_np[y_padded : y_padded + h_padded, x_padded : x_padded + w_padded] = (
            np.array(watermarked_viewframe)
        )

        # Draw corner brackets
        import cv2

        img_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        # Calculate line_thickness based on image size
        min_dim = min(result_np.shape[0], result_np.shape[1])
        line_thickness = calculate_line_thickness(min_dim)
        draw_viewframe(
            img_bgr,
            x,
            y,
            w,
            h,
            method=BRACKET_METHOD_DISTINCTIVE,
            corner_length_pct=0.15,
            line_thickness=line_thickness,
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(img_rgb, mode="RGB")

        # Add backend info to coords
        coords["backend"] = "custom_model"
        coords["encoding"] = "roco32" if len(message) == 4 else "roco"

        return result_pil, binary_str, coords

    def verify(
        self,
        img_source: Union[str, Image.Image, np.ndarray],
        original_message: Optional[str] = None,
        padding: int = 4,
    ) -> Dict[str, Any]:
        """Verify watermark from image using VideoSeal.

        Args:
            img_source: Watermarked image (PIL Image, numpy array, or file path)
            original_message: Optional original message for bit accuracy calculation
            padding: Pixels to pad from each edge to exclude bracket arms (default: 4)

        Returns:
            Dict with verification results
        """
        # Load and preprocess image - handle different input types
        if isinstance(img_source, Image.Image):
            img = img_source.convert("RGB")
        elif isinstance(img_source, np.ndarray):
            img = Image.fromarray(img_source, mode="RGB")
        else:
            img = load_image(img_source)

        img_np = np.array(img)

        # Crop to centered square
        img_square = crop_to_centered_square(img_np)

        # Detect viewframe
        result = self.viewframe_detector.detect(
            img_square, method=viewframe_config.detection_method
        )

        if result is None:
            # Fallback to default viewframe
            result = get_default_viewframe_coords(
                img_square.shape[:2], margin_pct=0.10
            )

        x, y, w, h = result["x"], result["y"], result["width"], result["height"]

        # Extract viewframe region with padding to exclude bracket arms
        x_padded, y_padded = x + padding, y + padding
        w_padded, h_padded = w - 2 * padding, h - 2 * padding
        viewframe_region = img_square[
            y_padded : y_padded + h_padded, x_padded : x_padded + w_padded
        ]
        viewframe_pil = Image.fromarray(viewframe_region, mode="RGB")

        # Convert viewframe to bytes
        viewframe_bytes = io.BytesIO()
        viewframe_pil.save(viewframe_bytes, format="PNG")
        viewframe_bytes.seek(0)

        # Determine encoding based on original_message length (if provided)
        if original_message and len(original_message) == 4:
            # Use ROCO32 decoding (256 bits)
            result_dict = self.videoseal_backend.verify_bytes(
                viewframe_bytes.getvalue(), original_message
            )
        else:
            # Use ROCO decoding (32 bits)
            result_dict = self.videoseal_backend.verify_bytes_roco(
                viewframe_bytes.getvalue(),
                original_message,
                padding=0,  # Already padded
            )

        # Update viewframe info
        h_img, w_img = img_square.shape[:2]
        result_dict["viewframe"] = {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "x_percent": float(x / w_img),
            "y_percent": float(y / h_img),
            "width_percent": float(w / w_img),
            "height_percent": float(h / h_img),
            "ratio": float((w * h) / (w_img * h_img)),
            "size": int(min(w, h)),
            "backend": "custom_model",
        }
        result_dict["backend"] = "custom_model"
        result_dict["encoding"] = (
            "roco32" if original_message and len(original_message) == 4 else "roco"
        )

        return result_dict

    def embed_bytes_roco(
        self,
        image_bytes: bytes,
        message: str,
        margin_pct: float = 0.10,
        padding: int = 4,
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """Embed watermark into image bytes (ROCO/ROCO32 encoding, up to 4 chars).

        Args:
            image_bytes: Input image as bytes
            message: 1-4 character message from ROCO alphabet
            margin_pct: Margin percentage for viewframe
            padding: Pixels to pad from each edge to exclude bracket arms (default: 4)

        Returns:
            Tuple of (watermarked_image_bytes, binary_string, coords_dict)
        """
        # Convert bytes to PIL Image
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Embed watermark
        result_pil, binary_str, coords = self.embed(
            img_pil, message, margin_pct, padding
        )

        # Convert back to bytes
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)

        return buf.getvalue(), binary_str, coords

    def verify_bytes_roco(
        self,
        image_bytes: bytes,
        original_message: Optional[str] = None,
        padding: int = 4,
    ) -> Dict[str, Any]:
        """Verify watermark from image bytes (ROCO/ROCO32 encoding).

        Args:
            image_bytes: Watermarked image as bytes
            original_message: Optional original message for bit accuracy
            padding: Pixels to pad from each edge to exclude bracket arms (default: 4)

        Returns:
            Dict with verification results
        """
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.verify(img_pil, original_message, padding)
