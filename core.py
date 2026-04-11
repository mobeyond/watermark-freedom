import numpy as np
import cv2
import os
import sys
import warnings
from PIL import Image
from contextlib import contextmanager
import torch
from torchvision.utils import save_image
from typing import Optional, Dict, Any, Tuple, Union
from notebooks.inference_utils import (
    unnormalize_img,
    msg2str,
    default_transform,
    load_model_from_checkpoint,
)
from watermark_anything.data.metrics import msg_predict_inference
from watermark_utils import (
    load_image,
    crop_to_centered_square,
    pil_to_cv2,
    cv2_to_pil,
    roco_encode_to_binary_tensor,
    roco_decode_from_binary_tensor,
)
from viewframe import (
    draw_corner_brackets,
    get_corner_color,
    SUPPORTED_BRACKET_METHODS,
    DEFAULT_BRACKET_METHOD,
    get_default_viewframe_coords,
    calculate_line_thickness,
    calculate_viewframe_padding,
)
from viewframe_detector import ViewframeDetector
from viewframe_config import viewframe_config

# Constants
WAM_INPUT_SIZE = 256  # WAM model trained on 256x256
MIN_VIEWFRAME_SIZE = 180  # Minimum viewframe for reliable watermarking
DEFAULT_MARGIN_PERCENT = 0.10  # 10% margin for centered square
LINE_THICKNESS = 3  # Corner bracket line thickness
CORNER_LENGTH_RATIO = 0.15  # Corner bracket length as fraction of region
DEFAULT_SCALING_W = 2.0  # Default watermark strength
LOWER_SCALING_W = 2.0  # Same as default - large viewframes need full strength
VIEWFRAME_PADDING = 4  # Pixels to pad from viewframe edge to exclude bracket arms


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class WatermarkManager:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        viewframe_detector: Optional[ViewframeDetector] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        exp_dir = "checkpoints"
        json_path = os.path.join(exp_dir, "params.json")
        ckpt_path = os.path.join(exp_dir, "wam_mit.pth")

        with suppress_stdout():
            self.wam = (
                load_model_from_checkpoint(json_path, ckpt_path).to(self.device).eval()
            )
        print(f"Model loaded successfully from {ckpt_path}")

        self.viewframe_detector = viewframe_detector or ViewframeDetector(
            line_thickness=LINE_THICKNESS, brightness_threshold=200, min_region_area=100
        )

    def _preprocess_image(
        self, source: Union[str, Image.Image]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        img = load_image(source)
        cv_img = pil_to_cv2(img)
        cv_img = crop_to_centered_square(cv_img)
        img = cv2_to_pil(cv_img)
        img_pt = default_transform(img).unsqueeze(0).to(self.device)
        return img_pt, cv_img

    def _get_viewframe_region(
        self,
        w: int,
        h: int,
        mode: str,
        params: Optional[Dict[str, int]] = None,
        margin_percent: float = DEFAULT_MARGIN_PERCENT,
    ) -> Tuple[int, int, int, int]:
        """Calculate viewframe region (x, y, width, height)."""
        if mode == "pixels" and params:
            x, y = params["x"], params["y"]
            width, height = params["width"], params["height"]
        elif mode == "percentage" and params:
            x = int(params["x_percent"] * w)
            y = int(params["y_percent"] * h)
            width = int(params["width_percent"] * w)
            height = int(params["height_percent"] * h)
        else:  # corners mode (default)
            margin = int(min(w, h) * margin_percent)
            raw_width = min(w, h) - 2 * margin
            # Ensure odd side length for symmetric alignment
            if raw_width % 2 == 0:
                raw_width += 1
            width = height = raw_width
            x = y = margin
        return x, y, width, height

    def _detect_viewframe(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Auto-detect viewframe from image. Returns (x, y, width, height) or None."""
        if isinstance(img, torch.Tensor):
            img_np = img[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype("uint8")
        else:
            img_np = img

        result = self.viewframe_detector.detect(
            img_np, method=viewframe_config.detection_method
        )
        if result:
            return result["x"], result["y"], result["width"], result["height"]
        return None

    def _get_fallback_viewframe(
        self, h: int, w: int, margin_percent: float = DEFAULT_MARGIN_PERCENT
    ) -> Tuple[int, int, int, int]:
        """Get fallback viewframe using the same margin calculation as embed.

        This ensures that when detection fails, the fallback region matches
        the default embedding region (centered square with margin).

        Uses get_default_viewframe_coords() for consistent behavior across
        the codebase.
        """
        coords = get_default_viewframe_coords((h, w), margin_percent)
        return coords["x"], coords["y"], coords["width"], coords["height"]

    def _recommend_scaling_w(self, viewframe_size: int) -> float:
        """Recommend scaling_w based on viewframe size.

        Larger viewframes (>WAM_INPUT_SIZE) get lower strength to avoid over-embedding.
        Smaller viewframes use default strength for robustness.
        """
        if viewframe_size > WAM_INPUT_SIZE:
            return LOWER_SCALING_W
        return DEFAULT_SCALING_W

    def embed(
        self,
        image_source: Union[str, Image.Image],
        message: str,
        mask_mode: str,
        mask_params: Optional[Dict] = None,
        margin_percent: float = DEFAULT_MARGIN_PERCENT,
        scaling_w: Optional[float] = None,
        bracket_method: str = DEFAULT_BRACKET_METHOD,
    ) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
        img_pt, cv_img = self._preprocess_image(image_source)
        h, w = img_pt.shape[2:]

        x, y, width, height = self._get_viewframe_region(
            w, h, mask_mode, mask_params, margin_percent
        )
        viewframe_size = min(width, height)

        # Warn if viewframe is smaller than minimum
        if viewframe_size < MIN_VIEWFRAME_SIZE:
            warnings.warn(
                f"Viewframe size ({viewframe_size}px) is smaller than recommended minimum "
                f"({MIN_VIEWFRAME_SIZE}px). Watermark quality may be reduced.",
                UserWarning,
            )

        # Determine scaling_w
        if scaling_w is None:
            scaling_w = self._recommend_scaling_w(viewframe_size)

        # Inform user about scaling_w choice
        if viewframe_size > WAM_INPUT_SIZE:
            print(
                f"Info: Using scaling_w={scaling_w} for large viewframe ({viewframe_size}px > {WAM_INPUT_SIZE}px)"
            )
        elif viewframe_size < MIN_VIEWFRAME_SIZE:
            print(
                f"Info: Using scaling_w={scaling_w} for small viewframe ({viewframe_size}px)"
            )

        coords = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "x_percent": x / w,
            "y_percent": y / h,
            "width_percent": width / w,
            "height_percent": height / h,
            "viewframe_size": int(viewframe_size),
            "scaling_w": float(scaling_w),
        }

        # Calculate dynamic padding based on image size
        dynamic_padding = calculate_viewframe_padding(min(h, w))

        # 1. CROP to region (with padding to exclude bracket arms)
        x_padded = x + dynamic_padding
        y_padded = y + dynamic_padding
        width_padded = width - 2 * dynamic_padding
        height_padded = height - 2 * dynamic_padding
        cropped = img_pt[
            :,
            :,
            y_padded : y_padded + height_padded,
            x_padded : x_padded + width_padded,
        ]

        # 2. RESIZE to WAM input size
        cropped_256 = torch.nn.functional.interpolate(
            cropped,
            size=(WAM_INPUT_SIZE, WAM_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        # 3. EMBED watermark
        wm_msg_tensor = roco_encode_to_binary_tensor(message)
        wm_msg = wm_msg_tensor.unsqueeze(0).to(self.device)

        original_scaling_w = self.wam.scaling_w
        self.wam.scaling_w = scaling_w

        try:
            outputs = self.wam.embed(cropped_256, wm_msg)
        finally:
            self.wam.scaling_w = original_scaling_w

        # 4. RESIZE back to padded viewframe size
        watermarked_crop = torch.nn.functional.interpolate(
            outputs["imgs_w"],
            size=(height_padded, width_padded),
            mode="bilinear",
            align_corners=False,
        )

        # 5. PLACE back into original image (with padding)
        img_w = img_pt.clone()
        img_w[
            :,
            :,
            y_padded : y_padded + height_padded,
            x_padded : x_padded + width_padded,
        ] = watermarked_crop

        # 6. Draw corner brackets
        img_np = (
            unnormalize_img(img_w).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        )
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        corner_length = int(min(width, height) * CORNER_LENGTH_RATIO)
        # Line thickness based on image size (after crop_to_square)
        line_thickness = calculate_line_thickness(min(h, w))

        draw_corner_brackets(
            img_bgr,
            x,
            y,
            width,
            height,
            corner_length,
            line_thickness,
            method=bracket_method,
            alpha=0.7,
        )
        result_img = img_bgr

        # Convert back to RGB tensor
        img_rgb_result = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        overlay_pil = cv2_to_pil(img_rgb_result)
        img_w = default_transform(overlay_pil).unsqueeze(0).to(self.device)

        binary_message_str = "".join(map(str, wm_msg_tensor.int().tolist()))

        return img_w, binary_message_str, coords

    def verify(
        self,
        image_source: Union[str, Image.Image],
        original_message: Optional[str] = None,
        viewframe_coords: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        img_pt, cv_img = self._preprocess_image(image_source)
        h, w = img_pt.shape[2:]

        detected = self._detect_viewframe(cv_img)
        if detected:
            x, y, width, height = detected
        else:
            x, y, width, height = self._get_fallback_viewframe(h, w)

        viewframe_size = min(width, height)

        # Warn if viewframe is smaller than minimum
        if viewframe_size < MIN_VIEWFRAME_SIZE:
            warnings.warn(
                f"Viewframe size ({viewframe_size}px) is smaller than recommended minimum "
                f"({MIN_VIEWFRAME_SIZE}px). Detection quality may be reduced.",
                UserWarning,
            )

        # Calculate dynamic padding based on image size
        dynamic_padding = calculate_viewframe_padding(min(h, w))

        # Crop and resize for WAM (with padding to exclude bracket arms)
        x_padded = x + dynamic_padding
        y_padded = y + dynamic_padding
        width_padded = width - 2 * dynamic_padding
        height_padded = height - 2 * dynamic_padding
        cropped = img_pt[
            :,
            :,
            y_padded : y_padded + height_padded,
            x_padded : x_padded + width_padded,
        ]
        cropped_256 = torch.nn.functional.interpolate(
            cropped,
            size=(WAM_INPUT_SIZE, WAM_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        preds = self.wam.detect(cropped_256)["preds"]

        mask_preds = torch.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        pred_message_tensor = msg_predict_inference(bit_preds, mask_preds).cpu().float()

        readable_message, is_valid, bitflips = roco_decode_from_binary_tensor(
            pred_message_tensor[0]
        )

        total_bits = 32
        bit_error_rate = (bitflips / total_bits) * 100 if bitflips >= 0 else -1

        binary_str = msg2str(pred_message_tensor[0].numpy())

        results = {
            "binary_message": binary_str,
            "readable_message": readable_message,
            "bit_error_rate_percent": bit_error_rate,
            "corrected_bitflips": bitflips,
            "ecc_valid": is_valid,
            "bit_accuracy": None,
            "viewframe": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "x_percent": float(x / w),
                "y_percent": float(y / h),
                "width_percent": float(width / w),
                "height_percent": float(height / h),
                "ratio": float((width * height) / (w * h)),
                "size": int(viewframe_size),
            },
        }

        if original_message:
            original_binary_tensor = roco_encode_to_binary_tensor(original_message)
            results["bit_accuracy"] = (
                (pred_message_tensor[0] == original_binary_tensor).float().mean().item()
            )

        return results

    def verify_tensor(
        self, img_tensor: torch.Tensor, original_message: Optional[str] = None
    ) -> dict:
        """Verify watermark from a tensor (for optimizer use)."""
        h, w = img_tensor.shape[2:]

        detected = self._detect_viewframe(img_tensor)
        if detected:
            x, y, width, height = detected
        else:
            x, y, width, height = self._get_fallback_viewframe(h, w)

        # Calculate dynamic padding based on image size
        dynamic_padding = calculate_viewframe_padding(min(h, w))

        # Crop with padding to exclude bracket arms
        x_padded = x + dynamic_padding
        y_padded = y + dynamic_padding
        width_padded = width - 2 * dynamic_padding
        height_padded = height - 2 * dynamic_padding
        cropped = img_tensor[
            :,
            :,
            y_padded : y_padded + height_padded,
            x_padded : x_padded + width_padded,
        ]
        cropped_256 = torch.nn.functional.interpolate(
            cropped,
            size=(WAM_INPUT_SIZE, WAM_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        preds = self.wam.detect(cropped_256)["preds"]

        mask_preds = torch.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        pred_message_tensor = msg_predict_inference(bit_preds, mask_preds).cpu().float()

        readable_message, is_valid, bitflips = roco_decode_from_binary_tensor(
            pred_message_tensor[0]
        )

        total_bits = 32
        correct_bits = total_bits - bitflips if bitflips >= 0 else 0

        results = {
            "readable_message": readable_message,
            "ecc_valid": is_valid,
            "correct_bits": correct_bits,
            "bitflips": bitflips,
        }

        if original_message:
            original_binary_tensor = roco_encode_to_binary_tensor(original_message)
            results["bit_accuracy"] = (
                (pred_message_tensor[0] == original_binary_tensor).float().mean().item()
            )
            results["correct_bits"] = int(
                (pred_message_tensor[0] == original_binary_tensor).sum().item()
            )

        return results
