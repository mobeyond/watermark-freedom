"""
Viewframe Detector Module
=========================
Handles viewframe detection logic independently from watermarking.
The viewframe has 4 corner brackets (L-shaped lines) marking a region.

Detection Strategy: Look for distinctive pixel values (1 or 254).
These are unlikely in natural images, making detection robust.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Must match viewframe.py
BRACKET_BRIGHT = 254
BRACKET_DARK = 1


class ViewframeDetector:
    """Detects viewframe corners from images with L-shaped corner brackets."""

    def __init__(
        self,
        line_thickness: int = 3,
        brightness_threshold: int = 200,
        darkness_threshold: int = 55,
        min_region_area: int = 100,
        debug: bool = False,
    ):
        self.line_thickness = line_thickness
        self.brightness_threshold = brightness_threshold
        self.darkness_threshold = darkness_threshold
        self.min_region_area = min_region_area
        self.debug = debug

    def detect(
        self, image: np.ndarray, method: str = "diagonal"
    ) -> Optional[Dict[str, Any]]:
        """Detect viewframe from image."""
        h, w = image.shape[:2]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if method == "diagonal":
            result = self._detect_diagonal(gray, h, w)
        elif method == "direct":
            result = self._detect_direct(gray, h, w)
        elif method == "adaptive":
            result = self._detect_adaptive(gray, h, w)
        else:
            return None

        if result and self._validate_result(result, h, w):
            return result

        return None

    def _detect_diagonal(
        self, gray: np.ndarray, h: int, w: int
    ) -> Optional[Dict[str, Any]]:
        """Find corner brackets using morphology + largest contour per corner."""
        bright_mask = gray == BRACKET_BRIGHT
        dark_mask = gray == BRACKET_DARK
        bracket_mask = bright_mask | dark_mask
        binary = (bracket_mask * 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) < 4:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [c for c in contours if cv2.contourArea(c) >= 100]

        if len(contours) < 4:
            return None

        mid_h, mid_w = h // 2, w // 2

        corner_contours = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx, cy = x + cw // 2, y + ch // 2

            if cx < mid_w and cy < mid_h:
                corner_contours.append(("tl", cnt, x, y, cw, ch))
            elif cx > mid_w and cy < mid_h:
                corner_contours.append(("tr", cnt, x, y, cw, ch))
            elif cx < mid_w and cy > mid_h:
                corner_contours.append(("bl", cnt, x, y, cw, ch))
            elif cx > mid_w and cy > mid_h:
                corner_contours.append(("br", cnt, x, y, cw, ch))

        if len(corner_contours) < 4:
            return None

        # Get the best (largest) contour for each corner
        best = {}
        for name, cnt, x, y, cw, ch in corner_contours:
            if name not in best:
                best[name] = (x, y, cw, ch)

        if len(best) < 4:
            return None

        tl_x, tl_y, _, _ = best["tl"]
        tr_x, tr_y, tr_w, _ = best["tr"]
        bl_x, bl_y, _, bl_h = best["bl"]

        offset = self.line_thickness
        x = tl_x + offset
        y = tl_y + offset
        width = (tr_x + tr_w) - tl_x - 2 * offset
        height = (bl_y + bl_h) - tl_y - 2 * offset

        min_dim = min(width, height)
        width = height = min_dim

        return {
            "x": max(0, x),
            "y": max(0, y),
            "width": max(0, width),
            "height": max(0, height),
            "confidence": 1.0,
            "method": "diagonal",
        }

    def _detect_direct(
        self, gray: np.ndarray, h: int, w: int
    ) -> Optional[Dict[str, Any]]:
        """Direct detection: Find corner brackets using morphology + largest contour."""
        bright_mask = gray == BRACKET_BRIGHT
        dark_mask = gray == BRACKET_DARK
        combined_mask = bright_mask | dark_mask
        binary = (combined_mask * 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) < 4:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [c for c in contours if cv2.contourArea(c) >= 100]

        if len(contours) < 4:
            return None

        mid_h, mid_w = h // 2, w // 2

        corner_contours = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx, cy = x + cw // 2, y + ch // 2

            if cx < mid_w and cy < mid_h:
                corner_contours.append(("tl", x, y, cw, ch))
            elif cx > mid_w and cy < mid_h:
                corner_contours.append(("tr", x, y, cw, ch))
            elif cx < mid_w and cy > mid_h:
                corner_contours.append(("bl", x, y, cw, ch))
            elif cx > mid_w and cy > mid_h:
                corner_contours.append(("br", x, y, cw, ch))

        if len(corner_contours) < 4:
            return None

        best = {}
        for name, x, y, cw, ch in corner_contours:
            if name not in best:
                best[name] = (x, y, cw, ch)

        if len(best) < 4:
            return None

        tl_x, tl_y, _, _ = best["tl"]
        tr_x, tr_y, tr_w, _ = best["tr"]
        bl_x, bl_y, _, bl_h = best["bl"]

        offset = self.line_thickness
        x = tl_x + offset
        y = tl_y + offset
        width = (tr_x + tr_w) - tl_x - 2 * offset
        height = (bl_y + bl_h) - tl_y - 2 * offset

        min_dim = min(width, height)
        width = height = min_dim

        return {
            "x": max(0, x),
            "y": max(0, y),
            "width": max(0, width),
            "height": max(0, height),
            "confidence": 1.0,
            "method": "direct",
        }

    def _detect_adaptive(
        self, gray: np.ndarray, h: int, w: int
    ) -> Optional[Dict[str, Any]]:
        """Adaptive detection: Uses contour detection with morphology."""
        bright_mask = gray == BRACKET_BRIGHT
        dark_mask = gray == BRACKET_DARK
        combined_mask = bright_mask | dark_mask
        binary = (combined_mask * 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) < 4:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [c for c in contours if cv2.contourArea(c) >= 100]

        if len(contours) < 4:
            return None

        mid_h, mid_w = h // 2, w // 2

        corner_contours = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx, cy = x + cw // 2, y + ch // 2

            if cx < mid_w and cy < mid_h:
                corner_contours.append(("tl", x, y, cw, ch))
            elif cx > mid_w and cy < mid_h:
                corner_contours.append(("tr", x, y, cw, ch))
            elif cx < mid_w and cy > mid_h:
                corner_contours.append(("bl", x, y, cw, ch))
            elif cx > mid_w and cy > mid_h:
                corner_contours.append(("br", x, y, cw, ch))

        if len(corner_contours) < 4:
            return None

        best = {}
        for name, x, y, cw, ch in corner_contours:
            if name not in best:
                best[name] = (x, y, cw, ch)

        if len(best) < 4:
            return None

        tl_x, tl_y, _, _ = best["tl"]
        tr_x, tr_y, tr_w, _ = best["tr"]
        bl_x, bl_y, _, bl_h = best["bl"]

        x = tl_x + self.line_thickness
        y = tl_y + self.line_thickness
        width = (tr_x + tr_w) - tl_x - 2 * self.line_thickness
        height = (bl_y + bl_h) - tl_y - 2 * self.line_thickness

        min_dim = min(width, height)
        width = height = min_dim

        return {
            "x": max(0, x),
            "y": max(0, y),
            "width": max(0, width),
            "height": max(0, height),
            "confidence": 0.7,
            "method": "adaptive",
        }

    def _validate_result(self, result: Dict[str, Any], h: int, w: int) -> bool:
        """Validate detection result."""
        x, y = result["x"], result["y"]
        width, height = result["width"], result["height"]

        if width <= 0 or height <= 0:
            return False
        if x < 0 or y < 0:
            return False
        if x + width > w or y + height > h:
            return False
        if width < 10 or height < 10:
            return False
        if width > w * 0.95 or height > h * 0.95:
            return False

        return True

    def get_region_coordinates(
        self, result: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Extract (x, y, width, height) tuple from result dict."""
        return (result["x"], result["y"], result["width"], result["height"])

    def get_normalized_coordinates(
        self, result: Dict[str, Any], img_h: int, img_w: int
    ) -> Dict[str, float]:
        """Get normalized coordinates (0-1 range) for result."""
        return {
            "x_percent": result["x"] / img_w,
            "y_percent": result["y"] / img_h,
            "width_percent": result["width"] / img_w,
            "height_percent": result["height"] / img_h,
            "ratio": (result["width"] * result["height"]) / (img_w * img_h),
        }


def detect_viewframe(
    image: np.ndarray, method: str = "diagonal"
) -> Optional[Dict[str, Any]]:
    """Quick viewframe detection function."""
    detector = ViewframeDetector()
    return detector.detect(image, method=method)
