"""
Viewframe Detector Module
=========================
Handles viewframe detection logic independently from watermarking.
The viewframe has 4 corner brackets (L-shaped lines) marking a region.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


class ViewframeDetector:
    """Detects viewframe corners from images with L-shaped corner brackets."""

    def __init__(
        self,
        line_thickness: int = 3,
        brightness_threshold: int = 200,
        pixel_tolerance: int = 2,
        min_region_area: int = 100,
        debug: bool = False
    ):
        """
        Initialize viewframe detector.

        Args:
            line_thickness: Thickness of corner bracket lines (for offset calculation)
            brightness_threshold: Minimum pixel value to consider as bracket (default 200 for blended brackets)
            pixel_tolerance: Tolerance for pixel matching (helps with anti-aliasing)
            min_region_area: Minimum region area to consider valid
            debug: Enable debug output
        """
        self.line_thickness = line_thickness
        self.brightness_threshold = brightness_threshold
        self.pixel_tolerance = pixel_tolerance
        self.min_region_area = min_region_area
        self.debug = debug

    def detect(
        self,
        image: np.ndarray,
        method: str = 'direct'
    ) -> Optional[Dict[str, Any]]:
        """
        Detect viewframe from image.

        Args:
            image: Input image (BGR OpenCV format or grayscale)
            method: Detection method ('direct', 'adaptive', 'multi')

        Returns:
            Dict with keys: x, y, width, height, confidence, method
            Returns None if detection fails
        """
        h, w = image.shape[:2]

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if method == 'direct':
            result = self._detect_direct(gray, h, w)
        elif method == 'adaptive':
            result = self._detect_adaptive(gray, h, w)
        elif method == 'multi':
            # Try direct first, fall back to adaptive
            result = self._detect_direct(gray, h, w)
            if result is None:
                result = self._detect_adaptive(gray, h, w)

        if result and self._validate_result(result, h, w):
            return result

        return None

    def _detect_direct(
        self,
        gray: np.ndarray,
        h: int,
        w: int
    ) -> Optional[Dict[str, Any]]:
        """
        Direct detection: Find bright corner brackets.
        Searches for bright pixels near expected corner positions.
        """
        # Find bright pixels
        bright_mask = (gray >= self.brightness_threshold).astype(np.uint8) * 255
        rows, cols = np.where(bright_mask > 0)

        if len(rows) == 0:
            return None

        # Search radius for finding corners (larger to support varying margins)
        search_radius = min(w, h) * 0.3

        def find_corner_in_quadrant(quadrant):
            """Find the innermost bright pixel in the given quadrant."""
            # Define quadrant boundaries
            if quadrant == 'tl':  # Top-left
                mask = (rows < h//2) & (cols < w//2)
            elif quadrant == 'tr':  # Top-right
                mask = (rows < h//2) & (cols > w//2)
            elif quadrant == 'bl':  # Bottom-left
                mask = (rows > h//2) & (cols < w//2)
            else:  # Bottom-right
                mask = (rows > h//2) & (cols > w//2)

            if mask.sum() == 0:
                return None

            quadrant_rows = rows[mask]
            quadrant_cols = cols[mask]

            # Find corner based on quadrant
            if quadrant == 'tl':
                # Top-left: minimum row and minimum col
                min_row_idx = np.argmin(quadrant_rows)
                min_col_idx = np.argmin(quadrant_cols)
                return quadrant_cols[min_col_idx], quadrant_rows[min_row_idx]
            elif quadrant == 'tr':
                # Top-right: minimum row and maximum col
                min_row_idx = np.argmin(quadrant_rows)
                max_col_idx = np.argmax(quadrant_cols)
                return quadrant_cols[max_col_idx], quadrant_rows[min_row_idx]
            elif quadrant == 'bl':
                # Bottom-left: maximum row and minimum col
                max_row_idx = np.argmax(quadrant_rows)
                min_col_idx = np.argmin(quadrant_cols)
                return quadrant_cols[min_col_idx], quadrant_rows[max_row_idx]
            else:  # Bottom-right
                # Bottom-right: maximum row and maximum col
                max_row_idx = np.argmax(quadrant_rows)
                max_col_idx = np.argmax(quadrant_cols)
                return quadrant_cols[max_col_idx], quadrant_rows[max_row_idx]

        # Find all 4 corners
        tl_col, tl_row = find_corner_in_quadrant('tl')
        tr_col, tr_row = find_corner_in_quadrant('tr')
        bl_col, bl_row = find_corner_in_quadrant('bl')
        br_col, br_row = find_corner_in_quadrant('br')

        if not all([tl_col, tl_row, tr_col, tr_row, bl_col, bl_row, br_col, br_row]):
            return None

        # Calculate viewframe bounds
        offset = self.line_thickness
        x = tl_col + offset
        y = tl_row + offset
        width = tr_col - tl_col - 2 * offset
        height = br_row - tl_row - 2 * offset

        result = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'confidence': 1.0,
            'method': 'direct',
        }

        if self._validate_result(result, h, w):
            return result
        return None

    def _detect_adaptive(
        self,
        gray: np.ndarray,
        h: int,
        w: int
    ) -> Optional[Dict[str, Any]]:
        """
        Adaptive detection: Uses contour detection for non-pure-white brackets.
        More robust for blended/transparent brackets or images with compression artifacts.
        """
        # Apply adaptive threshold to find bright regions
        adapt_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to connect bracket arms
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        morphed = cv2.morphologyEx(adapt_thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        # Collect corner candidates from all contours
        corner_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_region_area:
                continue

            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)

            if w_rect > 5 and h_rect > 5:
                corner_candidates.append({
                    'tl': (x_rect, y_rect),
                    'tr': (x_rect + w_rect, y_rect),
                    'bl': (x_rect, y_rect + h_rect),
                    'br': (x_rect + w_rect, y_rect + h_rect),
                })

        if len(corner_candidates) < 2:
            return None

        # Find outermost corners in each direction
        all_tl = [(c['tl'][0], c['tl'][1]) for c in corner_candidates]
        all_tr = [(c['tr'][0], c['tr'][1]) for c in corner_candidates]
        all_bl = [(c['bl'][0], c['bl'][1]) for c in corner_candidates]
        all_br = [(c['br'][0], c['br'][1]) for c in corner_candidates]

        # Find the bounding box of all corner candidates
        min_x = min(c[0] for c in all_tl)
        min_y = min(c[1] for c in all_tl)
        max_x = max(c[0] for c in all_tr)
        max_y = max(c[1] for c in all_br)

        x = min_x
        y = min_y
        width = max_x - min_x
        height = max_y - min_y

        # Lower confidence for adaptive method
        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'confidence': 0.7,
            'method': 'adaptive',
        }

    def _validate_result(
        self,
        result: Dict[str, Any],
        h: int,
        w: int
    ) -> bool:
        """Validate detection result."""
        x, y = result['x'], result['y']
        width, height = result['width'], result['height']

        # Basic bounds check
        if width <= 0 or height <= 0:
            return False
        if x < 0 or y < 0:
            return False
        if x + width > w or y + height > h:
            return False

        # Reasonable size check (not too small, not too large)
        if width < 10 or height < 10:
            return False
        if width > w * 0.95 or height > h * 0.95:
            return False

        return True

    def detect_multiple(
        self,
        image: np.ndarray,
        methods: list = None
    ) -> Dict[str, Any]:
        """
        Try multiple detection methods and return best result.

        Args:
            image: Input image
            methods: List of methods to try (default: ['direct', 'adaptive'])

        Returns:
            Dict with 'best_result' and 'all_results' keys
        """
        if methods is None:
            methods = ['direct', 'adaptive']

        all_results = {}
        best_result = None
        best_confidence = 0

        for method in methods:
            result = self.detect(image, method=method)
            if result:
                all_results[method] = result
                if result['confidence'] > best_confidence:
                    best_confidence = result['confidence']
                    best_result = result

        return {
            'best_result': best_result,
            'all_results': all_results,
        }

    def get_region_coordinates(
        self,
        result: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Extract (x, y, width, height) tuple from result dict."""
        return (
            result['x'],
            result['y'],
            result['width'],
            result['height']
        )

    def get_normalized_coordinates(
        self,
        result: Dict[str, Any],
        img_h: int,
        img_w: int
    ) -> Dict[str, float]:
        """Get normalized coordinates (0-1 range) for result."""
        x = result['x'] / img_w
        y = result['y'] / img_h
        width = result['width'] / img_w
        height = result['height'] / img_h
        ratio = (result['width'] * result['height']) / (img_w * img_h)

        return {
            'x_percent': x,
            'y_percent': y,
            'width_percent': width,
            'height_percent': height,
            'ratio': ratio,
        }


# Simple detection function for quick use
def detect_viewframe(
    image: np.ndarray,
    method: str = 'direct'
) -> Optional[Dict[str, Any]]:
    """
    Quick viewframe detection function.

    Args:
        image: Input image (BGR or grayscale)
        method: Detection method

    Returns:
        Dict with viewframe coordinates or None
    """
    detector = ViewframeDetector()
    return detector.detect(image, method=method)
