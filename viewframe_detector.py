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
BRACKET_BRIGHT = (255, 128, 255)  # Magenta
BRACKET_DARK = (0, 255, 0)  # Pure green

# Size threshold for line thickness calculation (must match viewframe.py)
SIZE_THRESHOLD_150 = 150


def _calculate_line_thickness(min_dim: int) -> int:
    """Calculate bracket line thickness based on image size.

    cv2.line draws thicker lines due to antialiasing:
    - thickness=1 → actual 1px wide
    - thickness=2 → actual 3px wide

    Args:
        min_dim: Minimum dimension of the image (width or height)

    Returns:
        Line thickness parameter for cv2.line (1 for small images, 2 for larger)
    """
    return 1 if min_dim <= SIZE_THRESHOLD_150 else 2


def _detect_actual_line_thickness(
    bracket_mask: np.ndarray, corner_row: int, corner_col: int, h: int, w: int
) -> int:
    """Detect actual line thickness from the bracket pixels.

    cv2.line draws centered lines. The thickness is the PIXEL SPAN
    of a bracket arm, measured PERPENDICULAR to the arm direction.

    For a TL bracket at (corner_row, corner_col):
    - Horizontal arm extends RIGHT along row corner_row
    - Vertical arm extends DOWN along col corner_col
    - The thickness is how many rows/cols the arm SPANS

    We measure thickness at a position AWAY from the corner intersection.

    Args:
        bracket_mask: Boolean mask of bracket pixels
        corner_row, corner_col: Position of the bracket corner (outermost pixel)
        h, w: Image dimensions

    Returns:
        Detected line_thickness parameter for cv2.line (1, 2, or 3)
    """
    # The corner is at the bracket intersection. To measure line thickness,
    # we need to look at a position along the arm but away from the corner.

    # For the horizontal arm: measure rows that have bracket pixels at a col away from corner
    # For the vertical arm: measure cols that have bracket pixels at a row away from corner

    # Measure horizontal arm thickness (at row corner_row, col = corner_col + 5)
    test_col = min(corner_col + 5, w - 1)
    rows_thick = 0
    for r in range(max(0, corner_row - 3), min(corner_row + 8, h)):
        if bracket_mask[r, test_col]:
            rows_thick += 1

    # Measure vertical arm thickness (at col corner_col, row = corner_row + 5)
    test_row = min(corner_row + 5, h - 1)
    cols_thick = 0
    for c in range(max(0, corner_col - 3), min(corner_col + 8, w)):
        if bracket_mask[test_row, c]:
            cols_thick += 1

    # cv2.line actual pixel spans:
    # - thickness=1 → 1 pixel
    # - thickness=2 → 3 pixels (centered line, antialiasing)
    # - thickness=3 → 5 pixels

    # Use the measurement that's more reliable (away from corner)
    pixel_span = max(rows_thick, cols_thick)

    # Map pixel span to line_thickness parameter
    if pixel_span <= 1:
        return 1  # Actual 1px
    elif pixel_span <= 3:
        return 2  # Actual 3px
    else:
        return 3  # Actual 5px+


class ViewframeDetector:
    """Detects viewframe corners from images with L-shaped corner brackets.

    Bracket configuration (AT viewframe corners, arms extending INWARD toward center):
    - Top-Left: corner at (x,y), arms extend RIGHT and DOWN (TOWARD center)
    - Top-Right: corner at (x+width,y), arms extend LEFT and DOWN (TOWARD center)
    - Bottom-Left: corner at (x,y+height), arms extend RIGHT and UP (TOWARD center)
    - Bottom-Right: corner at (x+width,y+height), arms extend LEFT and UP (TOWARD center)

    Detection strategy:
    1. Find all bracket pixels (value 1 or 254)
    2. In each quadrant, find the bracket CORNER (vertex where arms meet):
       - TL: minimum y, then minimum x (the corner furthest from center in TL direction)
       - TR: minimum y, then maximum x
       - BL: maximum y, then minimum x
       - BR: maximum y, then maximum x
    3. The bracket corners ARE the viewframe corners
    4. Enforce symmetric square: all corners equidistant from image center

    Note: The viewframe is extracted from the bracket corners (vertex positions),
    and the bracket arms extend INTO the viewframe region (inward-pointing).
    """

    def __init__(
        self,
        line_thickness: int = 1,  # Overridden dynamically in detect()
        brightness_threshold: int = 200,
        darkness_threshold: int = 55,
        min_region_area: int = 100,
        start_margin: float = 0.30,  # Start scanning from 30% margin
        debug: bool = False,
    ):
        self.line_thickness = line_thickness
        self.brightness_threshold = brightness_threshold
        self.darkness_threshold = darkness_threshold
        self.min_region_area = min_region_area
        self.start_margin = start_margin
        self.debug = debug

    def detect(
        self, image: np.ndarray, method: str = "diagonal"
    ) -> Optional[Dict[str, Any]]:
        """Detect viewframe from image."""
        h, w = image.shape[:2]
        # Calculate dynamic line_thickness based on image size
        min_dim = min(h, w)
        self.line_thickness = _calculate_line_thickness(min_dim)

        if method == "diagonal":
            # For diagonal method, we need the original color image for bracket detection
            if len(image.shape) == 3:
                img_bgr = image
            else:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            result = self._detect_diagonal(img_bgr, h, w)
        elif method == "direct":
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            result = self._detect_direct(gray, h, w)
        elif method == "adaptive":
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            result = self._detect_adaptive(gray, h, w)
        else:
            return None

        if result and self._validate_result(result, h, w):
            return result

        return None

    def _detect_diagonal(
        self, img_bgr: np.ndarray, h: int, w: int
    ) -> Optional[Dict[str, Any]]:
        """Detect viewframe by finding bracket corners in each quadrant.

        For INWARD-pointing brackets (brackets drawn INSIDE viewframe edges):
        - Top-Left bracket: corner at (x,y), arms extend RIGHT and DOWN (TOWARD center)
        - Top-Right bracket: corner at (x+width,y), arms extend LEFT and DOWN (TOWARD center)
        - Bottom-Left bracket: corner at (x,y+height), arms extend RIGHT and UP (TOWARD center)
        - Bottom-Right bracket: corner at (x+width,y+height), arms extend LEFT and UP (TOWARD center)

        The bracket corner IS the viewframe corner.

        Strategy:
        1. Find all bracket pixels (magenta or green)
        2. In each quadrant, find the bracket CORNER (vertex where arms meet):
           - TL: minimum y AND minimum x (the corner furthest from center in TL direction)
           - TR: minimum y AND maximum x (the corner furthest from center in TR direction)
           - BL: maximum y AND minimum x (the corner furthest from center in BL direction)
           - BR: maximum y AND maximum x (the corner furthest from center in BR direction)
        3. The bracket corners ARE the viewframe corners
        4. Enforce symmetric square: all corners equidistant from image center
        5. FINE-TUNING: For odd-sized images, enforce diagonal alignment (corners on true diagonals)
        """
        # Find bracket pixels by checking for distinctive colors
        # BRACKET_BRIGHT = (255, 128, 255) - Magenta (RGB) = (255, 128, 255) in BGR too
        # BRACKET_DARK = (0, 255, 0) - Pure green (RGB) = (0, 255, 0) in BGR too

        # Find magenta pixels (BGR: 255, 128, 255)
        magenta_mask = (
            (img_bgr[:, :, 0] == 255)
            & (img_bgr[:, :, 1] == 128)
            & (img_bgr[:, :, 2] == 255)
        )
        # Find green pixels (BGR: 0, 255, 0)
        green_mask = (
            (img_bgr[:, :, 0] == 0)
            & (img_bgr[:, :, 1] == 255)
            & (img_bgr[:, :, 2] == 0)
        )
        bracket_mask = magenta_mask | green_mask
        bracket_pixels = np.column_stack(np.where(bracket_mask))

        if len(bracket_pixels) < 4:
            return None

        # Image center (use float for precision in calculations)
        center_y, center_x = h // 2, w // 2

        # Find bracket pixels in each quadrant
        tl_quadrant = bracket_pixels[
            (bracket_pixels[:, 0] < center_y) & (bracket_pixels[:, 1] < center_x)
        ]
        tr_quadrant = bracket_pixels[
            (bracket_pixels[:, 0] < center_y) & (bracket_pixels[:, 1] >= center_x)
        ]
        bl_quadrant = bracket_pixels[
            (bracket_pixels[:, 0] >= center_y) & (bracket_pixels[:, 1] < center_x)
        ]
        br_quadrant = bracket_pixels[
            (bracket_pixels[:, 0] >= center_y) & (bracket_pixels[:, 1] >= center_x)
        ]

        if (
            len(tl_quadrant) == 0
            or len(tr_quadrant) == 0
            or len(bl_quadrant) == 0
            or len(br_quadrant) == 0
        ):
            return None

        # For INWARD-pointing brackets, find the CORNER in each quadrant
        # TL: min x, then min y (furthest to outer edge in TL quadrant)
        tl_min_x = np.min(tl_quadrant[:, 1])
        tl_at_min_x = tl_quadrant[tl_quadrant[:, 1] == tl_min_x]
        tl_y, tl_x = np.min(tl_at_min_x[:, 0]), tl_min_x

        # TR: max x, then min y (furthest to outer edge in TR quadrant)
        tr_max_x = np.max(tr_quadrant[:, 1])
        tr_at_max_x = tr_quadrant[tr_quadrant[:, 1] == tr_max_x]
        tr_y, tr_x = np.min(tr_at_max_x[:, 0]), tr_max_x

        # BL: min x, then max y (furthest to outer edge in BL quadrant)
        bl_min_x = np.min(bl_quadrant[:, 1])
        bl_at_min_x = bl_quadrant[bl_quadrant[:, 1] == bl_min_x]
        bl_y, bl_x = np.max(bl_at_min_x[:, 0]), bl_min_x

        # BR: max x, then max y (furthest to outer edge in BR quadrant)
        br_max_x = np.max(br_quadrant[:, 1])
        br_at_max_x = br_quadrant[br_quadrant[:, 1] == br_max_x]
        br_y, br_x = np.max(br_at_max_x[:, 0]), br_max_x

        # Compute distances from image center to each corner
        dist_tl = np.sqrt(
            (tl_x - center_x) * (tl_x - center_x)
            + (tl_y - center_y) * (tl_y - center_y)
        )
        dist_tr = np.sqrt(
            (tr_x - center_x) * (tr_x - center_x)
            + (tr_y - center_y) * (tr_y - center_y)
        )
        dist_bl = np.sqrt(
            (bl_x - center_x) * (bl_x - center_x)
            + (bl_y - center_y) * (bl_y - center_y)
        )
        dist_br = np.sqrt(
            (br_x - center_x) * (br_x - center_x)
            + (br_y - center_y) * (br_y - center_y)
        )

        # Use average distance for symmetric square
        avg_dist = (dist_tl + dist_tr + dist_bl + dist_br) / 4

        # Compute symmetric corner positions (on diagonal lines from center)
        # Use the average of detected half-sizes for better precision
        detected_half_sizes = [
            (center_x - tl_x),  # TL x distance
            (tr_x - center_x),  # TR x distance
            (center_x - bl_x),  # BL x distance
            (br_x - center_x),  # BR x distance
            (center_y - tl_y),  # TL y distance
            (tr_y - center_y),  # TR y distance (negative, but we want absolute)
            (bl_y - center_y),  # BL y distance
            (br_y - center_y),  # BR y distance
        ]
        avg_half_size = np.mean([abs(d) for d in detected_half_sizes])
        half_size = int(avg_half_size)  # Use int for proper centering

        sym_tl_x = int(center_x - half_size)
        sym_tl_y = int(center_y - half_size)
        sym_tr_x = int(center_x + half_size)
        sym_tr_y = int(center_y - half_size)
        sym_bl_x = int(center_x - half_size)
        sym_bl_y = int(center_y + half_size)
        sym_br_x = int(center_x + half_size)
        sym_br_y = int(center_y + half_size)

        # FINE-TUNING: For odd-sized images, acknowledge geometric constraints.
        # With an odd-sized image and symmetric viewframe:
        # - TL and BR corners are ON the 45° diagonal (y = x)
        # - TR and BL corners are ~1px OFF the 135° diagonal (y = max_idx - x)
        # This is unavoidable for a symmetric viewframe on odd-sized images.
        # The current symmetric positioning is correct - no adjustment needed.

        # Viewframe is defined by the bracket corners (full region, same as embedding)
        # Add offset to account for line thickness (brackets extend beyond true corner)
        # DETECT actual line thickness from the bracket pixels
        min_dim = min(h, w)
        detected_thickness = _detect_actual_line_thickness(
            bracket_mask, sym_tl_y, sym_tl_x, h, w
        )
        # Use detected thickness if available, otherwise fall back to formula
        if detected_thickness > 0:
            offset = detected_thickness // 2
            self.line_thickness = detected_thickness  # Store for later use
        else:
            line_thickness = _calculate_line_thickness(min_dim)
            offset = line_thickness // 2
            self.line_thickness = line_thickness
        x = sym_tl_x + offset
        y = sym_tl_y + offset
        width = sym_tr_x - sym_tl_x - 2 * offset
        height = sym_br_y - sym_tl_y - 2 * offset

        # Enforce odd side length for symmetric alignment with odd-sized images
        if width % 2 == 0:
            width += 1
            height += 1

        # Enforce square
        min_dim = min(width, height)
        width = height = max(0, int(min_dim))

        return {
            "x": max(0, x),
            "y": max(0, y),
            "width": max(0, width),
            "height": max(0, height),
            "confidence": 1.0,
            "method": "diagonal",
        }

    def _find_nearest_bracket_outward(
        self,
        gray: np.ndarray,
        bracket_pixels: np.ndarray,
        h: int,
        w: int,
        start_pos: Tuple[int, int],
        center_y: int,
        center_x: int,
        direction: str,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest bracket pixel scanning outward from start position.

        Args:
            direction: 'tl', 'tr', 'bl', or 'br'
        """
        start_y, start_x = start_pos

        # Determine scan direction based on corner
        if direction == "tl":
            # Scan up-left: decreasing y, decreasing x
            for y in range(start_y, -1, -1):
                for x in range(start_x, -1, -1):
                    if gray[y, x] == BRACKET_BRIGHT or gray[y, x] == BRACKET_DARK:
                        return (y, x)
        elif direction == "tr":
            # Scan up-right: decreasing y, increasing x
            for y in range(start_y, -1, -1):
                for x in range(start_x, w):
                    if gray[y, x] == BRACKET_BRIGHT or gray[y, x] == BRACKET_DARK:
                        return (y, x)
        elif direction == "bl":
            # Scan down-left: increasing y, decreasing x
            for y in range(start_y, h):
                for x in range(start_x, -1, -1):
                    if gray[y, x] == BRACKET_BRIGHT or gray[y, x] == BRACKET_DARK:
                        return (y, x)
        elif direction == "br":
            # Scan down-right: increasing y, increasing x
            for y in range(start_y, h):
                for x in range(start_x, w):
                    if gray[y, x] == BRACKET_BRIGHT or gray[y, x] == BRACKET_DARK:
                        return (y, x)

        return None

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
