import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any


# Special pixel values for brackets - unlikely in natural images
BRACKET_BRIGHT = 254
BRACKET_DARK = 1

# Supported bracket overlay methods
BRACKET_METHOD_DISTINCTIVE = "distinctive"  # Uses pixel values 254/1
BRACKET_METHOD_ALPHA = "alpha"  # Uses alpha blending

SUPPORTED_BRACKET_METHODS = [BRACKET_METHOD_DISTINCTIVE, BRACKET_METHOD_ALPHA]
DEFAULT_BRACKET_METHOD = BRACKET_METHOD_DISTINCTIVE


def crop_to_centered_square(img: np.ndarray) -> np.ndarray:
    """Crop image to centered square before viewframe processing.

    Args:
        img: BGR image (H, W, 3) or grayscale (H, W)

    Returns:
        Centered square image with min(H, W) dimensions
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    # Center crop
    if h > w:
        y_offset = (h - w) // 2
        return img[y_offset : y_offset + w, :]
    elif w > h:
        x_offset = (w - h) // 2
        return img[:, x_offset : x_offset + h]
    return img  # Already square


def detect_viewframe(
    img: np.ndarray, method: str = "diagonal"
) -> Optional[Dict[str, Any]]:
    """Detect viewframe from image with VALIDATION.

    Detection validation criteria (ALL must pass):
    1. Exactly 4 corners found (tl, tr, bl, br)
    2. Detected square is centered (geometric center aligns with image center, tolerance 10%)
    3. Margin ratio is between 5% and 25%

    If ANY validation fails, returns None (caller should use default 15% margin).

    Args:
        img: BGR image (should be pre-processed to square first)
        method: Detection algorithm ('diagonal', 'direct', 'adaptive')

    Returns:
        Dict with keys: x, y, width, height, x_percent, y_percent, width_percent, height_percent, margin_pct, confidence, method
        OR None if detection fails validation
    """
    h, w = img.shape[:2]

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if method == "diagonal":
        result = _detect_diagonal(gray, h, w)
    elif method == "direct":
        result = _detect_direct(gray, h, w)
    elif method == "adaptive":
        result = _detect_adaptive(gray, h, w)
    else:
        return None

    if result is None:
        return None

    if not _validate_detection(result, h, w):
        return None

    result["x_percent"] = result["x"] / w
    result["y_percent"] = result["y"] / h
    result["width_percent"] = result["width"] / w
    result["height_percent"] = result["height"] / h

    margin_pct = result["y"] / min(h, w)
    result["margin_pct"] = margin_pct

    return result


def _detect_diagonal(gray: np.ndarray, h: int, w: int) -> Optional[Dict[str, Any]]:
    bright_mask = gray == BRACKET_BRIGHT
    dark_mask = gray == BRACKET_DARK
    bracket_mask = bright_mask | dark_mask
    binary = (bracket_mask * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    offset = 3
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


def _detect_direct(gray: np.ndarray, h: int, w: int) -> Optional[Dict[str, Any]]:
    bright_mask = gray == BRACKET_BRIGHT
    dark_mask = gray == BRACKET_DARK
    combined_mask = bright_mask | dark_mask
    binary = (combined_mask * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    offset = 3
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


def _detect_adaptive(gray: np.ndarray, h: int, w: int) -> Optional[Dict[str, Any]]:
    bright_mask = gray == BRACKET_BRIGHT
    dark_mask = gray == BRACKET_DARK
    combined_mask = bright_mask | dark_mask
    binary = (combined_mask * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    offset = 3
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
        "confidence": 0.7,
        "method": "adaptive",
    }


def _validate_detection(result: Dict[str, Any], h: int, w: int) -> bool:
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

    image_center_x = w / 2
    image_center_y = h / 2
    detected_center_x = x + width / 2
    detected_center_y = y + height / 2

    center_tolerance = 0.10
    x_center_delta = abs(detected_center_x - image_center_x) / w
    y_center_delta = abs(detected_center_y - image_center_y) / h
    if x_center_delta > center_tolerance or y_center_delta > center_tolerance:
        return False

    min_dim = min(h, w)
    margin_pct = y / min_dim
    if not (0.05 <= margin_pct <= 0.25):
        return False

    return True


def get_corner_color(image, pt, length):
    """Get the appropriate color for a corner based on the underlying image content."""
    x0 = max(pt[0], 0)
    y0 = max(pt[1], 0)
    x1 = min(pt[0] + length, image.shape[1])
    y1 = min(pt[1] + length, image.shape[0])
    region = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return (
        (BRACKET_DARK, BRACKET_DARK, BRACKET_DARK)
        if mean > 128
        else (BRACKET_BRIGHT, BRACKET_BRIGHT, BRACKET_BRIGHT)
    )


def draw_transparent_corner_brackets(
    img, x, y, width, height, corner_length, line_thickness, alpha=0.7
):
    """Draw corner brackets with distinctive pixel values.

    Uses pixel values 1 (black) and 254 (white) which are distinctive.
    The 'alpha' parameter is kept for API compatibility but lines are solid
    for reliable detection.

    Args:
        img: BGR image (modified in place)
        x, y: Top-left corner of viewframe
        width, height: Viewframe dimensions
        corner_length: Length of each bracket arm
        line_thickness: Line thickness
        alpha: Opacity (kept for compatibility, currently draws solid)
    """
    # Top-left corner
    color = get_corner_color(img, (x, y), corner_length)
    cv2.line(img, (x, y), (x + corner_length, y), color, line_thickness)
    cv2.line(img, (x, y), (x, y + corner_length), color, line_thickness)

    # Top-right corner
    color = get_corner_color(img, (x + width - corner_length, y), corner_length)
    cv2.line(img, (x + width, y), (x + width - corner_length, y), color, line_thickness)
    cv2.line(img, (x + width, y), (x + width, y + corner_length), color, line_thickness)

    # Bottom-left corner
    color = get_corner_color(img, (x, y + height - corner_length), corner_length)
    cv2.line(
        img, (x, y + height), (x + corner_length, y + height), color, line_thickness
    )
    cv2.line(
        img, (x, y + height), (x, y + height - corner_length), color, line_thickness
    )

    # Bottom-right corner
    color = get_corner_color(
        img, (x + width - corner_length, y + height - corner_length), corner_length
    )
    cv2.line(
        img,
        (x + width, y + height),
        (x + width - corner_length, y + height),
        color,
        line_thickness,
    )
    cv2.line(
        img,
        (x + width, y + height),
        (x + width, y + height - corner_length),
        color,
        line_thickness,
    )


def draw_alpha_blend_corner_brackets(
    img, x, y, width, height, corner_length, line_thickness, alpha=0.7
):
    """Draw corner brackets with alpha blending (subtle overlay)."""
    overlay = img.copy()

    color = get_corner_color(img, (x, y), corner_length)
    cv2.line(overlay, (x, y), (x + corner_length, y), color, line_thickness)
    cv2.line(overlay, (x, y), (x, y + corner_length), color, line_thickness)

    color = get_corner_color(img, (x + width - corner_length, y), corner_length)
    cv2.line(
        overlay, (x + width, y), (x + width - corner_length, y), color, line_thickness
    )
    cv2.line(
        overlay, (x + width, y), (x + width, y + corner_length), color, line_thickness
    )

    color = get_corner_color(img, (x, y + height - corner_length), corner_length)
    cv2.line(
        overlay, (x, y + height), (x + corner_length, y + height), color, line_thickness
    )
    cv2.line(
        overlay, (x, y + height), (x, y + height - corner_length), color, line_thickness
    )

    color = get_corner_color(
        img, (x + width - corner_length, y + height - corner_length), corner_length
    )
    cv2.line(
        overlay,
        (x + width, y + height),
        (x + width - corner_length, y + height),
        color,
        line_thickness,
    )
    cv2.line(
        overlay,
        (x + width, y + height),
        (x + width, y + height - corner_length),
        color,
        line_thickness,
    )

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_corner_brackets(
    img,
    x,
    y,
    width,
    height,
    corner_length,
    line_thickness,
    method=DEFAULT_BRACKET_METHOD,
    alpha=0.7,
):
    """Draw corner brackets using the specified method.

    Args:
        img: BGR image (modified in place)
        x, y: Top-left corner of viewframe
        width, height: Viewframe dimensions
        corner_length: Length of each bracket arm
        line_thickness: Line thickness
        method: Bracket drawing method ('distinctive' or 'alpha')
        alpha: Opacity for alpha blending method (0.0-1.0)

    Raises:
        ValueError: If method is not supported
    """
    if method == BRACKET_METHOD_DISTINCTIVE:
        draw_transparent_corner_brackets(
            img, x, y, width, height, corner_length, line_thickness, alpha=1.0
        )
    elif method == BRACKET_METHOD_ALPHA:
        draw_alpha_blend_corner_brackets(
            img, x, y, width, height, corner_length, line_thickness, alpha=alpha
        )
    else:
        raise ValueError(
            f"Unsupported bracket method: {method}. "
            f"Supported: {SUPPORTED_BRACKET_METHODS}"
        )


def draw_viewframe(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    method: str = "distinctive",
    corner_length_pct: float = 0.15,
    line_thickness: int = 2,
    alpha: float = 0.7,
) -> np.ndarray:
    """Draw viewframe overlay on image (ALWAYS draws, regardless of image size).

    Args:
        img: BGR image (modified in place)
        x, y: Top-left of viewframe region
        width, height: Viewframe dimensions
        method: 'distinctive' (254/1 pixels) or 'alpha' (blended)
        corner_length_pct: Length of bracket arms as % of min_dim
        line_thickness: Line thickness in pixels
        alpha: Opacity for alpha blending method

    Returns:
        Image with viewframe drawn
    """
    min_dim = min(img.shape[0], img.shape[1])
    corner_length = int(min_dim * corner_length_pct)

    draw_corner_brackets(
        img,
        x,
        y,
        width,
        height,
        corner_length=corner_length,
        line_thickness=line_thickness,
        method=method,
        alpha=alpha,
    )
    return img


def get_default_viewframe_coords(
    img_shape: Tuple[int, int], margin_pct: float = 0.15
) -> Dict[str, Any]:
    """Get default viewframe coordinates using specified margin percentage.

    Args:
        img_shape: (height, width) of image
        margin_pct: Margin as fraction (0.15 = 15%)

    Returns:
        Dict with normalized and pixel coordinates
    """
    h, w = img_shape[:2]
    min_dim = min(h, w)
    m = int(min_dim * margin_pct)
    x = y = m
    width = height = min_dim - 2 * m

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "x_percent": x / w,
        "y_percent": y / h,
        "width_percent": width / w,
        "height_percent": height / h,
        "margin_pct": margin_pct,
        "confidence": 1.0,
        "method": "default",
    }


def process_viewframe(
    img: np.ndarray,
    detection_method: str = "diagonal",
    overlay_method: str = "distinctive",
    default_margin: float = 0.15,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Complete viewframe processing: crop -> detect -> validate -> overlay.

    Args:
        img: Input BGR image (any size)
        detection_method: Algorithm for detecting viewframe ('diagonal', 'direct', 'adaptive')
        overlay_method: Method for drawing brackets ('distinctive', 'alpha')
        default_margin: Fallback margin if detection fails validation

    Returns:
        Tuple of (output_image_with_overlay, coordinates_dict)
    """
    img_square = crop_to_centered_square(img)

    coords = detect_viewframe(img_square, method=detection_method)

    if coords is None:
        coords = get_default_viewframe_coords(img_square.shape, default_margin)

    result_img = img_square.copy()
    draw_viewframe(
        result_img,
        coords["x"],
        coords["y"],
        coords["width"],
        coords["height"],
        method=overlay_method,
    )

    return result_img, coords
