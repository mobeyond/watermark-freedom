import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any

# Special pixel values for brackets - unlikely in natural images
# Using distinctive RGB values that won't be affected by watermark embedding
BRACKET_BRIGHT = (255, 128, 255)  # Magenta - distinctive, unlikely in natural images
BRACKET_DARK = (0, 255, 0)  # Pure green - distinctive, unlikely to be used by watermark

# Supported bracket overlay methods
BRACKET_METHOD_DISTINCTIVE = "distinctive"  # Uses pixel values 254/1
BRACKET_METHOD_ALPHA = "alpha"  # Uses alpha blending

SUPPORTED_BRACKET_METHODS = [BRACKET_METHOD_DISTINCTIVE, BRACKET_METHOD_ALPHA]
DEFAULT_BRACKET_METHOD = BRACKET_METHOD_DISTINCTIVE

# Size threshold for line thickness calculation
SIZE_THRESHOLD_150 = 150  # Images <=150px get thinner lines


def calculate_line_thickness(min_dim: int) -> int:
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


def calculate_viewframe_padding(min_dim: int) -> int:
    """Calculate padding to exclude bracket arms from viewframe region.

    Padding should match the actual bracket arm thickness to ensure
    the extracted viewframe region matches pixel-to-pixel.

    Args:
        min_dim: Minimum dimension of the image (width or height)

    Returns:
        Padding in pixels (matches calculate_line_thickness)
    """
    return calculate_line_thickness(min_dim)


# Import ViewframeDetector for delegation
from viewframe_detector import ViewframeDetector

# Import crop_to_centered_square from utils to avoid duplication
from watermark_utils import crop_to_centered_square


def detect_viewframe(
    img: np.ndarray, method: str = "diagonal"
) -> Optional[Dict[str, Any]]:
    """Detect viewframe from image by delegating to ViewframeDetector.

    This is a thin wrapper around ViewframeDetector for convenience.
    The actual detection logic lives in viewframe_detector.py to avoid
    code duplication.

    Args:
        img: BGR image (should be pre-processed to square first)
        method: Detection algorithm ('diagonal', 'direct', 'adaptive')

    Returns:
        Dict with keys: x, y, width, height, x_percent, y_percent,
        width_percent, height_percent, margin_pct, confidence, method
        OR None if detection fails validation
    """
    h, w = img.shape[:2]

    # Create detector and detect viewframe
    detector = ViewframeDetector()
    result = detector.detect(img, method=method)

    if result is None:
        return None

    # Add additional fields that were previously computed here
    result["x_percent"] = result["x"] / w
    result["y_percent"] = result["y"] / h
    result["width_percent"] = result["width"] / w
    result["height_percent"] = result["height"] / h

    margin_pct = result["y"] / min(h, w)
    result["margin_pct"] = margin_pct

    # Include the detected line_thickness for padding calculation
    result["detected_line_thickness"] = detector.line_thickness

    return result


def get_corner_color(image, pt, length):
    """Get the appropriate color for a corner based on the underlying image content."""
    x0 = max(pt[0], 0)
    y0 = max(pt[1], 0)
    x1 = min(pt[0] + length, image.shape[1])
    y1 = min(pt[1] + length, image.shape[0])
    region = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return BRACKET_DARK if mean > 128 else BRACKET_BRIGHT


def draw_transparent_corner_brackets(
    img, x, y, width, height, corner_length, line_thickness, alpha=0.7
):
    """Draw corner brackets AT viewframe corners, arms extending INWARD.

    Uses distinctive pixel values for reliable detection.
    The 'alpha' parameter is kept for API compatibility but lines are solid
    for reliable detection.

    Bracket positions (AT viewframe corners, arms extending INWARD toward center):
    - Top-Left: corner at (x, y), arms extend to (x+corner_length, y) and (x, y+corner_length)
    - Top-Right: corner at (x+width, y), arms extend to (x+width-corner_length, y) and (x+width, y+corner_length)
    - Bottom-Left: corner at (x, y+height), arms extend to (x+corner_length, y+height) and (x, y+height-corner_length)
    - Bottom-Right: corner at (x+width, y+height), arms extend to (x+width-corner_length, y+height) and (x+width, y+height-corner_length)

    The bracket corner (vertex) IS the viewframe corner.
    The arms extend INWARD (toward the viewframe center), marking the boundary.

    Args:
        img: BGR image (modified in place)
        x, y: Top-left corner of viewframe (also bracket corner position)
        width, height: Viewframe dimensions
        corner_length: Length of each bracket arm
        line_thickness: Line thickness
        alpha: Opacity (kept for compatibility, currently draws solid)
    """
    # Top-left corner (at (x,y), arms extend RIGHT and DOWN INTO viewframe)
    color = get_corner_color(img, (x, y), corner_length)
    cv2.line(img, (x, y), (x + corner_length, y), color, line_thickness)
    cv2.line(img, (x, y), (x, y + corner_length), color, line_thickness)

    # Top-right corner (at (x+width, y), arms extend LEFT and DOWN INTO viewframe)
    color = get_corner_color(img, (x + width - corner_length, y), corner_length)
    cv2.line(img, (x + width, y), (x + width - corner_length, y), color, line_thickness)
    cv2.line(img, (x + width, y), (x + width, y + corner_length), color, line_thickness)

    # Bottom-left corner (at (x, y+height), arms extend RIGHT and UP INTO viewframe)
    color = get_corner_color(img, (x, y + height - corner_length), corner_length)
    cv2.line(
        img, (x, y + height), (x + corner_length, y + height), color, line_thickness
    )
    cv2.line(
        img, (x, y + height), (x, y + height - corner_length), color, line_thickness
    )

    # Bottom-right corner (at (x+width, y+height), arms extend LEFT and UP INTO viewframe)
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
    """Draw corner brackets AT viewframe corners with alpha blending (subtle overlay).

    Bracket positions (AT viewframe corners, arms extending INWARD toward center):
    - Top-Left: corner at (x, y), arms extend to (x+corner_length, y) and (x, y+corner_length)
    - Top-Right: corner at (x+width, y), arms extend to (x+width-corner_length, y) and (x+width, y+corner_length)
    - Bottom-Left: corner at (x, y+height), arms extend to (x+corner_length, y+height) and (x, y+height-corner_length)
    - Bottom-Right: corner at (x+width, y+height), arms extend to (x+width-corner_length, y+height) and (x+width, y+height-corner_length)
    """
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
    line_thickness: Optional[int] = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """Draw viewframe overlay on image (ALWAYS draws, regardless of image size).

    Args:
        img: BGR image (modified in place)
        x, y: Top-left of viewframe region
        width, height: Viewframe dimensions
        method: 'distinctive' (254/1 pixels) or 'alpha' (blended)
        corner_length_pct: Length of bracket arms as % of min_dim
        line_thickness: Line thickness in pixels (None = auto: 2 for <150px, 3 for >=150px)
        alpha: Opacity for alpha blending method

    Returns:
        Image with viewframe drawn
    """
    min_dim = min(img.shape[0], img.shape[1])
    # Auto-calculate line_thickness based on image size if not specified
    # cv2.line draws thicker due to antialiasing: thickness=1 → 1px, thickness=2 → 3px
    if line_thickness is None:
        line_thickness = 1 if min_dim <= 150 else 2
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

    Always applies the specified margin regardless of image size.
    Ensures odd side length for symmetric alignment with odd-sized images.
    """
    h, w = img_shape[:2]
    min_dim = min(h, w)
    m = int(min_dim * margin_pct)
    x = y = m
    width = height = min_dim - 2 * m

    # Ensure odd side length
    if width % 2 == 0:
        width += 1
        height += 1

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "x_percent": x / w if w > 0 else 0,
        "y_percent": y / h if h > 0 else 0,
        "width_percent": width / w if w > 0 else 0,
        "height_percent": height / h if h > 0 else 0,
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
