import numpy as np
import cv2


# Special pixel values for brackets - unlikely in natural images
BRACKET_BRIGHT = 254
BRACKET_DARK = 1

# Supported bracket overlay methods
BRACKET_METHOD_DISTINCTIVE = "distinctive"  # Uses pixel values 254/1
BRACKET_METHOD_ALPHA = "alpha"  # Uses alpha blending

SUPPORTED_BRACKET_METHODS = [BRACKET_METHOD_DISTINCTIVE, BRACKET_METHOD_ALPHA]
DEFAULT_BRACKET_METHOD = BRACKET_METHOD_DISTINCTIVE


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
