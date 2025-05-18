import numpy as np
import cv2

def get_inner_square_region(image):
    """
    Takes a square image and returns the coordinates for the inner square region.
    Returns (top, left, bottom, right) coordinates.
    """
    assert image.shape[0] == image.shape[1], "Input image must be square. Use crop_to_centered_square first."
    min_dim = image.shape[0]
    center = (min_dim // 2, min_dim // 2)
    radius = min_dim // 2
    inner_side = int(radius * np.sqrt(2))
    half_side = inner_side // 2
    
    top = center[1] - half_side
    left = center[0] - half_side
    bottom = top + inner_side
    right = left + inner_side
    
    return (top, left, bottom, right)

def get_corner_color(image, pt, length):
    """Get the appropriate color for a corner based on the underlying image content"""
    x0 = max(pt[0], 0)
    y0 = max(pt[1], 0)
    x1 = min(pt[0]+length, image.shape[1])
    y1 = min(pt[1]+length, image.shape[0])
    region = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return (0,0,0) if mean > 128 else (255,255,255)

def draw_alpha_line(img, pt1, pt2, color, alpha, thickness):
    """Draw a line with alpha transparency"""
    line_img = np.zeros_like(img)
    cv2.line(line_img, pt1, pt2, color + (int(alpha * 255),), thickness)
    mask = line_img[..., 3] > 0
    img[mask] = line_img[mask]

def draw_viewframe_overlay(image, corner_length_ratio=0.1, line_thickness=10, transparency=0.5):
    """
    Takes a square image and draws the transparent viewframe overlay.
    Returns the overlayed image.
    """
    assert image.shape[0] == image.shape[1], "Input image must be square. Use crop_to_centered_square first."
    square_img = image.copy()
    min_dim = square_img.shape[0]
    center = (min_dim // 2, min_dim // 2)
    radius = min_dim // 2
    inner_side = int(radius * np.sqrt(2))
    half_side = inner_side // 2
    corner_length = int(inner_side * corner_length_ratio)
    
    # Coordinates of the inner square's corners
    tl = (center[0] - half_side, center[1] - half_side)
    tr = (center[0] + half_side, center[1] - half_side)
    bl = (center[0] - half_side, center[1] + half_side)
    br = (center[0] + half_side, center[1] + half_side)
    
    # Create overlay with alpha channel
    overlay_rgba = np.zeros((min_dim, min_dim, 4), dtype=np.uint8)
    
    # Get colors for each corner
    color_tl = get_corner_color(square_img, tl, corner_length)
    color_tr = get_corner_color(square_img, (tr[0] - corner_length, tr[1]), corner_length)
    color_bl = get_corner_color(square_img, (bl[0], bl[1] - corner_length), corner_length)
    color_br = get_corner_color(square_img, (br[0] - corner_length, br[1] - corner_length), corner_length)
    
    # Draw the corner lines
    alpha = transparency
    draw_alpha_line(overlay_rgba, tl, (tl[0] + corner_length, tl[1]), color_tl, alpha, line_thickness)
    draw_alpha_line(overlay_rgba, tl, (tl[0], tl[1] + corner_length), color_tl, alpha, line_thickness)
    draw_alpha_line(overlay_rgba, tr, (tr[0] - corner_length, tr[1]), color_tr, alpha, line_thickness)
    draw_alpha_line(overlay_rgba, tr, (tr[0], tr[1] + corner_length), color_tr, alpha, line_thickness)
    draw_alpha_line(overlay_rgba, bl, (bl[0] + corner_length, bl[1]), color_bl, alpha, line_thickness)
    draw_alpha_line(overlay_rgba, bl, (bl[0], bl[1] - corner_length), color_bl, alpha, line_thickness)
    draw_alpha_line(overlay_rgba, br, (br[0] - corner_length, br[1]), color_br, alpha, line_thickness)
    draw_alpha_line(overlay_rgba, br, (br[0], br[1] - corner_length), color_br, alpha, line_thickness)
    
    # Blend the overlay with the original image
    square_img_bgra = cv2.cvtColor(square_img, cv2.COLOR_BGR2BGRA)
    overlay_alpha = overlay_rgba[..., 3:4] / 255.0
    blended_bgra = (overlay_rgba[..., :3] * overlay_alpha +
                    square_img_bgra[..., :3] * (1 - overlay_alpha)).astype(np.uint8)
    overlayed_img = cv2.cvtColor(blended_bgra, cv2.COLOR_BGRA2BGR)
    
    return overlayed_img 