"""
Geometric Attacks for Watermark Robustness Testing
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def crop(img: torch.Tensor, ratio: float = 0.75) -> torch.Tensor:
    """Center crop and upscale to original size."""
    B, C, H, W = img.shape
    new_h, new_w = int(H * ratio), int(W * ratio)
    top, left = (H - new_h) // 2, (W - new_w) // 2
    cropped = img[:, :, top:top+new_h, left:left+new_w]
    upscaled = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    return upscaled.clamp(0, 1)


def resize(img: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Downscale and upscale to original size."""
    B, C, H, W = img.shape
    new_h, new_w = int(H * scale), int(W * scale)
    downscaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
    upscaled = F.interpolate(downscaled, size=(H, W), mode='bilinear', align_corners=False)
    return upscaled.clamp(0, 1)


def rotate(img: torch.Tensor, angle: float = 15.0) -> torch.Tensor:
    """Rotate image by angle in degrees."""
    B, C, H, W = img.shape
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_np, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rotated = torch.from_numpy(rotated).float() / 255.0
    rotated = rotated.permute(2, 0, 1).unsqueeze(0)
    return rotated.to(img.device)


def flip(img: torch.Tensor, horizontal: bool = True) -> torch.Tensor:
    """Flip image horizontally or vertically."""
    if horizontal:
        return torch.flip(img, dims=[3])
    else:
        return torch.flip(img, dims=[2])
