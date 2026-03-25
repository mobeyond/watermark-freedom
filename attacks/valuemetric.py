"""
Value/Metric Attacks for Watermark Robustness Testing
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import cv2


def jpeg(img: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Apply JPEG compression."""
    B, C, H, W = img.shape
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed = np.array(compressed)
    compressed = torch.from_numpy(compressed).float() / 255.0
    compressed = compressed.permute(2, 0, 1).unsqueeze(0)
    return compressed.to(img.device)


def noise(img: torch.Tensor, sigma: float = 20.0) -> torch.Tensor:
    """Add Gaussian noise."""
    sigma_norm = sigma / 255.0
    noise_tensor = torch.randn_like(img) * sigma_norm
    return (img + noise_tensor).clamp(0, 1)


def blur(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur."""
    sigma = kernel_size / 6.0
    x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel_2d, padding=padding, groups=3)
    return blurred.clamp(0, 1)


def brightness(img: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Adjust brightness."""
    return (img * factor).clamp(0, 1)


def contrast(img: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Adjust contrast."""
    mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
    adjusted = (img - mean) * factor + mean
    return adjusted.clamp(0, 1)


def saturation(img: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Adjust saturation."""
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)
    saturated = gray + (img - gray) * factor
    return saturated.clamp(0, 1)


def median_filter(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply median filter."""
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    filtered = cv2.medianBlur(img_np, kernel_size)
    filtered = torch.from_numpy(filtered).float() / 255.0
    filtered = filtered.permute(2, 0, 1).unsqueeze(0)
    return filtered.to(img.device)
