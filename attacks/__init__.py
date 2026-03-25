"""
Attack Suite for WAM Watermark Robustness Testing

Adapted from powerplant/benchmarks/ss_improver.py
"""

from .geometric import crop, resize, rotate, flip
from .valuemetric import jpeg, noise, blur, brightness, contrast, saturation

__all__ = [
    'crop', 'resize', 'rotate', 'flip',
    'jpeg', 'noise', 'blur', 'brightness', 'contrast', 'saturation'
]
