"""Watermark backends for watermark-freedom.

Provides pluggable watermarking implementations:
- WAM: Watermark Anything with localized messages (default)
- VideoSeal: Facebook's VideoSeal watermarking model
"""

from .videoseal_backend import VideoSealBackend

__all__ = ["VideoSealBackend"]
