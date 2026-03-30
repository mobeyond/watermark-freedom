"""Viewframe Configuration

System configuration for viewframe detection. This is read at runtime
and affects how viewframes are detected from images.

Usage:
    from viewframe_config import viewframe_config

    # Get detection method
    method = viewframe_config.detection_method

    # Change method at runtime
    viewframe_config.detection_method = "adaptive"
"""

from typing import Literal

DetectionMethod = Literal["diagonal", "direct", "adaptive"]
SUPPORTED_DETECTION_METHODS = ["diagonal", "direct", "adaptive"]
DEFAULT_DETECTION_METHOD = "diagonal"


class ViewframeConfig:
    """Configuration container for viewframe detection.

    Attributes:
        detection_method: Method to use for viewframe detection.
            - "diagonal": Uses contour detection with largest contour per corner
            - "direct": Direct contour-based detection
            - "adaptive": Adaptive detection with lower confidence
        fallback_to_centered: If True, fallback to centered square when detection fails.
        debug: Enable debug mode for detection.
    """

    def __init__(
        self,
        detection_method: DetectionMethod = DEFAULT_DETECTION_METHOD,
        fallback_to_centered: bool = True,
        debug: bool = False,
    ):
        self._detection_method = detection_method
        self._fallback_to_centered = fallback_to_centered
        self._debug = debug

    @property
    def detection_method(self) -> DetectionMethod:
        """Get current detection method."""
        return self._detection_method

    @detection_method.setter
    def detection_method(self, value: DetectionMethod) -> None:
        """Set detection method with validation."""
        if value not in SUPPORTED_DETECTION_METHODS:
            raise ValueError(
                f"Unsupported detection method: {value}. "
                f"Supported: {SUPPORTED_DETECTION_METHODS}"
            )
        self._detection_method = value

    @property
    def fallback_to_centered(self) -> bool:
        """Whether to fallback to centered square when detection fails."""
        return self._fallback_to_centered

    @fallback_to_centered.setter
    def fallback_to_centered(self, value: bool) -> None:
        self._fallback_to_centered = value

    @property
    def debug(self) -> bool:
        """Enable debug mode for detection."""
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "detection_method": self._detection_method,
            "fallback_to_centered": self._fallback_to_centered,
            "debug": self._debug,
        }


viewframe_config = ViewframeConfig()
