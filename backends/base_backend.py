"""
Shared Base Class for Watermark Backends.

Provides common functionality for:
- Viewframe coordinate calculation
- Padding-aware region extraction
- Bracket drawing
- Image format conversion
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, Dict, Any, Tuple, Union
import torchvision.transforms as T

# Shared constants
VIEWFRAME_PADDING = 4  # Pixels to pad from viewframe edge to exclude bracket arms
CORNER_LENGTH_RATIO = 0.15  # Corner bracket length as fraction of region
LINE_THICKNESS_BASE = 3  # Base line thickness for corner brackets


class BaseWatermarkBackend:
    """Base class providing shared watermarking procedures.

    Shared procedures:
    1. Crop image to centered square
    2. Calculate viewframe coordinates from margin
    3. Apply padding to exclude bracket arms
    4. Crop/extract embed region
    5. Draw corner brackets
    6. Format conversion (PIL ↔ numpy ↔ tensor)

    Backend-specific procedures (to be implemented by subclasses):
    - embed_watermark(): Actual watermark embedding algorithm
    - detect_watermark(): Actual watermark detection algorithm
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._transform = T.ToTensor()

    # =========================================================================
    # Shared Utility Methods
    # =========================================================================

    def _crop_to_centered_square(self, image: np.ndarray) -> np.ndarray:
        """Crop numpy array to centered square with ODD side length.

        Ensures the output has an ODD side length for proper center pixel alignment.
        If the min dimension is even, crops to (min_dim - 1).

        Center preservation: The original center point is always contained within
        the cropped region. Extra pixel (if any) is cropped from bottom/right.
        """
        h, w = image.shape[:2]
        min_dim = min(h, w)

        # Ensure odd side length for proper center pixel alignment
        if min_dim % 2 == 0:
            min_dim -= 1

        # Calculate offsets to crop from each side
        # Extra pixel (if any) goes to bottom/right to preserve center
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return image[top : top + min_dim, left : left + min_dim]

    def _get_viewframe_coords(
        self, img_shape: Tuple[int, int], margin_pct: float = 0.10
    ) -> Dict[str, Any]:
        """Calculate viewframe coordinates from margin percentage.

        For ODD-sized images, ensures the viewframe is PERFECTLY CENTERED
        by using an EVEN viewframe width/height. This allows all 4 corners
        to lie exactly on the true diagonals:
        - 45° diagonal: y = x
        - 135° diagonal: y = (size-1) - x

        Args:
            img_shape: (height, width) of the image
            margin_pct: Margin as fraction of min dimension (0.10 = 10%)

        Returns:
            Dict with x, y, width, height and percentage versions
        """
        h, w = img_shape[:2]
        min_dim = min(h, w)

        # Calculate margin (round to nearest integer)
        margin = int(min_dim * margin_pct)

        # Calculate viewframe size: ensure ODD size for symmetric alignment
        # with odd-sized images. This ensures all corners are on true diagonals.
        raw_size = min_dim - 2 * margin
        if raw_size % 2 == 0:
            # If raw size is even, add 1 to make it odd
            size = raw_size + 1
        else:
            size = raw_size

        x = y = margin
        width = height = size

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
        }

    def _get_embed_region_coords(
        self, viewframe_coords: Dict[str, Any], padding: int = VIEWFRAME_PADDING
    ) -> Tuple[int, int, int, int]:
        """Calculate embed region coordinates (viewframe minus padding).

        Args:
            viewframe_coords: Viewframe coordinates dict
            padding: Pixels to exclude from each edge (default: VIEWFRAME_PADDING)

        Returns:
            (x, y, width, height) of the embed region
        """
        x = viewframe_coords["x"] + padding
        y = viewframe_coords["y"] + padding
        width = viewframe_coords["width"] - 2 * padding
        height = viewframe_coords["height"] - 2 * padding
        return x, y, max(0, width), max(0, height)

    def _crop_to_embed_region_numpy(
        self, img: np.ndarray, x: int, y: int, width: int, height: int
    ) -> np.ndarray:
        """Crop numpy array to embed region."""
        return img[y : y + height, x : x + width]

    def _crop_to_embed_region_tensor(
        self, img: torch.Tensor, x: int, y: int, width: int, height: int
    ) -> torch.Tensor:
        """Crop tensor to embed region.

        Args:
            img: Tensor of shape (B, C, H, W)

        Returns:
            Cropped tensor
        """
        return img[:, :, y : y + height, x : x + width]

    def _place_back_numpy(
        self, img: np.ndarray, crop: np.ndarray, x: int, y: int
    ) -> np.ndarray:
        """Place cropped region back into image (numpy)."""
        h, w = crop.shape[:2]
        result = img.copy()
        result[y : y + h, x : x + w] = crop
        return result

    def _place_back_tensor(
        self, img: torch.Tensor, crop: torch.Tensor, x: int, y: int
    ) -> torch.Tensor:
        """Place cropped region back into image (tensor)."""
        h, w = crop.shape[2:]
        result = img.clone()
        result[:, :, y : y + h, x : x + w] = crop
        return result

    # =========================================================================
    # Format Conversion Methods
    # =========================================================================

    def _pil_to_numpy(self, img: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array (RGB)."""
        return np.array(img.convert("RGB"))

    def _numpy_to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert numpy array (RGB) to PIL Image."""
        return Image.fromarray(arr)

    def _pil_to_tensor(self, img: Union[str, Image.Image]) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        if isinstance(img, str):
            img = Image.open(img)
        tensor = self._transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.cpu()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).byte()
        return Image.fromarray(tensor.numpy())

    # =========================================================================
    # Bracket Drawing Methods
    # =========================================================================

    def _get_corner_length(self, viewframe_size: int) -> int:
        """Calculate corner bracket length."""
        return int(viewframe_size * CORNER_LENGTH_RATIO)

    def _get_line_thickness(self, viewframe_size: int) -> int:
        """Calculate line thickness for corner brackets."""
        return max(2, int(viewframe_size * 0.012))

    # =========================================================================
    # Backend-Specific Methods (to be implemented by subclasses)
    # =========================================================================

    def embed_watermark(self, img: Any, message: str) -> Tuple[Any, str]:
        """Embed watermark into image.

        Must be implemented by subclass.

        Args:
            img: Image (format depends on backend)
            message: Message to embed

        Returns:
            (watermarked_image, binary_message)
        """
        raise NotImplementedError("Subclass must implement embed_watermark()")

    def detect_watermark(self, img: Any) -> Dict[str, Any]:
        """Detect watermark from image.

        Must be implemented by subclass.

        Args:
            img: Image (format depends on backend)

        Returns:
            Detection result dict
        """
        raise NotImplementedError("Subclass must implement detect_watermark()")
