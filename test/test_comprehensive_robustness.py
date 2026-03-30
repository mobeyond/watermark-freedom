"""
Comprehensive Watermark Robustness Test Suite
==============================================

This test suite validates watermark embedding and verification across:
1. Color effects (brightness, contrast, saturation, hue, JPEG compression)
2. Image sizes (256x256 to 1024x1024+)
3. Attack scenarios (cropping, resize, noise, blur, flip)

Usage:
    pytest test/test_comprehensive_robustness.py -v
    pytest test/test_comprehensive_robustness.py::TestColorEffects -v
    pytest test/test_comprehensive_robustness.py::TestImageSizes -v
    pytest test/test_comprehensive_robustness.py::TestAttackScenarios -v
"""

import os
import sys
import random
import string
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import WatermarkManager
from watermark_utils import (
    load_image,
    roco_encode_to_binary_tensor,
    roco_decode_from_binary_tensor,
)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================


@dataclass
class TestConfig:
    """Global test configuration."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    test_images_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "assets" / "images"
    )

    # Test messages (ROCO only supports: A-Z, 4, 6, 7, 9, ., #)
    test_messages: List[str] = field(
        default_factory=lambda: [
            "ABC",
            "XYZ",
            "AAA",
            "BBB",
            "CCC",
            "WAT",
            "ERM",
            "WIN",
            "WIN",
            "WIN",
            "WWW",
            "...",
            "###",
        ]
    )

    # Default message for repeated tests
    default_message: str = "TEST"

    # Number of test iterations per configuration
    iterations: int = 3

    # Success threshold (percentage)
    success_threshold: float = 80.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Global config instance
config = TestConfig()


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def wm_manager():
    """Initialize WatermarkManager once for all tests."""
    print(f"\n{'=' * 60}")
    print(f"Initializing WatermarkManager on device: {config.device}")
    print(f"{'=' * 60}")
    manager = WatermarkManager(device=torch.device(config.device))
    return manager


@pytest.fixture(scope="module")
def test_images() -> List[Path]:
    """Get list of test images."""
    images = list(config.test_images_dir.glob("*.png")) + list(
        config.test_images_dir.glob("*.jpg")
    )
    if not images:
        pytest.skip(f"No test images found in {config.test_images_dir}")
    return images


@pytest.fixture
def sample_image() -> Image.Image:
    """Load a sample test image."""
    images = list(config.test_images_dir.glob("*.png"))
    if images:
        return load_image(images[0])
    # Fallback: create a simple test image
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    return img


@pytest.fixture
def random_message() -> str:
    """Generate a random 3-character message."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=3))


# =============================================================================
# ATTACK FUNCTIONS (for testing)
# =============================================================================


def apply_crop(img_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    """Center crop and upscale to original size.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        ratio: Crop ratio (0.0 to 1.0), 0.5 means 50% crop
    """
    B, C, H, W = img_tensor.shape
    new_h, new_w = int(H * ratio), int(W * ratio)
    top, left = (H - new_h) // 2, (W - new_w) // 2
    cropped = img_tensor[:, :, top : top + new_h, left : left + new_w]
    upscaled = torch.nn.functional.interpolate(
        cropped, size=(H, W), mode="bilinear", align_corners=False
    )
    return upscaled.clamp(0, 1)


def apply_resize(img_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Downscale and upscale to original size.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        scale: Scale factor (e.g., 0.5 means downscale to 50%)
    """
    B, C, H, W = img_tensor.shape
    new_h, new_w = int(H * scale), int(W * scale)
    downscaled = torch.nn.functional.interpolate(
        img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    upscaled = torch.nn.functional.interpolate(
        downscaled, size=(H, W), mode="bilinear", align_corners=False
    )
    return upscaled.clamp(0, 1)


def apply_brightness(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust brightness.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        factor: Brightness factor (1.0 = no change, 1.5 = 50% brighter)
    """
    return (img_tensor * factor).clamp(0, 1)


def apply_contrast(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        factor: Contrast factor (1.0 = no change, 1.5 = more contrast)
    """
    mean = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).view(1, 3, 1, 1)
    adjusted = (img_tensor - mean) * factor + mean
    return adjusted.clamp(0, 1)


def apply_saturation(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust saturation.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        factor: Saturation factor (1.0 = no change, 0.0 = grayscale)
    """
    gray = (
        0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    )
    gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)
    saturated = gray + (img_tensor - gray) * factor
    return saturated.clamp(0, 1)


def apply_gaussian_noise(img_tensor: torch.Tensor, sigma: float = 20.0) -> torch.Tensor:
    """Add Gaussian noise.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        sigma: Noise standard deviation (0-255 scale)
    """
    sigma_norm = sigma / 255.0
    noise_tensor = torch.randn_like(img_tensor) * sigma_norm
    return (img_tensor + noise_tensor).clamp(0, 1)


def apply_gaussian_blur(img_tensor: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        kernel_size: Kernel size (odd number)
    """
    sigma = kernel_size / 6.0
    x = (
        torch.arange(kernel_size, dtype=torch.float32, device=img_tensor.device)
        - kernel_size // 2
    )
    kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
    # Expand kernel for all channels
    kernel_2d = kernel_2d.repeat(img_tensor.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    blurred = torch.nn.functional.conv2d(
        img_tensor, kernel_2d, padding=padding, groups=img_tensor.shape[1]
    )
    return blurred.clamp(0, 1)


def apply_jpeg_compression(img_tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Apply JPEG compression simulation.

    Args:
        img_tensor: Image tensor [B, C, H, W]
        quality: JPEG quality (1-100, higher is better)
    """
    # Convert to numpy, apply compression, back to tensor
    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    import io

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed = np.array(compressed)

    result = torch.from_numpy(compressed).float() / 255.0
    result = result.permute(2, 0, 1).unsqueeze(0)
    return result.to(img_tensor.device)


def apply_horizontal_flip(img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply horizontal flip."""
    return torch.flip(img_tensor, dims=[3])


def apply_vertical_flip(img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply vertical flip."""
    return torch.flip(img_tensor, dims=[2])


# =============================================================================
# HELPER CLASSES
# =============================================================================


@dataclass
class TestResult:
    """Single test result."""

    name: str
    success: bool
    message_embedded: str
    message_decoded: str
    bit_accuracy: float
    bit_error_rate: float
    ecc_valid: bool
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        status = "PASS" if self.success else "FAIL"
        return f"{self.name}: {status} | Embedded: {self.message_embedded} | Decoded: {self.message_decoded} | BER: {self.bit_error_rate:.1f}%"


@dataclass
class TestSuiteResult:
    """Aggregated test suite results."""

    test_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    results: List[TestResult] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    def add_result(self, result: TestResult):
        self.results.append(result)
        self.total_tests += 1
        if result.success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        self.success_rate = (
            (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"Test Suite: {self.test_name}",
            f"{'=' * 60}",
            f"Total:  {self.total_tests:4d}",
            f"Passed: {self.passed_tests:4d}",
            f"Failed: {self.failed_tests:4d}",
            f"Rate:   {self.success_rate:5.1f}%",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)


# =============================================================================
# CORE TEST FUNCTIONS
# =============================================================================


def embed_and_verify(
    wm_manager: WatermarkManager,
    image: Image.Image,
    message: str,
    margin_percent: float = 0.15,
) -> Tuple[TestResult, Image.Image]:
    """Embed watermark and verify, returning result and watermarked image."""
    import tempfile
    import os
    from notebooks.inference_utils import unnormalize_img

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        image.save(temp_path)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            watermarked_tensor, binary_msg, coords = wm_manager.embed(
                temp_path,
                message,
                mask_mode="corners",
                mask_params=None,
                margin_percent=margin_percent,
            )

        img_np = (
            unnormalize_img(watermarked_tensor)
            .squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        img_np = (img_np * 255).astype(np.uint8)
        wm_image = Image.fromarray(img_np)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            wm_path = f.name
            wm_image.save(wm_path)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                verify_result = wm_manager.verify(wm_path, original_message=message)

            decoded_msg = verify_result.get("readable_message", "")
            bit_accuracy = verify_result.get("bit_accuracy", 0.0)
            bit_error_rate = verify_result.get("bit_error_rate_percent", 100.0)
            ecc_valid = verify_result.get("ecc_valid", False)

            success = decoded_msg == message

            result = TestResult(
                name=f"Verify '{message}'",
                success=success,
                message_embedded=message,
                message_decoded=decoded_msg,
                bit_accuracy=bit_accuracy,
                bit_error_rate=bit_error_rate,
                ecc_valid=ecc_valid,
                metadata={
                    "bitflips": verify_result.get("corrected_bitflips", -1),
                    "viewframe_size": coords.get("viewframe_size", 0),
                },
            )

            return result, wm_image
        finally:
            os.unlink(wm_path)
    finally:
        os.unlink(temp_path)


def embed_and_verify_with_attack(
    wm_manager: WatermarkManager,
    image: Image.Image,
    message: str,
    attack_func,
    attack_params: Dict,
    margin_percent: float = 0.15,
) -> TestResult:
    """Embed watermark, apply attack, verify."""
    import tempfile
    import os
    from notebooks.inference_utils import unnormalize_img, normalize_img

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        image.save(temp_path)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            watermarked_tensor, binary_msg, coords = wm_manager.embed(
                temp_path,
                message,
                mask_mode="corners",
                mask_params=None,
                margin_percent=margin_percent,
            )

        wm_unnorm = unnormalize_img(watermarked_tensor)
        attacked_unnorm = attack_func(wm_unnorm, **attack_params)
        attacked = normalize_img(torch.clamp(attacked_unnorm, 0, 1))

        img_np = (
            unnormalize_img(attacked).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        )
        img_np = (img_np * 255).astype(np.uint8)
        wm_image = Image.fromarray(img_np)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            wm_path = f.name
            wm_image.save(wm_path)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                verify_result = wm_manager.verify(wm_path, original_message=message)

            decoded_msg = verify_result.get("readable_message", "")
            bit_accuracy = verify_result.get("bit_accuracy", 0.0)
            bit_error_rate = verify_result.get("bit_error_rate_percent", 100.0)
            ecc_valid = verify_result.get("ecc_valid", False)

            success = decoded_msg == message
            attack_name = f"{attack_func.__name__}({attack_params})"

            result = TestResult(
                name=f"Attack: {attack_name} | Verify '{message}'",
                success=success,
                message_embedded=message,
                message_decoded=decoded_msg,
                bit_accuracy=bit_accuracy,
                bit_error_rate=bit_error_rate,
                ecc_valid=ecc_valid,
                metadata={
                    "attack": attack_name,
                    "attack_params": attack_params,
                    "bitflips": verify_result.get("corrected_bitflips", -1),
                },
            )

            return result
        finally:
            os.unlink(wm_path)
    finally:
        os.unlink(temp_path)


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestColorEffects:
    """Test watermark robustness under color transformations."""

    @pytest.fixture(autouse=True)
    def setup(self, wm_manager, sample_image):
        self.wm = wm_manager
        self.image = sample_image

    def test_brightness_darkening(self):
        """Test with reduced brightness (darker images)."""
        print("\n--- Testing Brightness: Darkening ---")
        results = []

        # Test brightness factors from 0.3 to 1.0
        for factor in [0.3, 0.5, 0.7, 0.9, 1.0]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm, self.image, msg, apply_brightness, {"factor": factor}
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Brightness Darkening: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Brightness darkening success rate {rate:.1f}% below threshold"
        )

    def test_brightness_lightening(self):
        """Test with increased brightness (brighter images)."""
        print("\n--- Testing Brightness: Lightening ---")
        results = []

        # Test brightness factors from 1.0 to 2.0
        for factor in [1.0, 1.2, 1.5, 1.8, 2.0]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm, self.image, msg, apply_brightness, {"factor": factor}
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Brightness Lightening: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Brightness lightening success rate {rate:.1f}% below threshold"
        )

    def test_contrast_reduction(self):
        """Test with reduced contrast."""
        print("\n--- Testing Contrast: Reduction ---")
        results = []

        for factor in [0.3, 0.5, 0.7, 0.9]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm, self.image, msg, apply_contrast, {"factor": factor}
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Contrast Reduction: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Contrast reduction success rate {rate:.1f}% below threshold"
        )

    def test_contrast_increase(self):
        """Test with increased contrast."""
        print("\n--- Testing Contrast: Increase ---")
        results = []

        for factor in [1.2, 1.5, 1.8, 2.0]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm, self.image, msg, apply_contrast, {"factor": factor}
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Contrast Increase: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Contrast increase success rate {rate:.1f}% below threshold"
        )

    def test_saturation_grayscale(self):
        """Test with saturation reduction (toward grayscale)."""
        print("\n--- Testing Saturation: Grayscale ---")
        results = []

        for factor in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm, self.image, msg, apply_saturation, {"factor": factor}
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Saturation/Grayscale: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Saturation/grayscale success rate {rate:.1f}% below threshold"
        )

    def test_saturation_oversaturated(self):
        """Test with increased saturation (oversaturated)."""
        print("\n--- Testing Saturation: Oversaturated ---")
        results = []

        for factor in [1.5, 2.0, 2.5, 3.0]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm, self.image, msg, apply_saturation, {"factor": factor}
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Saturation Oversaturated: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Saturation oversaturated success rate {rate:.1f}% below threshold"
        )

    def test_jpeg_quality_high(self):
        """Test with high-quality JPEG compression."""
        print("\n--- Testing JPEG: High Quality ---")
        results = []

        for quality in [90, 95, 100]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm,
                    self.image,
                    msg,
                    apply_jpeg_compression,
                    {"quality": quality},
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  JPEG High Quality: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"JPEG high quality success rate {rate:.1f}% below threshold"
        )

    def test_jpeg_quality_medium(self):
        """Test with medium-quality JPEG compression."""
        print("\n--- Testing JPEG: Medium Quality ---")
        results = []

        for quality in [60, 70, 80]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm,
                    self.image,
                    msg,
                    apply_jpeg_compression,
                    {"quality": quality},
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  JPEG Medium Quality: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"JPEG medium quality success rate {rate:.1f}% below threshold"
        )

    def test_jpeg_quality_low(self):
        """Test with low-quality JPEG compression."""
        print("\n--- Testing JPEG: Low Quality ---")
        results = []

        for quality in [30, 40, 50]:
            for msg in config.test_messages[:3]:
                result = embed_and_verify_with_attack(
                    self.wm,
                    self.image,
                    msg,
                    apply_jpeg_compression,
                    {"quality": quality},
                )
                results.append(result)
                print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  JPEG Low Quality: {passed}/{len(results)} ({rate:.1f}%)")
        # Lower threshold for aggressive compression
        assert rate >= 50.0, (
            f"JPEG low quality success rate {rate:.1f}% below 50% threshold"
        )


class TestImageSizes:
    """Test watermark verification across different image sizes."""

    @pytest.fixture(autouse=True)
    def setup(self, wm_manager):
        self.wm = wm_manager

    def _resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size."""
        return image.resize(size, Image.Resampling.LANCZOS)

    @pytest.mark.parametrize("size", [256, 384, 512, 768, 1024])
    def test_different_sizes_baseline(self, sample_image, size):
        """Test baseline verification at different sizes (no attacks)."""
        print(f"\n--- Testing Image Size: {size}x{size} (Baseline) ---")
        resized = self._resize_image(sample_image, (size, size))

        results = []
        for msg in config.test_messages[:5]:
            result, _ = embed_and_verify(self.wm, resized, msg)
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Size {size}x{size} Baseline: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Size {size}x{size} success rate {rate:.1f}% below threshold"
        )

    @pytest.mark.parametrize("size", [256, 384, 512, 768, 1024])
    def test_different_sizes_with_jpeg(self, sample_image, size):
        """Test verification at different sizes with JPEG compression."""
        print(f"\n--- Testing Image Size: {size}x{size} (JPEG q=80) ---")
        resized = self._resize_image(sample_image, (size, size))

        results = []
        for msg in config.test_messages[:3]:
            result = embed_and_verify_with_attack(
                self.wm, resized, msg, apply_jpeg_compression, {"quality": 80}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Size {size}x{size} + JPEG: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Size {size}x{size} + JPEG success rate {rate:.1f}% below threshold"
        )

    @pytest.mark.parametrize("size", [256, 512, 1024])
    def test_different_sizes_with_noise(self, sample_image, size):
        """Test verification at different sizes with noise."""
        print(f"\n--- Testing Image Size: {size}x{size} (Noise σ=10) ---")
        resized = self._resize_image(sample_image, (size, size))

        results = []
        for msg in config.test_messages[:3]:
            result = embed_and_verify_with_attack(
                self.wm, resized, msg, apply_gaussian_noise, {"sigma": 10}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Size {size}x{size} + Noise: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Size {size}x{size} + Noise success rate {rate:.1f}% below threshold"
        )


class TestAttackScenarios:
    """Test watermark robustness under various attack scenarios."""

    @pytest.fixture(autouse=True)
    def setup(self, wm_manager, sample_image):
        self.wm = wm_manager
        self.image = sample_image

    # -------------------------------------------------------------------------
    # Cropping Attacks (≤50%)
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("ratio", [0.90, 0.75, 0.60, 0.50, 0.40])
    def test_center_crop(self, ratio):
        """Test watermark after center cropping (preserves center)."""
        print(f"\n--- Testing Crop Attack: {ratio * 100:.0f}% Retention ---")
        results = []

        for msg in config.test_messages[:5]:
            result = embed_and_verify_with_attack(
                self.wm, self.image, msg, apply_crop, {"ratio": ratio}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Crop {ratio * 100:.0f}%: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Crop {ratio * 100:.0f}% success rate {rate:.1f}% below threshold"
        )

    def test_extreme_crop_50_percent(self):
        """Test watermark after 50% center crop (maximum tested)."""
        print("\n--- Testing Crop Attack: 50% (Maximum) ---")
        results = []

        for iteration in range(5):
            for msg in config.test_messages[:5]:
                result = embed_and_verify_with_attack(
                    self.wm, self.image, msg, apply_crop, {"ratio": 0.50}
                )
                results.append(result)
                print(f"  [{iteration + 1}] {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Crop 50% Extreme: {passed}/{len(results)} ({rate:.1f}%)")
        # More lenient for extreme cropping
        assert rate >= 60.0, (
            f"Crop 50% extreme success rate {rate:.1f}% below 60% threshold"
        )

    # -------------------------------------------------------------------------
    # Resize Attacks
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("scale", [0.8, 0.6, 0.5, 0.4])
    def test_downscale_attack(self, scale):
        """Test watermark after downscaling and upscaling."""
        print(f"\n--- Testing Resize Attack: {scale * 100:.0f}% Scale ---")
        results = []

        for msg in config.test_messages[:5]:
            result = embed_and_verify_with_attack(
                self.wm, self.image, msg, apply_resize, {"scale": scale}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Resize {scale * 100:.0f}%: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Resize {scale * 100:.0f}% success rate {rate:.1f}% below threshold"
        )

    @pytest.mark.parametrize("scale", [1.25, 1.5, 2.0])
    def test_upscale_attack(self, scale):
        """Test watermark after upscaling."""
        print(f"\n--- Testing Resize Attack: {scale * 100:.0f}% Scale ---")
        results = []

        for msg in config.test_messages[:5]:
            result = embed_and_verify_with_attack(
                self.wm, self.image, msg, apply_resize, {"scale": scale}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Resize {scale * 100:.0f}%: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Resize {scale * 100:.0f}% success rate {rate:.1f}% below threshold"
        )

    # -------------------------------------------------------------------------
    # Noise Attacks
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("sigma", [5, 10, 15, 20])
    def test_gaussian_noise(self, sigma):
        """Test watermark with Gaussian noise."""
        print(f"\n--- Testing Noise Attack: σ={sigma} ---")
        results = []

        for msg in config.test_messages[:5]:
            result = embed_and_verify_with_attack(
                self.wm, self.image, msg, apply_gaussian_noise, {"sigma": sigma}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Noise σ={sigma}: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Noise σ={sigma} success rate {rate:.1f}% below threshold"
        )

    # -------------------------------------------------------------------------
    # Blur Attacks
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
    def test_gaussian_blur(self, kernel_size):
        """Test watermark with Gaussian blur."""
        print(f"\n--- Testing Blur Attack: Kernel={kernel_size} ---")
        results = []

        for msg in config.test_messages[:5]:
            result = embed_and_verify_with_attack(
                self.wm,
                self.image,
                msg,
                apply_gaussian_blur,
                {"kernel_size": kernel_size},
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Blur Kernel={kernel_size}: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Blur Kernel={kernel_size} success rate {rate:.1f}% below threshold"
        )

    # -------------------------------------------------------------------------
    # Flip Attacks
    # -------------------------------------------------------------------------
    def test_horizontal_flip(self):
        """Test watermark with horizontal flip."""
        print("\n--- Testing Flip Attack: Horizontal ---")
        results = []

        for msg in config.test_messages[:10]:
            result = embed_and_verify_with_attack(
                self.wm, self.image, msg, apply_horizontal_flip, {}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  H-Flip: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"H-Flip success rate {rate:.1f}% below threshold"
        )

    def test_vertical_flip(self):
        """Test watermark with vertical flip."""
        print("\n--- Testing Flip Attack: Vertical ---")
        results = []

        for msg in config.test_messages[:10]:
            result = embed_and_verify_with_attack(
                self.wm, self.image, msg, apply_vertical_flip, {}
            )
            results.append(result)
            print(f"  {result}")

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  V-Flip: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"V-Flip success rate {rate:.1f}% below threshold"
        )

    # -------------------------------------------------------------------------
    # Combined Attacks
    # -------------------------------------------------------------------------
    def test_combined_jpeg_and_crop(self):
        """Test watermark with JPEG + Crop combined attack."""
        import tempfile
        import os
        from notebooks.inference_utils import unnormalize_img, normalize_img

        print("\n--- Testing Combined Attack: JPEG + Crop ---")
        results = []

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            self.image.save(temp_path)

        try:
            for msg in config.test_messages[:5]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    watermarked_tensor, _, _ = self.wm.embed(
                        temp_path, msg, mask_mode="corners"
                    )

                wm_unnorm = unnormalize_img(watermarked_tensor)
                jpeg_tensor = apply_jpeg_compression(wm_unnorm, quality=75)
                cropped_tensor = apply_crop(jpeg_tensor, ratio=0.75)
                attacked = normalize_img(torch.clamp(cropped_tensor, 0, 1))

                img_np = (
                    unnormalize_img(attacked)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                img_np = (img_np * 255).astype(np.uint8)
                wm_image = Image.fromarray(img_np)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    wm_path = f.name
                    wm_image.save(wm_path)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        verify_result = self.wm.verify(wm_path, original_message=msg)

                    decoded_msg = verify_result.get("readable_message", "")
                    success = decoded_msg == msg

                    result = TestResult(
                        name=f"Combined: JPEG+Crop | '{msg}'",
                        success=success,
                        message_embedded=msg,
                        message_decoded=decoded_msg,
                        bit_accuracy=verify_result.get("bit_accuracy", 0.0),
                        bit_error_rate=verify_result.get(
                            "bit_error_rate_percent", 100.0
                        ),
                        ecc_valid=verify_result.get("ecc_valid", False),
                        metadata={"attack": "jpeg75_crop75"},
                    )
                    results.append(result)
                    print(f"  {result}")
                finally:
                    os.unlink(wm_path)
        finally:
            os.unlink(temp_path)

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Combined JPEG+Crop: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= 50.0, (
            f"Combined attack success rate {rate:.1f}% below 50% threshold"
        )

    def test_combined_noise_and_blur(self):
        """Test watermark with Noise + Blur combined attack."""
        import tempfile
        import os
        from notebooks.inference_utils import unnormalize_img, normalize_img

        print("\n--- Testing Combined Attack: Noise + Blur ---")
        results = []

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            self.image.save(temp_path)

        try:
            for msg in config.test_messages[:5]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    watermarked_tensor, _, _ = self.wm.embed(
                        temp_path, msg, mask_mode="corners"
                    )

                wm_unnorm = unnormalize_img(watermarked_tensor)
                noised = apply_gaussian_noise(wm_unnorm, sigma=10)
                blurred = apply_gaussian_blur(noised, kernel_size=5)
                attacked = normalize_img(torch.clamp(blurred, 0, 1))

                img_np = (
                    unnormalize_img(attacked)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                img_np = (img_np * 255).astype(np.uint8)
                wm_image = Image.fromarray(img_np)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    wm_path = f.name
                    wm_image.save(wm_path)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        verify_result = self.wm.verify(wm_path, original_message=msg)

                    decoded_msg = verify_result.get("readable_message", "")
                    success = decoded_msg == msg

                    result = TestResult(
                        name=f"Combined: Noise+Blur | '{msg}'",
                        success=success,
                        message_embedded=msg,
                        message_decoded=decoded_msg,
                        bit_accuracy=verify_result.get("bit_accuracy", 0.0),
                        bit_error_rate=verify_result.get(
                            "bit_error_rate_percent", 100.0
                        ),
                        ecc_valid=verify_result.get("ecc_valid", False),
                        metadata={"attack": "noise10_blur5"},
                    )
                    results.append(result)
                    print(f"  {result}")
                finally:
                    os.unlink(wm_path)
        finally:
            os.unlink(temp_path)

        passed = sum(1 for r in results if r.success)
        rate = passed / len(results) * 100
        print(f"\n  Combined Noise+Blur: {passed}/{len(results)} ({rate:.1f}%)")
        assert rate >= config.success_threshold, (
            f"Combined attack success rate {rate:.1f}% below threshold"
        )


# =============================================================================
# MAIN RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
