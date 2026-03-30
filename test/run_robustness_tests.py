#!/usr/bin/env python3
"""
Standalone Watermark Robustness Test Runner
============================================

A standalone script to run comprehensive watermark robustness tests
without requiring pytest. Can be used for quick verification.

Usage:
    python test/run_robustness_tests.py
    python test/run_robustness_tests.py --color-effects
    python test/run_robustness_tests.py --attacks
    python test/run_robustness_tests.py --sizes
    python test/run_robustness_tests.py --quick
"""

import os
import sys
import argparse
import random
import string
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import WatermarkManager
from watermark_utils import load_image


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TestConfig:
    """Test configuration."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    test_images_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "assets" / "images"
    )
    abnormal_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "abnormal"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "test_output_robustness"
    )
    save_images: bool = True

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
            "WAT",
            "ERM",
        ]
    )

    default_message: str = "TEST"
    success_threshold: float = 80.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# IMAGE LOGGING HELPER
# =============================================================================


def setup_output_dir(output_dir: Path, test_name: str) -> Path:
    """Create timestamped output directory for test results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{test_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_watermarked_image(
    output_dir: Path,
    category: str,
    attack_name: str,
    message: str,
    image: Image.Image,
    success: bool,
    index: int = 0,
) -> str:
    """Save watermarked image with descriptive filename.

    Returns the saved file path.
    """
    status = "PASS" if success else "FAIL"
    safe_attack = (
        attack_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("'", "")
    )
    safe_msg = message.replace(" ", "_")

    filename = f"{category}_{safe_attack}_{safe_msg}_{status}_{index:03d}.png"
    filepath = output_dir / filename
    image.save(filepath)
    return str(filepath)


# =============================================================================
# ATTACK FUNCTIONS
# =============================================================================


def apply_crop(img_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    B, C, H, W = img_tensor.shape
    new_h, new_w = int(H * ratio), int(W * ratio)
    top, left = (H - new_h) // 2, (W - new_w) // 2
    cropped = img_tensor[:, :, top : top + new_h, left : left + new_w]
    upscaled = torch.nn.functional.interpolate(
        cropped, size=(H, W), mode="bilinear", align_corners=False
    )
    return upscaled.clamp(0, 1)


def apply_resize(img_tensor: torch.Tensor, scale: float) -> torch.Tensor:
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
    return (img_tensor * factor).clamp(0, 1)


def apply_contrast(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    mean = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).view(1, 3, 1, 1)
    adjusted = (img_tensor - mean) * factor + mean
    return adjusted.clamp(0, 1)


def apply_saturation(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    gray = (
        0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    )
    gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)
    saturated = gray + (img_tensor - gray) * factor
    return saturated.clamp(0, 1)


def apply_gaussian_noise(img_tensor: torch.Tensor, sigma: float = 20.0) -> torch.Tensor:
    sigma_norm = sigma / 255.0
    noise_tensor = torch.randn_like(img_tensor) * sigma_norm
    return (img_tensor + noise_tensor).clamp(0, 1)


def apply_gaussian_blur(img_tensor: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    sigma = kernel_size / 6.0
    x = (
        torch.arange(kernel_size, dtype=torch.float32, device=img_tensor.device)
        - kernel_size // 2
    )
    kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    padding = kernel_size // 2
    blurred = torch.nn.functional.conv2d(
        img_tensor, kernel_2d, padding=padding, groups=3
    )
    return blurred.clamp(0, 1)


def apply_jpeg_compression(img_tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    import io

    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed = np.array(compressed)
    result = torch.from_numpy(compressed).float() / 255.0
    result = result.permute(2, 0, 1).unsqueeze(0)
    return result.to(img_tensor.device)


def apply_horizontal_flip(img_tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(img_tensor, dims=[3])


def apply_vertical_flip(img_tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(img_tensor, dims=[2])


# =============================================================================
# TEST RESULT CLASSES
# =============================================================================


@dataclass
class TestResult:
    name: str
    success: bool
    message_embedded: str
    message_decoded: str
    bit_accuracy: float
    bit_error_rate: float
    ecc_valid: bool

    def __repr__(self):
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.name}: Embedded='{self.message_embedded}' Decoded='{self.message_decoded}' BER={self.bit_error_rate:.1f}%"


@dataclass
class TestSuiteResult:
    name: str
    results: List[TestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0

    def add(self, result: TestResult):
        self.results.append(result)

    def summary(self) -> str:
        icon = "✓" if self.success_rate >= 80 else "✗"
        return (
            f"{icon} {self.name}: {self.passed}/{self.total} ({self.success_rate:.1f}%)"
        )


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def embed_and_verify(wm, image, message, output_dir=None, category="baseline", index=0):
    """Embed and verify a watermark."""
    import tempfile
    import os
    from notebooks.inference_utils import unnormalize_img

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        image.save(temp_path)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wm_tensor, _, _ = wm.embed(
                temp_path, message, mask_mode="corners", margin_percent=0.15
            )

        img_np = (
            unnormalize_img(wm_tensor)
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
                verify_result = wm.verify(wm_path, original_message=message)

            decoded = verify_result.get("readable_message", "")
            success = decoded == message

            if output_dir is not None:
                save_watermarked_image(
                    output_dir, category, "baseline", message, wm_image, success, index
                )

            return TestResult(
                name=f"Baseline '{message}'",
                success=success,
                message_embedded=message,
                message_decoded=decoded,
                bit_accuracy=verify_result.get("bit_accuracy", 0.0),
                bit_error_rate=verify_result.get("bit_error_rate_percent", 100.0),
                ecc_valid=verify_result.get("ecc_valid", False),
            )
        finally:
            os.unlink(wm_path)
    finally:
        os.unlink(temp_path)


def embed_verify_attack(
    wm,
    image,
    message,
    attack_func,
    output_dir=None,
    category="attack",
    index=0,
    **attack_params,
):
    """Embed, apply attack, verify."""
    import tempfile
    import os
    from notebooks.inference_utils import unnormalize_img, normalize_img

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        image.save(temp_path)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wm_tensor, _, _ = wm.embed(
                temp_path, message, mask_mode="corners", margin_percent=0.15
            )

        wm_unnorm = unnormalize_img(wm_tensor)
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
                verify_result = wm.verify(wm_path, original_message=message)

            decoded = verify_result.get("readable_message", "")
            attack_name = f"{attack_func.__name__}({attack_params})"
            success = decoded == message

            if output_dir is not None:
                save_watermarked_image(
                    output_dir, category, attack_name, message, wm_image, success, index
                )

            return TestResult(
                name=f"{attack_name} | '{message}'",
                success=success,
                message_embedded=message,
                message_decoded=decoded,
                bit_accuracy=verify_result.get("bit_accuracy", 0.0),
                bit_error_rate=verify_result.get("bit_error_rate_percent", 100.0),
                ecc_valid=verify_result.get("ecc_valid", False),
            )
        finally:
            os.unlink(wm_path)
    finally:
        os.unlink(temp_path)


def test_color_effects(wm, image, messages, output_dir=None):
    """Test color effect robustness."""
    suite = TestSuiteResult("Color Effects")

    print("\n" + "=" * 60)
    print("TESTING COLOR EFFECTS")
    print("=" * 60)

    idx = 0
    for factor in [0.5, 0.7, 1.0, 1.3, 1.5]:
        for msg in messages[:3]:
            result = embed_verify_attack(
                wm,
                image,
                msg,
                apply_brightness,
                output_dir,
                "brightness",
                idx,
                factor=factor,
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    for factor in [0.5, 0.8, 1.0, 1.3, 1.5]:
        for msg in messages[:3]:
            result = embed_verify_attack(
                wm,
                image,
                msg,
                apply_contrast,
                output_dir,
                "contrast",
                idx,
                factor=factor,
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    for factor in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for msg in messages[:3]:
            result = embed_verify_attack(
                wm,
                image,
                msg,
                apply_saturation,
                output_dir,
                "saturation",
                idx,
                factor=factor,
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    for quality in [60, 75, 85, 95, 100]:
        for msg in messages[:3]:
            result = embed_verify_attack(
                wm,
                image,
                msg,
                apply_jpeg_compression,
                output_dir,
                "jpeg",
                idx,
                quality=quality,
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    print(f"\n{suite.summary()}")
    return suite


def test_image_sizes(wm, image, messages, output_dir=None):
    """Test different image sizes."""
    suite = TestSuiteResult("Image Sizes")

    print("\n" + "=" * 60)
    print("TESTING IMAGE SIZES")
    print("=" * 60)

    idx = 0
    for size in [256, 384, 512, 768, 1024]:
        resized = image.resize((size, size), Image.Resampling.LANCZOS)
        print(f"\n--- Size: {size}x{size} ---")

        for msg in messages[:5]:
            result = embed_and_verify(wm, resized, msg, output_dir, f"size_{size}", idx)
            suite.add(result)
            print(f"  {result}")
            idx += 1

        for msg in messages[:3]:
            result = embed_verify_attack(
                wm,
                resized,
                msg,
                apply_jpeg_compression,
                output_dir,
                f"size_{size}_jpeg",
                idx,
                quality=80,
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    print(f"\n{suite.summary()}")
    return suite


def test_cropping_attacks(wm, image, messages, output_dir=None):
    """Test cropping attacks (≤50%)."""
    suite = TestSuiteResult("Cropping Attacks (≤50%)")

    print("\n" + "=" * 60)
    print("TESTING CROPPING ATTACKS (≤50%)")
    print("=" * 60)

    idx = 0
    for ratio in [0.90, 0.75, 0.60, 0.50, 0.40]:
        print(f"\n--- Crop: {ratio * 100:.0f}% Retention ---")
        for msg in messages[:5]:
            result = embed_verify_attack(
                wm, image, msg, apply_crop, output_dir, "crop", idx, ratio=ratio
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    print(f"\n{suite.summary()}")
    return suite


def test_geometric_attacks(wm, image, messages, output_dir=None):
    """Test geometric attacks."""
    suite = TestSuiteResult("Geometric Attacks")

    print("\n" + "=" * 60)
    print("TESTING GEOMETRIC ATTACKS")
    print("=" * 60)

    idx = 0
    for scale in [0.5, 0.75, 1.0, 1.5, 2.0]:
        for msg in messages[:3]:
            result = embed_verify_attack(
                wm, image, msg, apply_resize, output_dir, "resize", idx, scale=scale
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    for sigma in [5, 10, 15, 20]:
        for msg in messages[:3]:
            result = embed_verify_attack(
                wm,
                image,
                msg,
                apply_gaussian_noise,
                output_dir,
                "noise",
                idx,
                sigma=sigma,
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    for kernel in [3, 5, 7, 9]:
        for msg in messages[:3]:
            result = embed_verify_attack(
                wm,
                image,
                msg,
                apply_gaussian_blur,
                output_dir,
                "blur",
                idx,
                kernel_size=kernel,
            )
            suite.add(result)
            print(f"  {result}")
            idx += 1

    for msg in messages[:5]:
        result = embed_verify_attack(
            wm, image, msg, apply_horizontal_flip, output_dir, "hflip", idx
        )
        suite.add(result)
        print(f"  {result}")
        idx += 1

    for msg in messages[:5]:
        result = embed_verify_attack(
            wm, image, msg, apply_vertical_flip, output_dir, "vflip", idx
        )
        suite.add(result)
        print(f"  {result}")
        idx += 1

    print(f"\n{suite.summary()}")
    return suite


def test_combined_attacks(wm, image, messages, output_dir=None):
    """Test combined attacks."""
    import tempfile
    import os
    from notebooks.inference_utils import unnormalize_img, normalize_img

    suite = TestSuiteResult("Combined Attacks")

    print("\n" + "=" * 60)
    print("TESTING COMBINED ATTACKS")
    print("=" * 60)

    combined_attacks = [
        ("JPEG+Crop", lambda x: apply_crop(apply_jpeg_compression(x, 75), 0.75)),
        ("Noise+Blur", lambda x: apply_gaussian_blur(apply_gaussian_noise(x, 10), 5)),
        ("Crop+Noise", lambda x: apply_gaussian_noise(apply_crop(x, 0.75), 10)),
        (
            "JPEG+Noise",
            lambda x: apply_gaussian_noise(apply_jpeg_compression(x, 75), 10),
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        image.save(temp_path)

    try:
        idx = 0
        for name, attack in combined_attacks:
            print(f"\n--- {name} ---")
            for msg in messages[:5]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wm_tensor, _, _ = wm.embed(
                        temp_path, msg, mask_mode="corners", margin_percent=0.15
                    )

                wm_unnorm = unnormalize_img(wm_tensor)
                attacked_unnorm = attack(wm_unnorm)
                attacked = normalize_img(torch.clamp(attacked_unnorm, 0, 1))

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

                if output_dir is not None:
                    save_watermarked_image(
                        output_dir,
                        "combined",
                        name,
                        msg,
                        wm_image,
                        success=False,
                        index=idx,
                    )

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    wm_path = f.name
                    wm_image.save(wm_path)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        verify_result = wm.verify(wm_path, original_message=msg)

                    decoded = verify_result.get("readable_message", "")
                    success = decoded == msg

                    if output_dir is not None:
                        save_watermarked_image(
                            output_dir, "combined", name, msg, wm_image, success, idx
                        )

                    result = TestResult(
                        name=f"{name} | '{msg}'",
                        success=success,
                        message_embedded=msg,
                        message_decoded=decoded,
                        bit_accuracy=verify_result.get("bit_accuracy", 0.0),
                        bit_error_rate=verify_result.get(
                            "bit_error_rate_percent", 100.0
                        ),
                        ecc_valid=verify_result.get("ecc_valid", False),
                    )
                    suite.add(result)
                    print(f"  {result}")
                    idx += 1
                finally:
                    os.unlink(wm_path)
    finally:
        os.unlink(temp_path)

    print(f"\n{suite.summary()}")
    return suite


def test_abnormal_images(wm, abnormal_dir, expected_messages=None):
    """Test pre-watermarked images from abnormal directory.

    Args:
        wm: WatermarkManager instance
        abnormal_dir: Path to directory containing pre-watermarked images
        expected_messages: Optional dict mapping filename to expected message

    Returns:
        TestSuiteResult
    """
    suite = TestSuiteResult("Abnormal Images")

    print("\n" + "=" * 60)
    print("TESTING ABNORMAL IMAGES")
    print("=" * 60)

    if not abnormal_dir.exists():
        print(f"Abnormal directory not found: {abnormal_dir}")
        return suite

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    abnormal_images = sorted(
        [
            f
            for f in abnormal_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not abnormal_images:
        print("No images found in abnormal directory")
        return suite

    # Default expected messages (can be overridden)
    default_messages = expected_messages or {}

    for img_path in abnormal_images:
        filename = img_path.name

        # Get expected message if provided, otherwise use None (no comparison)
        expected = default_messages.get(filename)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = wm.verify(str(img_path), original_message=expected)

            decoded = result.get("readable_message", "")
            ecc_valid = result.get("ecc_valid", False)
            bit_accuracy = result.get("bit_accuracy")
            bit_error_rate = result.get("bit_error_rate_percent", 100.0)

            # Success if either:
            # 1. We have an expected message and it matches
            # 2. ECC is valid (watermark detected regardless of message)
            if expected:
                success = decoded == expected
                comparison = f"Expected='{expected}' Decoded='{decoded}'"
            else:
                success = ecc_valid
                comparison = f"Decoded='{decoded}' ECC={ecc_valid}"

            suite.add(
                TestResult(
                    name=f"'{filename}'",
                    success=success,
                    message_embedded=expected or "N/A",
                    message_decoded=decoded,
                    bit_accuracy=bit_accuracy or 0.0,
                    bit_error_rate=bit_error_rate,
                    ecc_valid=ecc_valid,
                )
            )

            status = "✓" if success else "✗"
            print(f"  {status} {filename}: {comparison} BER={bit_error_rate:.1f}%")

            # Print viewframe info
            if "viewframe" in result:
                vf = result["viewframe"]
                print(
                    f"      Viewframe: ({vf['x']},{vf['y']}) {vf['width']}x{vf['height']}"
                )

        except Exception as e:
            suite.add(
                TestResult(
                    name=f"'{filename}'",
                    success=False,
                    message_embedded=expected or "N/A",
                    message_decoded="ERROR",
                    bit_accuracy=0.0,
                    bit_error_rate=100.0,
                    ecc_valid=False,
                )
            )
            print(f"  ✗ {filename}: ERROR - {e}")

    print(f"\n{suite.summary()}")
    return suite


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Watermark Robustness Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument(
        "--color-effects", action="store_true", help="Run color effects tests only"
    )
    parser.add_argument(
        "--sizes", action="store_true", help="Run image sizes tests only"
    )
    parser.add_argument("--attacks", action="store_true", help="Run attack tests only")
    parser.add_argument(
        "--combined", action="store_true", help="Run combined attack tests only"
    )
    parser.add_argument(
        "--abnormal",
        action="store_true",
        help="Test pre-watermarked images from abnormal directory",
    )
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save watermarked images"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("WATERMARK ROBUSTNESS TEST SUITE")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {TestConfig().device}")
    print("=" * 70)

    config = TestConfig()
    save_images = not args.no_save
    output_dir = None

    if save_images:
        output_dir = setup_output_dir(
            config.output_dir, "quick" if args.quick else "full"
        )
        print(f"Images will be saved to: {output_dir}")

    # Initialize manager
    print("\nInitializing WatermarkManager...")
    wm = WatermarkManager(device=torch.device(config.device))

    # Load test image
    images = list(config.test_images_dir.glob("*.png")) + list(
        config.test_images_dir.glob("*.jpg")
    )

    if images:
        image = load_image(images[0])
        print(f"Loaded test image: {images[0]}")
    else:
        print("No test images found, creating synthetic image...")
        image = Image.new("RGB", (512, 512), color=(128, 128, 128))

    messages = TestConfig().test_messages

    # Run tests
    all_results = []

    if args.quick:
        print("\n" + "=" * 60)
        print("QUICK TEST MODE")
        print("=" * 60)

        suite = TestSuiteResult("Quick Baseline")
        print("\n--- Baseline Tests ---")
        for i, msg in enumerate(messages[:5]):
            result = embed_and_verify(wm, image, msg, output_dir, "baseline", i)
            suite.add(result)
            print(f"  {result}")
        print(f"\n{suite.summary()}")
        all_results.append(suite)

        suite = TestSuiteResult("Quick JPEG (q=75)")
        print("\n--- JPEG Tests ---")
        for i, msg in enumerate(messages[:5]):
            result = embed_verify_attack(
                wm,
                image,
                msg,
                apply_jpeg_compression,
                output_dir,
                "jpeg",
                i,
                quality=75,
            )
            suite.add(result)
            print(f"  {result}")
        print(f"\n{suite.summary()}")
        all_results.append(suite)

        suite = TestSuiteResult("Quick Crop (75%)")
        print("\n--- Crop 75% Tests ---")
        for i, msg in enumerate(messages[:5]):
            result = embed_verify_attack(
                wm, image, msg, apply_crop, output_dir, "crop", i, ratio=0.75
            )
            suite.add(result)
            print(f"  {result}")
        print(f"\n{suite.summary()}")
        all_results.append(suite)

    else:
        run_all = not any(
            [args.color_effects, args.sizes, args.attacks, args.combined, args.abnormal]
        )

        if run_all or args.color_effects:
            all_results.append(test_color_effects(wm, image, messages, output_dir))

        if run_all or args.sizes:
            all_results.append(test_image_sizes(wm, image, messages))

        if run_all or args.attacks:
            all_results.append(test_cropping_attacks(wm, image, messages, output_dir))
            all_results.append(test_geometric_attacks(wm, image, messages, output_dir))

        if run_all or args.combined:
            all_results.append(test_combined_attacks(wm, image, messages, output_dir))

        if run_all or args.abnormal:
            all_results.append(test_abnormal_images(wm, config.abnormal_dir))

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_tests = sum(s.total for s in all_results)
    total_passed = sum(s.passed for s in all_results)
    total_failed = sum(s.failed for s in all_results)
    overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    for suite in all_results:
        print(f"  {suite.summary()}")

    print(f"\n{'=' * 50}")
    print(f"OVERALL: {total_passed}/{total_tests} ({overall_rate:.1f}%)")
    print(f"{'=' * 50}")

    # Save to JSON if requested
    if args.output:
        import json

        output = {
            "timestamp": datetime.now().isoformat(),
            "device": TestConfig().device,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_success_rate": overall_rate,
            "suites": [
                {
                    "name": s.name,
                    "total": s.total,
                    "passed": s.passed,
                    "failed": s.failed,
                    "success_rate": s.success_rate,
                    "results": [
                        {
                            "name": r.name,
                            "success": r.success,
                            "embedded": r.message_embedded,
                            "decoded": r.message_decoded,
                            "bit_accuracy": r.bit_accuracy,
                            "bit_error_rate": r.bit_error_rate,
                            "ecc_valid": r.ecc_valid,
                        }
                        for r in s.results
                    ],
                }
                for s in all_results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0 if overall_rate >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
