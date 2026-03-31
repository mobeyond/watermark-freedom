"""VideoSeal Backend for Watermark Freedom

Uses Facebook Research's VideoSeal model for watermarking.
- 256-bit capacity
- Flexible resolution (no fixed size requirement)
- MIT licensed
"""

import os
import sys
import torch
from PIL import Image
from typing import Optional, Dict, Any, Tuple, Union
import torchvision.transforms as T

PYTHON312_PATH = "/usr/bin/python3.12"
VIDEOSEAL_SITE = "/home/h/.local/lib/python3.12/site-packages"


class VideoSealBackend:
    """Watermarking backend using VideoSeal model."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None
        self._transform = T.ToTensor()
        self._n_bits = 256

    def _get_model(self):
        if self._model is None:
            sys.path.insert(0, VIDEOSEAL_SITE)
            os.chdir(VIDEOSEAL_SITE)
            import videoseal

            self._model = videoseal.load("videoseal_1.0")
            self._model.eval()
            self._model.to(self.device)
            print(f"VideoSeal model loaded (256 bits)")

        return self._model

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.cpu()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).byte()
        return Image.fromarray(tensor.numpy())

    def _pil_to_tensor(self, img: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(img, str):
            img = Image.open(img)
        tensor = self._transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def embed(
        self,
        image_source: Union[str, Image.Image],
        message: str,
    ) -> Tuple[Image.Image, str, Dict[str, Any]]:
        model = self._get_model()

        img_tensor = self._pil_to_tensor(image_source)
        if img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3, :, :]

        h, w = img_tensor.shape[2:]
        b, c, h, w = img_tensor.shape

        msg_bits = self._message_to_bits(message, self._n_bits)
        msg_tensor = torch.tensor([msg_bits], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = model.embed(img_tensor, msg_tensor)
            watermarked = outputs["imgs_w"]

        result_img = self._tensor_to_pil(watermarked)

        binary_str = "".join(str(b) for b in msg_bits)

        coords = {
            "x": 0,
            "y": 0,
            "width": w,
            "height": h,
            "x_percent": 0.0,
            "y_percent": 0.0,
            "width_percent": 1.0,
            "height_percent": 1.0,
            "viewframe_size": min(h, w),
            "backend": "videoseal",
        }

        return result_img, binary_str, coords

    def verify(
        self,
        image_source: Union[str, Image.Image],
        original_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = self._get_model()

        img_tensor = self._pil_to_tensor(image_source)
        if img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3, :, :]

        h, w = img_tensor.shape[2:]

        with torch.no_grad():
            detected = model.detect(img_tensor)
            preds = detected["preds"]

        msg_bits = (preds[0, 1:] > 0.5).long().cpu().tolist()
        binary_str = "".join(str(b) for b in msg_bits)

        readable = self._bits_to_message(msg_bits)

        result = {
            "binary_message": binary_str,
            "readable_message": readable,
            "bit_error_rate_percent": None,
            "corrected_bitflips": None,
            "ecc_valid": None,
            "bit_accuracy": None,
            "viewframe": {
                "x": 0,
                "y": 0,
                "width": w,
                "height": h,
                "x_percent": 0.0,
                "y_percent": 0.0,
                "width_percent": 1.0,
                "height_percent": 1.0,
                "ratio": 1.0,
                "size": min(h, w),
            },
        }

        if original_message:
            orig_bits = self._message_to_bits(original_message, self._n_bits)
            accuracy = sum(1 for p, o in zip(msg_bits, orig_bits) if p == o) / len(
                orig_bits
            )
            result["bit_accuracy"] = accuracy

        return result

    def _message_to_bits(self, message: str, n_bits: int) -> list:
        bits = []
        for char in message:
            val = ord(char) if isinstance(char, str) else char
            for i in range(8):
                bits.append((val >> (7 - i)) & 1)
        while len(bits) < n_bits:
            bits.append(0)
        return bits[:n_bits]

    def _bits_to_message(self, bits: list, bytes_needed: int = 32) -> str:
        message = []
        for i in range(0, min(len(bits), bytes_needed * 8), 8):
            byte_bits = bits[i : i + 8]
            if len(byte_bits) < 8:
                break
            val = sum(b << (7 - j) for j, b in enumerate(byte_bits))
            if 32 <= val <= 126:
                message.append(chr(val))
            else:
                message.append("?")
        return "".join(message)
