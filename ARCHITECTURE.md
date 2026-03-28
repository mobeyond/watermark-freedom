# Watermark Freedom - System Architecture & Processing Logic

## Overview

This system implements **localized watermark embedding** using the WAM (Watermark Anything Model) architecture. The watermark is embedded only within a designated "viewframe" region, marked by corner brackets, rather than across the entire image.

---

## System Components

### 1. WAM Model (WAM - Watermark Anything Model)
- **Architecture**: A neural network trained to embed hidden messages into images
- **Trained Input Size**: 256×256 pixels
- **Message Capacity**: 32 bits (encoded as max 3 ASCII characters via ROCO encoding)
- **Location**: `checkpoints/wam_mit.pth`

### 2. Viewframe System
- A rectangular region within the image where watermark is embedded
- Marked by 4 corner brackets (L-shaped white lines)
- Allows **localized watermarking** - only the viewframe region is modified

### 3. ROCO Encoding
- **ROCO-ECC**: Robust Cyclic Coding with Error Correction
- Encodes up to 3 ASCII characters into 32 bits
- Provides forward error correction for robustness

---

## Processing Pipeline: Embedding

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBED WATERMARK                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 1: PREPROCESS IMAGE                                  │
    │ ────────────────────────────────────────────────────────  │
    │ • Load image from source (file path or PIL Image)        │
    │ • Convert to OpenCV format (BGR)                         │
    │ • Crop to centered square (maintains aspect ratio)       │
    │ • Convert back to PIL, then to PyTorch tensor           │
    │ • Apply default_transform (normalize to [-1, 1])         │
    │ • Move to device (GPU/CPU)                               │
    │                                                           │
    │ Output: img_pt [1, 3, H, W], cv_img (OpenCV BGR)        │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 2: CALCULATE VIEWFRAME REGION                       │
    │ ────────────────────────────────────────────────────────  │
    │ Based on mask_mode:                                      │
    │                                                          │
    │ • 'corners' (default): Centered square with 15% margin   │
    │   margin = int(min(H, W) × 0.15)                         │
    │   x = y = margin                                         │
    │   width = height = min(H, W) - 2×margin                 │
    │                                                          │
    │ • 'pixels': Use exact pixel coordinates from params      │
    │ • 'percentage': Convert percentages to pixels            │
    │                                                          │
    │ Example for 512×512 image (corners mode):               │
    │   margin = 76, viewframe = 360×360 at (76, 76)          │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 3: CROP TO VIEWFRAME REGION                         │
    │ ────────────────────────────────────────────────────────  │
    │ cropped = img_pt[:, :, y:y+height, x:x+width]           │
    │                                                           │
    │ Only the viewframe region will be watermarked.           │
    │ Rest of image remains unchanged.                         │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 4: RESIZE TO 256×256 FOR WAM                        │
    │ ────────────────────────────────────────────────────────  │
    │ cropped_256 = interpolate(cropped, size=(256, 256))     │
    │                                                           │
    │ CRITICAL: WAM model was trained on 256×256 inputs.       │
    │ This resizing is mandatory for proper embedding.         │
    │                                                           │
    │ ⚠️  SMALL IMAGE ISSUE: If viewframe is small (e.g.,     │
    │    46×46 for a 64×64 image), upscaling to 256×256        │
    │    introduces interpolation artifacts that may degrade   │
    │    watermark quality.                                    │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 5: ENCODE MESSAGE TO BINARY TENSOR                  │
    │ ────────────────────────────────────────────────────────  │
    │ wm_msg_tensor = roco_encode_to_binary_tensor(message)    │
    │                                                           │
    │ Input: "ABC" (max 3 chars)                               │
    │ Output: [32] tensor of 0s and 1s                        │
    │                                                           │
    │ Example: "ABC" → [1,0,1,0,0,0,01,...] (32 bits)        │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 6: EMBED WATERMARK WITH WAM MODEL                   │
    │ ────────────────────────────────────────────────────────  │
    │ wm_msg = wm_msg_tensor.unsqueeze(0).to(device)          │
    │ outputs = self.wam.embed(cropped_256, wm_msg)           │
    │                                                           │
    │ The WAM model:                                           │
    │ • Takes 256×256 image and 32-bit message                │
    │ • Embeds message into image features                     │
    │ • Returns watermarked image                              │
    │                                                           │
    │ outputs['imgs_w'] = watermarked [1, 3, 256, 256] tensor │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 7: RESIZE BACK TO VIEWFRAME SIZE                    │
    │ ────────────────────────────────────────────────────────  │
    │ watermarked_crop = interpolate(outputs['imgs_w'],       │
    │                                  size=(height, width))   │
    │                                                           │
    │ Scale the watermarked region back to original size.      │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 8: PLACE WATERMARKED REGION BACK                    │
    │ ────────────────────────────────────────────────────────  │
    │ img_w = img_pt.clone()                                   │
    │ img_w[:, :, y:y+height, x:x+width] = watermarked_crop   │
    │                                                           │
    │ The watermarked crop replaces the original region.       │
    │ Everything outside the viewframe is untouched.           │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 9: DRAW VIEWFRAME CORNER BRACKETS                   │
    │ ────────────────────────────────────────────────────────  │
    │ Convert tensor → numpy → BGR OpenCV format              │
    │                                                           │
    │ Calculate bracket parameters:                            │
    │ • corner_length = min(width, height) × 0.15             │
    │ • line_thickness = max(2, min(width, height) × 0.012)   │
    │ • opacity = 0.95 (95% white)                             │
    │                                                           │
    │ Draw 8 L-shaped lines (2 per corner):                    │
    │ • Top-left, Top-right, Bottom-left, Bottom-right         │
    │                                                           │
    │ Alpha blend brackets with original image:                │
    │ result = original × (1 - opacity × mask) + white × opacity│
    │                                                           │
    │ Convert back to RGB tensor                               │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 10: RETURN RESULTS                                  │
    │ ────────────────────────────────────────────────────────  │
    │ Returns:                                                 │
    │ • img_w: Watermarked image tensor [1, 3, H, W]          │
    │ • binary_message_str: Binary string (e.g., "10100110...")│
    │ • coords: Dictionary with viewframe coordinates          │
    │                                                         │
    │ coords = {                                              │
    │   'x': x, 'y': y, 'width': width, 'height': height,    │
    │   'x_percent': x/w, 'y_percent': y/h,                   │
    │   'width_percent': width/w, 'height_percent': height/h │
    │ }                                                       │
    └───────────────────────────────────────────────────────────┘
```

---

## Processing Pipeline: Verification

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFY WATERMARK                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 1: PREPROCESS IMAGE (same as embed)                  │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 2: DETECT VIEWFRAME CORNERS                         │
    │ ────────────────────────────────────────────────────────  │
    │ detected = _detect_viewframe_corners(cv_img)             │
    │                                                           │
    │ The ViewframeDetector:                                    │
    │ • Converts image to grayscale                             │
    │ • Thresholds for bright pixels (>200)                    │
    │ • Finds contours and detects L-shaped corners            │
    │ • Returns bounding box of detected viewframe             │
    │                                                           │
    │ If detection fails → fallback to centered square         │
    │ (70% of min dimension, centered)                         │
    │                                                           │
    │ Output: (x, y, width, height) or fallback coords         │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 3: CROP & RESIZE TO 256×256                         │
    │ ────────────────────────────────────────────────────────  │
    │ cropped = img_pt[:, :, y:y+height, x:x+width]           │
    │ cropped_256 = interpolate(cropped, size=(256, 256))     │
    │                                                           │
    │ Same as embedding - extract viewframe and scale.         │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 4: DETECT WATERMARK WITH WAM MODEL                  │
    │ ────────────────────────────────────────────────────────  │
    │ preds = self.wam.detect(cropped_256)["preds"]           │
    │                                                           │
    │ The WAM detector outputs:                                │
    │ • preds[:, 0, :, :] - Mask prediction (which pixels     │
    │                     contain watermark info)              │
    │ • preds[:, 1:, :, :] - Bit predictions (the actual bits) │
    │                                                           │
    │ Shape: [1, 33, 256, 256] (1 channel mask + 32 bit maps) │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 5: INFERENCE MESSAGE FROM PREDICTIONS               │
    │ ────────────────────────────────────────────────────────  │
    │ mask_preds = sigmoid(preds[:, 0, :, :])                 │
    │ bit_preds = preds[:, 1:, :, :]                          │
    │ pred_message_tensor = msg_predict_inference(bit_preds,  │
    │                                             mask_preds) │
    │                                                           │
    │ msg_predict_inference:                                   │
    │ • Uses mask to focus on watermark-containing regions     │
    │ • Aggregates bit predictions across spatial dimensions   │
    │ • Applies threshold (0.5) to get binary decisions       │
    │                                                           │
    │ Output: [32] tensor of predicted bits (floats)          │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 6: DECODE BINARY TO READABLE MESSAGE               │
    │ ────────────────────────────────────────────────────────  │
    │ readable_message, is_valid, bitflips =                   │
    │     roco_decode_from_binary_tensor(pred_message_tensor)  │
    │                                                           │
    │ ROCO decoder:                                            │
    │ • Extracts data bits and parity bits from 32-bit string │
    │ • Applies Reed-Solomon error correction                  │
    │ • Decodes corrected bits to ASCII characters             │
    │ • Returns readable message and validation status         │
    │                                                           │
    │ Output examples:                                         │
    │ • "ABC", True, 0  → Perfect decode                       │
    │ • "ABC", True, 2  → 2 bit errors corrected               │
    │ • "XY.", False, -1 → Decode failed (invalid ECC)        │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 7: CALCULATE METRICS                                │
    │ ────────────────────────────────────────────────────────  │
    │ total_bits = 32                                          │
    │ bit_error_rate = (bitflips / 32) × 100                  │
    │                                                           │
    │ If original_message provided:                             │
    │ bit_accuracy = (pred == original).mean() × 100          │
    └───────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 8: RETURN RESULTS                                   │
    │ ────────────────────────────────────────────────────────  │
    │ Returns dictionary:                                      │
    │ {                                                        │
    │   'binary_message': "10100110...",                       │
    │   'readable_message': "ABC",                             │
    │   'bit_error_rate_percent': 3.12,                       │
    │   'corrected_bitflips': 1,                              │
    │   'ecc_valid': True,                                    │
    │   'bit_accuracy': 0.96875 (if original provided),        │
    │   'viewframe': {                                        │
    │       'x': 76, 'y': 76, 'width': 360, 'height': 360,   │
    │       'x_percent': 0.148, ...                           │
    │       'ratio': 0.499 (viewframe / image area)           │
    │   }                                                     │
    │ }                                                       │
    └───────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why Localized Watermarking?
- **Targeted embedding**: Only modifies the viewframe region
- **Preserves image quality**: Areas outside viewframe are untouched
- **Easier detection**: Known location simplifies verification
- **Flexible positioning**: Supports pixel, percentage, or auto modes

### 2. Why 256×256 for WAM?
- **Model constraint**: WAM was trained on 256×256 inputs
- **Consistency**: Fixed input size ensures predictable behavior
- **Trade-off**: Small images must be upscaled (quality loss)

### 3. Why 95% Opacity for Brackets?
- **Detection reliability**: High brightness (value 255 × 0.95 ≈ 240)
- **Aesthetic balance**: Visible but not overwhelming
- **Detector compatibility**: Threshold set at 200 catches these easily

### 4. Why ROCO Encoding?
- **Error correction**: Can recover from bit flips during attacks
- **Compact**: 32 bits for max 3 chars with redundancy
- **Robustness**: Designed for watermarking use cases

---

## Small Image Limitations

### The Problem
For images smaller than ~350×350 pixels:

| Image Size | Viewframe (15% margin) | Upscaling Factor | Quality |
|------------|----------------------|------------------|---------|
| 64×64      | 46×46                | 5.6×             | ✗ Poor  |
| 128×128    | 90×90                | 2.8×             | ✗ Poor  |
| 256×256    | 180×180              | 1.4×             | ⚠️ Questionable |
| 512×512    | 360×360              | 0.7× (downscale) | ✓ Good  |

### Why It Fails
1. **Upscaling artifacts**: Bilinear interpolation blurs the small region
2. **Model mismatch**: WAM trained on natural 256×256 images
3. **Information density**: Fewer pixels to carry the watermark signal
4. **Double scaling**: Upscale → embed → downscale loses information

### Recommendation
**Minimum image size: 350×350 pixels** for reliable watermarking.

---

## File References

```
core.py                           # Main WatermarkManager class
├── embed()                       # Watermark embedding pipeline
├── verify()                      # Watermark verification pipeline
├── verify_tensor()               # Tensor-based verification
├── _preprocess_image()           # Image loading and preprocessing
└── _detect_viewframe_corners()   # Viewframe detection

viewframe_detector.py             # Viewframe corner detection
viewframe.py                      # Viewframe geometry utilities
watermark_utils.py                # Utility functions
roco_core.py / roco_ecc.py        # ROCO encoding/decoding
notebooks/inference_utils.py      # Model loading, transforms
watermark_anything/models/        # WAM model architecture
```

---

## Example Workflow

```python
from core import WatermarkManager

# Initialize
manager = WatermarkManager()

# Embed watermark
image_path = "input.jpg"
message = "ABC"  # Max 3 characters

watermarked_img, binary_msg, coords = manager.embed(
    image_path,
    message,
    mask_mode='corners',  # or 'pixels', 'percentage'
    margin_percent=0.15
)

print(f"Viewframe: {coords}")
# Output: {'x': 76, 'y': 76, 'width': 360, 'height': 360, ...}

# Verify watermark
result = manager.verify("watermarked.jpg")
print(f"Message: {result['readable_message']}")  # "ABC"
print(f"ECC Valid: {result['ecc_valid']}")       # True
print(f"Bit Error Rate: {result['bit_error_rate_percent']}%")
```

---

*Generated for Watermark Freedom System Documentation*
