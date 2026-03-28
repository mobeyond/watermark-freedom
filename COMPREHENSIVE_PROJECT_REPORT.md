# Comprehensive Technical Report: Watermark Freedom Project

**Date:** March 25, 2026
**Report Type:** Deep Technical Analysis
**Scope:** Watermark Tool Architecture + Attack-Based Self-Optimizing Mechanism

---

## Table of Contents

1. Executive Summary
2. Part 1: Watermark Tool Analysis
3. Part 2: Attack-Based Self-Optimizing Mechanism
4. System Integration & Data Flow
5. Strengths, Weaknesses, and Recommendations
6. Appendix: File Structure and Key Metrics

---

## 1. Executive Summary

The Watermark Freedom project implements a sophisticated **localized image watermarking system** with an innovative **attack-based self-optimizing mechanism**. The system is built upon the WAM (Watermark Anything) architecture from Meta, adapted for robust watermark embedding with viewframe detection and multi-attack resilience.

### Key Components Analyzed:

**Part 1 - Watermark Tool:**
- VAE-based embedder for message encoding
- SAM-based extractor for watermark detection  
- ROCO encoding system with BCH error correction
- Viewframe-based localized watermarking
- Flask API for web integration

**Part 2 - Self-Optimizing Mechanism:**
- Evolutionary search algorithm
- 10-attack benchmark suite
- Z-score based statistical evaluation
- Multi-generational parameter optimization

---

## 2. Part 1: Watermark Tool Analysis

### 2.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      WAM WATERMARK FREEDOM SYSTEM                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    EMBEDDING PIPELINE                             │ │
│  │                                                                  │ │
│  │   Input Image → Preprocess → Crop ROI → Resize 256x256          │ │
│  │                          ↓                                       │ │
│  │   Message "ABC" → ROCO Encode → 32-bit Tensor                   │ │
│  │                          ↓                                       │ │
│  │   ┌────────────────────────────────────────────────────────────┐ │ │
│  │   │                  VAE EMBEDDER                              │ │ │
│  │   │   ┌──────────┐    ┌─────────────┐    ┌──────────────┐     │ │ │
│  │   │   │ Encoder  │───>│ MsgProcessor│───>│   Decoder    │     │ │ │
│  │   │   │ 4ch out  │    │ binary+concat│   │ 36ch input   │     │ │ │
│  │   │   └──────────┘    └─────────────┘    └──────────────┘     │ │ │
│  │   └────────────────────────────────────────────────────────────┘ │ │
│  │                          ↓                                       │ │
│  │   Watermarked ROI → Resize → Place → Draw Corners               │ │
│  │                                                                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    DETECTION PIPELINE                             │ │
│  │                                                                  │ │
│  │   Watermarked Image → Detect Corners → Crop ROI → Resize        │ │
│  │                          ↓                                       │ │
│  │   ┌────────────────────────────────────────────────────────────┐ │ │
│  │   │                 SAM EXTRACTOR                              │ │ │
│  │   │   ┌──────────┐    ┌─────────────┐    ┌─────────────────┐  │ │ │
│  │   │   │ ImageEnc │───>│ PixelDecoder│───>│ (1+32)xHxW     │  │ │ │
│  │   │   │   ViT    │    │ 16x upscale │    │ Mask + Bits     │  │ │ │
│  │   │   └──────────┘    └─────────────┘    └─────────────────┘  │ │ │
│  │   └────────────────────────────────────────────────────────────┘ │ │
│  │                          ↓                                       │ │
│  │   Aggregate Bits → ROCO Decode → ECC Correct → Message          │ │
│  │                                                                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Analysis

#### 2.2.1 VAE Embedder (`watermark_anything/models/embedder.py`)

**Architecture:** Variational Autoencoder with Message Processing

```python
# Configuration from configs/embedder.yaml
vae_small:
  encoder:
    ch: 32                      # Base channel count
    ch_mult: [1, 1, 1, 2]       # Multiplier per resolution level  
    z_channels: 4               # Latent dimension
    num_res_blocks: 2           # ResNet blocks per level
    
  msg_processor:
    nbits: 32                   # Bits processed (matches ECC codeword)
    hidden_size: 64             # Message embedding dimension (nbits * 2)
    msg_processor_type: 'binary+concat'
    
  decoder:
    z_channels: 68              # 4 (latent) + 64 (message) = 68
    tanh_out: True              # Output range [-1, 1]
```

**Encoding Flow:**
1. **Image Encoding:** 256×256 RGB → 4-channel latent (16×16 spatial)
2. **Message Processing:** 
   - 32-bit message split into 32 individual bits
   - Each bit (0/1) selects embedding from layer of size 64
   - Embeddings summed → 64-dimensional message vector
   - Message vector tiled to match latent spatial dimensions
3. **Concatenation:** Latent (4ch) + Message (64ch) = 68ch combined
4. **Decoding:** 68ch → 3ch watermarked image (256×256)

**Mathematical Formulation:**
```
E: ℝ^(3×256×256) → ℝ^(4×16×16)           (Encoder)
M: {0,1}^32 → ℝ^64                       (Message Embedding)
D: ℝ^(68×16×16) → ℝ^(3×256×256)          (Decoder)

Watermarked = D(cat(E(Image), M(Message)))
```

#### 2.2.2 SAM Extractor (`watermark_anything/models/extractor.py`)

**Architecture:** Segment Anything Model variant with ViT backbone

```python
# Configuration from configs/extractor.yaml
sam_base:
  encoder:
    img_size: 256
    embed_dim: 768              # Transformer embedding dimension
    depth: 12                   # Transformer layers
    num_heads: 12               # Multi-head attention heads
    patch_size: 16              # Patch size for ViT
    
  pixel_decoder:
    embed_dim: 768
    nbits: 32
    upscale_stages: [4, 2, 2]   # 4×2×2 = 16x upscale factor
```

**Detection Flow:**
1. **Patch Embedding:** 256×256 → 16×16 patches × 768 dims
2. **ViT Encoding:** 12-layer transformer with attention
3. **Pixel Decoding:** Feature map → (1+32) channels at 256×256
   - Channel 0: Watermark presence probability
   - Channels 1-32: Bit predictions per pixel

**Output Structure:**
```
preds: [B, 33, 256, 256]
  preds[:, 0, :, :]     → mask probability (sigmoid → [0,1])
  preds[:, 1:, :, :]    → 32 bit predictions (raw logits)
```

#### 2.2.3 ROCO Encoding System (`roco_core.py`, `roco_ecc.py`)

**ROCO = Resilient Optimized Code Operation**

**Character Encoding:**
```
Allowed Characters (32 total = 5 bits each):
  '.' (padding) = 0
  'A'-'Z'        = 1-26
  '4', '6', '7', '9' = 27-30
  '#'            = 31
```

**Encoding Pipeline:**
```
Step 1: Character → Binary Encoding
  "ABC" → A(00001) B(00010) C(00011) → 15 bits

Step 2: Add Version Bit
  0 + 00001 00010 00011 → 16 bits (version=0)

Step 3: BCH Error Correction (t=2)
  16-bit data + 16-bit ECC → 32-bit codeword

Step 4: Tensor Conversion
  32-bit codeword → torch.Tensor([b0, b1, ..., b31])
```

**BCH Parameters:**
```python
BCH = bchlib.BCH(0x43, 2)  # m=6, t=2
# - Primitive polynomial: x^6 + x + 1 = 0x43
# - Correction capability: 2 bit errors
# - Code rate: 16/32 = 0.5
```

**Error Correction Demonstration:**
```
Original:  "ABC" → 0000100001000011 (data) + ECC
Transmitted with 2 bit flips
Decoded:   ECC corrects 2 errors → "ABC" ✓

Original:  "ABC" → data + ECC  
Transmitted with 3 bit flips
Decoded:   ECC reports invalid (3 > t=2) → error detected
```

#### 2.2.4 Viewframe Detection System (`viewframe.py`, `core.py`)

**Purpose:** Localized watermarking with automatic ROI detection

**Corner Bracket Design:**
```
┌─────────────────────────────────────────────────────────┐
│                                                        │
│     ┌───                                        ───┐   │
│     │                                                   │
│     │           WATERMARK REGION                       │
│     │                                                   │
│     └───                                        ───┐   │
│                                                        │
└─────────────────────────────────────────────────────────┘

Corner Properties:
  - Line thickness: 3 pixels (thick for visibility)
  - Length: 15% of ROI side
  - Color: Pure white (255, 255, 255)
  - Detection: Simple threshold at value 255
```

**Detection Algorithm:**
```python
def _detect_viewframe_corners(cv_img):
    # 1. Find pure white pixels (value = 255)
    bright_mask = (gray == 255).astype(np.uint8) * 255
    
    # 2. Locate 4 corner regions
    rows, cols = np.where(bright_mask > 0)
    
    # 3. Find outer edges in each quadrant
    tl_x = cols[(rows < h//2) & (cols < w//2)].min()
    tl_y = rows[(rows < h//2) & (cols < w//2)].min()
    # ... similar for other corners
    
    # 4. Compute ROI bounding box
    x = tl_x + offset
    y = tl_y + offset  
    width = tr_x - tl_x - 2*offset
    height = bl_y - tl_y - 2*offset
    
    return x, y, width, height
```

### 2.3 Core Implementation (`core.py`)

#### 2.3.1 WatermarkManager Class

**Embed Method:**
```python
def embed(image_source, message, mask_mode, mask_params=None):
    # 1. Preprocess image (center crop, square)
    img_pt, cv_img = self._preprocess_image(image_source)
    
    # 2. Determine ROI coordinates
    if mask_mode == 'corners':
        # Use viewframe calculation
        x, y, width, height = get_viewframe_roi(cv_img)
    
    # 3. Crop and resize to 256x256
    cropped = img_pt[:, :, y:y+height, x:x+width]
    cropped_256 = F.interpolate(cropped, size=(256, 256), ...)
    
    # 4. Encode message with ROCO
    wm_msg_tensor = roco_encode_to_binary_tensor(message)  # 32 bits
    wm_msg = wm_msg_tensor.unsqueeze(0).to(self.device)
    
    # 5. Embed with WAM
    outputs = self.wam.embed(cropped_256, wm_msg)
    
    # 6. Resize back and place in original
    watermarked_crop = F.interpolate(outputs['imgs_w'], 
                                     size=(height, width), ...)
    img_w[:, :, y:y+height, x:x+width] = watermarked_crop
    
    # 7. Draw corner brackets (visible marker)
    draw_corner_brackets(img_bgr, x, y, width, height)
    
    return img_w, binary_message_str, coords
```

**Verify Method:**
```python
def verify(image_source, original_message=None, viewframe_coords=None):
    # 1. Preprocess image
    img_pt, cv_img = self._preprocess_image(image_source)
    
    # 2. Auto-detect viewframe corners
    detected = self._detect_viewframe_corners(cv_img)
    if detected:
        x, y, width, height = detected
    else:
        # Fallback: centered square
        x, y, width, height = get_default_roi(img_pt.shape)
    
    # 3. Crop to ROI and resize
    cropped = img_pt[:, :, y:y+height, x:x+width]
    cropped_256 = F.interpolate(cropped, size=(256, 256), ...)
    
    # 4. Run detector
    preds = self.wam.detect(cropped_256)["preds"]
    
    # 5. Extract mask and bit predictions
    mask_preds = torch.sigmoid(preds[:, 0, :, :])
    bit_preds = preds[:, 1:, :, :]
    
    # 6. Aggregate bits using inference method
    pred_message_tensor = msg_predict_inference(bit_preds, mask_preds)
    
    # 7. Decode with ECC
    readable_message, is_valid, bitflips = roco_decode_from_binary_tensor(pred_message_tensor[0])
    
    return {
        'readable_message': readable_message,
        'ecc_valid': is_valid,
        'corrected_bitflips': bitflips,
        'bit_error_rate_percent': (bitflips/32)*100,
        # ... additional metadata
    }
```

### 2.4 Message Inference (`watermark_anything/data/metrics.py`)

**Three Inference Methods:**

**1. Hard Method:**
```python
# Binarize each pixel prediction first
preds = preds > threshold  # B, K, H, W → binary
# Select based on mask, average per bit position
for each bit position k:
    selected = preds[k][mask == 1]  # All pixels predicting this bit
    aggregated = selected.mean()     # Majority vote
    final_bit = 1 if aggregated > 0.5 else 0
```

**2. Semihard Method:**
```python
# Select raw predictions based on mask, average, then binarize
for each bit position k:
    selected = preds[k][mask == 1]  # Keep raw values
    aggregated = selected.mean()     # Average confidence
    final_bit = 1 if aggregated > 0.5 else 0
```

**3. Soft Method:**
```python
# Weighted average using mask as soft weights
preds = sum(preds * masks) / sum(masks)  # B, K
final_bits = (preds > 0.5).int()
```

### 2.5 JND Attenuation (`watermark_anything/modules/jnd.py`)

**Purpose:** Perceptual quality preservation

**Just Noticeable Distortion Calculation:**
```python
def compute_jnd(image):
    # Luminance masking: visual system less sensitive in bright areas
    la = luminance_masking(image)  # High in bright regions
    
    # Contrast masking: less sensitive near edges
    cm = contrast_masking(image)   # High near edges
    
    # Combined JND map
    hmaps = la + cm - clc * min(la, cm)  # clc = correlation coeff
    
    return hmaps
```

**Attenuation Application:**
```python
# Instead of direct blending:
# imgs_w = scaling_i * imgs + scaling_w * preds_w

# Apply JND-based attenuation:
imgs_w = scaling_i * imgs + scaling_w * preds_w * (1 - alpha * hmaps)
# Watermark less visible where human vision is sensitive
```

### 2.6 API Interface (`app.py`)

**Flask Endpoints:**

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/` | GET | - | HTML interface |
| `/watermark` | POST | Image + message + mask params | Watermarked image + metadata |
| `/verify` | POST | Watermarked image + original message | Decoded message + statistics |

**Mask Mode Options:**
1. **corners:** Default viewframe corners (automatic)
2. **pixels:** Exact pixel coordinates
3. **percentage:** Normalized coordinates (0-1)

---

## 3. Part 2: Attack-Based Self-Optimizing Mechanism

### 3.1 Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SELF-OPTIMIZING OPTIMIZER ARCHITECTURE               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  PHASE 1: INITIAL CANDIDATE GENERATION                          │  │
│  │                                                                 │  │
│  │  Generate 10 predefined variants:                               │  │
│  │    baseline (sw=2.0, si=1.0)                                   │  │
│  │    sw_30 (sw=3.0, si=1.0)                                      │  │
│  │    sw_15 (sw=1.5, si=1.0)                                      │  │
│  │    sw_25 (sw=2.5, si=1.0)                                      │  │
│  │    sw_25_si09 (sw=2.5, si=0.9)                                 │  │
│  │    ...                                                         │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  PHASE 2: ATTACK BENCHMARKING                                   │  │
│  │                                                                 │  │
│  │  For each candidate:                                            │  │
│  │    For each test image (20 images):                             │  │
│  │      Embed watermark with current variant                       │  │
│  │      For each attack (10 types):                                │  │
│  │        Apply attack to watermarked image                       │  │
│  │        Verify watermark                                        │  │
│  │        Compute z-score from correct bits                       │  │
│  │    Compute median z-score per attack                           │  │
│  │    Compute overall median z-score                              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  PHASE 3: ACCEPT/REJECT DECISION                                │  │
│  │                                                                 │  │
│  │  Accept if:                                                     │  │
│  │    (delta_z >= 0.10) AND (wins >= 2 attacks)                   │  │
│  │                                                                 │  │
│  │  Where:                                                         │  │
│  │    delta_z = candidate_z - best_z                              │  │
│  │    wins = count(attacks where candidate > best)                │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                         ┌────┴────┐                                    │
│                         │ ACCEPT  │ REJECT                             │
│                         │         │                                    │
│                         ▼         ▼                                    │
│                  ┌──────────┐  ┌─────────┐                            │
│                  │ Update   │  │ Discard │                            │
│                  │ Best    │  │ Variant │                            │
│                  └────┬─────┘  └─────────┘                            │
│                       │                                               │
│                       ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  PHASE 4: CONTINUATION GENERATIONS (if improvements found)      │  │
│  │                                                                 │  │
│  │  For N generations:                                             │  │
│  │    Generate 12 fine-tuned candidates around best:              │  │
│  │      sw ± 0.25, sw ± 0.5                                        │  │
│  │      si ± 0.1                                                  │  │
│  │    Benchmark all 12                                            │  │
│  │    Accept best improvement                                     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Attack Suite (`attacks/`)

#### 3.2.1 Geometric Attacks (`attacks/geometric.py`)

**1. Crop Attack:**
```python
def crop(img: torch.Tensor, ratio: float = 0.75) -> torch.Tensor:
    """Center crop and upscale to original size."""
    B, C, H, W = img.shape
    new_h, new_w = int(H * ratio), int(W * ratio)
    top, left = (H - new_h) // 2, (W - new_w) // 2
    cropped = img[:, :, top:top+new_h, left:left+new_w]
    upscaled = F.interpolate(cropped, size=(H, W), mode='bilinear', ...)
    return upscaled.clamp(0, 1)
```
- **Effect:** Removes edge content, forces watermark into smaller area
- **Parameters:** ratio (0.5-1.0), default 0.75
- **Robustness Required:** Spatial redundancy in watermark encoding

**2. Resize Attack:**
```python
def resize(img: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Downscale and upscale to original size."""
    B, C, H, W = img.shape
    new_h, new_w = int(H * scale), int(W * scale)
    downscaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', ...)
    upscaled = F.interpolate(downscaled, size=(H, W), mode='bilinear', ...)
    return upscaled.clamp(0, 1)
```
- **Effect:** Frequency loss from downsampling, interpolation artifacts
- **Parameters:** scale (0.1-1.0), default 0.5
- **Robustness Required:** Frequency-domain watermark placement

**3. Rotate Attack:**
```python
def rotate(img: torch.Tensor, angle: float = 15.0) -> torch.Tensor:
    """Rotate image by angle in degrees."""
    B, C, H, W = img.shape
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_np, M, (W, H), 
                            flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_REPLICATE)
    rotated = torch.from_numpy(rotated).float() / 255.0
    return rotated.permute(2, 0, 1).unsqueeze(0)
```
- **Effect:** Misaligns watermark region, interpolation artifacts
- **Parameters:** angle (-45 to 45 degrees), default 15
- **Robustness Required:** Rotation-invariant features

**4. Flip Attack:**
```python
def flip(img: torch.Tensor, horizontal: bool = True) -> torch.Tensor:
    """Flip image horizontally or vertically."""
    if horizontal:
        return torch.flip(img, dims=[3])  # Flip width
    else:
        return torch.flip(img, dims=[2])  # Flip height
```
- **Effect:** Mirrors watermark, may break directional encoding
- **Parameters:** horizontal (boolean)
- **Robustness Required:** Symmetric encoding

#### 3.2.2 Value/Metric Attacks (`attacks/valuemetric.py`)

**1. JPEG Compression:**
```python
def jpeg(img: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Apply JPEG compression."""
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed = torch.from_numpy(np.array(compressed)).float() / 255.0
    return compressed.permute(2, 0, 1).unsqueeze(0)
```
- **Effect:** DCT coefficient quantization, blocking artifacts
- **Parameters:** quality (10-100), tested at 75 and 50
- **Robustness Required:** Low-frequency watermark embedding

**2. Gaussian Noise:**
```python
def noise(img: torch.Tensor, sigma: float = 20.0) -> torch.Tensor:
    """Add Gaussian noise."""
    sigma_norm = sigma / 255.0
    noise_tensor = torch.randn_like(img) * sigma_norm
    return (img + noise_tensor).clamp(0, 1)
```
- **Effect:** Random perturbation of all pixels
- **Parameters:** sigma (5-50), default 20
- **Robustness Required:** Statistical aggregation across pixels

**3. Gaussian Blur:**
```python
def blur(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur."""
    sigma = kernel_size / 6.0
    x = torch.arange(kernel_size, dtype=torch.float32, device=img.device)
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    padded = F.pad(img, (kernel_size//2,)*4, mode='reflect')
    blurred = F.conv2d(padded, kernel_2d, groups=3)
    return blurred.clamp(0, 1)
```
- **Effect:** Low-pass filtering, loss of high-frequency details
- **Parameters:** kernel_size (3-17), default 5
- **Robustness Required:** Low-frequency watermark components

**4. Brightness Adjustment:**
```python
def brightness(img: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Adjust brightness."""
    return (img * factor).clamp(0, 1)
```
- **Effect:** Scales all pixel values uniformly
- **Parameters:** factor (0.5-2.0), default 1.5
- **Robustness Required:** Relative encoding (not absolute values)

**5. Contrast Adjustment:**
```python
def contrast(img: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Adjust contrast."""
    mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
    adjusted = (img - mean) * factor + mean
    return adjusted.clamp(0, 1)
```
- **Effect:** Expands/contracts value range around midpoint
- **Parameters:** factor (0.5-2.0), default 1.5
- **Robustness Required:** Non-linear invariant features

**6. Saturation Adjustment:**
```python
def saturation(img: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Adjust saturation."""
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)
    saturated = gray + (img - gray) * factor
    return saturated.clamp(0, 1)
```
- **Effect:** Moves colors toward/away from grayscale
- **Parameters:** factor (0.5-2.0), default 1.5
- **Robustness Required:** Luminance-based encoding

### 3.3 Optimization Algorithm

#### 3.3.1 Z-Score Metric

**Statistical Foundation:**
```python
def compute_z_score(correct_bits: int, total_bits: int = 32) -> float:
    """
    Compute z-score measuring how many standard deviations 
    the correct bit count is from random chance.
    
    Under null hypothesis (random guessing):
      - Expected correct: total_bits / 2 = 16
      - Variance: total_bits / 4 = 8
      - Std dev: sqrt(8) ≈ 2.83
    """
    expected = total_bits / 2        # 16 bits (random chance)
    std = math.sqrt(total_bits / 4)  # 2.83
    return (correct_bits - expected) / std
```

**Interpretation:**
| Z-Score | Correct Bits | Interpretation |
|---------|-------------|----------------|
| < 0 | < 16 | Below random chance |
| 0 | 16 | Random chance |
| 1 | 19 | 1 std dev above random |
| 2 | 22 | 2 std dev above random (good) |
| 3 | 25 | 3 std dev above random (excellent) |
| 4 | 28 | 4 std dev above random (near perfect) |
| 5.66 | 32 | Perfect detection |

**Statistical Significance:**
- z > 1.96: p < 0.05 (significant)
- z > 2.58: p < 0.01 (highly significant)
- z > 3.29: p < 0.001 (very highly significant)

#### 3.3.2 WAMVariant Class

**Parameter Definitions:**
```python
class WAMVariant:
    def __init__(
        self,
        scaling_w: float = 2.0,    # Watermark intensity
        scaling_i: float = 1.0,    # Image preservation
        label: str = "variant"
    ):
        self.scaling_w = scaling_w
        self.scaling_i = scaling_i
        self.label = label
```

**Parameter Semantics:**

**scaling_w (Watermark Scaling):**
- Controls strength of watermark embedding
- Higher = more robust but less invisible
- Formula: `watermark_delta = scaling_w * embedder_output`
- Range tested: 0.5 to 5.0

**scaling_i (Image Scaling):**
- Controls preservation of original image
- Lower = watermark more prominent
- Formula: `watermarked = scaling_i * original + scaling_w * watermark_delta`
- Range tested: 0.5 to 1.5

**Interaction:**
```
scaling_w ↑, scaling_i = 1.0  → Stronger watermark, same image
scaling_w ↑, scaling_i < 1.0  → Stronger watermark, less image (more distortion)
scaling_w ↓, scaling_i = 1.0  → Weaker watermark, same image
scaling_w = 2.0, scaling_i = 1.0 → Baseline balance
```

#### 3.3.3 Candidate Generation

**Initial Candidates (10 variants):**
```python
initial_candidates = [
    WAMVariant(label="baseline", scaling_w=2.0, scaling_i=1.0),
    WAMVariant(label="sw_30",    scaling_w=3.0, scaling_i=1.0),
    WAMVariant(label="sw_15",    scaling_w=1.5, scaling_i=1.0),
    WAMVariant(label="sw_25",    scaling_w=2.5, scaling_i=1.0),
    WAMVariant(label="sw_25si09",scaling_w=2.5, scaling_i=0.9),
    WAMVariant(label="sw_20si08",scaling_w=2.0, scaling_i=0.8),
    WAMVariant(label="sw_35",    scaling_w=3.5, scaling_i=1.0),
    WAMVariant(label="sw_40",    scaling_w=4.0, scaling_i=1.0),
    WAMVariant(label="sw_10",    scaling_w=1.0, scaling_i=1.0),
    WAMVariant(label="sw_05",    scaling_w=0.5, scaling_i=1.0),
]
```

**Continuation Candidates (12 per winner):**
```python
def _continuation_candidates(winner: WAMVariant) -> list:
    candidates = []
    
    # Fine sweeps on scaling_w (±0.25, ±0.5)
    for delta in [0.25, 0.5, -0.25, -0.5]:
        new_sw = winner.scaling_w + delta
        if 0.5 <= new_sw <= 5.0:
            candidates.append(WAMVariant(
                label=f"sw_{new_sw:.2f}",
                scaling_w=new_sw,
                scaling_i=winner.scaling_i,
            ))
    
    # Fine sweeps on scaling_i (±0.1)
    for delta in [0.1, -0.1]:
        new_si = winner.scaling_i + delta
        if 0.5 <= new_si <= 1.5:
            candidates.append(WAMVariant(
                label=f"si_{new_si:.2f}",
                scaling_w=winner.scaling_w,
                scaling_i=new_si,
            ))
    
    return candidates
```

#### 3.3.4 Acceptance Criteria

**Multi-Criteria Decision:**
```python
# Configuration
accept_delta = 0.10      # Minimum z-score improvement
accept_wins = 2          # Minimum attacks improved

# Accept if BOTH conditions met:
delta = candidate_z - best_z
wins = sum(1 for attack in attacks if candidate[attack] > best[attack])
accept = (delta >= accept_delta) and (wins >= accept_wins)
```

**Rationale:**
- **Delta requirement:** Ensures meaningful overall improvement
- **Wins requirement:** Ensures improvement is not just in one attack
- **Combined:** Balances robustness with generalization

#### 3.3.5 Benchmark Pipeline

**Full Benchmark Process:**
```python
def benchmark_variant(variant: WAMVariant) -> dict:
    start_time = time.time()
    attack_z_scores = {name: [] for name, _ in ATTACKS}
    
    # Apply variant parameters
    variant.apply(self.manager)
    
    try:
        for img_idx, img_file in enumerate(self.test_image_files):
            message = self.test_messages[img_idx % len(self.test_messages)]
            
            # Embed watermark
            img_tensor, _, coords = self.manager.embed(img_file, message, 'corners')
            
            # Test each attack
            for attack_name, attack_fn in ATTACKS:
                # Apply attack
                attacked = attack_fn(img_tensor)
                
                # Verify watermark
                result = self.manager.verify_tensor(attacked, message)
                
                # Compute z-score
                z = compute_z_score(result['correct_bits'], 32)
                attack_z_scores[attack_name].append(z)
    
    finally:
        variant.restore(self.manager)
    
    # Compute median z-scores per attack
    median_z = {}
    for attack_name, z_scores in attack_z_scores.items():
        median_z[attack_name] = float(np.median(z_scores))
    
    # Overall median
    all_medians = list(median_z.values())
    overall_median = float(np.median(all_medians))
    
    return {
        'variant': variant.to_dict(),
        'median_z': median_z,
        'overall_median_z': overall_median,
        'elapsed_seconds': time.time() - start_time,
    }
```

**Computation Cost:**
```
Per Variant:
  Test Images: 20
  × Attacks: 10
  × Embed + Verify: 1 operation each
  = 200 watermark operations per variant

Initial Phase:
  10 variants × 200 ops = 2,000 operations

Continuation Phase (per generation):
  12 variants × 200 ops = 2,400 operations
  
Total (5 generations):
  2,000 + (5 × 2,400) = 14,000 operations
```

### 3.4 Optimization Loop

**Pseudocode:**
```python
# Initialization
best_variant = WAMVariant(label="baseline", scaling_w=2.0, scaling_i=1.0)
best_z = compute_z_score(16)  # Random chance baseline
best_median_z = {}

# Phase 1: Initial Rounds
for round in range(initial_rounds):
    candidate = initial_candidates[round]
    result = benchmark(candidate)
    
    delta = result['overall_median_z'] - best_z
    wins = count_wins(result['median_z'], best_median_z)
    
    accept = (delta >= accept_delta) and (wins >= accept_wins)
    
    if accept or best_z == 0:  # Always accept first
        best_z = result['overall_median_z']
        best_variant = candidate
        best_median_z = result['median_z']
        print(f"✓ ACCEPTED: z={best_z:.2f} (delta={delta:+.2f}, wins={wins})")
    else:
        print(f"✗ REJECTED: z={result['overall_median_z']:.2f}")
    
    results.append(result)

# Phase 2: Continuation Generations
for gen in range(generations):
    candidates = generate_continuation_candidates(best_variant)
    
    for candidate in candidates:
        result = benchmark(candidate)
        
        delta = result['overall_median_z'] - best_z
        wins = count_wins(result['median_z'], best_median_z)
        
        accept = (delta >= accept_delta) and (wins >= accept_wins)
        
        if accept:
            best_z = result['overall_median_z']
            best_variant = candidate
            best_median_z = result['median_z']
            print(f"✓ ACCEPTED: z={best_z:.2f} (delta={delta:+.2f}, wins={wins})")
        else:
            print(f"✗ REJECTED: z={result['overall_median_z']:.2f}")
        
        results.append(result)

# Output
print(f"OPTIMIZATION COMPLETE")
print(f"Best z-score: {best_z:.2f}")
print(f"Best variant: {best_variant.label}")
print(f"Params: {best_variant.to_dict()}")
```

### 3.5 Results Storage

**JSON Log Structure:**
```json
[
  {
    "variant": {
      "label": "sw_25",
      "scaling_w": 2.5,
      "scaling_i": 1.0
    },
    "median_z": {
      "clean": 3.45,
      "jpeg_75": 2.87,
      "jpeg_50": 1.92,
      "noise_20": 2.15,
      "crop_75": 2.67,
      "crop_50": 1.45,
      "resize_05": 2.34,
      "rotate_15": 1.78,
      "blur_5": 2.01
    },
    "overall_median_z": 2.15,
    "elapsed_seconds": 45.67
  }
]
```

---

## 4. System Integration & Data Flow

### 4.1 Component Dependencies

```
app.py (Flask API)
    ├── core.py (WatermarkManager)
    │       ├── roco_core.py (Message encoding)
    │       ├── roco_ecc.py (Error correction)
    │       ├── viewframe.py (Corner detection)
    │       └── watermark_utils.py (Helper functions)
    │
    └── watermark_utils.py
            └── notebooks/inference_utils.py
                    ├── watermark_anything/models/wam.py
                    │       ├── embedder.py (VAE-based)
                    │       ├── extractor.py (SAM-based)
                    │       └── augmentation/augmenter.py
                    │
                    └── watermark_anything/modules/
                            ├── vae.py (VAE components)
                            ├── vit.py (Vision Transformer)
                            ├── pixel_decoder.py
                            ├── msg_processor.py
                            └── jnd.py (Just Noticeable Distortion)

optimizer/wam_optimizer.py (Self-optimizing)
    ├── core.py (WatermarkManager)
    ├── attacks/geometric.py (Crop, Resize, Rotate, Flip)
    └── attacks/valuemetric.py (JPEG, Noise, Blur, etc.)
```

### 4.2 Data Flow Diagrams

**Watermark Embedding Flow:**
```
User Image (任意尺寸)
    ↓
Preprocess (center crop to square)
    ↓
Crop to Viewframe ROI
    ↓
Resize to 256×256
    ↓
ROCO Encode ("ABC" → 32-bit tensor)
    ↓
WAM Embed:
  ├── VAE Encoder (256×256 → 4ch latent)
  ├── Msg Processor (32 bits → 64ch embedding)
  ├── Concat (4ch + 64ch = 68ch)
  └── VAE Decoder (68ch → 256×256 watermarked)
    ↓
Resize back to ROI size
    ↓
Place into original image
    ↓
Draw corner brackets (visible marker)
    ↓
Return watermarked image + metadata
```

**Watermark Verification Flow:**
```
Watermarked Image
    ↓
Preprocess (center crop to square)
    ↓
Detect Viewframe Corners (value=255 threshold)
    ↓
Crop to Detected ROI
    ↓
Resize to 256×256
    ↓
WAM Detect:
  ├── ViT Encoder (256×256 → features)
  └── Pixel Decoder (features → 33ch output)
    ↓
Separate Mask + Bit Predictions
    ↓
Aggregate Bits (semihard inference)
    ↓
ROCO Decode (32-bit → message)
    ↓
ECC Correction (BCH t=2)
    ↓
Return decoded message + statistics
```

### 4.3 Configuration Files

**Model Configuration (`checkpoints/params.json`):**
```json
{
  "embedder_config": "configs/embedder.yaml",
  "extractor_config": "configs/extractor.yaml",
  "augmentation_config": "configs/all_augs_multi_wm.yaml",
  "attenuation_config": "configs/attenuation.yaml",
  "nbits": 32,
  "img_size": 256,
  "scaling_w": 2.0,
  "scaling_i": 1.0,
  "roll_probability": 0.2
}
```

**Training Augmentations (`configs/all_augs_multi_wm.yaml`):**
```yaml
augs:
  identity: 1
  jpeg: 1            # Quality 40-80
  resize: 1          # 0.7-1.5x
  crop: 1            # 0.33-1.0x
  rotate: 1          # -10 to 10 degrees
  hflip: 1
  perspective: 1     # 0.1-0.5 distortion
  gaussian_blur: 1   # Kernel 3-17
  median_filter: 1   # Kernel 3-7
  brightness: 1      # 0.5-2x
  contrast: 1        # 0.5-2x
  saturation: 1      # 0.5-2x
  hue: 1             # -0.1 to 0.1
```

---

## 5. Strengths, Weaknesses, and Recommendations

### 5.1 Strengths

#### Watermarking System

**1. Localized Watermarking**
- Viewframe detection enables automatic ROI identification
- Corner brackets provide visual marker for human verification
- Can detect and extract watermark even if image is partially modified
- ROI-based approach allows watermark placement flexibility

**2. Robust Message Encoding**
- ROCO encoding with 5-bit character mapping (32 characters)
- BCH ECC (t=2) corrects up to 2 bit errors
- 32-bit total codeword (16 data + 16 ECC)
- Supports alphanumeric + special characters

**3. Strong Detection Architecture**
- SAM-based extractor with ViT backbone
- Per-pixel bit predictions aggregated for robustness
- Multiple inference methods (hard, semihard, soft)
- Mask-guided attention for focused detection

**4. Perceptual Quality Preservation**
- JND-based attenuation adapts to image content
- Configurable scaling_w/scaling_i for quality/robustness trade-off
- Can tune watermark strength independently of image preservation

**5. Flexible Deployment**
- Flask API for web integration
- Command-line tools (mark.py, verify.py)
- Configurable mask modes (corners, pixels, percentage)

#### Self-Optimizing Mechanism

**1. Systematic Evaluation**
- 10 canonical attack types covering geometric and value distortions
- Statistical z-score metric for performance comparison
- Median aggregation per attack type (robust to outliers)

**2. Evolutionary Search**
- Initial broad exploration (10 variants)
- Fine-grained continuation (12 candidates per generation)
- Multi-criteria acceptance (delta + wins)

**3. Transparent Results**
- JSON log with per-attack z-scores
- Variant parameters clearly documented
- Timing metrics for performance evaluation

### 5.2 Weaknesses

#### Watermarking System

**1. Viewframe Dependency**
- Corner brackets must be visible for automatic detection
- Severe cropping or rotation may hide all corners
- Fallback to centered square reduces accuracy
- No recovery mechanism for destroyed viewframes

**2. Fixed Resolution**
- All processing at 256×256
- Bilinear interpolation may blur fine details
- No multi-scale processing
- High-resolution images lose detail

**3. Message Length Limit**
- Maximum 3 characters in allowed alphabet
- No support for longer messages
- Limited character set (A-Z, 4,6,7,9, ., #)
- Cannot encode URLs, hashes, or meaningful identifiers

**4. No Perceptual Metrics Output**
- PSNR not calculated in verify()
- No SSIM or LPIPS for perceptual quality
- Viewer cannot assess watermark invisibility
- Missing quality feedback loop

#### Self-Optimizing Mechanism

**1. Limited Attack Coverage**
- No deep learning attacks (model distillation, adversarial examples)
- No combined attacks (rotate + JPEG + crop)
- Fixed attack parameters (no ranges)
- No perceptual attacks (cropping based on content)

**2. Narrow Parameter Space**
- Only tuning scaling_w and scaling_i
- No exploration of other WAM parameters
- No architecture-level changes
- Limited optimization scope

**3. Static Test Dataset**
- Random images may not represent real-world content
- No diversity in image characteristics
- No real image dataset integration
- Limited generalization testing

**4. No Early Stopping**
- Runs full optimization regardless of improvement
- No convergence detection
- Wasteful computation on diminishing returns

**5. No Robustness to Detection Attacks**
- Does not test against viewframe removal
- No testing for false positives
- No adversarial training on detector
- Single-objective optimization

### 5.3 Recommendations

#### High Priority

**1. Expand Attack Suite**
```python
# Add to attacks/__init__.py
from .geometric import perspective, shear, elastic
from .valuemetric import sharpen, median, contrast_stretch, histogram_eq
from .combined import rotate_jpeg_crop  # Combined attacks
```

**2. Add Perceptual Metrics**
```python
# In core.py verify() method
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def verify(self, image_source, original_message, viewframe_coords=None):
    # ... existing code ...
    results['psnr'] = psnr(img_np, unwatermarked_np)
    results['ssim'] = structural_similarity(img_np, unwatermarked_np, channel_axis=2)
```

**3. Improve Viewframe Detection**
```python
# In viewframe.py
# Add Hough line detection for partial corners
# Add contour-based detection for corrupted corners
# Add fallback detection strategies
```

**4. Support Longer Messages**
```python
# In roco_core.py
# Add support for multi-message encoding
# Or increase payload to 6 characters with higher ECC overhead
```

#### Medium Priority

**5. Multi-Resolution Processing**
- Pyramid-based detection for robustness to scaling
- Scale-invariant feature detection

**6. Real-Image Test Dataset**
```python
# In wam_optimizer.py
# Replace random images with COCO or LAION subset
# Ensure diversity in content, texture, colors
```

**7. Combined Attack Testing**
```python
# In attacks/
# Add composite attacks: rotate(jpeg(crop(img)))
# More realistic attack scenarios
```

**8. Early Stopping Mechanism**
```python
# In WAMOptimizer.run()
# Track improvement over last N generations
# Stop if no significant improvement for K generations
```

#### Low Priority

**9. Configuration File for Attacks**
- Load attack parameters from YAML/JSON
- Easy to add/remove attacks without code changes

**10. Visual Benchmark Dashboard**
- Streamlit or Dash app for real-time optimization monitoring
- Per-attack performance charts

**11. Model Checkpoint Evolution**
- Save best model configuration
- Allow loading optimized parameters directly

**12. API Integration**
- Webhook callback on optimization complete
- Batch processing for multiple images

---

## 6. Appendix

### 6.1 Key Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Z-Score | (correct_bits - 16) / 2.83 | Std devs above random |
| Bit Accuracy | correct_bits / 32 | Raw accuracy |
| BER | bitflips / 32 × 100% | Error percentage |
| PSNR | 20 log10(255) - 10 log10(MSE) | Image quality (dB) |
| IoU | intersection / union | Mask overlap |

### 6.2 Parameter Ranges Tested

| Parameter | Min | Max | Default | Step |
|-----------|-----|-----|---------|------|
| scaling_w | 0.5 | 5.0 | 2.0 | 0.25 |
| scaling_i | 0.5 | 1.5 | 1.0 | 0.1 |

### 6.3 Attack Parameters

| Attack | Parameter | Values Tested |
|--------|-----------|---------------|
| JPEG | quality | 75, 50 |
| Crop | ratio | 0.75, 0.50 |
| Resize | scale | 0.5 |
| Rotate | angle | 15° |
| Noise | sigma | 20 |
| Blur | kernel_size | 5 |
| Brightness | factor | 1.5 |
| Contrast | factor | 1.5 |
| Saturation | factor | 1.5 |

### 6.4 File Structure Summary

```
watermark-freedom/
├── app.py                    # Flask API server
├── core.py                   # WatermarkManager class
├── mark.py                   # CLI watermarking tool
├── verify.py                 # CLI verification tool
├── viewframe.py              # Viewframe detection/drawing
├── watermark_utils.py        # Utility functions
├── roco_core.py              # Message encoding/decoding
├── roco_ecc.py               # BCH error correction
│
├── configs/
│   ├── embedder.yaml         # VAE embedder config
│   ├── extractor.yaml        # SAM extractor config
│   ├── all_augs_multi_wm.yaml# Training augmentations
│   └── attenuation.yaml      # JND attenuation config
│
├── checkpoints/
│   ├── wam_mit.pth          # Model weights
│   └── params.json           # Model parameters
│
├── optimizer/
│   ├── wam_optimizer.py      # Self-optimizing mechanism
│   └── improvement_log.json  # Previous test results
│
├── attacks/
│   ├── __init__.py           # Attack suite exports
│   ├── geometric.py          # Geometric attacks
│   └── valuemetric.py        # Value/metric attacks
│
└── watermark_anything/
    ├── models/
    │   ├── wam.py            # WAM main model
    │   ├── embedder.py       # VAE embedder
    │   └── extractor.py      # SAM extractor
    ├── modules/
    │   ├── vae.py            # VAE components
    │   ├── vit.py            # Vision Transformer
    │   ├── pixel_decoder.py  # Pixel decoder
    │   ├── msg_processor.py  # Message processor
    │   └── jnd.py            # JND attenuation
    ├── augmentation/
    │   └── augmenter.py      # Training augmentations
    └── data/
        └── metrics.py        # Evaluation metrics
```

---

*End of Report*
