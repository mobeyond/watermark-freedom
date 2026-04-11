# Updated Procedures Summary: Odd-Length Square Cropping with Center Preservation

## Overview

Updated the centered square cropping procedure to ensure **ODD side lengths** for proper center pixel alignment, while **preserving the original center point** within the cropped region.

### Key Principles

1. **Odd Side Length**: Output always has an odd side length for a true center pixel
2. **Center Preservation**: The original geometric center is always contained within the crop
3. **Symmetric Cropping**: Extra pixels (when needed) are removed from bottom/right edges

---

## Modified Files

### 1. `watermark_utils.py` - `crop_to_centered_square()`

**Location:** Lines 26-54

**Updated Logic:**
```python
def crop_to_centered_square(image):
    """Returns the largest centered square crop with ODD side length.

    Works with both PIL Image and numpy arrays (cv2 format).

    Center preservation:
    - If original min_dim is ODD: crops to that size (perfect center alignment)
    - If original min_dim is EVEN: crops to (min_dim - 1), center preserved within crop

    The original center point is always contained within the cropped region.
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    min_dim = min(h, w)

    # Ensure odd side length for proper center pixel alignment
    if min_dim % 2 == 0:
        min_dim -= 1

    # Calculate offsets to crop from each side
    # Extra pixel (if any) goes to bottom/right to preserve center
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2

    if isinstance(image, np.ndarray):
        return image[top:top+min_dim, left:left+min_dim]
    else:
        return image.crop((left, top, left + min_dim, top + min_dim))
```

---

### 2. `backends/base_backend.py` - `_crop_to_centered_square()`

**Location:** Lines 50-68

**Updated Logic:**
```python
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
    return image[top:top+min_dim, left:left+min_dim]
```

---

## Complete Processing Pipeline

### Embedding Process

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOAD IMAGE                                                  │
│    Input: Any rectangular image (W × H)                        │
│    Example: 300 × 256                                         │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CROP TO CENTERED SQUARE (ODD SIDE LENGTH)                   │
│    min_dim = min(W, H)                                         │
│    If min_dim is EVEN: min_dim -= 1                            │
│                                                                │
│    Example:                                                    │
│      Input:  300 × 256                                         │
│      min_dim: 256 → 255 (made odd)                             │
│      Output: 255 × 255 (square with odd side length)           │
│                                                                │
│    Center pixel is at: (127, 127) for 255×255 image           │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CALCULATE VIEWFRAME COORDINATES                              │
│    margin = int(min_dim × margin_pct)                          │
│                                                                │
│    Example (10% margin on 255×255):                            │
│      margin = int(255 × 0.10) = 25                             │
│      viewframe: x=25, y=25, width=205, height=205             │
│      (viewframe also has ODD side length: 205)                │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. APPLY PADDING FOR EMBED REGION                               │
│    padding = 4 (to exclude bracket line pixels)                │
│                                                                │
│    Example:                                                    │
│      embed_x = 25 + 4 = 29                                     │
│      embed_y = 25 + 4 = 29                                     │
│      embed_width = 205 - 8 = 197                               │
│      embed_height = 205 - 8 = 197                              │
│      (embed region also has ODD side length: 197)             │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. CROP TO EMBED REGION                                        │
│    Extract region at (29, 29) with size 197×197               │
│                                                                │
│    Center pixel is at: (98, 98) for 197×197 embed region      │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. EMBED WATERMARK (backend-specific)                           │
│    - VideoSeal: Embed directly at flexible resolution          │
│    - WAM: Resize to 256×256, embed, resize back               │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. PLACE WATERMARK BACK                                         │
│    Copy watermarked region back to (29, 29) position           │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. DRAW CORNER BRACKETS                                        │
│    Draw L-shaped brackets at viewframe corners (25, 25)        │
│    Corner length: ~30 pixels (15% of 205)                      │
│    Line thickness: 2-3 pixels                                  │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. OUTPUT FINAL IMAGE                                          │
│    Save as PNG                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Verification Process

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOAD WATERMARKED IMAGE                                       │
│    Input: Watermarked image (any size)                         │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CROP TO CENTERED SQUARE (ODD SIDE LENGTH)                   │
│    Same as embedding: ensures ODD side length                  │
│    Example: 300×256 → 255×255                                 │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DETECT VIEWFRAME (AUTO-DETECTION)                            │
│    Detect corner brackets using diagonal detection             │
│                                                                │
│    If detection succeeds:                                      │
│      - Use detected coordinates                                │
│      - Sets coords['detected'] = True                          │
│                                                                │
│    If detection fails:                                         │
│      - Fall back to default 10% margin                         │
│      - Sets coords['detected'] = False                         │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. LOG DETECTED VIEWFRAME (EXACT, NO PADDING)                  │
│    Save image at exact detected position                       │
│    Example: 205×205 at (25, 25)                                │
│    File: {session_id}_02_detected_viewframe.png               │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. APPLY PADDING FOR EXTRACT REGION                             │
│    padding = 4 (same as embedding)                             │
│                                                                │
│    Example:                                                    │
│      extract_x = 25 + 4 = 29                                   │
│      extract_y = 25 + 4 = 29                                   │
│      extract_width = 205 - 8 = 197                             │
│      extract_height = 205 - 8 = 197                            │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. CROP TO EXTRACT REGION                                       │
│    Extract region at (29, 29) with size 197×197               │
│    File: {session_id}_03_extracted_viewframe.png              │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. DETECT WATERMARK (backend-specific)                          │
│    - VideoSeal: Run model.detect() on extracted region         │
│    - WAM: Resize, detect, decode with ROCO                     │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. DECODE MESSAGE                                              │
│    ROCO decoding with ECC correction                           │
│    Returns: (message, is_valid, bitflips_corrected)           │
└─────────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. OUTPUT VERIFICATION RESULTS                                 │
│    - Decoded message                                           │
│    - ECC validity                                              │
│    - Bit accuracy (if original message provided)               │
│    - Viewframe coordinates (detected or fallback)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Properties

### Odd Side Length Guarantees

| Stage | Input (Example) | Output (Example) | Center Pixel |
|-------|-----------------|------------------|--------------|
| Original | 300 × 256 | - | (128.0, 150.0) |
| Cropped Square | - | 255 × 255 | (127, 127) |
| Viewframe | - | 205 × 205 | (102, 102) |
| Embed/Extract Region | - | 197 × 197 | (98, 98) |

### Center Preservation

**Original center is always contained within the crop:**

| Input | Original Center | Crop Bounds | Center in Crop? |
|-------|----------------|-------------|-----------------|
| 256×256 | (128.0, 128.0) | [0, 255) × [0, 255) | ✅ Yes |
| 255×255 | (127.5, 127.5) | [0, 255) × [0, 255) | ✅ Yes |
| 300×256 | (128.0, 150.0) | [0, 255) × [22, 277) | ✅ Yes |
| 300×300 | (150.0, 150.0) | [0, 299) × [0, 299) | ✅ Yes |

### Why Odd Side Length?

1. **True Center Pixel**: An odd-sized image has a single center pixel
   - 255 × 255 → center at (127, 127)
   - Even-sized (256 × 256) → no true center (between 127 and 128)

2. **Symmetric Operations**: Operations that expand/contract from center work cleanly
   - Margin calculations: `center ± offset`
   - Padding: symmetric on both sides

3. **Consistent Alignment**: Viewframe and embed regions maintain alignment
   - All regions have odd side lengths
   - Center pixels align perfectly

4. **Center Preservation**: Original center point is always within the crop
   - Important for maintaining image context
   - Ensures watermark is placed near original center

---

## Example: Complete Flow with Numbers

### Input Image: 300 × 256

**Step 1: Crop to Square (Odd)**
```
min_dim = min(300, 256) = 256
256 is even → use 255
Crop: left=22, top=0, size=255×255
Result: 255 × 255 (center at 127, 127)
```

**Step 2: Calculate Viewframe (10% margin)**
```
margin = int(255 × 0.10) = 25
viewframe: x=25, y=25, width=205, height=205
Result: 205 × 205 (center at 102, 102 relative to viewframe)
```

**Step 3: Calculate Embed Region (4px padding)**
```
embed_x = 25 + 4 = 29
embed_y = 25 + 4 = 29
embed_width = 205 - 8 = 197
embed_height = 205 - 8 = 197
Result: 197 × 197 (center at 98, 98 relative to embed region)
```

**Step 4: Bracket Calculation**
```
corner_length = int(205 × 0.15) = 30
line_thickness = max(2, int(205 × 0.012)) = 2
Brackets drawn at: (25, 25), (25, 230), (230, 25), (230, 230)
```

---

## Logging Files Generated

### During Embedding
```
/tmp/videoseal_logging/embed/
├── 01_input.png                 Original image
├── 02_cropped_square.png       Cropped to odd square (255×255)
├── 03_viewframe_before_embed.png  Viewframe region (205×205)
├── 04_viewframe_after_embed.png   Watermarked viewframe
├── 05_combined_with_watermark.png Combined image
└── 06_final_with_brackets.png     Final output
```

### During Verification
```
/tmp/videoseal_logging/verify/
├── 01_watermarked.png             Original watermarked image
├── 02_cropped_square.png          Cropped to odd square
├── {id}_02_detected_viewframe.png  DETECTED viewframe (exact, no padding)
└── {id}_03_extracted_viewframe.png Extracted region (with padding)
```

---

## Benefits

1. **Precise Centering**: Every region has a true center pixel
2. **Symmetric Margins**: Margin calculations work perfectly
3. **Consistent Behavior**: Same results regardless of input size parity
4. **Easier Debugging**: Center pixels are easily identifiable
5. **Better Alignment**: All regions (square, viewframe, embed) align perfectly
