# VideoSeal Backend TorchScript Migration

## Overview

This document describes the modifications made to switch the VideoSeal backend from using the full PyTorch model to the TorchScript model.

## Changes Made

### 1. Model Loading (`_get_model` method)

**Before:**
```python
def _get_model(self):
    import videoseal
    self._model = videoseal.load("videoseal_1.0")
```

**After:**
```python
def _get_model(self):
    import torch
    self._model = torch.jit.load(self._jit_model_path, map_location=str(self.device))
    self._model.eval()
```

**Key changes:**
- Uses `torch.jit.load()` instead of `videoseal.load()`
- Loads from pre-compiled `.jit` file instead of `.pth` checkpoint
- No dependency on `videoseal` package

### 2. Device Configuration

**Added CPU-only mode to avoid CUDA version mismatch:**
```python
def __init__(self, ...):
    # Force CPU for TorchScript to avoid CUDA version mismatch
    self._force_cpu = True
    if self._force_cpu:
        self.device = torch.device("cpu")
    else:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 3. Embed Method

**Changes to handle TorchScript return types:**

- TorchScript `embed()` returns a `torch.Tensor` directly (not a dict)
- Message tensor must be `torch.float32` (not `torch.long`)
- 32-bit messages are expanded to 256 bits by repeating 8 times

```python
# Extend 32 bits to 256 bits for TorchScript model
if self._n_bits == 32:
    msg_bits_256 = msg_bits * 8  # Repeat 32 bits 8 times = 256 bits
    msg_tensor = torch.tensor([msg_bits_256], dtype=torch.float32, device=self.device)
else:
    msg_tensor = torch.tensor([msg_bits], dtype=torch.float32, device=self.device)
```

**Handle different return types:**
```python
result = model.embed(img_tensor, msg_tensor)
if isinstance(result, torch.Tensor):
    watermarked = result
elif isinstance(result, tuple):
    watermarked = result[0]
else:
    watermarked = result["imgs_w"]
```

### 4. Verify Method

**Changes for detection:**

- TorchScript `detect()` returns tensor directly (not dict)
- Added majority voting for 32-bit mode

```python
# Detect
detected = model.detect(img_tensor)
if isinstance(detected, torch.Tensor):
    preds = detected
else:
    preds = detected["preds"]

# Majority voting for 32-bit mode
if self._n_bits == 32 and len(all_bits) >= 256:
    blocks = [all_bits[i*32:(i+1)*32] for i in range(8)]
    msg_bits = [1 if sum(block[bit_idx] for block in blocks) >= 4 else 0 for bit_idx in range(32)]
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `_use_torchscript` | `True` | Use TorchScript instead of full model |
| `_jit_model_path` | `/home/h/FLY/videoseal/ckpts/y_256b_img.jit` | Path to TorchScript model |
| `_force_cpu` | `True` | Force CPU to avoid CUDA version mismatch |

## Advantages of TorchScript

1. **No dependency issues** - Doesn't require `av`, `decord`, `ffmpeg` packages
2. **Faster loading** - Pre-compiled model loads instantly
3. **Standalone** - Single `.jit` file, no config/YAML needed
4. **Production-ready** - Simpler, less error-prone

## Troubleshooting

### CUDA Version Mismatch

If you see: `RuntimeError: Expected all tensors to be on the same device`

**Solution:** Keep `_force_cpu = True` (default)

### Low Detection Accuracy

If bit accuracy is low:
- Ensure message is repeated 8 times for 32-bit payloads
- Use majority voting in verify (already implemented)
- Check that the same image processing is used for embed and verify

## File Changed

- `backends/videoseal_backend.py`