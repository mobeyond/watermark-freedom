# GPU Acceleration Setup for Watermark Freedom

## Overview

The VideoSeal backend has been adapted to support GPU acceleration using CUDA. This document describes the changes made and how to use GPU-accelerated watermarking.

## Changes Made

### 1. Backend Changes (`backends/videoseal_backend.py`)

Added a `force_cpu` parameter to `VideoSealBackend.__init__()`:

```python
def __init__(
    self,
    device: Optional[torch.device] = None,
    margin_percent: float = 0.10,
    custom_model_path: Optional[str] = None,
    force_cpu: bool = False,  # New parameter - set to False for GPU
):
```

- **Default behavior**: `force_cpu=False` allows GPU usage if CUDA is available
- **Backward compatibility**: Set `force_cpu=True` to force CPU-only operation

### 2. GPU Test Script (`test_app_flow_gpu.py`)

A new GPU-enabled test script that:
- Extends `VideoSealBackend` with GPU support
- Tests margin percentage parameter with GPU acceleration
- Reports GPU memory usage
- Shows device information

### 3. Performance Benchmark (`test_app_flow_gpu_benchmark.py`)

A comparison script that benchmarks CPU vs GPU performance.

## Usage

### Basic GPU Usage

```python
from backends.videoseal_backend import VideoSealBackend

# Enable GPU (default behavior when force_cpu=False)
wm = VideoSealBackend(force_cpu=False)

# Or explicitly set device
import torch
wm = VideoSealBackend(device=torch.device("cuda"), force_cpu=False)
```

### Running GPU Tests

```bash
# Test margin parameter with GPU
python3 test_app_flow_gpu.py abnormal/seabackground.jpg

# Benchmark CPU vs GPU performance
python3 test_app_flow_gpu_benchmark.py abnormal/seabackground.jpg
```

### Example Output

```
Using device: cuda
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 2060 SUPER
[VideoSealBackendGPU] Using device: cuda

Testing margin_pct parameter on GPU (abnormal/seabackground.jpg, 470x470):
Device: cuda
----------------------------------------------------------------------
Margin    5% → viewframe 412 (actual 6.2%) | ✓
  Expected: 'ABC' | Decoded: 'ABC' | ECC: True
  Bit accuracy: 100.0%
----------------------------------------------------------------------
```

## Performance Notes

### Current Implementation

The `embed_bytes()` and `verify_bytes()` methods use subprocess calls for Python version compatibility. The GPU acceleration is most beneficial for:

1. **Direct `embed()` / `verify()` methods** - These run entirely on GPU
2. **Batch processing** - Multiple operations benefit from GPU parallelization

### Expected Speedup

- **Single operation**: Minimal speedup (subprocess overhead dominates)
- **Batch processing**: 5-10x speedup possible
- **Large images**: More significant gains due to parallel processing

## GPU Requirements

- **CUDA**: Version 11.0 or higher
- **GPU Memory**: At least 2GB VRAM recommended
- **PyTorch**: Built with CUDA support (`torch.cuda.is_available()` returns `True`)

## Checking GPU Availability

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
```

## Troubleshooting

### Issue: GPU not being used

**Solution**: Check that `force_cpu=False`:
```python
wm = VideoSealBackend(force_cpu=False)
print(f"Device: {wm.device}")  # Should show 'cuda'
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or clear GPU cache:
```python
import torch
torch.cuda.empty_cache()
```

### Issue: Slow performance

**Solution**: Use direct methods instead of `embed_bytes()`:
```python
# Instead of:
img_out, binary, coords = wm.embed_bytes(img_bytes, "ABC")

# Use:
from PIL import Image
img = Image.open(img_path)
img_out, binary, coords = wm.embed(img, "ABC")
```

## Future Improvements

1. Update subprocess scripts to use GPU
2. Add batch processing support for better GPU utilization
3. Implement mixed precision training for faster inference
4. Add GPU memory management for large-scale processing
