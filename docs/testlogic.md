Executive Summary

Apply the iterative optimization mechanism from powerplant/ss_improver.py to optimize the WAM (Watermark Anything Model) parameters in watermark-freedom for improved robustness against
attacks.

Background

Current State

watermark-freedom (WAM-based):
- VAE encoder-decoder with ViT detector
- 32-bit message capacity with BCH ECC
- Localized embedding with viewframe detection
- Parameters: scaling_w=2.0, scaling_i=1.0, JND attenuation

powerplant (Spread-Spectrum):
- SS codec with z-score optimization
- 12 attack types tested
- Final z-score: 9.38 (from 2.12 baseline)
- Key breakthroughs: prewhitening, alpha tuning, rotation search

Key Insight

The WAM model is fundamentally different from spread-spectrum:
- WAM: Learned neural network embedding
- SS: Carrier signal correlation

However, the optimization methodology can be adapted:
1. Attack suite for robustness testing
2. Hill-climbing with accept/reject logic
3. Z-score metric for bit accuracy
4. Parameter sweeps for optimal configuration

Proposed Architecture

1. Attack Suite (adapted from powerplant)

│ Attack             │ Method                │ Decode Strategy     │
├────────────────────┼───────────────────────┼─────────────────────┤
│ `clean`            │ None                  │ Standard            │
│ `jpeg_75/50/25`    │ JPEG compression      │ Standard            │
│ `noise_20/30`      │ Gaussian noise        │ Standard            │
│ `crop_75/50`       │ Center crop + upscale │ Multi-region search │
│ `resize_05/025`    │ Downscale + upscale   │ Multi-scale decode  │
│ `rotate_15/30`     │ Rotation              │ Rotation search     │
│ `brightness_05/20` │ Brightness adjustment │ Standard            │
│ `contrast_05/20`   │ Contrast adjustment   │ Standard            │
│ `blur_3/5`         │ Gaussian blur         │ Standard            │

2. Optimizable Parameters

Embedding Strength:
│ Parameter   │ Current │ Range   │ Impact             │
├─────────────┼─────────┼─────────┼────────────────────┤
│ `scaling_w` │ 2.0     │ 0.5-5.0 │ Watermark strength │
│ `scaling_i` │ 1.0     │ 0.5-1.5 │ Image preservation │

JND Attenuation:
│ Parameter │ Current │ Range    │ Impact           │
├───────────┼─────────┼──────────┼──────────────────┤
│ `alpha`   │ 1.0     │ 0.5-2.0  │ JND scale        │
│ `beta`    │ 0.117   │ 0.05-0.3 │ Contrast masking │

Message Processing:
│ Parameter  │ Current │ Range   │ Impact                     │
├────────────┼─────────┼─────────┼────────────────────────────┤
│ `msg_mult` │ 1.0     │ 0.5-3.0 │ Message embedding strength │

Viewframe:
│ Parameter             │ Current │ Range     │ Impact                │
├───────────────────────┼─────────┼───────────┼───────────────────────┤
│ `corner_length_ratio` │ 0.15    │ 0.05-0.25 │ Detection reliability │
│ `line_thickness`      │ 3       │ 1-5       │ Detection accuracy    │

3. Z-Score Metric for WAM

  # WAM uses 32-bit messages
  total_bits = 32
  expected = total_bits / 2  # 16 (random bits)
  std = math.sqrt(total_bits / 4)  # ~2.83
  z = (correct_bits - expected) / std

For 32-bit messages:
- z=9 requires ~29/32 correct (p < 1e-19)
- z=5 requires ~25/32 correct (p < 1e-6)
- z=2 requires ~20/32 correct (p < 0.05)

4. Accept/Reject Logic

  ACCEPT_DELTA = 0.10  # minimum z improvement
  ACCEPT_WINS = 2      # must improve at least 2 attacks

  accept = (delta_z >= ACCEPT_DELTA) and (wins >= ACCEPT_WINS)

5. Optimization Flow

  1. Load test images (100 random images)
  2. For each candidate configuration:
     a. Embed watermark with candidate params
     b. Apply each attack type
     c. Decode and compute z-score per attack
     d. Compute median z-score
     e. Accept if improves median z AND wins >= 2 attacks
  3. Write winner params to config file
  4. Repeat for next generation

