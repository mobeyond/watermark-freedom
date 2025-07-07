# Project Structure

## Main Components

- **app.py**: Flask application for watermarking and verification
  - Imports: Flask, PIL, torch, cv2, numpy, various custom modules
  - Functions:
    - `create_fixed_mask()`: Creates a fixed mask based on percentage coordinates
    - `get_viewframe_overlay_and_inner_square()`: Gets viewframe overlay and inner square region
  - Routes:
    - `/`: Serves the index.html template
    - `/watermark` (POST): Endpoint to watermark an image
      - Accepts cover image, message, mask parameters
      - Processes image and applies watermark using model
      - Returns watermarked image as downloadable PNG
    - `/verify` (POST): Endpoint to verify a watermarked image
      - Accepts watermarked image, mask parameters
      - Detects and extracts embedded message
      - Returns verification results with confidence

- **mark.py**: Command-line tool for watermarking images
  - Imports: PIL, torch, cv2, numpy, argparse, various custom modules
  - Functions:
    - `crop_to_centered_square()`: Crops image to centered square
    - `calculate_checksum()`: Calculates checksum for error detection
    - `robust_str_to_binary()`: Converts string message to binary tensor with error correction
    - `create_mask_from_pixels()`: Creates mask from pixel coordinates
    - `process_image()`: Processes image with watermarking based on mask type
  - Main functionality:
    - Parses command-line arguments for input image, message, and mask parameters
    - Calls `process_image()` to apply watermark and save output

- **verify.py**: Command-line tool for verifying watermarks in images
  - Imports: PIL, torch, argparse, various custom modules
  - Functions:
    - None (main functionality directly in main())
  - Main functionality:
    - Parses command-line arguments for input image and mask parameters
    - Loads model and processes image to extract embedded message
    - Displays verification results with confidence score

## Shared Functionality

- All scripts use the following shared components:
  - Model loading from checkpoint: `load_model_from_checkpoint()`
  - Image transformations: `default_transform()`, `unnormalize_img()`
  - Message processing: `msg_predict_inference()`, `msg2str()`

## Workflow Overview

1. **Watermarking**:
   - Use `mark.py` CLI tool or `/watermark` endpoint in `app.py`
   - Provide input image, message, and mask parameters
   - Output is watermarked image with embedded message

2. **Verification**:
   - Use `verify.py` CLI tool or `/verify` endpoint in `app.py`
   - Provide watermarked image and mask parameters
   - Output includes extracted message and confidence score




+-------------------------------------------------------+
|                   Top-level Scripts                   |
+-------------------------------------------------------+
| app.py:                                              |
|   - index()                                           |
|   - watermark_image()                                 |
|   - verify_watermark()                                |
|                                                       |
| mark.py:                                             |
|   - process_image()                                   |
|   - main()                                            |
|                                                       |
| train.py:                                            |
|   - get_parser()                                      |
|   - main()                                            |
|   - train_one_epoch()                                 |
|   - eval_full()                                       |
|   - eval_full_kwm()                                   |
|                                                       |
| verify.py:                                           |
|   - main()                                            |
|                                                       |
| viewframe.py:                                        |
|   - get_inner_square_region()                         |
|   - draw_viewframe_overlay()                          |
+-------------------------------------------------------+

+----------------------------+
|    Watermark Anything      |
|       Core Modules         |
+----------------------------+
| models/                    |
|   - Wam class              |
|     - forward()            |
|     - embed()              |
|     - detect()             |
|   - Embedder class         |
|     - get_random_msg()     |
|     - forward()            |
|   - Extractor class        |
|     - forward()            |
+----------------------------+
| modules/                   |
|   - MsgProcessor class     |
|     - forward()            |
|   - JND class              |
|     - forward()            |
|   - PixelDecoder class     |
|     - forward()            |
|   - Discriminator classes  |
|     - forward()            |
+----------------------------+
| augmentation/              |
|   - Augmenter class        |
|     - forward()            |
|   - Geometric transforms   |
|     - Identity             |
|     - Rotate               |
|     - Resize               |
|     - Crop                 |
|     - HorizontalFlip       |
+----------------------------+
| losses/                    |
|   - PerceptualLoss class   |
|     - forward()            |
|   - SSIM class             |
|     - forward()            |
|   - YUVLoss class          |
|     - forward()            |
+----------------------------+
| data/                      |
|   - ImageFolder class      |
|     - __getitem__()        |
|   - CocoImageIDWrapper     |
|     - __getitem__()        |
+----------------------------+

+--------------------------------------+
|           Utility Modules            |
+--------------------------------------+
| utils/                              |
|   - image.py:                        |
|     - jpeg_compress()                |
|     - detect_wm_hm()                 |
|   - logger.py:                       |
|     - MetricLogger class             |
|       - update()                     |
|       - log_every()                  |
|   - optim.py:                        |
|     - build_optimizer()              |
+--------------------------------------+

