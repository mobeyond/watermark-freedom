# Project Structure

## Root Directory
- .gitignore
- app.py
- core.py
- DldPic.cmp
- DldPic.txt
- LICENSE
- LICENSE-COCO
- mark.py
- README.md
- requirements.txt
- roco_core.py
- roco_ecc.py
- structure.md
- test_watermark.py
- train.py
- verify.py
- viewframe.py

## Assets Directory
- assets/
  - splash_wam.jpg
  - images/
    - alpaca.jpg
    - ducks.jpg
    - gauguin_256.jpg
    - seabackground.jpg
    - trex_bike.jpg
  - masks/
    - ducks_1.jpg
    - ducks_2.jpg

## Checkpoints Directory
- checkpoints/
  - params.json

## Configs Directory
- configs/
  - all_augs_multi_wm.yaml
  - all_augs.yaml
  - attenuation.yaml
  - embedder.yaml
  - extractor.yaml

## Notebooks Directory
- notebooks/
  - colab.ipynb
  - inference_utils.py
  - inference.ipynb

## Source Directory
- src/
  - components/

## Templates Directory
- templates/
  - index.html

## Tin Directory
- tin/
  - alpaca.jpg
  - cover.png
  - ducks.jpg
  - gauguin_256.jpg
  - Screenshot from 2025-06-11 15-53-07.png
  - Screenshot from 2025-06-11 15-59-19.png
  - Screenshot from 2025-06-11 16-01-48.png
  - seabackground.jpg
  - trex_bike.jpg

## Tin Watermarked Directory
- tin_watermarked/
  - alpaca.jpg
  - ducks.jpg
  - gauguin_256.jpg
  - seabackground.jpg
  - trex_bike.jpg

## Watermark Anything Directory
- watermark_anything/
  - augmentation/
    - __init__.py
    - augmenter.py
    - geometric.py
    - masks.py
    - valuemetric.py
  - data/
    - __init__.py
    - loader.py
    - metrics.py
    - transforms.py
  - losses/
    - __init__.py
    - detperceptual.py
    - perceptual.py
    - ssim.py
    - yuvloss.py
  - models/
    - __init__.py
    - embedder.py
    - extractor.py
    - wam.py
  - modules/
    - __init__.py
    - common.py
    - discriminator.py
    - jnd.py
    - msg_processor.py
    - pixel_decoder.py
    - vae.py
    - vit.py
  - utils/
    - __init__.py
    - dist.py
    - image.py
    - logger.py
    - optim.py

## Relations and Structure
The project is structured to separate different functionalities into distinct directories. The `watermark_anything` directory contains the core modules and components for the watermarking functionality, organized into subdirectories for augmentation, data handling, losses, models, modules, and utilities. The `assets` directory holds various images and masks used in the project. The `configs` directory contains configuration files for different settings and parameters. The `notebooks` directory includes Jupyter notebooks for data analysis and inference. The `src` directory is intended for source code components, although it currently only contains an empty `components` subdirectory. The `templates` directory holds HTML templates. The `tin` and `tin_watermarked` directories contain original and watermarked images, respectively. The root directory includes various Python scripts and configuration files necessary for the project's operation.
