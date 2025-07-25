# 🐤 Watermark Freedom (forked from Anything)

app.py:
This file is the main application file for a Flask web application that provides watermarking functionality. It initializes the Flask app and sets up the necessary routes for watermarking and verifying watermarks in images. The app uses a model initialized from the watermark_utils module to perform watermarking operations. It defines routes for the home page, watermarking an image, and verifying a watermark. The watermark_image_route function handles the watermarking process, which involves preprocessing the image, creating a watermark mask, embedding the watermark, and returning the watermarked image. The verify_watermark_route function handles the verification process, which involves preprocessing the watermarked image, extracting the watermark, and returning the verification results.

core.py:
This file contains core functions for watermarking and verification. It includes functions for preprocessing images, creating watermark masks, embedding watermarks, and verifying watermarks. The preprocess_image function loads and preprocesses an image from a source. The create_watermark_mask function creates a watermark mask based on the specified mode and parameters. The embed_watermark function embeds the watermark into the image. The verify_watermark function verifies the watermark in an image using ROCO ECC (Error Correction Code). The file also includes utility functions for device management, image transformation, and message encoding/decoding.

roco_core.py:
This file implements the ROCO (Resilient Optimized Code Operation) Core Encoder/Decoder. It handles the transformation of payload characters into bits and back. The encode_to_bits function encodes up to 3 alphanumeric characters to a 16-bit integer. The decode_from_bits function decodes a 16-bit integer to the original payload. The encode_string function encodes a payload to a hex string representation. The decode_string function decodes a hex string to a payload with error correction status.

roco_ecc.py:
This file implements the ROCO (Resilient Optimized Code Operation) ECC (Error Correction Code). It handles error correction code generation and verification. The encode_with_ecc function encodes a 16-bit payload with BCH ECC. The decode_with_ecc function decodes a 32-bit codeword with BCH ECC.

mark.py:
This file provides functionality to watermark an image and save the result. The watermark_image_and_save function loads an image, embeds a watermark, and saves the result. The main function parses command-line arguments and calls the watermark_image_and_save function.

verify.py:
This file provides functionality to verify a watermark in an image and print the results. The verify_image_and_print function loads a watermarked image, verifies the watermark using ROCO, and prints the results. The main function parses command-line arguments and calls the verify_image_and_print function.

viewframe.py:
This file provides functionality to draw a transparent viewframe overlay on a square image. The get_inner_square_region function returns the coordinates for the inner square region of a square image. The get_corner_color function gets the appropriate color for a corner based on the underlying image content. The draw_alpha_line function draws a line with alpha transparency. The draw_viewframe_overlay function draws the transparent viewframe overlay on a square image.

watermark_utils.py:
This file is a utility module that provides helper functions for watermarking operations. It includes functions such as init_model for initializing models, create_error_response for creating error responses, load_image for loading images, crop_to_centered_square for cropping images to a centered square, pil_to_cv2 for converting PIL images to OpenCV format, cv2_to_pil for converting OpenCV images to PIL format, and create_mask_from_coords for creating masks from coordinates.

Relation/Structure Diagram:
1. app.py (Flask application) -> core.py (Core functions)
2. app.py -> watermark_utils.py (Utility functions)
3. core.py -> roco_core.py (ROCO Core Encoder/Decoder)
4. core.py -> roco_ecc.py (ROCO ECC)
5. core.py -> viewframe.py (Viewframe overlay)
6. core.py -> watermark_utils.py (Utility functions)
7. mark.py -> core.py (Core functions)
8. mark.py -> watermark_utils.py (Utility functions)
9. verify.py -> core.py (Core functions)
10. verify.py -> watermark_utils.py (Utility functions)

The app.py file is the main entry point for the Flask application. It uses the core.py file for core functions and the watermark_utils.py file for utility functions. The core.py file uses the roco_core.py and roco_ecc.py files for ROCO-related operations, the viewframe.py file for drawing viewframe overlays, and the watermark_utils.py file for utility functions. The mark.py and verify.py files use the core.py file for their respective functionalities and the watermark_utils.py file for utility functions.




## Acknowledgments

If you find this repository useful, please consider giving a star :star: and please cite as:

```bibtex
@inproceedings{sander2025watermark,
  title={Watermark Anything with Localized Messages},
  author={Sander, Tom and Fernandez, Pierre and Durmus, Alain and Furon, Teddy and Douze, Matthijs},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
