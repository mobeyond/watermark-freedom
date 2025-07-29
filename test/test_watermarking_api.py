import os
import requests
import unittest
import random
import string
import base64
from PIL import Image
from io import BytesIO

TIN_DIR = './tin'
NUM_PAYLOADS = 20
PAYLOAD_LENGTH = 3
API_URL = 'http://127.0.0.1:5000'

class TestWatermarkingAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Check if Flask app is running
        try:
            r = requests.get(f'{API_URL}/')
            assert r.status_code == 200
            print('Flask app is running.')
        except Exception as e:
            raise RuntimeError(f'Flask app not running at {API_URL}. Start app.py first. Error: {e}')
        cls.images = [f for f in os.listdir(TIN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def generate_payloads(self, num):
        return [''.join(random.choices(string.ascii_uppercase, k=PAYLOAD_LENGTH)) for _ in range(num)]

    def test_api_watermark_and_verify(self):
        # x and y percent from 0 to 0.2 in steps of 0.05, mask always centered and symmetric
        percents = [i / 100.0 for i in range(0, 30, 5)]
        stats = {}
        total_tests = 0
        total_failures = 0
        total_mismatches = 0
        for filename in self.images:
            stats[filename] = {}
            img_path = os.path.join(TIN_DIR, filename)
            with Image.open(img_path) as img:
                w, h = img.size
            # The image is cropped to a square in the backend
            width = height = min(w, h)
            for pct in percents:
                # The mask is centered and symmetric, its size adapts to pct.
                # x and y are the offsets from the border, defining the top-left corner.
                # The mask spans from (x, y) to (width-x, height-y).
                x = int(width * pct)
                y = int(height * pct)
                mask_w = int(width * (1 - 2 * pct))
                mask_h = int(height * (1 - 2 * pct))

                # Ensure mask has a positive size
                if mask_w <= 0 or mask_h <= 0:
                    continue
                
                print(f"\n--- Verification Round ---")
                print(f"Image: {filename}")
                print(f"Mask parameters: x={x}, y={y}, width={mask_w}, height={mask_h}")
                payloads = self.generate_payloads(NUM_PAYLOADS)
                success = 0
                fail = 0
                mismatch = 0
                for payload in payloads:
                    temp_filename = f'temp_{filename}'
                    try:
                        with open(img_path, 'rb') as f:
                            files = {'cover': (filename, f, 'image/png')}
                            data = {
                                'message': payload,
                                'use_frame_corners': 'false',
                                'use_pixels': 'true',
                                'x_pixels': str(x),
                                'y_pixels': str(y),
                                'width_pixels': str(mask_w),
                                'height_pixels': str(mask_h)
                            }
                            r = requests.post(f'{API_URL}/watermark', files=files, data=data)
                        if r.status_code != 200:
                            print(f"Watermark API error: {r.text}")
                            fail += 1
                            total_failures += 1
                            continue
                        resp = r.json()
                        if 'image' not in resp:
                            print(f"No image in watermark API response: {resp}")
                            fail += 1
                            total_failures += 1
                            continue
                        watermarked_img_bytes = BytesIO(base64.b64decode(resp['image']))
                        with open(temp_filename, 'wb') as tempf:
                            tempf.write(watermarked_img_bytes.getvalue())
                        with open(temp_filename, 'rb') as wf:
                            files = {'watermarked': (temp_filename, wf, 'image/png')}
                            data = {'original_message': payload}
                            r2 = requests.post(f'{API_URL}/verify', files=files, data=data)
                        if r2.status_code != 200:
                            print(f"Verify API error: {r2.text}")
                            fail += 1
                            total_failures += 1
                            continue
                        resp2 = r2.json()
                        decoded_message = resp2.get('readable_message')
                        result_str = "SUCCESS" if decoded_message == payload else "FAIL"
                        print(f"Embedded: {payload}, Decoded: {decoded_message}, Result: {result_str}")
                        total_tests += 1
                        if decoded_message == payload:
                            success += 1
                        else:
                            fail += 1
                            total_failures += 1
                            if decoded_message is not None:
                                mismatch += 1
                                total_mismatches += 1
                    finally:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                stats[filename][pct] = {'success': success, 'fail': fail, 'mismatch': mismatch}
                current_failure_rate = (total_failures / total_tests) * 100 if total_tests else 0
                current_mismatch_rate = (total_mismatches / total_tests) * 100 if total_tests else 0
                print(f"Progress: {total_tests} tests done so far, Failure rate: {current_failure_rate:.2f}%, Mismatch rate: {current_mismatch_rate:.2f}%")
        for filename, positions in stats.items():
            print(f"\nImage: {filename}")
            for pct, res in positions.items():
                print(f"  Mask pct={pct:.2f}: Success={res['success']}, Fail={res['fail']}, Mismatch={res['mismatch']}")
        print(f"\nTotal tests: {total_tests}, Total failures: {total_failures}, Total mismatches: {total_mismatches}, Overall failure rate: {(total_failures / total_tests) * 100 if total_tests else 0:.2f}%, Overall mismatch rate: {(total_mismatches / total_tests) * 100 if total_tests else 0:.2f}%")
        for filename, positions in stats.items():
            for res in positions.values():
                self.assertGreaterEqual(res['success'], 1, f"Too many failures for {filename}")

if __name__ == "__main__":
    unittest.main()
