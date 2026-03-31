"""Simple HTTP server for VideoSeal watermarking.

Uses Python's built-in http.server - no Flask needed.
Requires Python 3.12 with videoseal package.

Run with: /usr/bin/python3.12 app_videoseal.py
"""

import os
import sys
import json
import base64
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

PYTHON312_SITE = "/home/h/.local/lib/python3.12/site-packages"

_watermarker = None


def get_watermarker():
    global _watermarker
    if _watermarker is None:
        sys.path.insert(0, PYTHON312_SITE)
        os.chdir(PYTHON312_SITE)
        sys.path.insert(0, "/home/h/FLY/watermark-freedom")
        from backends.videoseal_backend import VideoSealBackend

        _watermarker = VideoSealBackend()
        print("VideoSeal model loaded")
    return _watermarker


HTML_FORM = """
<!DOCTYPE html>
<html>
<head><title>VideoSeal Watermark</title></head>
<body>
<h1>VideoSeal Watermarking</h1>
<form action="/watermark" method="post" enctype="multipart/form-data">
  <h2>Embed Watermark</h2>
  <input type="file" name="cover" accept="image/*" required><br><br>
  Message: <input type="text" name="message" value="TEST" maxlength=32><br><br>
  <button type="submit">Watermark</button>
</form>
<form action="/verify" method="post" enctype="multipart/form-data">
  <h2>Verify Watermark</h2>
  <input type="file" name="watermarked" accept="image/*" required><br><br>
  Original Message: <input type="text" name="original_message"><br><br>
  <button type="submit">Verify</button>
</form>
<h2>Results</h2>
<pre>{results}</pre>
</body>
</html>
"""


class VideoSealHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_FORM.format(results="").encode())
        else:
            self.send_error(404)

    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")

        if self.path == "/watermark" and "multipart/form-data" in content_type:
            self.handle_watermark()
        elif self.path == "/verify" and "multipart/form-data" in content_type:
            self.handle_verify()
        else:
            self.send_error(404)

    def handle_watermark(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            boundary = content_type.split("boundary=")[1].encode()
            parts = body.split(b"--" + boundary)

            cover_data = None
            message = "TEST"

            for part in parts:
                if b'form-data; name="cover"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        cover_data = part[header_end + 4 :]
                        if cover_data.endswith(b"\r\n"):
                            cover_data = cover_data[:-2]
                elif b'form-data; name="message"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        msg_bytes = part[header_end + 4 :]
                        if msg_bytes.endswith(b"\r\n"):
                            msg_bytes = msg_bytes[:-2]
                        message = msg_bytes.decode("utf-8", errors="ignore").strip()

            if not cover_data:
                self.send_json({"error": "No image provided"})
                return

            wm = get_watermarker()
            img = Image.open(io.BytesIO(cover_data)).convert("RGB")
            result, binary, coords = wm.embed(img, message)

            output = io.BytesIO()
            result.save(output, format="PNG")
            img_bytes = output.getvalue()

            filename = f"watermarked_{message}.png"

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("X-Binary-Message", binary[:32])
            self.end_headers()
            self.wfile.write(img_bytes)

        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_verify(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            boundary = content_type.split("boundary=")[1].encode()
            parts = body.split(b"--" + boundary)

            img_data = None
            orig_msg = None

            for part in parts:
                if b'form-data; name="watermarked"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        img_data = part[header_end + 4 :]
                        if img_data.endswith(b"\r\n"):
                            img_data = img_data[:-2]
                elif b'form-data; name="original_message"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        msg_bytes = part[header_end + 4 :]
                        if msg_bytes.endswith(b"\r\n"):
                            msg_bytes = msg_bytes[:-2]
                        orig_msg = msg_bytes.decode("utf-8", errors="ignore").strip()

            if not img_data:
                self.send_json({"error": "No image provided"})
                return

            wm = get_watermarker()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            result = wm.verify(img, orig_msg)

            self.send_json(
                {
                    "status": "success",
                    "detected": result["readable_message"][:32],
                    "bit_accuracy": result.get("bit_accuracy"),
                    "viewframe": result.get("viewframe"),
                }
            )

        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())


def run(port=5001):
    server = HTTPServer(("0.0.0.0", port), VideoSealHandler)
    print(f"VideoSeal server running on http://0.0.0.0:{port}")
    print(f"Run with: /usr/bin/python3.12 {__file__}")
    server.serve_forever()


if __name__ == "__main__":
    run()
