"""Stage 6 test: OCR route, using a generated test image."""

from io import BytesIO
from PIL import Image, ImageDraw

from app import app

img = Image.new("RGB", (700, 300), "white")
draw = ImageDraw.Draw(img)
draw.multiline_text(
    (20, 20),
    "URGENT HIRING - Admin Assistant\n\nEasy money, no experience needed!\nContact us today!!!",
    fill="black",
)
buf = BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

client = app.test_client()
r = client.post("/extract-text", data={"image": (buf, "test.png")}, content_type="multipart/form-data")
print("status:", r.status_code)
print("body  :", r.get_json())
