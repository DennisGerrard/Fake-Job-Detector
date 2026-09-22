"""OCR: pulls text out of an uploaded job-ad image, splits into form fields."""

import pytesseract
from PIL import Image


def extract_text(file_stream) -> str:
    """file_stream: an uploaded file object (e.g. from Flask's request.files)."""
    image = Image.open(file_stream)
    return pytesseract.image_to_string(image).strip()


def guess_fields(raw_text: str) -> dict:
    """Heuristic split: first non-empty line -> title, rest -> description."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return {"title": "", "description": ""}

    title = lines[0]
    description = "\n".join(lines[1:]) if len(lines) > 1 else raw_text
    return {"title": title, "description": description}
