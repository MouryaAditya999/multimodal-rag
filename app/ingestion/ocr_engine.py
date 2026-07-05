"""
OCR module for scanned PDFs (PDFs where text is image-based).
Supports Tesseract and EasyOCR backends.
"""

import os
import tempfile
from PIL import Image


def _ocr_with_easyocr(image_path):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image_path)
    return " ".join([r[1] for r in results])


def _ocr_with_tesseract(image_path):
    import pytesseract
    return pytesseract.image_to_string(Image.open(image_path))


def ocr_image(image_path, engine="easyocr"):
    if engine == "easyocr":
        return _ocr_with_easyocr(image_path)
    elif engine == "tesseract":
        return _ocr_with_tesseract(image_path)
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")


def ocr_pdf_page(page, engine="easyocr"):
    pix = page.get_pixmap(dpi=300)
    img_data = pix.tobytes("png")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_data)
        tmp_path = tmp.name
    try:
        text = ocr_image(tmp_path, engine=engine)
        return text
    finally:
        os.unlink(tmp_path)


def is_scanned_pdf(pdf_path):
    """Check if a PDF has extractable text or is scanned."""
    import fitz
    doc = fitz.open(pdf_path)
    total_chars = 0
    for page in doc:
        total_chars += len(page.get_text().strip())
    doc.close()
    # If very little text (< 50 chars per page on avg), likely scanned
    return total_chars < 50 * doc.page_count
