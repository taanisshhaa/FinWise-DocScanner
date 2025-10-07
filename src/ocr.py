# src/ocr.py
import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from pytesseract import Output
from typing import List, Dict, Tuple

# If Tesseract is not on PATH, set it here:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def pdf_to_images(pdf_path: str, dpi: int = 300, poppler_path: str = None) -> List[Image.Image]:
    """
    Convert a PDF to a list of PIL Images (one per page).
    poppler_path: Windows users should pass the path to poppler's bin folder if not on PATH.
    """
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return images

def image_ocr_words(image: Image.Image, page: int = 0) -> Tuple[str, List[Dict]]:
    """
    Run pytesseract OCR on a PIL image and return:
      - full text (string)
      - words: list of dicts: {'text', 'bbox':[x0,y0,x1,y1], 'conf', 'page'}
    Bounding boxes returned are absolute pixel coords.
    """
    img = image.convert("RGB")
    width, height = img.size
    data = pytesseract.image_to_data(img, output_type=Output.DICT)  # word-level
    words = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i] != '' else -1
        if text == "" or conf <= 0:
            continue
        x = int(data['left'][i])
        y = int(data['top'][i])
        w = int(data['width'][i])
        h = int(data['height'][i])
        bbox = [x, y, x + w, y + h]
        words.append({'text': text, 'bbox': bbox, 'conf': conf, 'page': page, 'width': width, 'height': height})
    full_text = pytesseract.image_to_string(img)
    return full_text, words

def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """
    Normalize box to LayoutLM 0-1000 scale: [x0,y0,x1,y1]
    """
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(1000 * (x0 / max(1, width))))),
        max(0, min(1000, int(1000 * (y0 / max(1, height))))),
        max(0, min(1000, int(1000 * (x1 / max(1, width))))),
        max(0, min(1000, int(1000 * (y1 / max(1, height))))),
    ]
