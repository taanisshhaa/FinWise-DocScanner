# # src/app.py
# import os
# from flask import Flask, request, jsonify
# from PIL import Image
# from io import BytesIO
# from ocr import pdf_to_images, image_ocr_words, normalize_bbox
# from layout_model import LayoutModel
# from extractor import extract_from_text
# from scorer import score_financial_fields

# # If running as module, adjust imports:
# # e.g. python -m src.app from repo root; but here we'll assume you run `python src/app.py`

# app = Flask(__name__)

# # optional: set POPPLER_PATH if not on PATH
# POPPLER_PATH = None  # or r"C:\path\to\poppler\Library\bin"

# # initialize the layout model once
# layout_model = LayoutModel(device="cpu")  # change to "cuda" if GPU available

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     """
#     Accepts form multipart with 'file' (pdf/image).
#     Returns JSON with ocr_text, entities, regex_fields, risk_score, reasons
#     """
#     if 'file' not in request.files:
#         return jsonify({"error": "no file part 'file'"}), 400
#     f = request.files['file']
#     filename = f.filename.lower()
#     buf = f.read()
#     # handle images vs pdfs
#     if filename.endswith(".pdf"):
#         # write temporary file
#         tmp_path = "tmp_upload.pdf"
#         with open(tmp_path, "wb") as fw:
#             fw.write(buf)
#         pages = pdf_to_images(tmp_path, poppler_path=POPPLER_PATH)
#     else:
#         img = Image.open(BytesIO(buf)).convert("RGB")
#         pages = [img]

#     all_entities = []
#     full_text_pages = []
#     combined_text = ""

#     for p_idx, page_img in enumerate(pages):
#         txt, words = image_ocr_words(page_img, page=p_idx)
#         full_text_pages.append(txt)
#         combined_text += "\n" + txt
#         # Normalize boxes for layout model input:
#         # layout_model expects PIL image; words currently contain absolute boxes
#         entities = layout_model.predict_entities(page_img, words)
#         # add page number to each entity
#         for e in entities:
#             e['page'] = p_idx
#         all_entities.extend(entities)

#     # simple regex extraction on combined text
#     regex_fields = extract_from_text(combined_text)
#     regex_fields['full_text'] = combined_text

#     # scoring
#     score, reasons = score_financial_fields(regex_fields)

#     out = {
#         "ocr_text": combined_text,
#         "entities_model": all_entities,
#         "regex_fields": regex_fields,
#         "risk_score": score,
#         "risk_reasons": reasons
#     }
#     return jsonify(out)

# if __name__ == "__main__":
#     # run on port 8000
#     app.run(host="0.0.0.0", port=8000, debug=True)
# src/app.py
import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from io import BytesIO

# your project modules (assumes these files are in same src/ folder)
from ocr import pdf_to_images, image_ocr_words
from layout_model import LayoutModel
from extractor import extract_from_text
from scorer import score_financial_fields

# Flask app
app = Flask(__name__)

# If poppler isn't on PATH, set POPPLER_PATH to the bin folder, e.g.
# POPPLER_PATH = r"C:\poppler-23_08_0\Library\bin"
POPPLER_PATH = None

# Static UI dir (we will add index.html here)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Initialize layout model once (force CPU here; change to "cuda" if you have GPU)
layout_model = LayoutModel(device="cpu")

@app.route("/")
def index():
    # serve the UI file
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts multipart form with 'file' (pdf/image).
    Returns JSON with:
      - ocr_text
      - entities_model (list of {label, text, bbox, page})
      - regex_fields
      - risk_score, risk_reasons
      - page_images: list of {page, image (base64 PNG), width, height}
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part 'file'"}), 400

    f = request.files["file"]
    filename = (f.filename or "").lower()
    buf = f.read()

    # convert file -> list of PIL pages
    if filename.endswith(".pdf"):
        # write a temp file and convert using pdf2image
        tmp_path = "tmp_upload.pdf"
        with open(tmp_path, "wb") as fw:
            fw.write(buf)
        pages = pdf_to_images(tmp_path, poppler_path=POPPLER_PATH)
        # optionally remove tmp file here if desired
    else:
        img = Image.open(BytesIO(buf)).convert("RGB")
        pages = [img]

    all_entities = []
    combined_text = ""

    for p_idx, page_img in enumerate(pages):
        # run OCR (returns text and words list)
        txt, words = image_ocr_words(page_img, page=p_idx)
        combined_text += "\n" + txt

        # run layout-aware model on that page
        entities = layout_model.predict_entities(page_img, words)
        # attach page index
        for e in entities:
            e["page"] = p_idx
        all_entities.extend(entities)

    # regex-based extraction on combined text
    regex_fields = extract_from_text(combined_text)
    regex_fields["full_text"] = combined_text

    # scoring
    score, reasons = score_financial_fields(regex_fields)

    # encode page images as base64 PNGs so frontend can render them exactly as the model saw them
    page_images = []
    for p_idx, page_img in enumerate(pages):
        buffer = BytesIO()
        page_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        page_images.append({
            "page": p_idx,
            "image": b64,
            "width": page_img.width,
            "height": page_img.height
        })

    out = {
        "ocr_text": combined_text,
        "entities_model": all_entities,
        "regex_fields": regex_fields,
        "risk_score": score,
        "risk_reasons": reasons,
        "page_images": page_images
    }
    return jsonify(out)

if __name__ == "__main__":
    # Run on port 8000 (same as your docs)
    app.run(host="0.0.0.0", port=8000, debug=True)
