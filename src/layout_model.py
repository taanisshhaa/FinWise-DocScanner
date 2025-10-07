# src/layout_model.py
"""
Layout-aware token classification using LayoutLM (v1).
This implementation expects:
- `image` as a PIL.Image (RGB)
- `words` as a list of dicts: [{'text': 'word', 'bbox': [x0,y0,x1,y1]}, ...] in pixel coords
It returns a list of entities: [{'label': 'ANSWER', 'text': '135000', 'bbox': [x0,y0,x1,y1]}...]
"""

from collections import defaultdict, Counter
from typing import List, Dict
import torch
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image

# Choose a LayoutLM v1 model (fine-tuned on FUNSD). This avoids detectron2.
MODEL_ID = "mrm8488/layoutlm-finetuned-funsd"

class LayoutModel:
    def __init__(self, device: str = "cpu", model_id: str = MODEL_ID):
        self.device = torch.device(device)
        print(f"[LayoutModel] loading tokenizer and model '{model_id}' on {self.device} ...")
        
        # Load the fast tokenizer to support word_ids()
        from transformers import LayoutLMTokenizerFast
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_id)
        
        self.model = LayoutLMForTokenClassification.from_pretrained(model_id)
        self.model.to(self.device)

        self.id2label = getattr(self.model.config, "id2label", None)
        if self.id2label is None:
            self.id2label = {i: str(i) for i in range(self.model.config.num_labels)}
        print("[LayoutModel] loaded. num_labels=", self.model.config.num_labels)


    def _normalize_boxes(self, boxes: List[List[int]], width: int, height: int) -> List[List[int]]:
        # Normalize to 0-1000 as expected by LayoutLM
        norm = []
        for (x0, y0, x1, y1) in boxes:
            nx0 = max(0, min(1000, int(1000 * (x0 / width))))
            ny0 = max(0, min(1000, int(1000 * (y0 / height))))
            nx1 = max(0, min(1000, int(1000 * (x1 / width))))
            ny1 = max(0, min(1000, int(1000 * (y1 / height))))
            norm.append([nx0, ny0, nx1, ny1])
        return norm

    def predict_entities(self, image: Image.Image, words: List[Dict]) -> List[Dict]:
        """
        Inputs:
          - image (PIL.Image) in RGB
          - words: list of {'text': str, 'bbox': [x0,y0,x1,y1]} (pixel coords)
        Output:
          - entities: list of {'label': str, 'text': str, 'bbox': [x0,y0,x1,y1]} where bbox is pixel coords
        """
        if len(words) == 0:
            return []

        words_text = [w["text"] for w in words]
        boxes = [w["bbox"] for w in words]
        width, height = image.size
        norm_boxes = self._normalize_boxes(boxes, width, height)

        # Tokenize - tell tokenizer the input is already split into words
        encoding = self.tokenizer(
            words_text,
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # get the mapping token -> word index for the batch (only single image here)
        word_ids = encoding.word_ids(batch_index=0)  # list(len = seq_len), elements: None or word_index

        # Build token-level bbox list aligned with the tokenized input
        token_bboxes = []
        for wid in word_ids:
            if wid is None:
                token_bboxes.append([0, 0, 0, 0])
            else:
                token_bboxes.append(norm_boxes[wid])

        # Add bbox tensor to encoding (batch dimension)
        encoding["bbox"] = torch.tensor([token_bboxes], dtype=torch.long)

        # Move tensors to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits  # shape (batch, seq_len, num_labels)
            preds = logits.argmax(-1).squeeze(0).cpu().tolist()  # list of label ids for each token

        # Map tokens back to words: for each word index, collect token label ids then pick the most common
        labels_by_word = defaultdict(list)
        for token_idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            labels_by_word[wid].append(preds[token_idx])

        word_pred_label = {}
        for wid, label_ids in labels_by_word.items():
            # choose most common label id among sub-tokens
            most_common_id = Counter(label_ids).most_common(1)[0][0]
            word_pred_label[wid] = self.id2label.get(str(most_common_id), self.id2label.get(most_common_id, str(most_common_id)))

        # Build contiguous entities by walking words and grouping adjacent words with same entity label (ignoring 'O' or 'o')
        entities = []
        cur_label = None
        cur_words = []
        cur_box_coords = []  # pixel boxes for unioning

        def flush_entity():
            nonlocal cur_label, cur_words, cur_box_coords
            if cur_label and cur_label.upper() != "O":
                # compute union bbox in pixel coords
                xs0 = [b[0] for b in cur_box_coords]
                ys0 = [b[1] for b in cur_box_coords]
                xs1 = [b[2] for b in cur_box_coords]
                ys1 = [b[3] for b in cur_box_coords]
                union = [min(xs0), min(ys0), max(xs1), max(ys1)]
                entities.append({"label": cur_label, "text": " ".join(cur_words), "bbox": union})
            cur_label = None
            cur_words = []
            cur_box_coords = []

        for idx, w in enumerate(words):
            lab = word_pred_label.get(idx, "O")
            # Normalize label like "B-ANSWER"/"I-ANSWER" -> "ANSWER"
            if isinstance(lab, str) and "-" in lab:
                lab_type = lab.split("-", 1)[-1]
            else:
                lab_type = lab

            if lab_type.upper() != (cur_label.upper() if cur_label else None) and cur_label is not None:
                # label changed -> flush previous
                flush_entity()

            if lab_type.upper() == "O":
                # not an entity
                if cur_label is not None:
                    flush_entity()
                continue

            # continue or start new entity
            if cur_label is None:
                cur_label = lab_type
            cur_words.append(w["text"])
            cur_box_coords.append(w["bbox"])

        # final flush
        flush_entity()

        return entities


# quick test run when invoked directly
if __name__ == "__main__":
    # small smoke-test (requires PIL and torch)
    from PIL import ImageDraw
    img = Image.new("RGB", (800, 1000), "white")
    test_words = [
        {"text": "Account", "bbox": [50,50,140,80]},
        {"text": "No:", "bbox": [145,50,175,80]},
        {"text": "1234-5678-9876", "bbox": [180,50,360,80]},
        {"text": "Total", "bbox": [50,120,110,150]},
        {"text": "Due:", "bbox": [115,120,150,150]},
        {"text": "135000", "bbox": [155,120,260,150]},
    ]
    lm = LayoutModel(device="cpu")
    ents = lm.predict_entities(img, test_words)
    print("SMOKE: detected entities:", ents)
