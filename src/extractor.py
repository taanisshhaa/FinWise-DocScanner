# src/extractor.py
import re
from typing import List, Dict, Tuple
from datetime import datetime
import math

CURRENCY_RE = re.compile(r'(?<!\w)(?:Rs\.?|INR|₹)?\s*([0-9\.,]{3,})')  # loose
DATE_RE = re.compile(r'(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})')  # dd/mm/yyyy or similar
PAN_RE = re.compile(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b')  # Indian PAN format
ACC_RE = re.compile(r'(?:A/C|Account|Acct|Account No)[:\s]*([0-9\- ]{6,30})', re.I)
CREDIT_LIMIT_RE = re.compile(r'credit limit[:\s]*([0-9,\.]+)', re.I)
MIN_DUE_RE = re.compile(r'(min(?:imum)?(?:\s*due)?|minimum amount due)[:\s]*([0-9,\.]+)', re.I)
TOTAL_DUE_RE = re.compile(r'(total(?:\s*due|:)|amount due|amount payable)[:\s]*([0-9,\.]+)', re.I)
DUE_DATE_RE = re.compile(r'(due date|payment due)[:\s]*([\d\/\-\.\s]+)', re.I)

def extract_from_text(text: str) -> Dict:
    """
    Quick regex-based field extraction from plain OCR text.
    Returns a dict of discovered fields.
    """
    out = {}
    # numbers (first few matches)
    currencies = CURRENCY_RE.findall(text)
    if currencies:
        out['currency_candidates'] = [c.replace(',', '').replace(' ', '') for c in currencies][:8]
    # dates
    dates = DATE_RE.findall(text)
    if dates:
        out['date_candidates'] = dates[:6]
    # PAN
    m = PAN_RE.search(text)
    if m:
        out['pan'] = m.group(1)
    # account
    m = ACC_RE.search(text)
    if m:
        out['account'] = m.group(1).strip()
    # credit limit
    m = CREDIT_LIMIT_RE.search(text)
    if m:
        out['credit_limit'] = m.group(1).replace(',', '')
    # min due / total due
    m = MIN_DUE_RE.search(text)
    if m:
        out['min_due'] = m.group(2).replace(',', '')
    m = TOTAL_DUE_RE.search(text)
    if m:
        out['total_due'] = m.group(2).replace(',', '')
    m = DUE_DATE_RE.search(text)
    if m:
        out['due_date'] = m.group(2).strip()
    return out

def nearest_number_to_keyword(lines: List[str], keyword: str) -> Tuple[str, float]:
    """
    Very simple heuristic: find line with keyword and extract first currency number on that line.
    returns (number_str, score)
    """
    for line in lines:
        if keyword.lower() in line.lower():
            m = CURRENCY_RE.search(line)
            if m:
                return (m.group(1).replace(',', ''), 0.9)
    return ("", 0.0)
