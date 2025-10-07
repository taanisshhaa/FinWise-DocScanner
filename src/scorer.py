# src/scorer.py
from datetime import datetime
from typing import Dict, List, Tuple

def score_financial_fields(fields: Dict) -> Tuple[int, List[str]]:
    """
    fields: dictionary returned by extractor (combined info)
    Returns (score 0-100, [reasons])
    Simple explainable rules:
    """
    score = 0
    reasons = []

    total_due = None
    credit_limit = None
    min_due = None
    due_date = None

    if 'total_due' in fields:
        try:
            total_due = float(fields['total_due'])
        except:
            pass
    if 'credit_limit' in fields:
        try:
            credit_limit = float(fields['credit_limit'])
        except:
            pass
    if 'min_due' in fields:
        try:
            min_due = float(fields['min_due'])
        except:
            pass
    if 'due_date' in fields:
        # try parse dd/mm/yyyy or dd-mm-yyyy
        raw = fields['due_date']
        for fmt in ("%d/%m/%Y","%d-%m-%Y","%d.%m.%Y","%d/%m/%y","%Y-%m-%d"):
            try:
                due_date = datetime.strptime(raw.strip(), fmt).date()
                break
            except:
                continue

    # Rule: past due high risk
    if due_date:
        today = datetime.now().date()
        if due_date < today:
            score += 60
            reasons.append(f"Document has past due date ({due_date})")

    # Rule: utilization based
    if total_due is not None and credit_limit is not None and credit_limit > 0:
        util = (total_due / credit_limit) * 100
        if util >= 90:
            score += 60
            reasons.append(f"Credit utilization very high ({util:.0f}%)")
        elif util >= 70:
            score += 35
            reasons.append(f"Credit utilization high ({util:.0f}%)")
        elif util >= 50:
            score += 15
            reasons.append(f"Credit utilization moderate ({util:.0f}%)")

    # Rule: total due large relative to min_due
    if total_due and min_due:
        if total_due > min_due * 5:
            score += 20
            reasons.append("Total due is much larger than minimum due — risk of missed payments")

    # Rule: missing important items
    # If document mentions amounts but no due date or account number, moderate risk
    if (total_due or min_due) and 'due_date' not in fields:
        score += 10
        reasons.append("Monetary amounts present but due date missing")

    # If suspicious keywords
    suspicious_keywords = ['overdue','late payment','default','delinquent','warning']
    text = fields.get('full_text','').lower()
    for kw in suspicious_keywords:
        if kw in text:
            score += 40
            reasons.append(f"Found suspicious keyword '{kw}' in document")

    # cap score 0-100
    score = min(100, max(0, score))
    # If no reasons, low risk
    if not reasons:
        reasons.append("No high-risk patterns detected (default low risk)")
    return score, reasons
