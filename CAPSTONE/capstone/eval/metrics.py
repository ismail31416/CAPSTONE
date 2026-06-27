"""Answer parsing and metrics for POPE (accuracy) and VSR (acc/P/R/F1).

The LLM emits free-form text ending in ``Answer: [YES/NO]``. We parse that into
a binary prediction and compare against the ground-truth label.
"""

import re

_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\[?\s*(yes|no)\s*\]?", re.IGNORECASE)


def parse_yes_no(text: str):
    """Extract a binary prediction (1=yes, 0=no) from raw LLM output.

    Strategy: prefer an explicit ``Answer: YES/NO`` near the end; otherwise
    fall back to the last yes/no token found. Returns ``None`` if neither.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    matches = _ANSWER_RE.findall(text)
    if matches:
        return 1 if matches[-1].lower() == "yes" else 0

    tokens = re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)
    if tokens:
        return 1 if tokens[-1].lower() == "yes" else 0
    return None


def normalize_label(label):
    """Normalize a ground-truth label to {0, 1}.

    Accepts ``1/0``, ``"yes"/"no"``, ``True/False``. POPE labels are yes/no;
    VSR labels are 1/0 for whether the stated spatial relation holds.
    """
    if isinstance(label, str):
        s = label.strip().lower()
        if s in {"yes", "true", "1"}:
            return 1
        if s in {"no", "false", "0"}:
            return 0
        return None
    try:
        return 1 if int(label) == 1 else 0
    except (ValueError, TypeError):
        return None


def compute_metrics(predictions, labels):
    """Compute accuracy, precision, recall and F1 over binary lists.

    Pairs where either prediction or label is ``None`` are skipped (and
    counted as ``skipped``).
    """
    tp = tn = fp = fn = skipped = 0
    for pred, lab in zip(predictions, labels):
        if pred is None or lab is None:
            skipped += 1
            continue
        if pred == 1 and lab == 1:
            tp += 1
        elif pred == 0 and lab == 0:
            tn += 1
        elif pred == 1 and lab == 0:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "evaluated": total, "skipped": skipped,
    }
