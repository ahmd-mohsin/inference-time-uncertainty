import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_boxed(text: str) -> Optional[str]:
    depth = 0
    start = None
    i = 0
    while i < len(text):
        if text[i:i + 7] == r"\boxed{":
            if start is None:
                start = i + 7
                depth = 1
                i += 7
                continue
        if start is not None:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
        i += 1
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def extract_last_number(text: str) -> Optional[str]:
    for pat in [
        r"(?:answer\s*(?:is|=|:)\s*)([\-\+]?\d[\d,\.]*(?:/\d+)?)",
        r"(?:=\s*)([\-\+]?\d[\d,\.]*(?:/\d+)?)\s*$",
        r"([\-\+]?\d[\d,\.]*(?:/\d+)?)\s*$",
    ]:
        m = re.search(pat, text.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    return extract_last_number(text)


def _normalize_number(s: str) -> Optional[float]:
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    m = re.match(r"^([\-\+]?\d+)\s*/\s*(\d+)$", s)
    if m:
        den = int(m.group(2))
        if den != 0:
            return int(m.group(1)) / den
    return None


def _normalize_symbolic(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    return s.lower().strip()


def answers_match(pred: Optional[str], gold: str, tol: float = 1e-6) -> bool:
    if pred is None:
        return False
    pred = pred.strip()
    gold = gold.strip()
    if pred == gold:
        return True
    pn = _normalize_number(pred)
    gn = _normalize_number(gold)
    if pn is not None and gn is not None:
        return abs(pn - gn) < tol
    ps = _normalize_symbolic(pred)
    gs = _normalize_symbolic(gold)
    if ps == gs:
        return True
    try:
        from sympy import simplify, sympify
        diff = simplify(sympify(ps, evaluate=True) - sympify(gs, evaluate=True))
        if diff == 0:
            return True
    except Exception:
        pass
    return False


def score_prediction(pred_text: str, gold_answer: str) -> dict:
    extracted = extract_answer(pred_text)
    correct = answers_match(extracted, gold_answer)
    return {
        "extracted_answer": extracted,
        "gold_answer": gold_answer,
        "correct": correct,
        "has_boxed": extract_boxed(pred_text) is not None,
    }