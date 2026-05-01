import re
import math
import random
import logging
from pathlib import Path
from typing import Optional

import jsonlines
from datasets import load_dataset

logger = logging.getLogger(__name__)


# ============================================================
# Dataset Loaders
# ============================================================

def load_gsm8k(
    split: str = "train",
    n_problems: int = 500,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    logger.info(f"Loading GSM8K split='{split}' n={n_problems} seed={seed}")
    raw = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
    data = list(raw[split])
    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        gold = _extract_gsm8k_gold(item["answer"])
        problems.append({
            "problem_id": i,
            "question": item["question"],
            "answer_raw": item["answer"],
            "gold_answer": gold,
            "source": "gsm8k",
            "level": "",
            "problem_type": "",
        })
    logger.info(f"Loaded {len(problems)} GSM8K problems")
    return problems


def _extract_gsm8k_gold(answer_text: str) -> str:
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    lines = [ln.strip() for ln in answer_text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def load_math500(
    split: str = "test",
    n_problems: int = -1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    logger.info(f"Loading MATH500 split='{split}' n={n_problems}")
    raw = load_dataset("HuggingFaceH4/MATH-500", cache_dir=cache_dir)
    data = list(raw[split])
    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        level_raw = item.get("level", "")
        level_num = ""
        m = re.search(r"\d+", str(level_raw))
        if m:
            level_num = m.group(0)
        problems.append({
            "problem_id": i,
            "question": item["problem"],
            "gold_answer": item["answer"],
            "source": "math500",
            "level": level_num,
            "problem_type": item.get("type", ""),
        })
    logger.info(f"Loaded {len(problems)} MATH500 problems")
    return problems


def load_deepmath(
    split: str = "train",
    n_problems: int = -1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """zwhe99/DeepMath-103K: question + verifiable final_answer (+ difficulty, topic)."""
    logger.info(f"Loading DeepMath-103K split='{split}' n={n_problems} seed={seed}")
    raw = load_dataset("zwhe99/DeepMath-103K", cache_dir=cache_dir)
    if split not in raw:
        fallback = "train" if "train" in raw else next(iter(raw.keys()))
        logger.warning(
            "DeepMath-103K has no split %r; using %r (available: %s)",
            split,
            fallback,
            list(raw.keys()),
        )
        split = fallback
    data = list(raw[split])
    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        q = item.get("question") or item.get("problem") or ""
        gold = item.get("final_answer")
        if gold is None:
            gold = item.get("answer", "")
        problems.append({
            "problem_id": i,
            "question": q,
            "gold_answer": str(gold).strip() if gold is not None else "",
            "source": "deepmath",
            "level": str(item.get("difficulty", "")),
            "problem_type": str(item.get("topic", "")),
        })
    logger.info(f"Loaded {len(problems)} DeepMath problems")
    return problems


def load_aime_2024(n_problems: int = -1) -> list[dict]:
    """Load AIME 2024 from math-ai/aime24.

    Fields: id (int), problem (str), solution (str with \\boxed{answer}), url (str)
    30 problems total, split: test
    """
    logger.info("Loading AIME 2024 (math-ai/aime24)")
    raw = load_dataset("math-ai/aime24")
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        question = item.get("problem", "")
        sol = item.get("solution", "")
        gold = extract_boxed_answer(sol)
        if gold is None:
            gold = sol.strip()
        problems.append({
            "problem_id": i,
            "question": str(question),
            "gold_answer": str(gold).strip(),
            "source": "aime_2024",
            "level": "competition",
            "problem_type": "aime",
        })
    logger.info(f"Loaded {len(problems)} AIME 2024 problems")
    return problems


def load_aime_2025(n_problems: int = -1) -> list[dict]:
    """Load AIME 2025 from math-ai/aime25.

    Fields: problem (str), answer (str), id (str)
    30 problems total, split: test
    """
    logger.info("Loading AIME 2025 (math-ai/aime25)")
    raw = load_dataset("math-ai/aime25")
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        question = item.get("problem", "")
        answer = item.get("answer", "")
        problems.append({
            "problem_id": i,
            "question": str(question),
            "gold_answer": str(int(answer)) if isinstance(answer, (int, float)) else str(answer).strip(),
            "source": "aime_2025",
            "level": "competition",
            "problem_type": "aime",
        })
    logger.info(f"Loaded {len(problems)} AIME 2025 problems")
    return problems


def load_aime(year: int = 2025, n_problems: int = -1) -> list[dict]:
    """Dispatch to the correct AIME loader by year."""
    if year == 2024:
        return load_aime_2024(n_problems=n_problems)
    elif year == 2025:
        return load_aime_2025(n_problems=n_problems)
    else:
        logger.warning(f"AIME {year} not explicitly supported, trying aime25 as fallback")
        return load_aime_2025(n_problems=n_problems)


def _amo_bench_inner_answer(answer: str) -> str:
    """AMO-Bench stores gold like \\boxed{...}; normalize for answers_match."""
    s = str(answer).strip()
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, s)
    if matches:
        return matches[-1].strip()
    return s


def load_amc(n_problems: int = -1, cache_dir: Optional[str] = None) -> list[dict]:
    """Load the AMC 2023 dataset from math-ai/amc23."""
    logger.info("Loading AMC (via math-ai/amc23)")
    raw = load_dataset("math-ai/amc23", cache_dir=cache_dir)
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        url = item.get("url", "") or ""
        if "AMC_12A" in url:
            subtype = "amc12a"
        elif "AMC_12B" in url:
            subtype = "amc12b"
        else:
            subtype = "amc12"
        problems.append({
            "problem_id": i,
            "native_id": item.get("id", i),
            "question": item["question"],
            "gold_answer": str(item["answer"]).strip(),
            "source": "amc23",
            "level": "competition",
            "problem_type": subtype,
            "url": url,
        })
    logger.info(f"Loaded {len(problems)} AMC23 problems")
    return problems


def load_competition_math(
    n_problems: int = -1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """Load hendrycks/competition_math (the full MATH benchmark, 5K test problems)."""
    logger.info("Loading Competition MATH (hendrycks/competition_math)")
    raw = load_dataset("hendrycks/competition_math", cache_dir=cache_dir)
    data = list(raw["test"])
    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        solution_text = item.get("solution", "")
        gold = extract_boxed_answer(solution_text)
        if not gold:
            lines = [ln.strip() for ln in solution_text.strip().split("\n") if ln.strip()]
            gold = lines[-1] if lines else ""
        level_raw = item.get("level", "")
        level_num = ""
        m = re.search(r"\d+", str(level_raw))
        if m:
            level_num = m.group(0)
        problems.append({
            "problem_id": i,
            "question": item["problem"],
            "gold_answer": gold,
            "source": "competition_math",
            "level": level_num,
            "problem_type": item.get("type", ""),
        })
    logger.info(f"Loaded {len(problems)} Competition MATH problems")
    return problems


def load_olympiad_bench(n_problems: int = -1, seed: int = 42, cache_dir: Optional[str] = None, numeric_only: bool = True) -> list[dict]:
    logger.info("Loading OlympiadBench (math-ai/olympiadbench)")
    raw = load_dataset("math-ai/olympiadbench")
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])

    # Filter to text-only problems (skip image-based problems)
    data = [item for item in data if item.get("modality", "") == "Text-only"]
    logger.info(f"Filtered to {len(data)} text-only problems")

    # Filter to numeric-answer problems only
    if numeric_only:
        data = [item for item in data if item.get("answer_type", "") == "Numerical"]
        logger.info(f"Filtered to {len(data)} numerical-answer problems")

    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]

    problems = []
    for i, item in enumerate(data):
        question = item.get("question", "")

        # final_answer is a LIST, not a string
        answer = item.get("final_answer", [])
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer) if answer else ""

        # Strip LaTeX $ wrappers from answer
        answer = str(answer).strip().strip("$").strip()

        unit = item.get("unit", None)

        problems.append({
            "problem_id": i,
            "question": str(question),
            "gold_answer": answer,
            "source": "olympiad_bench",
            "level": "olympiad",
            "problem_type": item.get("subfield", "math"),
            "answer_type": item.get("answer_type", ""),
            "unit": str(unit) if unit else "",
            "is_multiple_answer": item.get("is_multiple_answer", False),
        })
    logger.info(f"Loaded {len(problems)} OlympiadBench problems")
    return problems


def load_amo_bench(
    n_problems: int = -1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    number_only: bool = True,
) -> list[dict]:
    """meituan-longcat/AMO-Bench: 50 Olympiad-style problems."""
    logger.info(f"Loading AMO-Bench n={n_problems} seed={seed}")
    raw = load_dataset("meituan-longcat/AMO-Bench", cache_dir=cache_dir)
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])

    if number_only:
        data = [item for item in data if item.get("answer_type", "") == "number"]
        logger.info(f"Filtered to {len(data)} number-type problems")

    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]

    problems = []
    for i, item in enumerate(data):
        q = item.get("prompt", "")
        gold = _amo_bench_inner_answer(item.get("answer", ""))
        problems.append({
            "problem_id": int(item.get("question_id", i)),
            "question": q,
            "gold_answer": gold,
            "source": "amo_bench",
            "level": "olympiad",
            "problem_type": str(item.get("answer_type", "")),
        })
    logger.info(f"Loaded {len(problems)} AMO-Bench problems")
    return problems


# ============================================================
# Prompt Formatting
# ============================================================

def format_prompt(problem: dict, model_name: str) -> str:
    question = problem["question"]
    system = (
        "You are a helpful math assistant. Solve the following problem step by step. "
        "Show your reasoning clearly. Put your final answer in \\boxed{}."
    )
    model_lower = model_name.lower()

    # Try tokenizer's built-in template first (most reliable)
    # Fall back to manual templates only if needed

    # Gemma needs forceful prompt
    if "gemma" in model_lower:
        system = (
            "Solve the following math problem completely. Show all calculations. "
            "You MUST compute the final numerical answer. "
            "Put your final answer in \\boxed{} at the end."
        )
        return (
            f"<start_of_turn>user\n{system}\n\n{question}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    # Qwen / DeepSeek / Nemotron (ChatML)
    if any(k in model_lower for k in ["qwen", "deepseek", "nemotron"]):
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # LLaMA 3
    if "llama" in model_lower:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # Ministral / Mistral (V3-Tekken template)
    if any(k in model_lower for k in ["ministral", "mistral"]):
        return f"[INST] {system}\n\n{question} [/INST]"

    # Fallback (ChatML)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ============================================================
# Answer Extraction
# ============================================================

def extract_boxed_answer(text: str) -> Optional[str]:
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


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract numeric answer from generated text.

    Handles multiple output formats:
    - \\boxed{answer}
    - "The answer is X"
    - "Answer: X"
    - "= X" at end of line
    - **X** (bold, common in Gemma/LLaMA)
    - "#### X" (GSM8K style)
    - Bare number at end of text
    """
    # 1. Try \boxed{} first (highest priority)
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # 2. Try "the answer is" / "answer:" patterns
    for pat in [
        r"(?:the\s+(?:final\s+)?answer\s+is)[:\s]*\$?([^\$\n]{1,60})\$?",
        r"(?:answer)[:\s]*\$?([^\$\n]{1,60})\$?",
        r"####\s*([\-\+]?\d[\d,\.]*(?:/\d+)?)",
    ]:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            ans = m.group(1).strip().strip("$").strip(".").strip()
            if ans:
                return ans

    # 3. Try bold answer format: **X** (common in Gemma, LLaMA)
    bold_match = re.findall(r"\*\*([^\*]{1,60})\*\*", text)
    if bold_match:
        for candidate in reversed(bold_match):
            candidate = candidate.strip().strip("$").strip()
            if re.match(r"^[\-\+]?\d", candidate) or re.match(r"^\\?(?:frac|sqrt)", candidate):
                return candidate

    # 4. Try "= X" at end of a line
    eq_match = re.search(r"=\s*\$?([^\$\n=]{1,40})\$?\s*$", text.strip(), re.MULTILINE)
    if eq_match:
        ans = eq_match.group(1).strip().strip("$").strip(".").strip()
        if ans and re.match(r"^[\-\+]?\d", ans):
            return ans

    # 5. Bare number at end of text
    bare_match = re.search(r"([\-\+]?\d[\d,\.]*(?:/\d+)?)\s*\.?\s*$", text.strip())
    if bare_match:
        return bare_match.group(1).replace(",", "").strip()

    return None


# ============================================================
# Answer Normalization & Matching
# ============================================================

def _normalize_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\circ", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("$", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\pi", "pi")
    s = s.lower()
    return s


def normalize_answer(answer: Optional[str]) -> str:
    """Normalize a math answer for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"^x\s*\\in\s*", "", answer).strip()
    answer = re.sub(r"^x\s*=\s*", "", answer).strip()
    answer = re.sub(r"\^\\circ", "", answer)
    answer = re.sub(r"\\circ", "", answer)
    answer = re.sub(r"\\?%", "", answer)
    answer = answer.replace(",", "").replace("\\,", "")
    answer = answer.strip("$").strip()
    try:
        val = float(answer)
        if math.isfinite(val):
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
            return f"{val:.8f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        pass
    return answer.lower().strip()


def answers_match(pred: Optional[str], gold: str, tol: float = 1e-6) -> bool:
    """Check if predicted answer matches gold answer.

    Handles: exact match, numeric tolerance, fractions, leading zeros,
    LaTeX normalization, multi-answer gold, percentage, sympy fallback.
    """
    if pred is None:
        return False

    # Strip $ signs from both
    pred = str(pred).strip().strip("$").strip()
    gold = str(gold).strip().strip("$").strip()

    # Multi-answer: check if pred matches any individual gold answer
    # Gold answers like "-1$,$2$,$-2" or "1, 3, 5, 15" or "f(x)=x,f(x)=-x"
    if "," in gold:
        gold_clean = gold.replace("$", "").strip()
        gold_parts = [g.strip() for g in gold_clean.split(",") if g.strip()]
        if len(gold_parts) > 1:
            for gp in gold_parts:
                if _single_answer_match(pred, gp, tol):
                    return True

    return _single_answer_match(pred, gold, tol)


def _single_answer_match(pred: str, gold: str, tol: float = 1e-6) -> bool:
    """Match a single predicted answer against a single gold answer."""
    pred = str(pred).strip().strip("$").strip()
    gold = str(gold).strip().strip("$").strip()

    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)

    if pred_n == gold_n:
        return True

    # Strip leading zeros
    pred_stripped = pred_n.lstrip("0") or "0"
    gold_stripped = gold_n.lstrip("0") or "0"
    if pred_stripped == gold_stripped:
        return True

    # Numeric comparison with tolerance (guarded against inf/nan)
    try:
        pv = float(pred_n)
        gv = float(gold_n)
        if math.isfinite(pv) and math.isfinite(gv):
            if abs(pv - gv) < tol:
                return True
            if abs(pv) < 1e15 and abs(gv) < 1e15 and pv == int(pv) and gv == int(gv) and int(pv) == int(gv):
                return True
    except (ValueError, TypeError, OverflowError):
        pass

    # Percentage comparison: "62.5%" vs "0.625" or "62.5"
    try:
        pred_pct = re.sub(r"\\?%", "", pred_n).strip()
        gold_pct = re.sub(r"\\?%", "", gold_n).strip()
        pv = float(pred_pct)
        gv = float(gold_pct)
        if math.isfinite(pv) and math.isfinite(gv):
            if abs(pv - gv) < tol:
                return True
            if abs(pv - gv * 100) < tol or abs(pv * 100 - gv) < tol:
                return True
    except (ValueError, TypeError, OverflowError):
        pass

    # Fraction comparison
    frac_pat = r"^([\-\+]?\d+)\s*/\s*(\d+)$"
    pm = re.match(frac_pat, pred_n)
    gm = re.match(frac_pat, gold_n)
    if pm and gm:
        try:
            pv = int(pm.group(1)) / int(pm.group(2))
            gv = int(gm.group(1)) / int(gm.group(2))
            if math.isfinite(pv) and math.isfinite(gv):
                return abs(pv - gv) < tol
        except (ZeroDivisionError, OverflowError):
            pass

    # Mixed: one is fraction, other is decimal
    if pm and not gm:
        try:
            pv = int(pm.group(1)) / int(pm.group(2))
            gv = float(gold_n)
            if math.isfinite(pv) and math.isfinite(gv):
                return abs(pv - gv) < tol
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    if gm and not pm:
        try:
            pv = float(pred_n)
            gv = int(gm.group(1)) / int(gm.group(2))
            if math.isfinite(pv) and math.isfinite(gv):
                return abs(pv - gv) < tol
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    # LaTeX normalization
    pred_l = _normalize_latex(pred)
    gold_l = _normalize_latex(gold)
    if pred_l == gold_l:
        return True

    pred_l2 = _normalize_latex(pred_n)
    gold_l2 = _normalize_latex(gold_n)
    if pred_l2 == gold_l2:
        return True

    # Sympy fallback
    try:
        from sympy import simplify, sympify
        p_expr = sympify(pred_l.replace("^", "**"))
        g_expr = sympify(gold_l.replace("^", "**"))
        if simplify(p_expr - g_expr) == 0:
            return True
    except Exception:
        pass

    return False


# ============================================================
# Cache / Save / Load
# ============================================================

def save_problems_cache(problems: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(problems)
    logger.info(f"Cached {len(problems)} problems to {path}")


def load_problems_cache(path: str) -> list[dict]:
    with jsonlines.open(path) as reader:
        problems = list(reader)
    logger.info(f"Loaded {len(problems)} problems from cache {path}")
    return problems


# ============================================================
# Dataset Dispatch
# ============================================================

def get_calibration_dataset(cfg: dict) -> list[dict]:
    cal = cfg["calibration"]
    name = cal.get("dataset", "gsm8k").lower()
    split = cal.get("split", "train")
    n = cal.get("n_problems", 500)
    seed = cal.get("seed", 42)
    if name == "gsm8k":
        return load_gsm8k(split=split, n_problems=n, seed=seed)
    elif name == "math500":
        return load_math500(split=split, n_problems=n, seed=seed)
    else:
        raise ValueError(f"Unsupported calibration dataset: {name}")


def get_inference_dataset(cfg: dict) -> list[dict]:
    ds = cfg["dataset"]
    name = ds["name"].lower()
    split = ds.get("split", "test")
    n = ds.get("n_problems", -1)
    seed = ds.get("seed", 42)
    if name == "gsm8k":
        return load_gsm8k(split=split, n_problems=n, seed=seed)
    elif name == "math500":
        return load_math500(split=split, n_problems=n, seed=seed)
    elif name == "deepmath":
        return load_deepmath(n_problems=n)
    elif name == "aime_2024":
        return load_aime_2024(n_problems=n)
    elif name == "aime_2025":
        return load_aime_2025(n_problems=n)
    elif name.startswith("aime"):
        year = int(name.split("_")[-1]) if "_" in name else 2025
        return load_aime(year=year, n_problems=n)
    elif name in ("amo", "amo_bench"):
        return load_amo_bench(n_problems=n)
    elif name == "amc":
        return load_amc(n_problems=n)
    elif name == "competition_math":
        return load_competition_math(n_problems=n, seed=seed)
    elif name == "olympiad_bench":
        return load_olympiad_bench(n_problems=n, seed=seed)
    else:
        raise ValueError(f"Unknown inference dataset: {name}")