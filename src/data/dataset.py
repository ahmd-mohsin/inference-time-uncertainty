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
    model_lower = model_name.lower()

    # Gemma needs a more forceful prompt
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

    system = (
        "You are a helpful math assistant. Solve the following problem step by step. "
        "Show your reasoning clearly. Put your final answer in \\boxed{}."
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

    # Ministral / Mistral
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
# LaTeX Normalization Helpers
# ============================================================

# LaTeX formatting wrappers that should be stripped (\text{...}, \mathrm{...}, etc.)
_TEXT_WRAPPERS = (
    "text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt",
    "operatorname", "textbf", "textit", "textrm", "rm", "bf", "it",
    "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
)


def _strip_text_wrappers(s: str) -> str:
    """Strip \\text{...}, \\mathrm{...}, \\operatorname{...}, etc.

    Handles nested wrappers like \\text{\\mathrm{x}} via repeated application.
    """
    for cmd in _TEXT_WRAPPERS:
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\\" + cmd + r"\s*\{([^{}]*)\}", r"\1", s)
            # Also handle \cmd without braces: \cmd x → x
            s = re.sub(r"\\" + cmd + r"\s+", "", s)
    return s


def _convert_frac_to_slash(s: str) -> str:
    """Convert \\frac{a}{b}, \\dfrac{a}{b}, \\tfrac{a}{b} → ((a)/(b)).

    Handles nested fractions by repeated application. Parentheses are added so
    "-\\frac{1}{2}" becomes "-((1)/(2))" which parses unambiguously.
    """
    # Normalize dfrac/tfrac → frac
    s = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", s)

    # Convert \frac{...}{...} repeatedly (innermost first)
    for _ in range(10):  # safety cap on nesting depth
        new = re.sub(
            r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
            r"((\1)/(\2))",
            s,
        )
        if new == s:
            break
        s = new

    # Handle \frac a b (no braces, single tokens)
    s = re.sub(r"\\frac\s+(\S)\s+(\S)", r"((\1)/(\2))", s)

    return s


def _convert_sqrt(s: str) -> str:
    """Convert \\sqrt{n} → sqrt(n), \\sqrt n → sqrt(n)."""
    for _ in range(5):
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\s+(\S+)", r"sqrt(\1)", s)
    return s


def _strip_left_right(s: str) -> str:
    """Strip \\left and \\right delimiters."""
    return s.replace(r"\left", "").replace(r"\right", "")


def _normalize_latex(s: str) -> str:
    """Aggressively normalize a LaTeX string for string-equality comparison."""
    if s is None:
        return ""
    s = str(s).strip()

    # Strip surrounding $ ... $ or $$ ... $$
    s = re.sub(r"^\${1,2}", "", s)
    s = re.sub(r"\${1,2}$", "", s)
    s = s.strip()

    # Strip \left \right
    s = _strip_left_right(s)

    # Strip text/format wrappers
    s = _strip_text_wrappers(s)

    # Convert fractions and square roots
    s = _convert_frac_to_slash(s)
    s = _convert_sqrt(s)

    # Spacing commands
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\:", "")
    s = s.replace(r"\;", "").replace(r"\ ", "").replace(r"\quad", "")
    s = s.replace(r"\qquad", "")

    # Degree/percent symbols
    s = s.replace(r"^\circ", "").replace(r"\circ", "").replace("°", "")
    s = s.replace(r"\%", "").replace("%", "")

    # Greek letters and operators
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")

    # Unicode minus → ASCII minus
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")

    # Strip remaining $ and braces (cosmetic at this point)
    s = s.replace("$", "")
    s = s.replace("{", "").replace("}", "")

    # Collapse all whitespace, lowercase, strip trailing punctuation
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    s = s.rstrip(".,;:")

    return s


# ============================================================
# Numeric Conversion
# ============================================================

def _try_float(s: str) -> Optional[float]:
    """Try parsing s as a float, including a/b fraction syntax. Returns None on failure."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    # Direct float
    try:
        v = float(s)
        if math.isfinite(v):
            return v
    except (ValueError, OverflowError):
        pass

    # Simple a/b
    m = re.match(
        r"^\s*\(?\s*([\-\+]?\d+(?:\.\d+)?)\s*\)?\s*/\s*\(?\s*([\-\+]?\d+(?:\.\d+)?)\s*\)?\s*$",
        s,
    )
    if m:
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            if den != 0 and math.isfinite(num) and math.isfinite(den):
                return num / den
        except (ValueError, OverflowError, ZeroDivisionError):
            pass

    # ((a)/(b)) — output of frac conversion
    m = re.match(
        r"^\(*\(([\-\+]?\d+(?:\.\d+)?)\)/\(([\-\+]?\d+(?:\.\d+)?)\)\)*$",
        s,
    )
    if m:
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            if den != 0 and math.isfinite(num) and math.isfinite(den):
                return num / den
        except (ValueError, OverflowError, ZeroDivisionError):
            pass

    return None


# ============================================================
# Set / Tuple Parsing
# ============================================================

def _try_parse_set(s: str) -> Optional[tuple]:
    """Parse {a,b,c}, (a,b,c), [a,b,c] into a tuple of normalized strings.

    Sets {...} are sorted (unordered semantics).
    Tuples (...) and lists [...] preserve order (e.g. coordinate points).

    Returns a tuple prefixed with "set" or "seq" so a set can never accidentally
    match a tuple with the same elements. Returns None if not a set/tuple/list.
    """
    if s is None:
        return None
    s = str(s).strip()
    s = _strip_left_right(s)
    s = re.sub(r"\\\{", "{", s).replace(r"\}", "}")

    pairs = [("{", "}", "set"), ("(", ")", "seq"), ("[", "]", "seq")]
    inner = None
    kind = None
    for o, c, k in pairs:
        if s.startswith(o) and s.endswith(c):
            inner = s[1:-1]
            kind = k
            break

    if inner is None:
        return None

    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if len(parts) < 2:
        return None

    normalized = [normalize_answer(p) for p in parts]
    if kind == "set":
        normalized = sorted(normalized)
    return (kind, *normalized)


# ============================================================
# Answer Normalization & Matching
# ============================================================

def normalize_answer(answer: Optional[str]) -> str:
    """Normalize a math answer for comparison.

    Returns a canonical string. Numeric answers are returned in their most
    compact decimal form; non-numeric answers are returned as a normalized
    LaTeX/lowercase string.
    """
    if answer is None:
        return ""
    answer = str(answer).strip()

    # Whitespace normalization
    answer = re.sub(r"\s+", " ", answer)

    # Strip common variable prefixes
    answer = re.sub(r"^\s*x\s*\\?in\s*", "", answer).strip()
    answer = re.sub(r"^\s*x\s*=\s*", "", answer).strip()

    # Strip degree/percent
    answer = re.sub(r"\^\\?circ", "", answer)
    answer = re.sub(r"\\?circ", "", answer)
    answer = re.sub(r"\\?%", "", answer)

    # Thousands separators and $ wrappers
    answer = answer.replace(",", "").replace(r"\,", "")
    answer = answer.strip("$").strip()

    # Trailing punctuation
    answer = answer.rstrip(".,;:")

    # Try numeric (handles ints, floats, simple fractions)
    val = _try_float(answer)
    if val is not None:
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.10f}".rstrip("0").rstrip(".")

    return answer.lower().strip()


def _single_answer_match(pred: str, gold: str, tol: float = 1e-6) -> bool:
    """Match a single predicted answer against a single gold answer."""
    if pred is None or gold is None:
        return False

    pred = str(pred).strip().strip("$").strip().rstrip(".,;:")
    gold = str(gold).strip().strip("$").strip().rstrip(".,;:")

    if not pred or not gold:
        return False

    # ---- Set/tuple comparison (must come before scalar attempts) ----
    pred_set = _try_parse_set(pred)
    gold_set = _try_parse_set(gold)
    if pred_set is not None and gold_set is not None:
        return pred_set == gold_set

    # ---- Direct normalized comparison ----
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)
    if pred_n == gold_n:
        return True

    # ---- Strip leading zeros ----
    if pred_n and gold_n:
        ps = pred_n.lstrip("0") or "0"
        gs = gold_n.lstrip("0") or "0"
        if ps == gs:
            return True

    # ---- Numeric comparison with tolerance ----
    pv = _try_float(pred_n)
    gv = _try_float(gold_n)
    if pv is not None and gv is not None:
        if abs(pv - gv) < tol:
            return True
        if abs(pv) < 1e15 and abs(gv) < 1e15:
            try:
                if pv == int(pv) and gv == int(gv) and int(pv) == int(gv):
                    return True
            except (ValueError, OverflowError):
                pass

    # ---- Percentage handling: "62.5%" ↔ "0.625" ↔ "62.5" ----
    pred_pct = re.sub(r"\\?%", "", pred_n).strip()
    gold_pct = re.sub(r"\\?%", "", gold_n).strip()
    pv = _try_float(pred_pct)
    gv = _try_float(gold_pct)
    if pv is not None and gv is not None:
        if abs(pv - gv) < tol:
            return True
        if abs(pv - gv * 100) < tol or abs(pv * 100 - gv) < tol:
            return True

    # ---- Aggressive LaTeX normalization ----
    pred_l = _normalize_latex(pred)
    gold_l = _normalize_latex(gold)
    if pred_l and gold_l and pred_l == gold_l:
        return True

    # Try numeric on LaTeX-normalized form (handles \frac → decimal)
    pv = _try_float(pred_l)
    gv = _try_float(gold_l)
    if pv is not None and gv is not None and abs(pv - gv) < tol:
        return True

    # ---- Sign-shuffling for negative fractions: -a/b vs a/-b vs -(a/b) ----
    def _neg_variants(s: str) -> set:
        out = {s}
        if s.startswith("-"):
            out.add(s[1:])
            out.add("-(" + s[1:] + ")")
        if s.startswith("("):
            out.add(s.strip("()"))
        return out

    for pv_s in _neg_variants(pred_l):
        for gv_s in _neg_variants(gold_l):
            if pv_s == gv_s:
                return True
            pv = _try_float(pv_s)
            gv = _try_float(gv_s)
            if pv is not None and gv is not None and abs(pv - gv) < tol:
                return True

    # ---- Fraction comparison (legacy a/b form) ----
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

    # ---- Sympy fallback for symbolic equivalence ----
    try:
        from sympy import simplify, sympify
        from sympy.parsing.sympy_parser import parse_expr

        def _prepare(expr_str: str) -> str:
            e = expr_str
            e = e.replace("^", "**")
            return e

        try:
            p_expr = parse_expr(_prepare(pred_l), evaluate=True)
            g_expr = parse_expr(_prepare(gold_l), evaluate=True)
            diff = simplify(p_expr - g_expr)
            if diff == 0:
                return True
            try:
                if abs(float(diff)) < tol:
                    return True
            except (TypeError, ValueError):
                pass
        except Exception:
            pass

        # Last-ditch: pure sympify on raw input (after stripping $)
        try:
            p_expr = sympify(pred.replace("$", "").replace(r"\dfrac", r"\frac"))
            g_expr = sympify(gold.replace("$", "").replace(r"\dfrac", r"\frac"))
            if simplify(p_expr - g_expr) == 0:
                return True
        except Exception:
            pass
    except ImportError:
        pass

    return False


def answers_match(pred: Optional[str], gold: str, tol: float = 1e-6) -> bool:
    """Check if predicted answer matches gold answer.

    Handles:
      - exact string match
      - numeric tolerance (ints, floats, simple a/b fractions)
      - leading-zero invariance
      - LaTeX normalization (\\frac, \\dfrac, \\tfrac, \\sqrt, \\text{}, \\mathrm{},
                             \\left/\\right, \\cdot, \\pi, etc.)
      - multi-answer gold ("1, 2, 3" — checks if pred matches any)
      - set vs ordered-tuple comparison (\\{a,b\\} unordered, (a,b) ordered)
      - percentage equivalence ("62.5%" ↔ "0.625")
      - sign-shuffling for negative fractions
      - sympy symbolic equivalence as last resort
    """
    if pred is None:
        return False

    pred = str(pred).strip().strip("$").strip()
    gold = str(gold).strip().strip("$").strip()

    if not pred or not gold:
        return False

    # ---- Multi-answer gold ("1, 2, 3" not wrapped in {} () [] ) ----
    if "," in gold:
        first = gold.lstrip()[:1]
        last = gold.rstrip()[-1:]
        is_wrapped = (first, last) in {("{", "}"), ("(", ")"), ("[", "]")}
        if not is_wrapped:
            gold_clean = gold.replace("$", "").strip()
            gold_parts = [g.strip() for g in gold_clean.split(",") if g.strip()]
            if len(gold_parts) > 1:
                for gp in gold_parts:
                    if _single_answer_match(pred, gp, tol):
                        return True
                # Also fall through to single-answer match below in case the
                # whole comma-separated string is the intended answer

    return _single_answer_match(pred, gold, tol)


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


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    """Sanity-check the answer matching against common LaTeX edge cases."""
    test_cases = [
        # Basic fraction equivalence
        (r"\frac{1}{2}", r"\dfrac{1}{2}", True),
        (r"\frac{1}{2}", r"\tfrac{1}{2}", True),
        (r"\frac{1}{2}", "0.5", True),
        (r"\frac{1}{2}", "1/2", True),
        (r"-\frac{1}{2}", r"-\dfrac{1}{2}", True),
        (r"-\frac{1}{2}", r"\frac{-1}{2}", True),
        (r"-\frac{25}{8}", "-3.125", True),
        ("-25/8", r"-\frac{25}{8}", True),

        # Text wrappers
        (r"\text{even}", "even", True),
        (r"\mathrm{even}", "even", True),
        (r"\mathbf{42}", "42", True),
        (r"\operatorname{even}", "even", True),

        # Square roots
        (r"\sqrt{2}", "sqrt(2)", True),
        (r"\sqrt 2", r"\sqrt{2}", True),
        (r"2\sqrt{3}", r"2\sqrt{3}", True),

        # Numbers
        ("33", "33", True),
        ("33.0", "33", True),
        ("0.625", "62.5%", True),
        ("-7", "-7", True),
        ("−7", "-7", True),  # unicode minus
        ("33.", "33", True),  # trailing punctuation
        ("33;", "33", True),
        ("  33  ", "33", True),
        ("$33$", "33", True),

        # Sets (unordered)
        (r"\{1,2,3\}", "{3,2,1}", True),
        (r"\{1, 2, 3\}", r"\{3, 1, 2\}", True),

        # Tuples (ordered)
        (r"\left(0,1\right)", "(0,1)", True),
        ("(1, 2, 3)", "(1, 2, 3)", True),
        ("(1, 2, 3)", "(3, 2, 1)", False),  # order matters for tuples
        ("(0, 1)", "(1, 0)", False),
        ("[1, 2]", "[2, 1]", False),
        (r"\{1, 2\}", "(1, 2)", False),  # set ≠ tuple

        # Pi
        (r"2\pi", "2pi", True),

        # Multi-answer gold
        ("2", "1, 2, 3", True),
        ("4", "1, 2, 3", False),

        # Units
        (r"5\text{ cm}", r"5 \text{ cm}", True),
        (r"5\text{cm}", "5cm", True),

        # Sympy fallback
        (r"\frac{1}{2} + \frac{1}{3}", r"\frac{5}{6}", True),

        # Negative cases
        (r"33", "34", False),
        (r"\frac{1}{2}", r"\frac{1}{3}", False),
        ("even", "odd", False),
        ("", "5", False),
        ("5", "", False),
        (None, "5", False),
    ]

    passed = 0
    failed = []
    for pred, gold, expected in test_cases:
        try:
            result = answers_match(pred, gold)
        except Exception as e:
            result = f"ERROR: {e}"
        if result == expected:
            passed += 1
        else:
            failed.append((pred, gold, expected, result))

    print(f"Passed: {passed}/{len(test_cases)}")
    if failed:
        print("\nFailures:")
        for pred, gold, expected, result in failed:
            print(f"  pred={pred!r:35} gold={gold!r:30} expected={expected} got={result}")