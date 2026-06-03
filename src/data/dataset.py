import re
import math
import random
import logging
from pathlib import Path
from typing import Optional

import jsonlines
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Patterns for stripping reasoning/thinking traces from various models
_THINK_PATTERNS = [
    # Qwen3, DeepSeek-R1: <think>...</think>
    (r"<think>", r"</think>"),
    # Some models use <reasoning>...</reasoning>
    (r"<reasoning>", r"</reasoning>"),
    # Some models use <reflection>...</reflection>
    (r"<reflection>", r"</reflection>"),
]

# Answer marker patterns (searched in post-reasoning tail)
_ANSWER_MARKERS = [
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?([^\$\n]{1,80})\$?",
    r"(?:the\s+)?(?:final\s+)?answer[:\s]*\$?([^\$\n]{1,80})\$?",
    r"(?:therefore|thus|hence|so)[,:\s]+(?:the\s+)?(?:answer\s+is\s+)?\$?([^\$\n]{1,60})\$?",
    r"####\s*([\-\+]?\d[\d,\.]*(?:/\d+)?)",
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
]

# ============================================================
# Dataset Loaders   (unchanged)
# ============================================================

def load_gsm8k(split="train", n_problems=500, seed=42, cache_dir=None):
    logger.info(f"Loading GSM8K split='{split}' n={n_problems} seed={seed}")
    raw = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
    data = list(raw[split]); random.seed(seed); random.shuffle(data)
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        gold = _extract_gsm8k_gold(item["answer"])
        problems.append({"problem_id": i, "question": item["question"],
                         "answer_raw": item["answer"], "gold_answer": gold,
                         "source": "gsm8k", "level": "", "problem_type": ""})
    logger.info(f"Loaded {len(problems)} GSM8K problems")
    return problems


def _extract_gsm8k_gold(answer_text):
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    lines = [ln.strip() for ln in answer_text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def load_math500(split="test", n_problems=-1, seed=42, cache_dir=None):
    logger.info(f"Loading MATH500 split='{split}' n={n_problems}")
    raw = load_dataset("HuggingFaceH4/MATH-500", cache_dir=cache_dir)
    data = list(raw[split]); random.seed(seed); random.shuffle(data)
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        level_raw = item.get("level", ""); m = re.search(r"\d+", str(level_raw))
        problems.append({"problem_id": i, "question": item["problem"],
                         "gold_answer": item["answer"], "source": "math500",
                         "level": m.group(0) if m else "", "problem_type": item.get("type", "")})
    logger.info(f"Loaded {len(problems)} MATH500 problems")
    return problems


def load_deepmath(split="train", n_problems=-1, seed=42, cache_dir=None):
    logger.info(f"Loading DeepMath-103K split='{split}' n={n_problems} seed={seed}")
    raw = load_dataset("zwhe99/DeepMath-103K", cache_dir=cache_dir)
    if split not in raw:
        fallback = "train" if "train" in raw else next(iter(raw.keys()))
        logger.warning("DeepMath-103K has no split %r; using %r", split, fallback)
        split = fallback
    data = list(raw[split]); random.seed(seed); random.shuffle(data)
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        q = item.get("question") or item.get("problem") or ""
        gold = item.get("final_answer")
        if gold is None: gold = item.get("answer", "")
        problems.append({"problem_id": i, "question": q,
                         "gold_answer": str(gold).strip() if gold is not None else "",
                         "source": "deepmath", "level": str(item.get("difficulty", "")),
                         "problem_type": str(item.get("topic", ""))})
    logger.info(f"Loaded {len(problems)} DeepMath problems")
    return problems


def load_aime_2024(n_problems=-1):
    logger.info("Loading AIME 2024 (math-ai/aime24)")
    raw = load_dataset("math-ai/aime24")
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        sol = item.get("solution", ""); gold = extract_boxed_answer(sol)
        if gold is None: gold = sol.strip()
        problems.append({"problem_id": i, "question": str(item.get("problem", "")),
                         "gold_answer": str(gold).strip(), "source": "aime_2024",
                         "level": "competition", "problem_type": "aime"})
    logger.info(f"Loaded {len(problems)} AIME 2024 problems")
    return problems


def load_aime_2025(n_problems=-1):
    logger.info("Loading AIME 2025 (math-ai/aime25)")
    raw = load_dataset("math-ai/aime25")
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        answer = item.get("answer", "")
        problems.append({"problem_id": i, "question": str(item.get("problem", "")),
                         "gold_answer": str(int(answer)) if isinstance(answer, (int, float)) else str(answer).strip(),
                         "source": "aime_2025", "level": "competition", "problem_type": "aime"})
    logger.info(f"Loaded {len(problems)} AIME 2025 problems")
    return problems


def load_aime(year=2025, n_problems=-1):
    if year == 2024: return load_aime_2024(n_problems=n_problems)
    if year == 2025: return load_aime_2025(n_problems=n_problems)
    logger.warning(f"AIME {year} not supported, fallback aime25")
    return load_aime_2025(n_problems=n_problems)


def _amo_bench_inner_answer(answer):
    s = str(answer).strip()
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", s)
    return matches[-1].strip() if matches else s


def load_amc(n_problems=-1, cache_dir=None):
    logger.info("Loading AMC (via math-ai/amc23)")
    raw = load_dataset("math-ai/amc23", cache_dir=cache_dir)
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        url = item.get("url", "") or ""
        subtype = "amc12a" if "AMC_12A" in url else "amc12b" if "AMC_12B" in url else "amc12"
        problems.append({"problem_id": i, "native_id": item.get("id", i),
                         "question": item["question"], "gold_answer": str(item["answer"]).strip(),
                         "source": "amc23", "level": "competition", "problem_type": subtype, "url": url})
    logger.info(f"Loaded {len(problems)} AMC23 problems")
    return problems


def load_competition_math(n_problems=-1, seed=42, cache_dir=None):
    logger.info("Loading Competition MATH (hendrycks/competition_math)")
    raw = load_dataset("hendrycks/competition_math", cache_dir=cache_dir)
    data = list(raw["test"]); random.seed(seed); random.shuffle(data)
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        solution_text = item.get("solution", ""); gold = extract_boxed_answer(solution_text)
        if not gold:
            lines = [ln.strip() for ln in solution_text.strip().split("\n") if ln.strip()]
            gold = lines[-1] if lines else ""
        m = re.search(r"\d+", str(item.get("level", "")))
        problems.append({"problem_id": i, "question": item["problem"], "gold_answer": gold,
                         "source": "competition_math", "level": m.group(0) if m else "",
                         "problem_type": item.get("type", "")})
    logger.info(f"Loaded {len(problems)} Competition MATH problems")
    return problems


def load_olympiad_bench(n_problems=-1, seed=42, cache_dir=None, numeric_only=True):
    logger.info("Loading OlympiadBench (math-ai/olympiadbench)")
    raw = load_dataset("math-ai/olympiadbench")
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    data = [item for item in data if item.get("modality", "") == "Text-only"]
    if numeric_only:
        data = [item for item in data if item.get("answer_type", "") == "Numerical"]
    random.seed(seed); random.shuffle(data)
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        answer = item.get("final_answer", [])
        if isinstance(answer, list):
            answer = ", ".join(str(a) for a in answer) if answer else ""
        answer = str(answer).strip().strip("$").strip()
        unit = item.get("unit", None)
        problems.append({"problem_id": i, "question": str(item.get("question", "")),
                         "gold_answer": answer, "source": "olympiad_bench", "level": "olympiad",
                         "problem_type": item.get("subfield", "math"),
                         "answer_type": item.get("answer_type", ""),
                         "unit": str(unit) if unit else "",
                         "is_multiple_answer": item.get("is_multiple_answer", False)})
    logger.info(f"Loaded {len(problems)} OlympiadBench problems")
    return problems


def load_amo_bench(n_problems=-1, seed=42, cache_dir=None, number_only=True):
    logger.info(f"Loading AMO-Bench n={n_problems} seed={seed}")
    raw = load_dataset("meituan-longcat/AMO-Bench", cache_dir=cache_dir)
    split = "test" if "test" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if number_only:
        data = [item for item in data if item.get("answer_type", "") == "number"]
    random.seed(seed); random.shuffle(data)
    if n_problems > 0: data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        gold = _amo_bench_inner_answer(item.get("answer", ""))
        problems.append({"problem_id": int(item.get("question_id", i)),
                         "question": item.get("prompt", ""), "gold_answer": gold,
                         "source": "amo_bench", "level": "olympiad",
                         "problem_type": str(item.get("answer_type", ""))})
    logger.info(f"Loaded {len(problems)} AMO-Bench problems")
    return problems


# ============================================================
# Prompt Formatting   (unchanged)
# ============================================================

def format_prompt(problem: dict, model_name: str) -> str:
    question = problem["question"]
    model_lower = model_name.lower()
    if "gemma" in model_lower:
        system = ("Solve the following math problem completely. Show all calculations. "
                  "You MUST compute the final numerical answer. "
                  "Put your final answer in \\boxed{} at the end.")
        return (f"<start_of_turn>user\n{system}\n\n{question}<end_of_turn>\n<start_of_turn>model\n")
    system = ("You are a helpful math assistant. Solve the following problem step by step. "
              "Show your reasoning clearly. Put your final answer in \\boxed{}.")
    if any(k in model_lower for k in ["qwen", "deepseek", "nemotron"]):
        return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
    if "llama" in model_lower:
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    if any(k in model_lower for k in ["ministral", "mistral"]):
        return f"[INST] {system}\n\n{question} [/INST]"
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")


# ============================================================
# Reasoning-model scaffold stripping
# ============================================================

# closed and (defensively) unclosed reasoning blocks emitted by R1 / Qwen3-thinking
_THINK_BLOCK = re.compile(
    r"<think>.*?</think>|<thinking>.*?</thinking>|<reasoning>.*?</reasoning>|"
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
    re.DOTALL | re.IGNORECASE,
)
_SPECIAL_TOKENS = re.compile(
    r"<\|[^|>]*\|>|<\/?s>|<end_of_turn>|<start_of_turn>\w*|\[/?INST\]|"
    r"<\|im_(?:start|end)\|>",
    re.IGNORECASE,
)


def _strip_reasoning(text: str) -> str:
    """Strip thinking/reasoning traces from model output.
    
    Handles:
    - Complete traces: <think>...</think> -> returns text after </think>
    - Truncated traces: <think>... (no closing tag) -> searches tail for answer
    - Multiple trace formats (Qwen3, DeepSeek-R1, etc.)
    - Nested or repeated traces
    """
    result = text
    
    for open_tag, close_tag in _THINK_PATTERNS:
        if close_tag in result:
            # Complete trace found — take everything after the LAST closing tag
            parts = result.split(close_tag)
            after = parts[-1].strip()
            if after:
                result = after
            else:
                # Closing tag is at the very end — check the part just before it
                # for a \boxed{} that might be inside the thinking
                # (some models put the final answer inside <think>)
                for part in reversed(parts[:-1]):
                    boxed = extract_boxed_answer(part)
                    if boxed:
                        return f"\\boxed{{{boxed}}}"
                # No boxed found, use last 1000 chars of the full text
                result = text[-1000:]
        elif re.search(open_tag, result):
            # Truncated trace — opening tag but no closing tag
            # Model likely hit token limit mid-thinking
            # Strategy: search the tail for \boxed{}, or fall back to last 500 chars
            boxed = extract_boxed_answer(text)
            if boxed:
                return f"\\boxed{{{boxed}}}"
            # Try the last 500 characters
            result = text[-500:]
    
    return result



# ============================================================
# Answer Extraction
# ============================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the LAST \\boxed{...} from text, handling nested braces.
    
    Returns the content inside the last \\boxed{}, or None if not found.
    Handles: \\boxed{}, \\fbox{}, nested braces like \\boxed{\\frac{1}{2}}.
    """
    if not text:
        return None
    
    # Find ALL \boxed{ occurrences and extract the last one
    last_answer = None
    i = 0
    while i < len(text):
        # Check for \boxed{ or \fbox{
        if text[i:i + 7] == r"\boxed{":
            start = i + 7
            depth = 1
            j = start
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                last_answer = text[start:j - 1].strip()
            i = j
            continue
        i += 1
    
    if last_answer is not None:
        return last_answer
    
    # Fallback: regex for simpler cases
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None



# answer-announcing markers, tried in order of reliability
_ANSWER_MARKERS = [
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?([^\$\n]{1,80})\$?",
    r"(?:the\s+)?(?:final\s+)?answer[:\s]*\$?([^\$\n]{1,80})\$?",
    r"(?:therefore|thus|hence|so)[,:\s]+(?:the\s+)?(?:answer\s+is\s+)?\$?([^\$\n]{1,60})\$?",
    r"####\s*([\-\+]?\d[\d,\.]*(?:/\d+)?)",
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
]


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the answer-of-record across diverse model output styles.

    Handles reasoning models (Qwen3 <think>, DeepSeek-R1 <think>),
    markdown bold, ChatML/LLaMA/Mistral special tokens, GSM8K #### X,
    fractions/roots, signed and comma-grouped numbers.
    
    Priority order:
    1. \\boxed{} anywhere in full text (last occurrence wins)
    2. \\boxed{} after stripping reasoning traces
    3. Explicit answer markers in post-reasoning tail
    4. Bold **X** format
    5. Trailing "= X" on last line
    6. Bare number at end
    7. Trailing LaTeX expression
    """
    if not text:
        return None

    # 1) Try \boxed{} on the FULL text first (catches boxed inside <think>)
    boxed = extract_boxed_answer(text)
    if boxed:
        # Clean up: if boxed contains thinking artifacts, clean them
        boxed = boxed.strip()
        # Sometimes the boxed content has trailing text from truncation
        # e.g., \boxed{23. However let me verify...}
        # Take only up to the first period/space that looks like prose
        prose_match = re.match(r"^([\-\+]?\d[\d,\.\\/\s\{\}\\a-zA-Z\^_]*?)(?:\.\s+[A-Z]|\.\s+however|\.\s+but|\.\s+let|\.\s+so|\.\s+wait)", boxed, re.IGNORECASE)
        if prose_match:
            boxed = prose_match.group(1).rstrip(".")
        return boxed

    # 2) Strip reasoning traces and try boxed again
    tail = _strip_reasoning(text).strip()
    
    boxed_tail = extract_boxed_answer(tail)
    if boxed_tail:
        return boxed_tail

    # 3) Explicit answer markers (search in tail)
    for pat in _ANSWER_MARKERS:
        m = None
        for m in re.finditer(pat, tail, re.IGNORECASE):
            pass  # keep the LAST occurrence
        if m:
            ans = m.group(1).strip().strip("$").strip("\\()").strip().rstrip(".,;:")
            if ans:
                return ans

    # 4) Bold **X** — scan original text for safety
    for cand in reversed(re.findall(r"\*\*([^\*]{1,80})\*\*", text)):
        cand = cand.strip().strip("$").strip()
        if re.match(r"^[\-\+]?\d", cand) or re.match(r"^\\?(?:frac|dfrac|sqrt)", cand):
            return cand.rstrip(".,;:")

    # 5) Trailing "= X" on last meaningful line
    for line in reversed([ln for ln in tail.splitlines() if ln.strip()]):
        eq = re.search(r"=\s*\$?([^\$\n=]{1,40})\$?\s*$", line.strip())
        if eq:
            ans = eq.group(1).strip().strip("$").rstrip(".,;:").strip()
            if ans and (re.match(r"^[\-\+]?\d", ans) or re.match(r"^\\?(?:frac|dfrac|sqrt)", ans)):
                return ans
        break  # only inspect the last non-empty line

    # 6) Bare number at the very end
    bare = re.search(r"([\-\+]?\d[\d,\.]*(?:/\d+)?)\s*\.?\s*$", tail)
    if bare:
        return bare.group(1).replace(",", "").strip()

    # 7) Trailing LaTeX expression
    latex_tail = re.search(
        r"([\-\+]?\\(?:dfrac|tfrac|frac|sqrt)\s*\{[^{}]*\}(?:\s*\{[^{}]*\})?)\s*\$?\.?\s*$",
        tail,
    )
    if latex_tail:
        return latex_tail.group(1).strip()

    return None


# ============================================================
# LaTeX Normalization Helpers   (unchanged below, + small additions)
# ============================================================

_TEXT_WRAPPERS = ("text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt",
                  "operatorname", "textbf", "textit", "textrm", "rm", "bf", "it",
                  "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle")


def _strip_text_wrappers(s):
    for cmd in _TEXT_WRAPPERS:
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\\" + cmd + r"\s*\{([^{}]*)\}", r"\1", s)
            s = re.sub(r"\\" + cmd + r"\s+", "", s)
    return s


def _convert_frac_to_slash(s):
    s = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", s)
    for _ in range(10):
        new = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        if new == s: break
        s = new
    s = re.sub(r"\\frac\s+(\S)\s+(\S)", r"((\1)/(\2))", s)
    return s


def _convert_sqrt(s):
    for _ in range(5):
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
        if new == s: break
        s = new
    return re.sub(r"\\sqrt\s+(\S+)", r"sqrt(\1)", s)


def _strip_left_right(s):
    return s.replace(r"\left", "").replace(r"\right", "")


def _latex_to_parseable(s: str) -> str:
    """Light LaTeX -> plain-math conversion used before numeric/sympy parsing."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"^\${1,2}", "", s); s = re.sub(r"\${1,2}$", "", s); s = s.strip()
    s = re.sub(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", s)
    s = _strip_left_right(s)
    s = _strip_text_wrappers(s)
    s = _convert_frac_to_slash(s)
    s = _convert_sqrt(s)
    for tok in (r"\!", r"\,", r"\:", r"\;", r"\ ", r"\quad", r"\qquad"):
        s = s.replace(tok, "")
    s = s.replace(r"^\circ", "").replace(r"\circ", "").replace("°", "")
    s = s.replace(r"\%", "").replace("%", "")
    s = s.replace(r"\pi", "pi").replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def _normalize_latex(s):
    if s is None: return ""
    s = _latex_to_parseable(s)
    s = re.sub(r"\s+", "", s).lower().rstrip(".,;:")
    return s


# ============================================================
# Numeric Conversion
# ============================================================

def _try_float(s):
    if s is None: return None
    s = str(s).strip()
    if not s: return None
    try:
        v = float(s)
        if math.isfinite(v): return v
    except (ValueError, OverflowError):
        pass
    # optional leading sign + simple a/b (with optional parens)
    m = re.match(r"^\s*([\-\+]?)\(?\s*([\-\+]?\d+(?:\.\d+)?)\s*\)?\s*/\s*\(?\s*([\-\+]?\d+(?:\.\d+)?)\s*\)?\s*$", s)
    if m:
        try:
            sign = -1.0 if m.group(1) == "-" else 1.0
            num = float(m.group(2)); den = float(m.group(3))
            if den != 0 and math.isfinite(num) and math.isfinite(den):
                return sign * num / den
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
    # optional sign + ((a)/(b)) from frac conversion
    m = re.match(r"^([\-\+]?)\(*\(([\-\+]?\d+(?:\.\d+)?)\)/\(([\-\+]?\d+(?:\.\d+)?)\)\)*$", s.replace(" ", ""))
    if m:
        try:
            sign = -1.0 if m.group(1) == "-" else 1.0
            num = float(m.group(2)); den = float(m.group(3))
            if den != 0 and math.isfinite(num) and math.isfinite(den):
                return sign * num / den
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
    return None


def _canonical_via_sympy(s: str) -> Optional[str]:
    """Canonical string for an expression via sympy; None on failure."""
    if not s or not re.search(r"[\d\\/^+\-*]", s):
        return None
    try:
        from sympy import nsimplify
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor)
        T = standard_transformations + (implicit_multiplication_application, convert_xor)
        expr = parse_expr(s.replace("^", "**"), transformations=T, evaluate=True)
        if expr.is_number:
            try:
                expr = nsimplify(expr, rational=True)
            except Exception:
                pass
        return str(expr)
    except Exception:
        return None


# ============================================================
# Set / Tuple Parsing   (unchanged)
# ============================================================

def _try_parse_set(s):
    if s is None: return None
    s = str(s).strip(); s = _strip_left_right(s)
    s = re.sub(r"\\\{", "{", s).replace(r"\}", "}")
    pairs = [("{", "}", "set"), ("(", ")", "seq"), ("[", "]", "seq")]
    inner = kind = None
    for o, c, k in pairs:
        if s.startswith(o) and s.endswith(c):
            inner = s[1:-1]; kind = k; break
    if inner is None: return None
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if len(parts) < 2: return None
    normalized = [normalize_answer(p) for p in parts]
    if kind == "set": normalized = sorted(normalized)
    return (kind, *normalized)


# ============================================================
# Answer Normalization & Matching
# ============================================================

def normalize_answer(answer: Optional[str]) -> str:
    """Canonical string for comparison AND vote clustering.

    Now converts LaTeX (\\frac, \\sqrt, \\cdot, \\pi, …) and falls back to a
    sympy canonical form, so \\frac{25}{8}, 25/8, and 3.125 collapse to ONE
    bucket instead of three. This is what keeps Maj@N from splitting a correct
    majority across equivalent surface forms.
    """
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"^\s*x\s*\\?in\s*", "", answer).strip()
    answer = re.sub(r"^\s*[a-zA-Z]\s*=\s*", "", answer).strip()  # strip 'x =', 'm =' …
    answer = re.sub(r"\^\\?circ", "", answer)
    answer = re.sub(r"\\?circ", "", answer)
    answer = re.sub(r"\\?%", "", answer)
    answer = answer.replace(",", "").replace(r"\,", "")
    answer = answer.strip("$").strip().rstrip(".,;:")

    # NEW: convert LaTeX to plain math before numeric / sympy parsing
    parseable = _latex_to_parseable(answer)

    # numeric (ints, floats, signed simple fractions, ((a)/(b)))
    val = _try_float(parseable)
    if val is None:
        val = _try_float(answer)
    if val is not None:
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.10f}".rstrip("0").rstrip(".")

    # NEW: symbolic canonical form (2\sqrt{3}, 2pi, a+b, …) -> one bucket
    canon = _canonical_via_sympy(parseable)
    if canon is not None:
        # if sympy reduced it to a number, emit compact decimal
        fv = _try_float(canon)
        if fv is not None:
            if fv == int(fv) and abs(fv) < 1e15:
                return str(int(fv))
            return f"{fv:.10f}".rstrip("0").rstrip(".")
        return re.sub(r"\s+", "", canon).lower()

    return re.sub(r"\s+", "", parseable).lower() or answer.lower().strip()


def _single_answer_match(pred, gold, tol=1e-6):
    if pred is None or gold is None: return False
    pred = str(pred).strip().strip("$").strip().rstrip(".,;:")
    gold = str(gold).strip().strip("$").strip().rstrip(".,;:")
    if not pred or not gold: return False

    pred_set = _try_parse_set(pred); gold_set = _try_parse_set(gold)
    if pred_set is not None and gold_set is not None:
        return pred_set == gold_set

    pred_n = normalize_answer(pred); gold_n = normalize_answer(gold)
    if pred_n == gold_n: return True
    if pred_n and gold_n and (pred_n.lstrip("0") or "0") == (gold_n.lstrip("0") or "0"):
        return True

    pv = _try_float(pred_n); gv = _try_float(gold_n)
    if pv is not None and gv is not None:
        if abs(pv - gv) < tol: return True
        if abs(pv) < 1e15 and abs(gv) < 1e15:
            try:
                if pv == int(pv) and gv == int(gv) and int(pv) == int(gv): return True
            except (ValueError, OverflowError): pass

    pred_pct = re.sub(r"\\?%", "", pred_n).strip(); gold_pct = re.sub(r"\\?%", "", gold_n).strip()
    pv = _try_float(pred_pct); gv = _try_float(gold_pct)
    if pv is not None and gv is not None:
        if abs(pv - gv) < tol: return True
        if abs(pv - gv * 100) < tol or abs(pv * 100 - gv) < tol: return True

    pred_l = _normalize_latex(pred); gold_l = _normalize_latex(gold)
    if pred_l and gold_l and pred_l == gold_l: return True
    pv = _try_float(pred_l); gv = _try_float(gold_l)
    if pv is not None and gv is not None and abs(pv - gv) < tol: return True

    def _neg_variants(s):
        out = {s}
        if s.startswith("-"): out.add(s[1:]); out.add("-(" + s[1:] + ")")
        if s.startswith("("): out.add(s.strip("()"))
        return out
    for pv_s in _neg_variants(pred_l):
        for gv_s in _neg_variants(gold_l):
            if pv_s == gv_s: return True
            pv = _try_float(pv_s); gv = _try_float(gv_s)
            if pv is not None and gv is not None and abs(pv - gv) < tol: return True

    frac_pat = r"^([\-\+]?\d+)\s*/\s*(\d+)$"
    pm = re.match(frac_pat, pred_n); gm = re.match(frac_pat, gold_n)
    if pm and gm:
        try:
            pv = int(pm.group(1)) / int(pm.group(2)); gv = int(gm.group(1)) / int(gm.group(2))
            if math.isfinite(pv) and math.isfinite(gv): return abs(pv - gv) < tol
        except (ZeroDivisionError, OverflowError): pass
    if pm and not gm:
        try:
            pv = int(pm.group(1)) / int(pm.group(2)); gv = float(gold_n)
            if math.isfinite(pv) and math.isfinite(gv): return abs(pv - gv) < tol
        except (ValueError, ZeroDivisionError, OverflowError): pass
    if gm and not pm:
        try:
            pv = float(pred_n); gv = int(gm.group(1)) / int(gm.group(2))
            if math.isfinite(pv) and math.isfinite(gv): return abs(pv - gv) < tol
        except (ValueError, ZeroDivisionError, OverflowError): pass

    try:
        from sympy import simplify, sympify
        from sympy.parsing.sympy_parser import parse_expr
        def _prepare(e): return e.replace("^", "**")
        try:
            p_expr = parse_expr(_prepare(pred_l), evaluate=True)
            g_expr = parse_expr(_prepare(gold_l), evaluate=True)
            diff = simplify(p_expr - g_expr)
            if diff == 0: return True
            try:
                if abs(float(diff)) < tol: return True
            except (TypeError, ValueError): pass
        except Exception: pass
        try:
            p_expr = sympify(pred.replace("$", "").replace(r"\dfrac", r"\frac"))
            g_expr = sympify(gold.replace("$", "").replace(r"\dfrac", r"\frac"))
            if simplify(p_expr - g_expr) == 0: return True
        except Exception: pass
    except ImportError:
        pass
    return False


def answers_match(pred: Optional[str], gold: str, tol: float = 1e-6) -> bool:
    if pred is None: return False
    pred = str(pred).strip().strip("$").strip()
    gold = str(gold).strip().strip("$").strip()
    if not pred or not gold: return False
    if "," in gold:
        first = gold.lstrip()[:1]; last = gold.rstrip()[-1:]
        if (first, last) not in {("{", "}"), ("(", ")"), ("[", "]")}:
            gold_parts = [g.strip() for g in gold.replace("$", "").split(",") if g.strip()]
            if len(gold_parts) > 1:
                for gp in gold_parts:
                    if _single_answer_match(pred, gp, tol): return True
    return _single_answer_match(pred, gold, tol)


# ============================================================
# Cache / Save / Load   (unchanged)
# ============================================================

def save_problems_cache(problems, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(problems)
    logger.info(f"Cached {len(problems)} problems to {path}")


def load_problems_cache(path):
    with jsonlines.open(path) as reader:
        problems = list(reader)
    logger.info(f"Loaded {len(problems)} problems from cache {path}")
    return problems


# ============================================================
# Dataset Dispatch   (unchanged)
# ============================================================

def get_calibration_dataset(cfg):
    cal = cfg["calibration"]; name = cal.get("dataset", "gsm8k").lower()
    split = cal.get("split", "train"); n = cal.get("n_problems", 500); seed = cal.get("seed", 42)
    if name == "gsm8k": return load_gsm8k(split=split, n_problems=n, seed=seed)
    if name == "math500": return load_math500(split=split, n_problems=n, seed=seed)
    raise ValueError(f"Unsupported calibration dataset: {name}")


def get_inference_dataset(cfg):
    ds = cfg["dataset"]; name = ds["name"].lower()
    split = ds.get("split", "test"); n = ds.get("n_problems", -1); seed = ds.get("seed", 42)
    if name == "gsm8k": return load_gsm8k(split=split, n_problems=n, seed=seed)
    if name == "math500": return load_math500(split=split, n_problems=n, seed=seed)
    if name == "deepmath": return load_deepmath(n_problems=n)
    if name == "aime_2024": return load_aime_2024(n_problems=n)
    if name == "aime_2025": return load_aime_2025(n_problems=n)
    if name.startswith("aime"):
        year = int(name.split("_")[-1]) if "_" in name else 2025
        return load_aime(year=year, n_problems=n)
    if name in ("amo", "amo_bench"): return load_amo_bench(n_problems=n)
    if name == "amc": return load_amc(n_problems=n)
    if name == "competition_math": return load_competition_math(n_problems=n, seed=seed)
    if name == "olympiad_bench": return load_olympiad_bench(n_problems=n, seed=seed)
    raise ValueError(f"Unknown inference dataset: {name}")