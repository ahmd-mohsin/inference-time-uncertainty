import re
import random
import logging
from pathlib import Path
from typing import Optional

import jsonlines
from datasets import load_dataset

from src.data.extra_loaders import load_amc, load_competition_math, load_olympiad_bench

logger = logging.getLogger(__name__)


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


def load_aime(
    year: int = 2025,
    n_problems: int = -1,
) -> list[dict]:
    # opencompass/AIME2025 is the available Hub source; use it for any requested year
    logger.info(f"Loading AIME {year} (via opencompass/AIME2025)")
    part_i  = load_dataset("opencompass/AIME2025", "AIME2025-I",  split="test")
    part_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    data = list(part_i) + list(part_ii)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        problems.append({
            "problem_id": i,
            "question": item["question"],
            "gold_answer": str(item["answer"]),
            "source": f"aime_{year}",
            "level": "competition",
            "problem_type": "aime",
        })
    logger.info(f"Loaded {len(problems)} AIME problems")
    return problems


def load_aime_2025(n_problems: int = -1) -> list[dict]:
    return load_aime(year=2025, n_problems=n_problems)


def _amo_bench_inner_answer(answer: str) -> str:
    """AMO-Bench stores gold like \\boxed{...}; normalize for answers_match."""
    s = str(answer).strip()
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, s)
    if matches:
        return matches[-1].strip()
    return s


def load_amo_bench(
    split: str = "train",
    n_problems: int = -1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """meituan-longcat/AMO-Bench: 50 Olympiad-style problems (prompt + answer)."""
    logger.info(f"Loading AMO-Bench split='{split}' n={n_problems} seed={seed}")
    raw = load_dataset("meituan-longcat/AMO-Bench", cache_dir=cache_dir)
    if split not in raw:
        fallback = "test" if "test" in raw else ("train" if "train" in raw else next(iter(raw.keys())))
        logger.warning(
            "AMO-Bench has no split %r; using %r (available: %s)",
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
        q = item.get("prompt") or item.get("question") or ""
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


def format_prompt(problem: dict, model_name: str) -> str:
    question = problem["question"]
    system = (
        "You are a helpful math assistant. Solve the following problem step by step. "
        "Show your reasoning clearly. Put your final answer in \\boxed{}."
    )
    model_lower = model_name.lower()
    if any(k in model_lower for k in ["qwen", "deepseek", "instruct"]):
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"System: {system}\n\nProblem: {question}\n\nSolution:"


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
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    for pat in [
        r"(?:the answer is|answer:|=)\s*([\-\+]?\d[\d,\.]*(?:/\d+)?)",
        r"([\-\+]?\d[\d,\.]*(?:/\d+)?)\s*$",
    ]:
        m = re.search(pat, text.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def _normalize_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\circ", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("{", "").replace("}", "")
    s = s.lower()
    return s


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    answer = answer.strip()
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"^x\s*\\in\s*", "", answer).strip()
    answer = re.sub(r"^x\s*=\s*", "", answer).strip()
    answer = re.sub(r"\^\\circ", "", answer)
    answer = re.sub(r"\\circ", "", answer)
    answer = answer.replace(",", "").replace("\\,", "")
    try:
        val = float(answer)
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.8f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        pass
    return answer.lower().strip()


def answers_match(pred: Optional[str], gold: str, tol: float = 1e-6) -> bool:
    if pred is None:
        return False

    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)

    if pred_n == gold_n:
        return True

    try:
        return abs(float(pred_n) - float(gold_n)) < tol
    except (ValueError, TypeError):
        pass

    frac_pat = r"^([\-\+]?\d+)\s*/\s*(\d+)$"
    pm = re.match(frac_pat, pred_n)
    gm = re.match(frac_pat, gold_n)
    if pm and gm:
        try:
            pv = int(pm.group(1)) / int(pm.group(2))
            gv = int(gm.group(1)) / int(gm.group(2))
            return abs(pv - gv) < tol
        except ZeroDivisionError:
            pass

    pred_l = _normalize_latex(pred)
    gold_l = _normalize_latex(gold)
    if pred_l == gold_l:
        return True

    pred_l2 = _normalize_latex(pred_n)
    gold_l2 = _normalize_latex(gold_n)
    if pred_l2 == gold_l2:
        return True

    try:
        from sympy import simplify, sympify
        p_expr = sympify(pred_l.replace("^", "**"))
        g_expr = sympify(gold_l.replace("^", "**"))
        if simplify(p_expr - g_expr) == 0:
            return True
    except Exception:
        pass

    return False


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
    elif name == "aime_2025":
        return load_aime_2025(n_problems=n)
    elif name.startswith("aime"):
        year = int(name.split("_")[-1]) if "_" in name else 2024
        return load_aime(year=year, n_problems=n)
    elif name == "amc":
        return load_amc(n_problems=n)
    elif name == "competition_math":
        return load_competition_math(n_problems=n, seed=seed)
    elif name == "olympiad_bench":
        return load_olympiad_bench(n_problems=n, seed=seed)
    elif name in ("amo", "amo_bench", "amobench"):
        return load_amo_bench(split=split, n_problems=n, seed=seed)
    else:
        raise ValueError(f"Unknown inference dataset: {name}")