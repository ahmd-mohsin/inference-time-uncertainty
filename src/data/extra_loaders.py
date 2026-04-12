import logging
import random
import re
from typing import Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_amc(n_problems: int = -1, cache_dir: Optional[str] = None) -> list[dict]:
    logger.info("Loading AMC (via AI-MO/aimo-validation-amc)")
    raw = load_dataset("AI-MO/aimo-validation-amc", cache_dir=cache_dir)
    split = "train" if "train" in raw else next(iter(raw.keys()))
    data = list(raw[split])
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        problems.append({
            "problem_id": i,
            "question": item["problem"],
            "gold_answer": str(item["answer"]),
            "source": "amc",
            "level": "competition",
            "problem_type": "amc12",
        })
    logger.info(f"Loaded {len(problems)} AMC problems")
    return problems


def load_competition_math(n_problems: int = -1, seed: int = 42, cache_dir: Optional[str] = None) -> list[dict]:
    logger.info("Loading Competition MATH (hendrycks/competition_math)")
    raw = load_dataset("hendrycks/competition_math", cache_dir=cache_dir)
    data = list(raw["test"])
    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        answer_text = item.get("solution", "")
        boxed = re.search(r"\\boxed\{([^}]+)\}", answer_text)
        gold = boxed.group(1) if boxed else answer_text.strip().split("\n")[-1]
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


def load_olympiad_bench(n_problems: int = -1, seed: int = 42, cache_dir: Optional[str] = None) -> list[dict]:
    logger.info("Loading OlympiadBench")
    try:
        raw = load_dataset("lmms-lab/OlympiadBench", cache_dir=cache_dir)
        data = list(raw["test_en"])
    except Exception:
        logger.warning("OlympiadBench not available, skipping")
        return []
    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        question = item.get("question", item.get("problem", ""))
        answer = item.get("final_answer", item.get("answer", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        problems.append({
            "problem_id": i,
            "question": str(question),
            "gold_answer": str(answer),
            "source": "olympiad_bench",
            "level": "olympiad",
            "problem_type": item.get("subject", "math"),
        })
    logger.info(f"Loaded {len(problems)} OlympiadBench problems")
    return problems
