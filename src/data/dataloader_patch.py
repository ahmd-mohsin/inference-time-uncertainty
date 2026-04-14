"""
PATCH: Replace load_math500 and load_competition_math in src/data/dataset.py
and src/data/extra_loaders.py respectively.

load_math500: Minor robustness fixes (strip gold answer, handle missing fields).
load_competition_math: Fix nested-brace boxed extraction bug.
"""

# ============================================================
# Put this in src/data/dataset.py (replaces existing load_math500)
# ============================================================

def load_math500(
    split: str = "test",
    n_problems: int = -1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """Load HuggingFaceH4/MATH-500.

    Fields in the dataset:
        problem  : str   — the problem text
        solution : str   — full solution with \\boxed{} answer
        answer   : str   — pre-extracted final answer (already unboxed)
        level    : str   — e.g. "Level 3"
        type     : str   — e.g. "Algebra", "Number Theory"
    """
    logger.info(f"Loading MATH-500 split='{split}' n={n_problems} seed={seed}")
    raw = load_dataset("HuggingFaceH4/MATH-500", cache_dir=cache_dir)
    data = list(raw[split])
    random.seed(seed)
    random.shuffle(data)
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        # Gold answer is pre-extracted in the 'answer' field — no boxed parsing needed
        gold = str(item.get("answer", "")).strip()

        # Extract numeric level from strings like "Level 3"
        level_raw = item.get("level", "")
        level_num = ""
        m = re.search(r"\d+", str(level_raw))
        if m:
            level_num = m.group(0)

        problems.append({
            "problem_id": i,
            "question": item["problem"],
            "gold_answer": gold,
            "source": "math500",
            "level": level_num,
            "problem_type": item.get("type", ""),
        })
    logger.info(f"Loaded {len(problems)} MATH-500 problems")
    return problems


# ============================================================
# Put this in src/data/extra_loaders.py (replaces existing load_competition_math)
# ============================================================

def _extract_boxed_from_solution(text: str) -> str:
    """Extract answer from \\boxed{} with proper nested-brace handling.

    The simple regex r'\\boxed\{([^}]+)\}' fails on nested braces like
    \\boxed{\\frac{1}{2}} — it would return '\\frac{1' instead of '\\frac{1}{2}'.
    This function uses depth tracking to handle arbitrary nesting.
    """
    # Find the last \\boxed{ occurrence (solutions sometimes have intermediate boxed)
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + 7  # skip past "\\boxed{"
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth > 0:
            i += 1
    if depth == 0:
        return text[start:i].strip()
    return ""


def load_competition_math(
    n_problems: int = -1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """Load hendrycks/competition_math (the full MATH benchmark, 5K test problems).

    Fields in the dataset:
        problem  : str — problem text
        solution : str — full solution text containing \\boxed{answer}
        level    : str — e.g. "Level 5"
        type     : str — e.g. "Algebra"
    """
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
        # Use depth-tracking extraction instead of simple regex
        gold = _extract_boxed_from_solution(solution_text)
        if not gold:
            # Fallback: last line of solution
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