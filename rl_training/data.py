# Build a TRL-compatible dataset from the repo's existing loaders.
#
# TRL GRPOTrainer needs a HF Dataset with a "prompt" column (string, standard format) and
# passes any OTHER columns to the reward function as kwargs aligned with completions. We
# carry "gold_answer" (for correctness) and "problem_id" (for harvesting/curriculum).
#
# Reuses src/data/dataset.py: get_inference_dataset, format_prompt.

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_dataset(dataset: str, model_name: str, n_problems: int = -1, seed: int = 42,
                  difficulty_json: str = "", hard_only: bool = True,
                  curriculum: bool = False, frag_lo: float = 0.02, frag_hi: float = 0.30,
                  frag_oversample: int = 3):
    """Return a datasets.Dataset with columns: prompt, gold_answer, problem_id.

    If difficulty_json is given (output of difficulty_prepass.py) and hard_only, keep only
    problems labeled hard (low pass@1, pass@k>0) — Component C targeting.

    EXPERIMENT A — FRAGILE-BAND CURRICULUM (curriculum=True): keep ALL problems but *oversample*
    the samplable-but-fragile band (base pass@1 in [frag_lo, frag_hi]) by repeating those rows
    `frag_oversample` times. Per-problem analysis showed 100% of GRPO's lost set has base
    pass@1 <= 0.10; concentrating rollouts on that band aims to lift BOTH pass@1 and large-k
    coverage. Requires difficulty_json with per-problem 'pass1'.
    """
    from datasets import Dataset
    # CODE SELF-REPAIR: dataset="repair:<path.jsonl>" — rows already have prompt + test/entry/mbpp(/pfail).
    # These extra columns flow to code_repair_reward as kwargs (TRL contract). No math loader needed.
    if isinstance(dataset, str) and dataset.startswith("repair:"):
        path = dataset.split("repair:", 1)[1]
        rows = [json.loads(l) for l in open(path) if l.strip()]
        cols = {"prompt": [r["prompt"] for r in rows], "test": [r["test"] for r in rows],
                "entry": [r.get("entry") for r in rows], "mbpp": [bool(r.get("mbpp")) for r in rows],
                "pfail": [r.get("pfail") for r in rows]}
        return Dataset.from_dict(cols)
    # §50 CONTROLLED TASKS: dataset="ctrl:<path.jsonl>" — rows {prompt, gold} from controlled_tasks.py.
    # Executable integer answers → correctness_reward (boxed/numeric match) works directly. Used by H-A/H-C.
    if isinstance(dataset, str) and dataset.startswith("ctrl:"):
        path = dataset.split("ctrl:", 1)[1]
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if n_problems > 0: rows = rows[:n_problems]
        return Dataset.from_dict({"prompt": [r["prompt"] for r in rows],
                                  "gold_answer": [str(r["gold"]) for r in rows],
                                  "problem_id": list(range(len(rows)))})
    # §64 COMPOSITIONAL CURRICULUM: dataset="comp:<pool.jsonl>" — rows {prompt, prog, test_inputs} from comp_tasks.
    # prog/test_inputs carried as JSON strings; comp_code_reward executes the emitted solve() and verifies exactly.
    if isinstance(dataset, str) and dataset.startswith("comp:"):
        path = dataset.split("comp:", 1)[1]
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if n_problems > 0: rows = rows[:n_problems]
        return Dataset.from_dict({"prompt": [r["prompt"] for r in rows],
                                  "prog": [json.dumps(r["prog"]) for r in rows],
                                  "test_inputs": [json.dumps(r["test_inputs"]) for r in rows]})
    # §73 MATH-TRAIN source (for the advantage-density gap-vs-D validation): pure MATH-train prompts, gold from \boxed{}.
    if isinstance(dataset, str) and dataset == "mathtrain":
        from datasets import load_dataset
        import re as _re
        d = None
        for did in ["EleutherAI/hendrycks_math", "hendrycks/competition_math", "lighteval/MATH"]:
            try:
                dd = load_dataset(did)
                d = dd["train"] if "train" in dd else dd[list(dd.keys())[0]]
                break
            except Exception:
                continue
        if d is None:
            raise SystemExit("MATH-train unavailable")
        rows = {"prompt": [], "gold_answer": [], "problem_id": []}
        _BOX = _re.compile(r"\\boxed\{(.+?)\}")
        n = 0
        for r in d:
            sol = r.get("solution", r.get("answer", ""))
            m = _BOX.findall(sol)
            if not m:
                continue
            from src.data.dataset import format_prompt
            rows["prompt"].append(format_prompt({"question": r.get("problem", r.get("question", ""))}, model_name))
            rows["gold_answer"].append(m[-1].strip()); rows["problem_id"].append(n); n += 1
            if n_problems > 0 and n >= n_problems:
                break
        return Dataset.from_dict(rows)
    # INTERVENTION arm B: GSM8K + 10% MATH-train MIXED (held-out MATH-500 is the disjoint 'test' split).
    if isinstance(dataset, str) and dataset == "mathmix":
        from src.data.dataset import load_gsm8k, load_math_full, format_prompt
        ng = n_problems if n_problems > 0 else 900
        g = load_gsm8k(split="train", n_problems=ng, seed=seed)
        m = load_math_full(n_problems=max(1, ng // 10), seed=seed, split="train")
        rows = {"prompt": [], "gold_answer": [], "problem_id": []}
        for i, p in enumerate(list(g) + list(m)):
            rows["prompt"].append(format_prompt(p, model_name))
            rows["gold_answer"].append(str(p.get("gold_answer", "")))
            rows["problem_id"].append(i)
        print(f"[mathmix] {len(g)} gsm8k + {len(m)} math = {len(rows['prompt'])} rows")
        return Dataset.from_dict(rows)
    from src.data.dataset import get_inference_dataset, format_prompt

    problems = get_inference_dataset({"dataset": {"name": dataset, "split": "test",
                                                  "n_problems": n_problems, "seed": seed}})

    keep_ids = None
    pass1 = {}
    if difficulty_json and os.path.exists(difficulty_json):
        diff = json.load(open(difficulty_json))
        # diff: {"per_problem": [{"problem_id":..., "label":..., "pass1":...}]}
        labels = {d["problem_id"]: d["label"] for d in diff.get("per_problem", [])}
        pass1 = {d["problem_id"]: d.get("pass1") for d in diff.get("per_problem", [])}
        if hard_only and not curriculum:
            keep_ids = {pid for pid, lab in labels.items() if lab == "hard"}

    rows = {"prompt": [], "gold_answer": [], "problem_id": []}
    n_frag = 0
    for p in problems:
        pid = int(p["problem_id"])
        if keep_ids is not None and pid not in keep_ids:
            continue
        reps = 1
        if curriculum:
            p1 = pass1.get(pid)
            if p1 is not None and frag_lo <= p1 <= frag_hi:
                reps = frag_oversample
                n_frag += 1
        for _ in range(reps):
            rows["prompt"].append(format_prompt(p, model_name))
            rows["gold_answer"].append(str(p.get("gold_answer", "")))
            rows["problem_id"].append(pid)

    if curriculum:
        print(f"[curriculum] fragile band [{frag_lo},{frag_hi}] oversampled {frag_oversample}x: "
              f"{n_frag} fragile problems, {len(rows['problem_id'])} total rows")
    return Dataset.from_dict(rows)
