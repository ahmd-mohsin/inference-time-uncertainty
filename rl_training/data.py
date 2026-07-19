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
