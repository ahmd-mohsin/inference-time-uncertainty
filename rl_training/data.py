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
                  difficulty_json: str = "", hard_only: bool = True):
    """Return a datasets.Dataset with columns: prompt, gold_answer, problem_id.

    If difficulty_json is given (output of difficulty_prepass.py) and hard_only, keep only
    problems labeled hard (low pass@1, pass@k>0) — Component C targeting.
    """
    from datasets import Dataset
    from src.data.dataset import get_inference_dataset, format_prompt

    problems = get_inference_dataset({"dataset": {"name": dataset, "split": "test",
                                                  "n_problems": n_problems, "seed": seed}})

    keep_ids = None
    if difficulty_json and os.path.exists(difficulty_json):
        diff = json.load(open(difficulty_json))
        # diff: {"per_problem": [{"problem_id":..., "label":"hard|solved|stuck", ...}]}
        labels = {d["problem_id"]: d["label"] for d in diff.get("per_problem", [])}
        if hard_only:
            keep_ids = {pid for pid, lab in labels.items() if lab == "hard"}

    rows = {"prompt": [], "gold_answer": [], "problem_id": []}
    for p in problems:
        if keep_ids is not None and p["problem_id"] not in keep_ids:
            continue
        rows["prompt"].append(format_prompt(p, model_name))
        rows["gold_answer"].append(str(p.get("gold_answer", "")))
        rows["problem_id"].append(int(p["problem_id"]))

    return Dataset.from_dict(rows)
