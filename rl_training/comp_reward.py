# Reward + dataset glue for the compositional-curriculum experiment (§64). The task's ground truth is executable
# (comp_tasks.verify_solution), so the reward is EXACT: 1.0 if the emitted `solve` reproduces the reference executor
# on the task's stored discriminating test inputs, else 0.0. Used by GRPO (--reward-mode comp) and iterative RFT.
import json
from rl_training.comp_tasks import verify_solution


def _coerce(x):
    """Columns may arrive as native lists or as JSON strings (safer through TRL/datasets). Normalize."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _reward_one(completion, prog, test_inputs, reference=None):
    task = {"prog": _coerce(prog), "test_inputs": _coerce(test_inputs)}
    try:
        return 1.0 if verify_solution(completion, task) else 0.0
    except Exception:
        return 0.0


def make_comp_reward():
    """TRL GRPO reward func. TRL passes the dataset's extra columns (prog, test_inputs) as kwargs (list-per-example)."""
    def comp_code_reward(completions, prog=None, test_inputs=None, **kw):
        n = len(completions)
        progs = prog if prog is not None else [None] * n
        tis = test_inputs if test_inputs is not None else [None] * n
        out = []
        for c, p, t in zip(completions, progs, tis):
            text = c if isinstance(c, str) else (c[-1]["content"] if c else "")
            out.append(_reward_one(text, p, t))
        return out
    comp_code_reward.__name__ = "comp_code_reward"
    return comp_code_reward


def load_comp_pool(path):
    """Return list of dataset rows {prompt, prog, test_inputs, reference} from a pool jsonl."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append({"prompt": r["prompt"], "prog": r["prog"],
                     "test_inputs": r["test_inputs"], "reference": r.get("reference", "")})
    return rows
