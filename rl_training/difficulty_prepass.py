# Component C: offline difficulty labeling (docs/RL.md §4.3).
#
# Sample the base model k times per problem, label each:
#   "solved" : pass@1 high (already easy)            -> not where expansion is needed
#   "hard"   : low pass@1 but pass@k > 0             -> THE target set (in-support, low-prob)
#   "stuck"  : pass@k == 0                            -> base can't solve at all (skip)
# train_grpo.py + data.build_dataset use this to focus GRPO + harvesting on "hard".
#
# Reuses src.data.dataset + verification_gap.run_gap.pass_at_k.

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_inference_dataset, format_prompt, extract_numeric_answer, answers_match
from verification_gap.run_gap import pass_at_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", default="aime_all")
    ap.add_argument("--n-problems", type=int, default=-1)
    ap.add_argument("--k", type=int, default=64, help="samples per problem")
    ap.add_argument("--pass1-solved-thresh", type=float, default=0.5,
                    help="pass@1 >= this => 'solved' (no headroom)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--output", default="rl_training/runs/difficulty.json")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoConfig

    problems = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test",
                                                  "n_problems": a.n_problems, "seed": 42}})
    try:
        cap = int(getattr(AutoConfig.from_pretrained(a.model_path, trust_remote_code=True),
                          "max_position_embeddings", a.max_new_tokens))
    except Exception:
        cap = a.max_new_tokens
    mml = min(a.max_new_tokens + 1024, cap)

    llm = LLM(model=a.model_path, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=a.tensor_parallel_size, max_model_len=mml,
              gpu_memory_utilization=0.9, enable_prefix_caching=True)
    sp = SamplingParams(n=a.k, max_tokens=mml - 1024, temperature=1.0, top_p=1.0,
                        stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([format_prompt(p, a.model_path) for p in problems], sp)

    per, counts = [], {"solved": 0, "hard": 0, "stuck": 0}
    for p, o in zip(problems, outs):
        gold = str(p.get("gold_answer", ""))
        mask = [bool((pred := extract_numeric_answer(s.text)) is not None and answers_match(pred, gold))
                for s in o.outputs]
        p1 = pass_at_k(mask, 1)
        pk = pass_at_k(mask, a.k)
        if pk == 0.0:
            label = "stuck"
        elif p1 >= a.pass1_solved_thresh:
            label = "solved"
        else:
            label = "hard"
        counts[label] += 1
        per.append({"problem_id": p["problem_id"], "label": label,
                    "pass1": p1, f"pass{a.k}": pk, "n_correct": int(sum(mask))})

    # Safeguard (docs/RL.md §4.3): if too few HARD problems for stable GRPO, relabel some
    # 'solved' as 'hard' (lowest-pass1 first) so the targeted set has >= MIN_HARD problems.
    MIN_HARD = 30
    if counts["hard"] < MIN_HARD:
        solved = sorted([p for p in per if p["label"] == "solved"], key=lambda x: x["pass1"])
        need = MIN_HARD - counts["hard"]
        for p in solved[:need]:
            p["label"] = "hard"; p["relabeled"] = True
            counts["hard"] += 1; counts["solved"] -= 1
        print(f"  (relabeled {min(need, len(solved))} solved->hard to reach MIN_HARD={MIN_HARD})")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model_path, "dataset": a.dataset, "k": a.k,
               "counts": counts, "per_problem": per}, open(a.output, "w"), indent=2)
    print(f"difficulty: {counts}  -> {a.output}")


if __name__ == "__main__":
    main()
