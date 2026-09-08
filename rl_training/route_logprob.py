# Routing measure (exp #1) — the LEFT side of the causal decomposition pi = rho * c.
#
# rho(m|q) = P_theta(enter strategy m | q). We measure it teacher-forced, with NO generation:
# the log-prob each policy assigns to the CANONICAL strategy-m opening seed (STRATEGY_PREFIX[m])
# given the problem prompt. Same (q,m) grid as strategy_probe's competence c(m,q), so
# Delta log rho (grpo-base) pairs 1:1 with Delta c for the make-or-break scatter (exp #3).
#
# Cheap: 14 strategies x N problems short-prefix forward passes. 8-GPU data-parallel over problems
# (shard-index/num-shards), then --merge. Mirrors score_bank_logprobs' teacher-forcing.
#
# Usage: python -m rl_training.route_logprob --model-path <dir> --dataset olympiad_bench \
#   --tag route_grpo --shard-index S --num-shards 8 --max-problems 150
# Merge: python -m rl_training.route_logprob --merge --tag route_grpo --num-shards 8 --output-dir ...

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.strategy_probe import STRATEGY_PREFIX, STRAT_NAMES


def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_training.model_utils import merge_adapter_if_needed

    model_path = merge_adapter_if_needed(a.model_path)
    problems = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test",
                                                  "n_problems": -1, "seed": 42}})
    if a.max_problems > 0:
        problems = problems[:a.max_problems]
    if a.num_shards > 1:
        problems = problems[a.shard_index::a.num_shards]

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def seq_logprob(prompt, seed):
        """sum and per-token logp of `seed` continuation given `prompt` (teacher-forced)."""
        pids = tok(prompt, return_tensors="pt").input_ids.cuda()
        full = tok(prompt + seed, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(full).logits[:, :-1, :].log_softmax(-1)
        tgt = full[:, 1:]
        tok_lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        start = pids.shape[1] - 1
        seg = tok_lp[start:]
        n = max(int(seg.numel()), 1)
        return float(seg.sum().item()), float(seg.sum().item() / n)

    per_problem = []
    for p in problems:
        prompt = format_prompt(p, model_path)
        route = {}
        for m in STRAT_NAMES:
            s, t = seq_logprob(prompt, STRATEGY_PREFIX[m])
            route[m] = {"logp_sum": s, "logp_tok": t}
        per_problem.append({"problem_id": p["problem_id"], "route": route})

    out = {"tag": a.tag, "model": model_path, "dataset": a.dataset,
           "n_problems": len(problems), "strategies": STRAT_NAMES, "per_problem": per_problem}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir) / (f"route_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json"
          if a.num_shards > 1 else f"route_{a.tag}.json"))
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(problems)} problems -> {fp}")


def merge(a):
    parts = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir) / f"route_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists():
            raise FileNotFoundError(f"missing shard {s}/{a.num_shards}: {fp}")
        parts.append(json.load(open(fp)))
    per_problem = [pp for part in parts for pp in part["per_problem"]]
    strategies = parts[0]["strategies"]
    # summary: mean per-token logp per strategy (comparable across policies on the same seed)
    agg = {m: [] for m in strategies}
    for p in per_problem:
        for m in strategies:
            agg[m].append(p["route"][m]["logp_tok"])
    rho_tok = {m: (sum(v) / len(v) if v else 0.0) for m, v in agg.items()}
    out = {"tag": a.tag, "model": parts[0]["model"], "dataset": parts[0]["dataset"],
           "n_problems": len(per_problem), "strategies": strategies,
           "rho_logp_tok_per_strategy": rho_tok, "per_problem": per_problem,
           "merged_from_shards": a.num_shards}
    fp = Path(a.output_dir) / f"route_{a.tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag}] MERGED {a.num_shards} shards, {len(per_problem)} problems")
    print("  mean per-token routing logp (higher = more likely to enter that strategy):")
    for m in sorted(rho_tok, key=lambda k: -rho_tok[k]):
        print(f"    {m:16} {rho_tok[m]:+.3f}")
    print(f"saved -> {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path")
    ap.add_argument("--dataset", default="olympiad_bench")
    ap.add_argument("--max-problems", type=int, default=150)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out")
    ap.add_argument("--tag", default="route")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else run(a)


if __name__ == "__main__":
    main()
