# Downstream go/no-go: strategy-STRATIFIED pass@k vs IID pass@k at MATCHED budget.
#
# The whole "exploit the hidden repertoire" story lives or dies here. For each problem we sample N
# free generations, record (correct, strategy-label). Then:
#   IID pass@k        : draw k of N uniformly, pass if any correct (Monte-Carlo).
#   STRATIFIED pass@k : draw k samples from k DISTINCT strategy groups (diversify first), pass if any
#                       correct (Monte-Carlo). This is "deliberately try different strategies".
#   ORACLE strategy   : the single strategy whose samples solve the problem best (ceiling of routing).
# If STRATIFIED >> IID at fixed k, controllable diversity converts to accuracy -> worth scaling.
# If ~equal, there is no inference-time win from strategy diversity.
#
# 8-GPU data-parallel over problems + merge. Usage:
#   python -m rl_training.stratified_passk --model-path <dir> --dataset olympiad_bench --tag sp_grpo \
#     --shard-index S --num-shards 8 --n-samples 32 --max-problems 150
#   python -m rl_training.stratified_passk --merge --tag sp_grpo --num-shards 8 --output-dir ...

import argparse, json, os, sys, random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.safe_match import safe_is_correct
from rl_training.strategy_probe import STRAT_NAMES
from rl_training.strategy_bank import STRAT_RE


def _classify(text):
    t = text or ""; best, bestn = "none", 0
    for m in STRAT_NAMES:
        n = len(STRAT_RE[m].findall(t))
        if n > bestn: bestn, best = n, m
    return best


def run(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    problems = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test", "n_problems": -1, "seed": 42}})
    if a.max_problems > 0: problems = problems[:a.max_problems]
    if a.num_shards > 1: problems = problems[a.shard_index::a.num_shards]
    try: cap = int(getattr(AutoConfig.from_pretrained(mp, trust_remote_code=True), "max_position_embeddings", 4096))
    except Exception: cap = 4096
    mml = min(4096, cap)
    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.85)); _eager = os.environ.get("EVAL_ENFORCE_EAGER", "1") == "1"
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
              max_model_len=mml, gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)
    sp = SamplingParams(n=a.n_samples, temperature=1.0, top_p=1.0, max_tokens=mml - 1024,
                        stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([format_prompt(p, mp) for p in problems], sp)
    per = []
    for p, o in zip(problems, outs):
        gold = str(p.get("gold_answer", ""))
        samples = [{"correct": bool(safe_is_correct(s.text, gold)[0]), "strategy": _classify(s.text)}
                   for s in o.outputs]
        per.append({"problem_id": p["problem_id"], "samples": samples})
    out = {"tag": a.tag, "model": mp, "dataset": a.dataset, "n_problems": len(problems),
           "n_samples": a.n_samples, "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir) / (f"sp_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json"
          if a.num_shards > 1 else f"sp_{a.tag}.json"))
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(problems)} problems -> {fp}")


def _passk_iid(samples, k, trials=400):
    n = len(samples)
    if k >= n: return float(any(s["correct"] for s in samples))
    hit = 0
    for _ in range(trials):
        if any(samples[i]["correct"] for i in random.sample(range(n), k)): hit += 1
    return hit / trials


def _passk_strat(samples, k, trials=400):
    """draw k samples from k DISTINCT strategy groups where possible (diversify first)."""
    n = len(samples)
    if k >= n: return float(any(s["correct"] for s in samples))
    by = {}
    for i, s in enumerate(samples): by.setdefault(s["strategy"], []).append(i)
    groups = list(by.values())
    hit = 0
    for _ in range(trials):
        random.shuffle(groups); picks = []
        gi = 0
        # round-robin across distinct strategy groups until k picks
        pools = [list(g) for g in groups]
        for pl in pools: random.shuffle(pl)
        while len(picks) < k:
            progressed = False
            for pl in pools:
                if pl and len(picks) < k:
                    picks.append(pl.pop()); progressed = True
            if not progressed: break
        if any(samples[i]["correct"] for i in picks): hit += 1
    return hit / trials


def merge(a):
    parts = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir) / f"sp_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): raise FileNotFoundError(f"missing shard {s}: {fp}")
        parts.append(json.load(open(fp)))
    per = [pp for part in parts for pp in part["per_problem"]]
    N = parts[0]["n_samples"]
    ks = [k for k in [1,2,4,8,16,32,64,128,256] if k <= N]
    iid = {k: sum(_passk_iid(p["samples"], k) for p in per) / len(per) for k in ks}
    strat = {k: sum(_passk_strat(p["samples"], k) for p in per) / len(per) for k in ks}
    # oracle strategy: best single strategy's solve rate per problem (ceiling)
    def oracle(p):
        by = {}
        for s in p["samples"]: by.setdefault(s["strategy"], []).append(s["correct"])
        return max((sum(v) / len(v) for v in by.values()), default=0.0)
    A_oracle = sum(oracle(p) for p in per) / len(per)
    out = {"tag": a.tag, "model": parts[0]["model"], "dataset": parts[0]["dataset"],
           "n_problems": len(per), "n_samples": N, "ks": ks,
           "iid_passk": iid, "stratified_passk": strat, "oracle_strategy_rate": A_oracle,
           "per_problem": per}
    fp = Path(a.output_dir) / f"sp_{a.tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag}] MERGED {len(per)} problems, N={N}")
    print(f"  {'k':>4} {'iid':>8} {'stratified':>11} {'gain':>7}")
    for k in ks:
        print(f"  {k:>4} {iid[k]:>8.3f} {strat[k]:>11.3f} {strat[k]-iid[k]:>+7.3f}")
    print(f"  oracle-strategy rate = {A_oracle:.3f}")
    print(f"saved -> {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--dataset", default="olympiad_bench")
    ap.add_argument("--max-problems", type=int, default=150); ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="sp")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else run(a)


if __name__ == "__main__":
    main()
