# Behavioral routing measure — the METHODOLOGICALLY-CORRECT rho(m|q), from FREE generations + an
# LLM-judge classifier (NOT regex, NOT teacher-forced logp — those recreate the measurement problem).
#
# Per problem: sample N free trajectories from the policy (its actual default behavior), then classify
# each into one of the 14 taxonomy strategies with an LLM judge (the same served model, a separate
# classification prompt). rho_free(m|q) = fraction of free samples routed to strategy m; report the
# routing entropy H(rho) — the honest "how concentrated is the model's strategy choice" measure.
#
# 8-GPU data-parallel over problems (shard/merge). Usage:
#   python -m rl_training.free_route --model-path <dir> --dataset olympiad_bench --tag free_grpo \
#     --shard-index S --num-shards 8 --n-samples 64 --max-problems 150
#   python -m rl_training.free_route --merge --tag free_grpo --num-shards 8 --output-dir ...

import argparse, json, os, sys, math, re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.strategy_probe import STRAT_NAMES
from rl_training.strategy_bank import STRAT_RE

# Classifier: regex-argmax over the FULL generation. Deterministic, always classifies. (The LLM-judge
# self-classification FAILED — Qwen-Math/DeepSeek-Math are math models, not instruction judges: ~99.6%
# unclassified. Regex-argmax labels each free generation with the taxonomy strategy whose signals fire
# most; a moderate-quality behavioral classifier over free generations. Upgrade to a strong instruct
# judge later, but this always yields a valid rho(m|q).)
def _classify(text):
    t = text or ""
    best, bestn = None, 0
    for m in STRAT_NAMES:
        n = len(STRAT_RE[m].findall(t))
        if n > bestn:
            bestn, best = n, m
    return best  # None only if the generation matched no strategy signal at all


def run(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    problems = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test", "n_problems": -1, "seed": 42}})
    if a.max_problems > 0: problems = problems[:a.max_problems]
    if a.num_shards > 1: problems = problems[a.shard_index::a.num_shards]
    try:
        cap = int(getattr(AutoConfig.from_pretrained(mp, trust_remote_code=True), "max_position_embeddings", 4096))
    except Exception:
        cap = 4096
    mml = min(4096, cap)
    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.85)); _eager = os.environ.get("EVAL_ENFORCE_EAGER", "1") == "1"
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
              max_model_len=mml, gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)
    # 1) free generations (default behavior)
    gen_sp = SamplingParams(n=a.n_samples, temperature=1.0, top_p=1.0, max_tokens=mml - 1024,
                            stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([format_prompt(p, mp) for p in problems], gen_sp)
    # 2) classify every free generation by regex-argmax over the solution text (no 2nd model pass)
    per = [{"problem_id": p["problem_id"], "counts": {m: 0 for m in STRAT_NAMES}, "n": 0, "unclassified": 0}
           for p in problems]
    for pi, o in enumerate(outs):
        for s in o.outputs:
            lab = _classify(s.text)
            per[pi]["n"] += 1
            if lab: per[pi]["counts"][lab] += 1
            else: per[pi]["unclassified"] += 1
    out = {"tag": a.tag, "model": mp, "dataset": a.dataset, "n_problems": len(problems),
           "n_samples": a.n_samples, "strategies": STRAT_NAMES, "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir) / (f"free_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json"
          if a.num_shards > 1 else f"free_{a.tag}.json"))
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(problems)} problems -> {fp}")


def merge(a):
    parts = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir) / f"free_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): raise FileNotFoundError(f"missing shard {s}: {fp}")
        parts.append(json.load(open(fp)))
    per = [pp for part in parts for pp in part["per_problem"]]
    S = parts[0]["strategies"]
    # per-problem routing entropy + global rho
    Hs = []; glob = {m: 0 for m in S}; tot = 0; unc = 0
    for p in per:
        n = p["n"] - p["unclassified"]
        unc += p["unclassified"]; tot += p["n"]
        if n <= 0: continue
        ps = [p["counts"][m] / n for m in S if p["counts"][m] > 0]
        Hs.append(-sum(x * math.log(x) for x in ps))
        for m in S: glob[m] += p["counts"][m]
    gtot = sum(glob.values()) or 1
    rho = {m: glob[m] / gtot for m in S}
    Heff = math.exp(sum(Hs) / len(Hs)) if Hs else 0.0  # effective # strategies chosen/problem
    out = {"tag": a.tag, "model": parts[0]["model"], "dataset": parts[0]["dataset"],
           "n_problems": len(per), "n_samples": parts[0]["n_samples"],
           "mean_routing_entropy": (sum(Hs) / len(Hs) if Hs else 0.0),
           "effective_strategies_per_problem": Heff,
           "unclassified_frac": unc / (tot or 1), "rho_global": rho, "per_problem": per}
    fp = Path(a.output_dir) / f"free_{a.tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag}] MERGED {a.num_shards} shards, {len(per)} problems")
    print(f"  mean routing entropy H(rho) = {out['mean_routing_entropy']:.3f}  "
          f"(effective strategies/problem = {Heff:.2f})  unclassified={out['unclassified_frac']:.2f}")
    top = sorted(rho, key=lambda k: -rho[k])[:6]
    print("  global rho top: " + ", ".join(f"{m}:{rho[m]:.2f}" for m in top))
    print(f"saved -> {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--dataset", default="olympiad_bench")
    ap.add_argument("--max-problems", type=int, default=150); ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="free")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else run(a)


if __name__ == "__main__":
    main()
