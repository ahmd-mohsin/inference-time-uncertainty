# Technique 2, EMPIRICAL form: sample the rollout-level mixture pi_mix = (1-eps) pi_RL + eps pi_base
# with REAL generations, score pass@k, and confirm it matches the closed form in mixture_passk.py.
#
# Rollout-level mixture = for each of the k samples per problem, flip a coin: w.p. eps generate from
# the FROZEN BASE model, else from the RL model. Then score all k together for pass@k. This needs
# both models loaded; we do it as two vLLM passes (n_base = round(eps*k) samples from base,
# n_rl = k - n_base from RL) which is exactly a rollout-level mixture and avoids per-sample routing
# overhead. NO retraining — both models already exist.
#
# Purpose: verify the closed-form guarantee empirically + produce real generations for inspection.
# Usage:
#   python -m rl_training.mixture_sampler --base BASE --rl RL_ADAPTER_OR_MERGED --dataset math500 \
#       --eps 0.1 --k 256 --n-problems -1 --output-dir runs/eval --tag mix_e10 [--level 5]

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _passk_from_counts(n_correct, n, ks):
    from math import comb
    out = {}
    for k in ks:
        if n_correct == 0:
            out[k] = 0.0
        elif n - n_correct < k:
            out[k] = 1.0
        else:
            out[k] = 1.0 - comb(n - n_correct, k) / comb(n, k)
    return out


def run(base_model, rl_model, dataset, eps, k, max_new_tokens, output_dir, tag,
        n_problems=-1, level="", temperature=1.0, gpu_mem=0.9):
    from vllm import LLM, SamplingParams
    from src.data.dataset import get_inference_dataset, format_prompt
    from rl_training.safe_match import safe_is_correct
    from rl_training.model_utils import merge_adapter_if_needed

    base_model = merge_adapter_if_needed(base_model)
    rl_model = merge_adapter_if_needed(rl_model)

    problems = get_inference_dataset({"dataset": {"name": dataset, "split": "test",
                                                  "n_problems": n_problems, "seed": 42}})
    if level:
        want = set(str(level).split(","))
        problems = [p for p in problems if str(p.get("level", "")) in want]
    prompts = [format_prompt(p, rl_model) for p in problems]
    gold = [str(p.get("gold_answer", "")) for p in problems]

    n_base = round(eps * k); n_rl = k - n_base
    print(f"[mix] eps={eps} k={k} -> {n_rl} RL + {n_base} base samples/problem; {len(problems)} problems")

    def gen(model, n):
        if n == 0:
            return [[] for _ in prompts]
        llm = LLM(model=model, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
                  max_model_len=max_new_tokens + 1024, gpu_memory_utilization=gpu_mem,
                  enforce_eager=True, enable_prefix_caching=True)
        sp = SamplingParams(n=n, max_tokens=max_new_tokens, temperature=temperature, top_p=1.0,
                            stop=["<|im_end|>", "<|endoftext|>"])
        outs = llm.generate(prompts, sp)
        del llm
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
        return [[s.text for s in o.outputs] for o in outs]

    # generate RL and base samples separately, then pool per problem = rollout-level mixture
    rl_txt = gen(rl_model, n_rl)
    base_txt = gen(base_model, n_base)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ks = [x for x in ks if x <= k]
    per_problem = []
    for p, g, rt, bt in zip(problems, gold, rl_txt, base_txt):
        texts = rt + bt
        nc = sum(1 for t in texts if safe_is_correct(t, g)[0])
        per_problem.append({"problem_id": p["problem_id"], "gold": g, "n_correct": nc,
                            "n_samples": len(texts), "pass_at_k": _passk_from_counts(nc, len(texts), ks)})
    curve = {str(kk): sum(pp["pass_at_k"][kk] for pp in per_problem) / len(per_problem) for kk in ks}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fp = os.path.join(output_dir, f"passk_{tag}.json")
    json.dump({"tag": tag, "eps": eps, "dataset": dataset, "level": level or "all",
               "n_problems": len(per_problem), "n_samples": k,
               "pass_at_k_curve": curve, "per_problem": per_problem}, open(fp, "w"), indent=2)
    print(f"[mix] pass@1={curve[str(ks[0])]:.4f} pass@{ks[-1]}={curve[str(ks[-1])]:.4f} -> {fp}")
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rl", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--output-dir", default="rl_training/runs/eval")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-problems", type=int, default=-1)
    ap.add_argument("--level", default="")
    a = ap.parse_args()
    run(a.base, a.rl, a.dataset, a.eps, a.k, a.max_new_tokens, a.output_dir, a.tag,
        n_problems=a.n_problems, level=a.level)


if __name__ == "__main__":
    main()
