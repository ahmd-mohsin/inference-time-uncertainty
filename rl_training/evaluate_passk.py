# pass@k evaluation — the decisive metric for docs/RL.md.
#
# Generates n_samples completions/problem with vLLM, computes the full pass@k curve.
# Run on base model, GRPO baseline, and our checkpoints; the success criterion is that
# OUR curve matches/beats GRPO at small k AND matches/beats the BASE model at large k
# (i.e. the Yue et al. sharpening crossover disappears).
#
# Reuses: src.data.dataset (get_inference_dataset, format_prompt, extract_boxed_answer,
# answers_match) and verification_gap.run_gap.pass_at_k (exact combinatorial formula).

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_inference_dataset, format_prompt, extract_boxed_answer, answers_match
from verification_gap.run_gap import pass_at_k
from rl_training.config import EvalConfig


def evaluate(cfg: EvalConfig):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig

    problems = get_inference_dataset({"dataset": {"name": cfg.dataset, "split": "test",
                                                  "n_problems": cfg.n_problems, "seed": cfg.seed}})
    # clamp context to the model's limit (Qwen2.5-7B caps at 32768)
    try:
        cap = int(getattr(AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True),
                          "max_position_embeddings", cfg.max_new_tokens))
    except Exception:
        cap = cfg.max_new_tokens
    max_model_len = min(cfg.max_new_tokens + 1024, cap)

    llm = LLM(model=cfg.model_path, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=cfg.tensor_parallel_size, max_model_len=max_model_len,
              gpu_memory_utilization=cfg.gpu_memory_utilization, enable_prefix_caching=True)
    sp = SamplingParams(n=cfg.n_samples, max_tokens=max_model_len - 1024,
                        temperature=cfg.temperature, top_p=cfg.top_p,
                        stop=["<|im_end|>", "<|endoftext|>"])

    prompts = [format_prompt(p, cfg.model_path) for p in problems]
    outs = llm.generate(prompts, sp)

    ks = [k for k in cfg.k_values if k <= cfg.n_samples]
    per_problem, curve_acc = [], {k: [] for k in ks}
    for p, o in zip(problems, outs):
        gold = str(p.get("gold_answer", ""))
        mask = []
        for s in o.outputs:
            pred = extract_boxed_answer(s.text)
            mask.append(bool(pred is not None and answers_match(pred, gold)))
        pk = {k: pass_at_k(mask, k) for k in ks}
        for k in ks:
            curve_acc[k].append(pk[k])
        per_problem.append({"problem_id": p["problem_id"], "gold": gold,
                            "n_correct": int(sum(mask)), "pass_at_k": pk})

    curve = {k: float(sum(v) / len(v)) for k, v in curve_acc.items()}
    out = {"tag": cfg.tag, "model": cfg.model_path, "dataset": cfg.dataset,
           "n_problems": len(problems), "n_samples": cfg.n_samples,
           "pass_at_k_curve": curve, "per_problem": per_problem}
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(cfg.output_dir) / f"passk_{cfg.tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{cfg.tag}] pass@k curve: " +
          " ".join(f"k={k}:{curve[k]:.3f}" for k in ks))
    print(f"saved -> {fp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", default=EvalConfig.dataset)
    ap.add_argument("--n-problems", type=int, default=EvalConfig.n_problems)
    ap.add_argument("--n-samples", type=int, default=EvalConfig.n_samples)
    ap.add_argument("--max-new-tokens", type=int, default=EvalConfig.max_new_tokens)
    ap.add_argument("--tensor-parallel-size", type=int, default=EvalConfig.tensor_parallel_size)
    ap.add_argument("--output-dir", default=EvalConfig.output_dir)
    ap.add_argument("--tag", default=EvalConfig.tag)
    a = ap.parse_args()
    cfg = EvalConfig(model_path=a.model_path, dataset=a.dataset, n_problems=a.n_problems,
                     n_samples=a.n_samples, max_new_tokens=a.max_new_tokens,
                     tensor_parallel_size=a.tensor_parallel_size,
                     output_dir=a.output_dir, tag=a.tag)
    evaluate(cfg)


if __name__ == "__main__":
    main()
