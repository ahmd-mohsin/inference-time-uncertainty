# pass@k evaluation — the decisive metric for docs/RL.md.
#
# Generates n_samples completions/problem with vLLM, computes the full pass@k curve.
# Run on base model, GRPO baseline, and our checkpoints; the success criterion is that
# OUR curve matches/beats GRPO at small k AND matches/beats the BASE model at large k
# (i.e. the Yue et al. sharpening crossover disappears).
#
# Reuses: src.data.dataset (get_inference_dataset, format_prompt, extract_numeric_answer,
# answers_match) and verification_gap.run_gap.pass_at_k (exact combinatorial formula).

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_inference_dataset, format_prompt, extract_numeric_answer, answers_match
from verification_gap.run_gap import pass_at_k
from rl_training.config import EvalConfig
from rl_training.safe_match import safe_is_correct


def _load_subset_ids(difficulty_json, labels):
    """problem_ids whose difficulty label is in `labels` (e.g. {'hard'}). None if no json."""
    if not difficulty_json or not os.path.exists(difficulty_json):
        return None
    d = json.load(open(difficulty_json))
    return {p["problem_id"] for p in d.get("per_problem", []) if p.get("label") in labels}


def evaluate(cfg: EvalConfig, shard_index=0, num_shards=1, difficulty_json="",
             subset_labels=("hard",), seed=None, level=""):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    from rl_training.model_utils import merge_adapter_if_needed

    # a trained arm's output dir (oursA/grpo/oursAB) is a bare LoRA adapter; vLLM needs a full
    # model, so merge into base first. The base arm passes a full HF id and this is a no-op.
    cfg.model_path = merge_adapter_if_needed(cfg.model_path)

    problems = get_inference_dataset({"dataset": {"name": cfg.dataset, "split": "test",
                                                  "n_problems": cfg.n_problems, "seed": cfg.seed}})
    # HARD-BAND SUBSET (methodology fix): MATH-500 is near-saturated at 1.5B (base solves most at
    # k=256), so the crossover has little room. Restricting to the difficulty-labeled 'hard' band
    # (low pass@1 but pass@k>0 — where coverage expansion is actually possible) is where the
    # crossover and the method's advantage should be largest. Filter BEFORE the shard stride so the
    # shards remain a disjoint cover of the SAME filtered set.
    keep = _load_subset_ids(difficulty_json, set(subset_labels))
    if keep is not None:
        problems = [p for p in problems if p["problem_id"] in keep]
        print(f"[subset] {len(problems)} problems with label in {set(subset_labels)} (from {difficulty_json})")
    # LEVEL filter (methodology): for datasets that carry a MATH difficulty level (math500,
    # competition_math), keep only the requested level(s) — e.g. --level 5 for the hard-at-large-k
    # subset that has coverage headroom AND enough problems for tight CIs.
    if level:
        want = set(str(level).split(","))
        problems = [p for p in problems if str(p.get("level", "")) in want]
        print(f"[level] {len(problems)} problems at level(s) {want}")
    # DATA-PARALLEL SHARDING: with num_shards>1 this process handles only a STRIDED slice of the
    # problems (problems[shard_index::num_shards]). Strided (not contiguous) so every shard gets a
    # balanced easy/hard mix. pass@k is independent across problems, so N shards on N GPUs give an
    # ~Nx speedup; each writes passk_{tag}.shardS-of-T.json, then --merge recombines them. TP>1
    # crashes in this container, so this data-parallel split is how we use all 8 GPUs per node.
    if num_shards > 1:
        problems = problems[shard_index::num_shards]
    # clamp context to the model's limit (Qwen2.5-7B caps at 32768)
    try:
        cap = int(getattr(AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True),
                          "max_position_embeddings", cfg.max_new_tokens))
    except Exception:
        cap = cfg.max_new_tokens
    max_model_len = min(cfg.max_new_tokens + 1024, cap)

    # Env overrides for stability when running many shards/node: EVAL_GPU_MEM lowers memory
    # pressure; EVAL_ENFORCE_EAGER=1 skips CUDA-graph capture (which hangs under 8-way contention).
    _gm = float(os.environ.get("EVAL_GPU_MEM", cfg.gpu_memory_utilization))
    _eager = os.environ.get("EVAL_ENFORCE_EAGER", "0") == "1"
    llm = LLM(model=cfg.model_path, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=cfg.tensor_parallel_size, max_model_len=max_model_len,
              gpu_memory_utilization=_gm, enable_prefix_caching=True,
              enforce_eager=_eager)
    # SEED (methodology fix): distinct sampling seed per replicate so 3 seeds give independent
    # pass@k estimates for bootstrap CIs. vLLM SamplingParams.seed makes generation reproducible.
    sp = SamplingParams(n=cfg.n_samples, max_tokens=max_model_len - 1024,
                        temperature=cfg.temperature, top_p=cfg.top_p,
                        stop=["<|im_end|>", "<|endoftext|>"],
                        seed=seed if seed is not None else None)

    prompts = [format_prompt(p, cfg.model_path) for p in problems]
    outs = llm.generate(prompts, sp)

    ks = [k for k in cfg.k_values if k <= cfg.n_samples]
    per_problem, curve_acc = [], {k: [] for k in ks}
    for p, o in zip(problems, outs):
        gold = str(p.get("gold_answer", ""))
        mask = [safe_is_correct(s.text, gold)[0] for s in o.outputs]
        pk = {k: pass_at_k(mask, k) for k in ks}
        for k in ks:
            curve_acc[k].append(pk[k])
        per_problem.append({"problem_id": p["problem_id"], "gold": gold,
                            "n_correct": int(sum(mask)), "pass_at_k": pk})

    curve = {k: (float(sum(v) / len(v)) if v else 0.0) for k, v in curve_acc.items()}
    out = {"tag": cfg.tag, "model": cfg.model_path, "dataset": cfg.dataset,
           "n_problems": len(problems), "n_samples": cfg.n_samples,
           "subset": list(subset_labels) if keep is not None else "all",
           "level": level or "all", "seed": seed,
           "pass_at_k_curve": curve, "per_problem": per_problem}
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if num_shards > 1:
        fp = Path(cfg.output_dir) / f"passk_{cfg.tag}.shard{shard_index}-of-{num_shards}.json"
    else:
        fp = Path(cfg.output_dir) / f"passk_{cfg.tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    tag = f"{cfg.tag} shard {shard_index}/{num_shards}" if num_shards > 1 else cfg.tag
    print(f"[{tag}] {len(problems)} problems | " +
          " ".join(f"k={k}:{curve[k]:.3f}" for k in ks))
    print(f"saved -> {fp}")
    return out


def merge_shards(output_dir, tag, num_shards):
    """Recombine passk_{tag}.shard*-of-N.json partials into the final passk_{tag}.json.
    Concatenates per_problem across shards and recomputes the aggregate pass@k curve over ALL
    problems. Fails loudly if a shard is missing (so we never publish a partial curve)."""
    od = Path(output_dir)
    parts = []
    for s in range(num_shards):
        fp = od / f"passk_{tag}.shard{s}-of-{num_shards}.json"
        if not fp.exists():
            raise FileNotFoundError(f"missing shard {s}/{num_shards}: {fp} — cannot merge")
        parts.append(json.load(open(fp)))
    per_problem = [pp for part in parts for pp in part["per_problem"]]
    # dedup by problem_id (defensive; strided shards should already be disjoint)
    seen, dedup = set(), []
    for pp in per_problem:
        if pp["problem_id"] in seen:
            continue
        seen.add(pp["problem_id"]); dedup.append(pp)
    per_problem = dedup
    ks = sorted(int(k) for k in per_problem[0]["pass_at_k"].keys())
    curve = {k: float(sum(pp["pass_at_k"][str(k)] if str(k) in pp["pass_at_k"]
                          else pp["pass_at_k"][k] for pp in per_problem) / len(per_problem))
             for k in ks}
    out = {"tag": tag, "model": parts[0]["model"], "dataset": parts[0]["dataset"],
           "n_problems": len(per_problem), "n_samples": parts[0]["n_samples"],
           "subset": parts[0].get("subset", "all"), "level": parts[0].get("level", "all"),
           "seed": parts[0].get("seed"),
           "pass_at_k_curve": curve, "per_problem": per_problem, "merged_from_shards": num_shards}
    fp = od / f"passk_{tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{tag}] MERGED {num_shards} shards, {len(per_problem)} problems | " +
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
    ap.add_argument("--shard-index", type=int, default=0, help="this shard's index [0, num_shards)")
    ap.add_argument("--num-shards", type=int, default=1, help="data-parallel shards (1 GPU each)")
    ap.add_argument("--merge", action="store_true",
                    help="merge passk_{tag}.shard*-of-{num_shards}.json into passk_{tag}.json and exit")
    ap.add_argument("--difficulty-json", default="",
                    help="restrict eval to labeled problems (methodology: hard-band subset)")
    ap.add_argument("--subset-labels", default="hard",
                    help="comma-sep difficulty labels to keep (e.g. 'hard' or 'hard,stuck')")
    ap.add_argument("--seed", type=int, default=None,
                    help="vLLM sampling seed for this replicate (multi-seed CIs)")
    ap.add_argument("--level", default="",
                    help="MATH difficulty level filter, e.g. '5' or '4,5' (math500/competition_math)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature override (default uses EvalConfig.temperature=1.0); for temp-robustness sweeps")
    a = ap.parse_args()
    if a.merge:
        merge_shards(a.output_dir, a.tag, a.num_shards)
        return
    cfg = EvalConfig(model_path=a.model_path, dataset=a.dataset, n_problems=a.n_problems,
                     n_samples=a.n_samples, max_new_tokens=a.max_new_tokens,
                     tensor_parallel_size=a.tensor_parallel_size,
                     output_dir=a.output_dir, tag=a.tag)
    if a.temperature is not None:
        cfg.temperature = a.temperature
    evaluate(cfg, shard_index=a.shard_index, num_shards=a.num_shards,
             difficulty_json=a.difficulty_json,
             subset_labels=tuple(s for s in a.subset_labels.split(",") if s),
             seed=a.seed, level=a.level)


if __name__ == "__main__":
    main()
