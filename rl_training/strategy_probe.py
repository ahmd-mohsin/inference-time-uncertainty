# Routing-vs-competence probe — "Knowing vs Choosing: what really collapses during RL reasoning?"
#
# Decomposes pi(y|q) = sum_m rho(m|q) * pi(y|q,m) into:
#   rho(m|q)   ROUTING / accessibility : how often the DEFAULT policy chooses strategy m.
#   c(m,q)     COMPETENCE              : P(correct | q, do(M=m)) — can it EXECUTE m when forced?
#
# For one policy model + a problem set + the 14-strategy taxonomy, per problem we sample:
#   DEFAULT : default_n plain samples          -> correctness + which strategies appear (regex)
#   FORCED  : forced_n samples per strategy m   -> forced-m correctness c(m,q) + adherence(m)
# So a single run yields, for this policy: default mode mass rho, forced adherence, forced
# competence c. Cross-policy join (does GRPO keep c while dropping rho?) is done in analysis.
#
# Reuses evaluate_passk's proven 8-GPU DATA-PARALLEL sharding (strided shards, --merge), the
# safe_is_correct verifier, src.data.dataset (format_prompt/get_inference_dataset), and the
# strategy_bank taxonomy (STRAT_RE) for the adherence classifier.
#
# Usage (one shard/GPU): python -m rl_training.strategy_probe --model-path <dir> --dataset olympiad_bench \
#   --tag probe_grpo --shard-index S --num-shards 8 --default-n 8 --forced-n 4
# Merge: python -m rl_training.strategy_probe --merge --tag probe_grpo --num-shards 8 --output-dir ...

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.safe_match import safe_is_correct
from rl_training.strategy_bank import STRAT_NAMES, STRAT_RE

# Natural-language forcing instruction per taxonomy strategy. Prepended to the problem so the
# model is explicitly routed to strategy m; adherence(m) then measures whether it complied.
STRATEGY_INSTRUCTION = {
    "substitution":   "Solve this by introducing a substitution or change of variables (e.g. let u = ...). Commit to the substitution approach even if another method looks easier.",
    "induction":      "Solve this using mathematical induction: state a base case and an inductive step. Use induction even if another method looks easier.",
    "contradiction":  "Solve this by contradiction: assume the negation and derive a contradiction. Use this approach even if another method looks easier.",
    "coordinate_geo": "Solve this using coordinate/analytic geometry: place the configuration on a coordinate system and use vectors or coordinates. Use this even if a synthetic argument looks easier.",
    "synthetic_geo":  "Solve this using synthetic geometry: reason directly about angles, triangles, circles, similarity/congruence, without coordinates. Use this even if coordinates look easier.",
    "trig":           "Solve this using trigonometry (trig identities, law of sines/cosines, angle sums). Use this even if another method looks easier.",
    "casework":       "Solve this by explicit casework: split into exhaustive cases and handle each. Use casework even if a unified argument looks easier.",
    "factoring":      "Solve this by algebraic factoring (factor expressions, difference of squares, complete the square, factor to find roots). Use this even if another method looks easier.",
    "calculus":       "Solve this using calculus (derivatives/integrals, critical points, optimization). Use calculus even if an algebraic method looks easier.",
    "number_theory":  "Solve this using number theory (modular arithmetic, divisibility, gcd/lcm, primes, residues). Use this even if another method looks easier.",
    "inequality":     "Solve this using inequality techniques (AM-GM, Cauchy-Schwarz, bounding). Use this even if another method looks easier.",
    "algebraic_manip":"Solve this by direct algebraic manipulation (expand, simplify, rearrange, common denominators). Use this even if another method looks easier.",
    "counting":       "Solve this using combinatorial counting (binomial coefficients, permutations, choose, pigeonhole). Use this even if another method looks easier.",
    "generating_fn":  "Solve this using generating functions or a recurrence relation. Use this even if another method looks easier.",
}


# High-adherence PREFIX intervention: seed the assistant turn with a strategy-specific opening so
# generation is forced to continue in that strategy (vs the weak instruction, which the model ignores
# ~80% of the time). The prefix is injected AFTER "...assistant\n" and prepended back for correctness.
STRATEGY_PREFIX = {
    "substitution":   "We solve this by substitution. Introduce a new variable: let ",
    "induction":      "We proceed by mathematical induction. Let \\(P(n)\\) denote the statement to prove. Base case: ",
    "contradiction":  "We argue by contradiction. Suppose, for the sake of contradiction, that ",
    "coordinate_geo": "We set up coordinates. Place the configuration in the coordinate plane so that ",
    "synthetic_geo":  "We use a synthetic geometry argument, reasoning directly about the angles and triangles. ",
    "trig":           "We use trigonometry. Applying trigonometric identities and the law of sines/cosines, ",
    "casework":       "We split the problem into exhaustive cases. Case 1: ",
    "factoring":      "We solve by algebraic factoring. Factor the key expression: ",
    "calculus":       "We use calculus. Taking the derivative and finding the critical points, ",
    "number_theory":  "We use number theory. Working modulo an appropriate integer, ",
    "inequality":     "We apply an inequality argument. By AM-GM (or Cauchy-Schwarz), ",
    "algebraic_manip":"We proceed by direct algebraic manipulation. Expanding and simplifying, ",
    "counting":       "We use a combinatorial counting argument. Counting the number of ways, ",
    "generating_fn":  "We use generating functions. Define the generating function ",
}


def _strategies_present(text):
    """Adherence classifier: which taxonomy strategies the completion exhibits (regex proxy)."""
    return [m for m in STRAT_NAMES if STRAT_RE[m].search(text or "")]


def _load_subset_ids(difficulty_json, labels):
    if not difficulty_json or not os.path.exists(difficulty_json):
        return None
    d = json.load(open(difficulty_json))
    return {p["problem_id"] for p in d.get("per_problem", []) if p.get("label") in labels}


def run(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    from rl_training.model_utils import merge_adapter_if_needed

    model_path = merge_adapter_if_needed(a.model_path)
    problems = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test",
                                                  "n_problems": a.n_problems, "seed": 42}})
    keep = _load_subset_ids(a.difficulty_json, set(s for s in a.subset_labels.split(",") if s))
    if keep is not None:
        problems = [p for p in problems if p["problem_id"] in keep]
    if a.max_problems > 0:
        problems = problems[:a.max_problems]
    if a.num_shards > 1:
        problems = problems[a.shard_index::a.num_shards]

    try:
        cap = int(getattr(AutoConfig.from_pretrained(model_path, trust_remote_code=True),
                          "max_position_embeddings", a.max_new_tokens))
    except Exception:
        cap = a.max_new_tokens
    max_model_len = min(a.max_new_tokens + 1024, cap)

    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.90))
    _eager = os.environ.get("EVAL_ENFORCE_EAGER", "0") == "1"
    llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=1, max_model_len=max_model_len,
              gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)

    def gen(prompts, n):
        sp = SamplingParams(n=n, max_tokens=max_model_len - 1024, temperature=a.temperature,
                            top_p=a.top_p, stop=["<|im_end|>", "<|endoftext|>"])
        return llm.generate(prompts, sp)

    # ---- DEFAULT condition: one prompt/problem, default_n samples ----
    default_prompts = [format_prompt(p, model_path) for p in problems]
    default_outs = gen(default_prompts, a.default_n)

    # ---- FORCED condition: for each strategy, one prompt/problem ----
    #  instruction mode: prepend a "solve using m" instruction to the question (model may ignore).
    #  prefix mode: seed the ASSISTANT turn with a strategy-specific opening so generation must
    #               continue in that strategy (high adherence). The seed is prepended back for scoring.
    # Batch ALL (problem x strategy) forced prompts into a single generate call for throughput.
    forced_prompts, forced_index, forced_seed = [], [], []
    for pi, p in enumerate(problems):
        for m in STRAT_NAMES:
            if a.force_mode == "prefix":
                forced_prompts.append(format_prompt(p, model_path) + STRATEGY_PREFIX[m])
                forced_seed.append(STRATEGY_PREFIX[m])
            else:
                fp = dict(p)
                fp["question"] = STRATEGY_INSTRUCTION[m] + "\n\nProblem:\n" + p["question"]
                forced_prompts.append(format_prompt(fp, model_path))
                forced_seed.append("")
            forced_index.append((pi, m))
    forced_outs = gen(forced_prompts, a.forced_n) if forced_prompts else []

    per_problem = []
    for p in problems:
        per_problem.append({"problem_id": p["problem_id"], "gold": str(p.get("gold_answer", "")),
                            "default": [], "forced": {m: [] for m in STRAT_NAMES}})
    for p, o in zip(per_problem, default_outs):
        gold = p["gold"]
        for s in o.outputs:
            p["default"].append({"correct": bool(safe_is_correct(s.text, gold)[0]),
                                 "strategies": _strategies_present(s.text)})
    for (pi, m), seed, o in zip(forced_index, forced_seed, forced_outs):
        gold = per_problem[pi]["gold"]
        for s in o.outputs:
            full = (seed + (s.text or ""))  # prepend the seeded prefix (not in vLLM output) for scoring
            per_problem[pi]["forced"][m].append({"correct": bool(safe_is_correct(full, gold)[0]),
                                                 "adhered": bool(STRAT_RE[m].search(full))})

    out = {"tag": a.tag, "model": model_path, "dataset": a.dataset, "n_problems": len(problems),
           "default_n": a.default_n, "forced_n": a.forced_n, "strategies": STRAT_NAMES,
           "per_problem": per_problem}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir) / (f"probe_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json"
          if a.num_shards > 1 else f"probe_{a.tag}.json"))
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(problems)} problems -> {fp}")


def _aggregate(per_problem, strategies, default_n):
    """Headline numbers: default mode mass rho(m), forced adherence, forced competence c(m)."""
    import statistics as st
    # rho(m): fraction of DEFAULT samples exhibiting m (routing/accessibility)
    rho = {m: [] for m in strategies}
    default_correct = []
    for p in per_problem:
        d = p["default"]
        if not d:
            continue
        default_correct.append(sum(x["correct"] for x in d) / len(d))
        for m in strategies:
            rho[m].append(sum(m in x["strategies"] for x in d) / len(d))
    rho_mean = {m: (sum(v) / len(v) if v else 0.0) for m, v in rho.items()}
    # forced adherence + competence per m
    adh, comp = {m: [] for m in strategies}, {m: [] for m in strategies}
    for p in per_problem:
        for m in strategies:
            f = p["forced"].get(m, [])
            if not f:
                continue
            adh[m].append(sum(x["adhered"] for x in f) / len(f))
            comp[m].append(sum(x["correct"] for x in f) / len(f))
    adh_mean = {m: (sum(v) / len(v) if v else 0.0) for m, v in adh.items()}
    comp_mean = {m: (sum(v) / len(v) if v else 0.0) for m, v in comp.items()}
    # summary scalars
    active = [m for m in strategies if rho_mean[m] > 0.05]  # routing-diversity: strategies actually used
    return {
        "default_pass1": (sum(default_correct) / len(default_correct) if default_correct else 0.0),
        "default_mode_mass_mean": (sum(rho_mean.values()) / len(rho_mean)),
        "routing_active_strategies": len(active),   # count with rho>5% (marginal diversity)
        "forced_adherence_mean": (sum(adh_mean.values()) / len(adh_mean)),
        "forced_competence_mean": (sum(comp_mean.values()) / len(comp_mean)),
        "rho_per_strategy": rho_mean, "adherence_per_strategy": adh_mean,
        "competence_per_strategy": comp_mean,
    }


def merge(a):
    parts = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir) / f"probe_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists():
            raise FileNotFoundError(f"missing shard {s}/{a.num_shards}: {fp}")
        parts.append(json.load(open(fp)))
    per_problem = [pp for part in parts for pp in part["per_problem"]]
    strategies = parts[0]["strategies"]
    summ = _aggregate(per_problem, strategies, parts[0]["default_n"])
    out = {"tag": a.tag, "model": parts[0]["model"], "dataset": parts[0]["dataset"],
           "n_problems": len(per_problem), "default_n": parts[0]["default_n"],
           "forced_n": parts[0]["forced_n"], "strategies": strategies,
           "summary": summ, "per_problem": per_problem, "merged_from_shards": a.num_shards}
    fp = Path(a.output_dir) / f"probe_{a.tag}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag}] MERGED {a.num_shards} shards, {len(per_problem)} problems")
    print(f"  default pass@1        = {summ['default_pass1']:.3f}")
    print(f"  default mode mass     = {summ['default_mode_mass_mean']:.3f}  "
          f"(routing-active strategies rho>5%: {summ['routing_active_strategies']}/{len(strategies)})")
    print(f"  forced adherence      = {summ['forced_adherence_mean']:.3f}")
    print(f"  forced competence c   = {summ['forced_competence_mean']:.3f}")
    print(f"saved -> {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path")
    ap.add_argument("--dataset", default="olympiad_bench")
    ap.add_argument("--n-problems", type=int, default=-1)
    ap.add_argument("--max-problems", type=int, default=200, help="cap after subset/shard-free filtering")
    ap.add_argument("--default-n", type=int, default=8)
    ap.add_argument("--forced-n", type=int, default=4)
    ap.add_argument("--force-mode", default="instruction", choices=["instruction", "prefix"],
                    help="prefix = high-adherence assistant-turn seeding; instruction = weak prompt")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--difficulty-json", default="")
    ap.add_argument("--subset-labels", default="hard")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
