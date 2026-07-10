# ============================================================================
# Wave-1 probes — 5 eval-only experiments on the pass@k / crossover story.
# All share ONE vLLM generation core, all resumable (per-problem JSONL + DONE flag), so an
# instance death continues from the last finished problem.
#
#   #9  gen_verify   : does the model VERIFY correct solutions it can no longer GENERATE?
#                      (generation-verification gap) — motivates harvesting.
#   #11 prompt_recover: for a model's UNsolved problems, do prompt REPHRASINGS recover them?
#                      (latent capability retrievable by prompting, not just sampling)
#   #5  modes        : cluster correct CoTs per problem -> how many DISTINCT solution modes?
#                      (does RL collapse solution diversity even where pass@k is unchanged?)
#   #10 brittleness  : pass@1 under prompt perturbations (paraphrase/format noise) -> robustness.
#   #6  ckpt_ensemble: sample from a MIX of checkpoints -> does the ensemble's pass@k beat any
#                      single checkpoint? (coverage distributed across training time) [multi-model]
#
# Usage (one probe per invocation, pin GPU):
#   python -m rl_training.wave1_probes --probe gen_verify   --model <path> --dataset math500 ...
#   python -m rl_training.wave1_probes --probe prompt_recover --model <path> ...
#   python -m rl_training.wave1_probes --probe modes        --model <path> ...
#   python -m rl_training.wave1_probes --probe brittleness  --model <path> ...
# (ckpt_ensemble is post-hoc: give it several per-problem generation dumps and it mixes them.)
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.model_utils import merge_adapter_if_needed
from rl_training.safe_match import safe_is_correct  # all answer-matching goes through this (timeout-guarded)


def get_llm(model, max_len):
    from vllm import LLM
    from transformers import AutoConfig
    model = merge_adapter_if_needed(model)
    try:
        cap = int(getattr(AutoConfig.from_pretrained(model, trust_remote_code=True),
                          "max_position_embeddings", max_len))
    except Exception:
        cap = max_len
    return LLM(model=model, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
               max_model_len=min(max_len + 1024, cap), gpu_memory_utilization=0.9,
               enable_prefix_caching=True), model


def sp(n, max_new, temp=1.0):
    from vllm import SamplingParams
    return SamplingParams(n=n, max_tokens=max_new, temperature=temp, top_p=1.0,
                          stop=["<|im_end|>", "<|endoftext|>"])


def load_problems(dataset, n):
    return get_inference_dataset({"dataset": {"name": dataset, "split": "test",
                                              "n_problems": n, "seed": 42}})


def done_ids(out_jsonl):
    if not os.path.exists(out_jsonl):
        return set()
    return {json.loads(l)["problem_id"] for l in open(out_jsonl) if l.strip()}


# ---------- #9: generation vs verification ----------
def _wrong_candidate(gold):
    """A PLAUSIBLE wrong answer for the negative control. We must present a known-wrong
    candidate alongside the gold one so verification is scored as DISCRIMINATION (YES-on-correct
    minus YES-on-wrong), never raw yes-rate — otherwise a yes-biased model 'verifies' everything
    and fabricates a gap. Numeric gold -> perturb the number (off-by-one / scaled); non-numeric
    -> a generic distractor. Returns a string distinct from gold."""
    g = str(gold).strip()
    try:
        val = float(g)
        if abs(val - round(val)) < 1e-9:          # integer-valued
            iv = int(round(val))
            cand = iv + 1 if iv != -1 else iv + 2  # +1 (avoid landing on 0 as the only option)
            return str(cand)
        return str(round(val + 1, 6))              # non-integer: shift by 1
    except (ValueError, TypeError):
        # non-numeric (fraction/set/expr): a simple, obviously-different string
        return "0" if g not in ("0", "$0$") else "1"


def _verify_yes_rate(llm, model, question, candidate, verify_k):
    """Ask the model verify_k times whether `candidate` is correct for `question`; return the
    fraction of PARSEABLE verdicts that were YES, and how many parsed (to flag base-model
    template mismatch where few verdicts parse)."""
    from verification_gap.verifier import build_verify_prompt, parse_verdict
    vprompt = build_verify_prompt(question, candidate, model)
    vout = llm.generate([vprompt], sp(verify_k, 512))[0]
    verds = [parse_verdict(s.text) for s in vout.outputs]
    verds = [v for v in verds if v is not None]
    return ((sum(verds) / len(verds)) if verds else None), len(verds)


def probe_gen_verify(a):
    """For each problem: (1) can the model GENERATE a correct answer in k samples? (2) does it
    VERIFY the GOLD answer as correct AND REJECT a plausible wrong answer? The verification
    signal is DISCRIMINATION = yes_rate(correct) - yes_rate(wrong) in [-1, 1]; the gen-verify
    GAP = can't-generate but discriminates (disc >= a.disc_thresh). Raw yes-rate alone is
    meaningless under a yes-biased model, hence the mandatory wrong-answer control."""
    llm, model = get_llm(a.model, a.max_new_tokens)
    probs = load_problems(a.dataset, a.n_problems)
    done = done_ids(a.out)
    probs = [p for p in probs if p["problem_id"] not in done]
    # (1) generation pass@k
    gen = llm.generate([format_prompt(p, model) for p in probs], sp(a.k, a.max_new_tokens - 1024))
    fout = open(a.out, "a")
    for p, o in zip(probs, gen):
        gold = str(p.get("gold_answer", ""))
        gen_correct = [safe_is_correct(s.text, gold)[0] for s in o.outputs]
        can_generate = any(gen_correct)
        # (2) verification with a NEGATIVE CONTROL: gold candidate AND a plausible wrong one.
        wrong = _wrong_candidate(gold)
        yes_correct, n_c = _verify_yes_rate(llm, model, p["question"], gold, a.verify_k)
        yes_wrong,   n_w = _verify_yes_rate(llm, model, p["question"], wrong, a.verify_k)
        # discrimination is only defined if BOTH sides produced parseable verdicts
        disc = (yes_correct - yes_wrong) if (yes_correct is not None and yes_wrong is not None) else None
        fout.write(json.dumps({
            "problem_id": p["problem_id"], "can_generate": can_generate,
            "gen_pass_frac": sum(gen_correct)/len(gen_correct),
            "yes_rate_correct": yes_correct, "yes_rate_wrong": yes_wrong,
            "n_verdicts_correct": n_c, "n_verdicts_wrong": n_w, "wrong_candidate": wrong,
            "discrimination": disc,
            # the gap claim: cannot generate a correct answer, yet reliably discriminates correct
            # from wrong when shown them -> the capability is present but inaccessible to sampling.
            "gap": (not can_generate) and (disc is not None and disc >= a.disc_thresh),
        }) + "\n"); fout.flush()
    fout.close(); Path(a.out + ".DONE").touch()
    _summarize_gen_verify(a.out)


def _summarize_gen_verify(out):
    R = [json.loads(l) for l in open(out) if l.strip()]
    ungen = [r for r in R if not r["can_generate"]]
    scored = [r for r in R if r.get("discrimination") is not None]
    ungen_scored = [r for r in ungen if r.get("discrimination") is not None]
    gap = [r for r in R if r["gap"]]
    def _mean(xs): return (sum(xs) / len(xs)) if xs else float("nan")
    print(f"#9 gen_verify: {len(R)} problems | {len(scored)} with valid discrimination "
          f"(verdicts parsed on both sides).")
    print(f"  mean discrimination (all)      = {_mean([r['discrimination'] for r in scored]):.3f} "
          f"(yes_correct - yes_wrong; >0 means real verification)")
    print(f"  mean discrimination (ungen)    = {_mean([r['discrimination'] for r in ungen_scored]):.3f} "
          f"on {len(ungen_scored)} problems the model CANNOT generate")
    print(f"  GAP (can't-generate & discriminates) = {len(gap)}  <- the gen-verify gap")
    if len(scored) < 0.5 * len(R):
        print(f"  WARNING: only {len(scored)}/{len(R)} problems had parseable verdicts on both "
              f"sides — likely a base (non-chat) model; interpret with care.")


# ---------- #11: prompt-rephrasing recovery ----------
PROMPT_VARIANTS = [
    None,  # default
    "Try a different approach than usual. ",
    "Think step by step and consider multiple methods before answering. ",
    "This is a hard problem; reason very carefully. ",
    "Solve it using an alternative or creative method. ",
]
def probe_prompt_recover(a):
    """For problems the model fails at default prompt (pass@k=0), do rephrasings recover them?"""
    llm, model = get_llm(a.model, a.max_new_tokens)
    probs = load_problems(a.dataset, a.n_problems)
    done = done_ids(a.out)
    probs = [p for p in probs if p["problem_id"] not in done]
    fout = open(a.out, "a")
    for p in probs:
        gold = str(p.get("gold_answer", ""))
        rec = {"problem_id": p["problem_id"], "variant_solved": {}}
        for vi, pref in enumerate(PROMPT_VARIANTS):
            q = (pref + p["question"]) if pref else p["question"]
            pr2 = dict(p); pr2["question"] = q
            out = llm.generate([format_prompt(pr2, model)], sp(a.k, a.max_new_tokens - 1024))[0]
            solved = any(safe_is_correct(s.text, gold)[0] for s in out.outputs)
            rec["variant_solved"][vi] = solved
        rec["default_solved"] = rec["variant_solved"][0]
        rec["recovered_by_variant"] = (not rec["variant_solved"][0]) and any(rec["variant_solved"][i] for i in range(1, len(PROMPT_VARIANTS)))
        fout.write(json.dumps(rec) + "\n"); fout.flush()
    fout.close(); Path(a.out + ".DONE").touch()
    R = [json.loads(l) for l in open(a.out) if l.strip()]
    unsolved = [r for r in R if not r["default_solved"]]
    rec = [r for r in R if r["recovered_by_variant"]]
    print(f"#11 prompt_recover: {len(R)} problems | default-unsolved={len(unsolved)} | "
          f"RECOVERED by a rephrasing={len(rec)} ({100*len(rec)/max(1,len(unsolved)):.1f}% of unsolved)")


# ---------- #5: solution-mode (LEXICAL) diversity ----------
# HONEST SCOPE: rl_training.semantic.embed_texts is TF-IDF word n-grams, NOT neural sentence
# embeddings (it is model-free by design so it survives DeepSpeed ZeRO-3 in the training reward).
# So this probe measures LEXICAL diversity of correct chains — distinct solution VOCABULARY /
# operator structure — not guaranteed-semantic method identity. We therefore (a) label it as
# lexical, and (b) never rely on one arbitrary threshold: we persist the full greedy-distinct
# count across a THRESHOLD SWEEP so the trend (does RL reduce diversity?) is judged by the whole
# curve, and also store mean pairwise distance (threshold-free) as the primary scalar.
_MODE_THRESH_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _greedy_modes(emb, thresh):
    """Greedy distinct-cluster count: a vector is a new mode if its cosine distance to every
    kept representative exceeds `thresh`."""
    kept = [emb[0]]
    for v in emb[1:]:
        if all(float(1 - v @ k) > thresh for k in kept):
            kept.append(v)
    return len(kept)


def probe_modes(a):
    """For problems the model solves, how LEXICALLY diverse are its correct chains? Reports, per
    problem: n_correct, mean pairwise cosine distance (threshold-free diversity), and distinct
    'modes' at each threshold in a sweep. Aggregate trend across arms answers: does RL collapse
    solution diversity even where pass@k is unchanged?"""
    from rl_training.semantic import embed_texts, pairwise_novelty
    llm, model = get_llm(a.model, a.max_new_tokens)
    probs = load_problems(a.dataset, a.n_problems)
    done = done_ids(a.out)
    probs = [p for p in probs if p["problem_id"] not in done]
    gen = llm.generate([format_prompt(p, model) for p in probs], sp(a.k, a.max_new_tokens - 1024))
    fout = open(a.out, "a")
    for p, o in zip(probs, gen):
        gold = str(p.get("gold_answer", ""))
        correct = [s.text for s in o.outputs if safe_is_correct(s.text, gold)[0]]
        rec = {"problem_id": p["problem_id"], "n_correct": len(correct),
               "mean_pairwise_dist": None, "modes_by_thresh": {}}
        if len(correct) >= 2:
            emb = embed_texts(correct)                       # (n,d) L2-normalized TF-IDF
            rec["mean_pairwise_dist"] = float(pairwise_novelty(emb).mean())  # threshold-free
            rec["modes_by_thresh"] = {str(t): _greedy_modes(emb, t) for t in _MODE_THRESH_SWEEP}
        elif len(correct) == 1:
            rec["mean_pairwise_dist"] = 0.0
            rec["modes_by_thresh"] = {str(t): 1 for t in _MODE_THRESH_SWEEP}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
    fout.close(); Path(a.out + ".DONE").touch()
    _summarize_modes(a.out)


def _summarize_modes(out):
    R = [json.loads(l) for l in open(out) if l.strip()]
    solved = [r for r in R if r["n_correct"] > 0]
    multi = [r for r in solved if r["n_correct"] >= 2]      # diversity only defined with >=2
    def _mean(xs): return (sum(xs) / len(xs)) if xs else float("nan")
    print(f"#5 modes (LEXICAL diversity): {len(solved)} solved, {len(multi)} with >=2 correct chains.")
    print(f"  mean pairwise TF-IDF distance = {_mean([r['mean_pairwise_dist'] for r in multi]):.3f} "
          f"(threshold-free; lower under RL => diversity collapse)")
    print(f"  distinct-mode count by threshold (mean over solved problems):")
    for t in _MODE_THRESH_SWEEP:
        vals = [r["modes_by_thresh"].get(str(t), 0) for r in solved]
        print(f"    thresh={t}: {_mean(vals):.2f} modes")


# ---------- #10: prompt-perturbation brittleness ----------
PERTURB = [
    lambda q: q,
    lambda q: q + " ",                          # trailing space
    lambda q: "Problem: " + q,                  # prefix
    lambda q: q.replace(". ", ".  "),           # spacing noise
    lambda q: q + "\nAnswer carefully.",        # suffix
]
def probe_brittleness(a):
    """pass@1 under prompt perturbations — does the model's accuracy wobble with trivial edits?"""
    llm, model = get_llm(a.model, a.max_new_tokens)
    probs = load_problems(a.dataset, a.n_problems)
    done = done_ids(a.out)
    probs = [p for p in probs if p["problem_id"] not in done]
    fout = open(a.out, "a")
    for p in probs:
        gold = str(p.get("gold_answer", ""))
        res = []
        for f in PERTURB:
            pr2 = dict(p); pr2["question"] = f(p["question"])
            out = llm.generate([format_prompt(pr2, model)], sp(1, a.max_new_tokens - 1024, temp=0.0))[0]
            res.append(safe_is_correct(out.outputs[0].text, gold)[0])
        fout.write(json.dumps({"problem_id": p["problem_id"], "perturb_correct": res,
                               "flip": len(set(res)) > 1}) + "\n"); fout.flush()
    fout.close(); Path(a.out + ".DONE").touch()
    R = [json.loads(l) for l in open(a.out) if l.strip()]
    flips = sum(r["flip"] for r in R)
    print(f"#10 brittleness: {len(R)} problems | answer FLIPPED under trivial perturbation = {flips} "
          f"({100*flips/max(1,len(R)):.1f}%)")


# ---------- #6: checkpoint-ensemble (post-hoc, NO GPU) ----------
def probe_ckpt_ensemble(a):
    """Coverage-union across checkpoints: given several passk eval JSONs (--ensemble-evals a,b,c),
    a problem is 'solved' if ANY checkpoint solves it within k. Tests whether coverage is
    distributed across training time and reassemblable for free at inference.
    Compares the ensemble's solvable-count to each single checkpoint's."""
    paths = [p for p in a.ensemble_evals.split(",") if p]
    evals = {p: {x["problem_id"]: x for x in json.load(open(p))["per_problem"]} for p in paths}
    any_e = next(iter(evals.values()))
    ids = sorted(any_e); K = sorted(int(k) for k in any_e[ids[0]]["pass_at_k"])
    def solv(ev, pid, k): return ev[pid]["pass_at_k"][str(k)] > 0.5
    rows = {}
    for k in K:
        singles = {p: sum(solv(ev, pid, k) for pid in ids) for p, ev in evals.items()}
        ens = sum(any(solv(ev, pid, k) for ev in evals.values()) for pid in ids)
        rows[k] = {"single_solvable": singles, "ensemble_solvable": ens, "best_single": max(singles.values())}
    json.dump({"k_values": K, "per_k": rows}, open(a.out, "w"), indent=2)
    print("#6 ckpt_ensemble (solvable problems, ensemble vs best single checkpoint):")
    for k in K:
        r = rows[k]
        gain = r["ensemble_solvable"] - r["best_single"]
        print(f"  k={k:>3}: ensemble={r['ensemble_solvable']} best_single={r['best_single']} "
              f"gain={gain:+d}{'  <-- ensemble beats any single' if gain>0 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    choices=["gen_verify", "prompt_recover", "modes", "brittleness", "ckpt_ensemble"])
    ap.add_argument("--model", default="", help="model path/id (not needed for ckpt_ensemble)")
    ap.add_argument("--ensemble-evals", default="", help="comma-sep passk eval jsons (ckpt_ensemble)")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n-problems", type=int, default=-1)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--verify-k", type=int, default=8)
    ap.add_argument("--disc-thresh", type=float, default=0.5,
                    help="#9: min (yes_correct - yes_wrong) discrimination to count a gen-verify gap")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--mode-thresh", type=float, default=0.3,
                    help="#5: cosine-distance threshold for a 'distinct' mode (a sweep is also reported)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    {"gen_verify": probe_gen_verify, "prompt_recover": probe_prompt_recover,
     "modes": probe_modes, "brittleness": probe_brittleness,
     "ckpt_ensemble": probe_ckpt_ensemble}[a.probe](a)


if __name__ == "__main__":
    main()
