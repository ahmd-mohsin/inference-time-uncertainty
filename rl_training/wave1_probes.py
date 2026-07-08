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
from src.data.dataset import (get_inference_dataset, format_prompt,
                              extract_numeric_answer, answers_match)
from rl_training.model_utils import merge_adapter_if_needed


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
def probe_gen_verify(a):
    """For each problem: (1) can the model GENERATE a correct answer in k samples? (2) given a
    correct candidate, can it VERIFY it as correct? Gap = verifies-but-can't-generate."""
    from verification_gap.verifier import build_verify_prompt, parse_verdict
    llm, model = get_llm(a.model, a.max_new_tokens)
    probs = load_problems(a.dataset, a.n_problems)
    done = done_ids(a.out)
    probs = [p for p in probs if p["problem_id"] not in done]
    # (1) generation pass@k
    gen = llm.generate([format_prompt(p, model) for p in probs], sp(a.k, a.max_new_tokens - 1024))
    fout = open(a.out, "a")
    for p, o in zip(probs, gen):
        gold = str(p.get("gold_answer", ""))
        gen_correct = [bool((pr := extract_numeric_answer(s.text)) is not None and answers_match(pr, gold))
                       for s in o.outputs]
        can_generate = any(gen_correct)
        # (2) verification: ask the model if the GOLD answer is correct for this problem
        vprompt = build_verify_prompt(p["question"], gold, model)
        vout = llm.generate([vprompt], sp(a.verify_k, 512))[0]
        # use the verifier's own VERDICT: YES/NO parser; verify_rate = fraction judged correct
        verds = [parse_verdict(s.text) for s in vout.outputs]
        verds = [v for v in verds if v is not None]
        verify_rate = (sum(verds) / len(verds)) if verds else 0.0
        fout.write(json.dumps({"problem_id": p["problem_id"], "can_generate": can_generate,
                               "gen_pass_frac": sum(gen_correct)/len(gen_correct),
                               "verify_rate": verify_rate,
                               "gap": (not can_generate) and verify_rate > 0.5}) + "\n"); fout.flush()
    fout.close(); Path(a.out + ".DONE").touch()
    _summarize_gen_verify(a.out)


def _summarize_gen_verify(out):
    R = [json.loads(l) for l in open(out) if l.strip()]
    gap = [r for r in R if r["gap"]]
    print(f"#9 gen_verify: {len(R)} problems | can't-generate-but-verifies (GAP)={len(gap)} "
          f"| mean verify_rate on ungenerated={sum(r['verify_rate'] for r in R if not r['can_generate'])/max(1,sum(1 for r in R if not r['can_generate'])):.3f}")


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
            solved = any(answers_match(extract_numeric_answer(s.text), gold) for s in out.outputs)
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


# ---------- #5: solution-mode clustering ----------
def probe_modes(a):
    """For problems the model solves, how many DISTINCT correct solution methods does it produce?
    Cluster correct CoTs by TF-IDF novelty (reuse Component A embedding)."""
    from rl_training.semantic import embed_texts
    import numpy as np
    llm, model = get_llm(a.model, a.max_new_tokens)
    probs = load_problems(a.dataset, a.n_problems)
    done = done_ids(a.out)
    probs = [p for p in probs if p["problem_id"] not in done]
    gen = llm.generate([format_prompt(p, model) for p in probs], sp(a.k, a.max_new_tokens - 1024))
    fout = open(a.out, "a")
    for p, o in zip(probs, gen):
        gold = str(p.get("gold_answer", ""))
        correct = [s.text for s in o.outputs if answers_match(extract_numeric_answer(s.text), gold)]
        n_modes = 0
        if len(correct) >= 2:
            emb = embed_texts(correct)               # (n,d) L2-normalized
            # greedy distinct-mode count: a rollout is a new mode if cos-dist>thresh to all kept
            kept = [emb[0]]
            for v in emb[1:]:
                if all(float(1 - v @ k) > a.mode_thresh for k in kept):
                    kept.append(v)
            n_modes = len(kept)
        elif len(correct) == 1:
            n_modes = 1
        fout.write(json.dumps({"problem_id": p["problem_id"], "n_correct": len(correct),
                               "n_distinct_modes": n_modes}) + "\n"); fout.flush()
    fout.close(); Path(a.out + ".DONE").touch()
    R = [json.loads(l) for l in open(a.out) if l.strip()]
    solved = [r for r in R if r["n_correct"] > 0]
    mm = sum(r["n_distinct_modes"] for r in solved) / max(1, len(solved))
    print(f"#5 modes: {len(solved)} solved problems | mean distinct correct modes = {mm:.2f}")


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
            res.append(bool(answers_match(extract_numeric_answer(out.outputs[0].text), gold)))
        fout.write(json.dumps({"problem_id": p["problem_id"], "perturb_correct": res,
                               "flip": len(set(res)) > 1}) + "\n"); fout.flush()
    fout.close(); Path(a.out + ".DONE").touch()
    R = [json.loads(l) for l in open(a.out) if l.strip()]
    flips = sum(r["flip"] for r in R)
    print(f"#10 brittleness: {len(R)} problems | answer FLIPPED under trivial perturbation = {flips} "
          f"({100*flips/max(1,len(R)):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    choices=["gen_verify", "prompt_recover", "modes", "brittleness"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n-problems", type=int, default=-1)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--verify-k", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--mode-thresh", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    {"gen_verify": probe_gen_verify, "prompt_recover": probe_prompt_recover,
     "modes": probe_modes, "brittleness": probe_brittleness}[a.probe](a)


if __name__ == "__main__":
    main()
