# Verification-Generation gap: per-problem pipeline.
#
#   Phase 1 (generate):  N chains per problem -> candidate answers + correctness.
#   Phase 2 (verify):    for every DISTINCT candidate answer, ask the model YES/NO; also
#                        verify the gold answer (so a correct candidate always exists) and
#                        a synthetic wrong answer (so a wrong candidate always exists) ->
#                        guarantees both classes for AUC even when all chains agree.
#   Phase 3 (metrics):   G = pass@k; V = AUC(P(YES) vs correctness); selection lift =
#                        verifier-best-of-N acc - majority-vote acc. One JSON per problem.
#
# ONE persistent vLLM (generation + verification share the engine). Data-parallel across
# GPUs via --shard-index/--num-shards (round-robin), then --merge-only writes summary.
#
# Usage:
#   python -m verification_gap.run_gap --model Qwen/Qwen3-4B --dataset aime_all \
#       --n-problems 90 --n-chains 16 --output-dir data/verification_gap_qwen4b \
#       [--shard-index i --num-shards 8] [--merge-only]

import argparse, json, logging, os, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification_gap.config import GapConfig
from verification_gap.verifier import (extract_boxed, answers_match,
                                        build_verify_prompt, parse_verdict)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _suffix(si, ns): return f"_shard{si}" if ns > 1 else ""


def pass_at_k(correct_mask, k, n_trials=1000):
    """P(>=1 correct in k draws without replacement) via the exact complement formula."""
    import numpy as np
    n = len(correct_mask); c = int(sum(correct_mask))
    if c == 0: return 0.0
    if c == n or k >= n: return 1.0 if c > 0 else 0.0
    # exact: 1 - C(n-c, k)/C(n, k)
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


def auc(scores, labels):
    """Rank AUC of scores separating labels (bool). 0.5 = chance; nan if one class only."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg: return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def synthetic_wrong(gold: str) -> str:
    """A guaranteed-wrong candidate (gold+1, or '0'), so the AUC always has a neg class."""
    try:
        return str(int(float(gold)) + 1)
    except (ValueError, TypeError):
        return "0" if str(gold).strip() != "0" else "1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=GapConfig.model_name)
    p.add_argument("--dataset", default=GapConfig.dataset)
    p.add_argument("--n-problems", type=int, default=GapConfig.n_problems)
    p.add_argument("--n-chains", type=int, default=GapConfig.n_chains)
    p.add_argument("--output-dir", default=GapConfig.output_dir)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    cfg = GapConfig(); cfg.model_name = args.model; cfg.dataset = args.dataset
    cfg.n_problems = args.n_problems; cfg.n_chains = args.n_chains
    # auto-derive output dir from model+dataset if caller passed the sentinel "auto",
    # so multi-model x multi-dataset sweeps never collide.
    if args.output_dir == "auto":
        tag = args.model.split("/")[-1].lower().replace(".", "")
        cfg.output_dir = f"data/vgap_{tag}_{args.dataset}"
    else:
        cfg.output_dir = args.output_dir
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge(out, args.num_shards); return

    from src.data.dataset import get_inference_dataset
    from verification_gap.run_gap import pass_at_k  # noqa (keep namespaced)
    problems = get_inference_dataset({"dataset": {"name": cfg.dataset, "split": "test",
                                                  "n_problems": cfg.n_problems, "seed": cfg.seed}})
    if args.num_shards > 1:
        problems = [pr for k, pr in enumerate(problems) if k % args.num_shards == args.shard_index]
        logger.info(f"SHARD {args.shard_index}/{args.num_shards}: {len(problems)} problems")

    from vllm import LLM, SamplingParams
    from src.data.dataset import format_prompt
    logger.info(f"Loading vLLM {cfg.model_name} (TP={cfg.tensor_parallel_size})")
    llm = LLM(model=cfg.model_name, dtype=cfg.dtype, tensor_parallel_size=cfg.tensor_parallel_size,
              trust_remote_code=True, max_model_len=cfg.max_new_tokens + 1024,
              enable_prefix_caching=True, gpu_memory_utilization=cfg.gpu_memory_utilization)

    gen_sp = SamplingParams(n=cfg.n_chains, max_tokens=cfg.max_new_tokens,
                            temperature=cfg.gen_temperature, top_p=cfg.gen_top_p,
                            stop=["<|im_end|>", "<|endoftext|>"])
    ver_sp = SamplingParams(n=cfg.n_verify_samples, max_tokens=cfg.verify_max_tokens,
                            temperature=cfg.verify_temperature,
                            stop=["<|im_end|>", "<|endoftext|>"])

    for i, prob in enumerate(problems):
        pid = prob.get("problem_id", i)
        fp = out / f"gap_{pid}.json"
        if fp.exists():
            logger.info(f"  problem {pid} done, skip"); continue
        q = prob["question"]; gold = prob.get("gold_answer", "")

        # ---- Phase 1: generate N chains ----
        gp = format_prompt(prob, cfg.model_name)
        gout = llm.generate([gp], gen_sp)[0]
        chains = []
        for o in gout.outputs:
            ans = extract_boxed(o.text)
            chains.append({"answer": ans, "correct": answers_match(ans, gold),
                           "truncated": o.finish_reason == "length",
                           "n_tokens": len(o.token_ids) if o.token_ids else 0})
        answers = [c["answer"] for c in chains]
        correct_mask = [c["correct"] for c in chains]

        # ---- Phase 2: build candidate set & verify ----
        # distinct non-blank produced answers + gold + synthetic-wrong (dedup, keep labels)
        cand = {}
        for a in answers:
            if a.strip(): cand.setdefault(a, answers_match(a, gold))
        cand.setdefault(gold, True)                       # guarantee a correct candidate
        sw = synthetic_wrong(gold); cand.setdefault(sw, False)  # guarantee a wrong candidate
        cand_list = list(cand.items())                    # [(answer, is_correct)]

        vprompts = [build_verify_prompt(q, a, cfg.model_name) for a, _ in cand_list]
        vouts = llm.generate(vprompts, ver_sp)
        verifier = []
        for (a, is_corr), vo in zip(cand_list, vouts):
            verdicts = [parse_verdict(s.text) for s in vo.outputs]
            verdicts = [v for v in verdicts if v is not None]
            p_yes = sum(verdicts) / len(verdicts) if verdicts else None
            verifier.append({"answer": a, "is_correct": is_corr, "p_yes": p_yes,
                             "produced": a in answers})

        # ---- Phase 3: metrics ----
        N = len(chains)
        G_passk = {k: pass_at_k(correct_mask, k) for k in [1, 2, 4, 8, 16, N] if k <= N}
        # majority vote
        nonblank = [a for a in answers if a.strip()]
        mv = Counter(nonblank).most_common(1)[0][0] if nonblank else ""
        mv_correct = answers_match(mv, gold)
        # verifier discrimination AUC over candidates that got a parseable verdict
        scored = [(v["p_yes"], v["is_correct"]) for v in verifier if v["p_yes"] is not None]
        V_auc = auc([s for s, _ in scored], [l for _, l in scored]) if scored else None
        # verifier-best-of-N selection: among PRODUCED candidates, pick highest p_yes
        produced_scored = [v for v in verifier if v["produced"] and v["p_yes"] is not None]
        if produced_scored:
            best = max(produced_scored, key=lambda v: v["p_yes"])
            vsel_correct = bool(best["is_correct"])
        else:
            vsel_correct = mv_correct  # fallback: nothing to select from

        rec = {
            "problem_id": pid, "gold": gold, "n_chains": N,
            "answers": answers, "correct_mask": correct_mask,
            "n_correct": int(sum(correct_mask)), "n_truncated": sum(c["truncated"] for c in chains),
            "G_pass_at_k": G_passk,
            "G": G_passk.get(N, 0.0),                 # headline generation = pass@N
            "majority_vote": mv, "mv_correct": mv_correct,
            "verifier": verifier,
            "V_auc": V_auc,                            # headline verification
            "gap": (V_auc - G_passk.get(N, 0.0)) if V_auc is not None else None,
            "verifier_select_correct": vsel_correct,
            "selection_lift": int(vsel_correct) - int(mv_correct),
        }
        json.dump(rec, open(fp, "w"), indent=2)
        logger.info(f"  pid {pid}: G(pass@{N})={rec['G']:.2f} V_auc="
                    f"{V_auc if V_auc is None else round(V_auc,2)} gap="
                    f"{rec['gap'] if rec['gap'] is None else round(rec['gap'],2)} "
                    f"mv={'OK' if mv_correct else 'X'} vsel={'OK' if vsel_correct else 'X'} "
                    f"trunc={rec['n_truncated']}/{N}")

    if args.num_shards == 1:
        merge(out, 1)
    logger.info(f"SHARD {args.shard_index} COMPLETE")


def merge(out: Path, num_shards: int):
    recs = [json.load(open(f)) for f in sorted(out.glob("gap_*.json"))]
    if not recs:
        logger.warning("no gap_*.json to merge"); return
    import statistics as st
    judged = [r for r in recs if r["V_auc"] is not None]
    summ = {
        "n_problems": len(recs),
        "mean_G": round(st.mean(r["G"] for r in recs), 3),
        "mean_V_auc": round(st.mean(r["V_auc"] for r in judged), 3) if judged else None,
        "mean_gap": round(st.mean(r["gap"] for r in judged), 3) if judged else None,
        "mv_accuracy": round(st.mean(r["mv_correct"] for r in recs), 3),
        "verifier_select_accuracy": round(st.mean(r["verifier_select_correct"] for r in recs), 3),
        "mean_selection_lift": round(st.mean(r["selection_lift"] for r in recs), 3),
        "total_truncation_rate": round(sum(r["n_truncated"] for r in recs) /
                                        sum(r["n_chains"] for r in recs), 3),
    }
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    logger.info(f"SUMMARY: {summ}")


if __name__ == "__main__":
    main()
