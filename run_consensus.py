#!/usr/bin/env python3
"""
run_consensus.py -- step-synchronous consensus decoding on a HF model.

Collects per-problem TRAJECTORIES (every step's branch set, consensus
distribution, decision, and resolution) plus the training signal needed to
later choose SFT / step-KL / preference-RL. Writes:

  <out_dir>/trajectories.jsonl   one rich record per problem
  <out_dir>/metrics.json         aggregate accuracy + decision/stop telemetry

Usage:
  python run_consensus.py \
    --config configs/dad_config_qwen3_8b.yaml \
    --dataset aime_2024 --n_problems -1 \
    --prompt_style stepwise \
    --out_dir data/inference_outputs/Qwen3-8B_aime_2024_consensus

No fallbacks: it uses src.data.dataset directly and will error loudly if the
config / dataset / model are not as expected.
"""
import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import yaml

from src.data.dataset import (
    get_inference_dataset, format_prompt, format_stepwise_prompt, answers_match,
)
from src.dad.consensus_decoder import (
    StepModel, ConsensusDecoder,
    extract_sft_examples, extract_kl_targets, extract_dpo_pairs,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_consensus")


# ----------------------------------------------------------------------
def load_model_and_tokenizer(cfg):
    """Heavy imports are local so the rest of the file is importable without a GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["name"]
    device = cfg["model"]["device"]
    log.info(f"loading {name} on {device}")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    # reasoning models need headroom; bump position embeddings if tiny
    if getattr(model.config, "max_position_embeddings", 1 << 30) < 32768:
        model.config.max_position_embeddings = 32768
    return model, tok


def build_prompt(problem, model_name, style):
    if style == "stepwise":
        return format_stepwise_prompt(problem, model_name)
    return format_prompt(problem, model_name)


def already_done(out_path):
    """Resume support: return the set of problem_ids already written."""
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["problem_id"])
                except Exception:
                    pass
    return done


# ----------------------------------------------------------------------
def make_record(problem, traj, correct):
    """Flatten a TrajectoryRecord into a JSONL row, keeping ALL signal needed
    to later build SFT / KL / DPO datasets."""
    steps = []
    for s in traj.steps:
        steps.append({
            "idx": s.idx,
            "D": s.D,
            "top_fraction": s.top_fraction,
            "entropy": s.entropy,
            "decision": s.decision,
            "resolution_method": s.resolution_method,
            "committed_step": s.committed_step,
            "distribution": s.distribution,     # consensus over candidate clusters
            "candidates": s.candidates,         # raw K branches
            "losing_steps": s.losing_steps,     # -> preference pairs
            "process_reward": s.process_reward,
            "prefix": s.prefix,                 # conditioning context (for KL/DPO)
            "lookahead": s.lookahead,
        })
    return {
        "problem_id": problem["problem_id"],
        "question": problem["question"],
        "gold_answer": problem.get("gold_answer", ""),
        "final_answer": traj.final_answer,
        "correct": correct,
        "stop_reason": traj.stop_reason,
        "n_steps": traj.n_steps,
        "n_model_calls": traj.n_model_calls,
        "wall_time_sec": traj.wall_time_sec,
        "committed_trajectory": traj.committed_trajectory,
        "steps": steps,
        # ready-to-use training signal (option 1/2/3)
        "sft_examples": extract_sft_examples(traj),
        "kl_targets": extract_kl_targets(traj),
        "dpo_pairs": extract_dpo_pairs(traj),
    }


def summarize(records):
    n = len(records)
    if n == 0:
        return {}
    acc = sum(r["correct"] for r in records) / n
    decisions = Counter()
    stops = Counter()
    calls, steps = [], []
    # "consensus calibration": of problems that ended on a clean boxed commit,
    # how often correct? -- the honest 'consensus != truth' check.
    clean_commit, clean_commit_correct = 0, 0
    dpo_pairs = kl_targets = sft_ex = 0
    for r in records:
        stops[r["stop_reason"]] += 1
        calls.append(r["n_model_calls"])
        steps.append(r["n_steps"])
        dpo_pairs += len(r["dpo_pairs"])
        kl_targets += len(r["kl_targets"])
        sft_ex += len(r["sft_examples"])
        for s in r["steps"]:
            decisions[s["decision"]] += 1
        if r["steps"] and r["steps"][-1]["decision"] == "commit" and r["stop_reason"] == "boxed":
            clean_commit += 1
            clean_commit_correct += int(r["correct"])
    calls.sort(); steps.sort()
    mid = lambda xs: xs[len(xs) // 2] if xs else 0
    return {
        "n_problems": n,
        "accuracy": acc,
        "n_correct": sum(r["correct"] for r in records),
        "mean_model_calls": sum(calls) / n,
        "median_model_calls": mid(calls),
        "mean_steps": sum(steps) / n,
        "median_steps": mid(steps),
        "stop_reason_distribution": dict(stops),
        "decision_distribution": dict(decisions),
        "consensus_calibration": {
            "clean_boxed_commit_problems": clean_commit,
            "of_which_correct": clean_commit_correct,
            "rate": (clean_commit_correct / clean_commit) if clean_commit else None,
        },
        "training_signal_totals": {
            "sft_examples": sft_ex,
            "kl_target_forks": kl_targets,
            "dpo_pairs": dpo_pairs,
        },
    }


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="aime_2024")
    ap.add_argument("--n_problems", type=int, default=-1)
    ap.add_argument("--prompt_style", choices=["stepwise", "default"],
                    default="stepwise")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", action="store_true")
    # optional CLI overrides of the decoder knobs (else taken from cfg['dad'])
    ap.add_argument("--branch_k", type=int)
    ap.add_argument("--commit_tau", type=float)
    ap.add_argument("--adjudicators", type=int)
    ap.add_argument("--lookahead_steps", type=int)
    ap.add_argument("--step_max_tokens", type=int)
    ap.add_argument("--max_steps", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg.setdefault("dataset", {})
    cfg["dataset"]["name"] = args.dataset
    cfg["dataset"]["n_problems"] = args.n_problems
    dad = cfg.setdefault("dad", {})
    for k in ["branch_k", "commit_tau", "adjudicators",
              "lookahead_steps", "step_max_tokens", "max_steps"]:
        v = getattr(args, k)
        if v is not None:
            dad[k] = v

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    metrics_path = out_dir / "metrics.json"

    problems = get_inference_dataset(cfg)
    log.info(f"{len(problems)} problems from {args.dataset}")

    done = already_done(traj_path) if args.resume else set()
    if done:
        log.info(f"resume: skipping {len(done)} already-done problems")

    model, tok = load_model_and_tokenizer(cfg)
    sm = StepModel.from_hf(model, tok, cfg)
    decoder = ConsensusDecoder(sm, cfg)
    model_name = cfg["model"]["name"]

    mode = "a" if (args.resume and traj_path.exists()) else "w"
    records = []
    # reload prior records so the final metrics cover the full set on resume
    if mode == "a":
        with open(traj_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    with open(traj_path, mode) as fout:
        for p in problems:
            if p["problem_id"] in done:
                continue
            t0 = time.time()
            prompt = build_prompt(p, model_name, args.prompt_style)
            try:
                traj = decoder.generate(prompt)
            except Exception as e:                # one bad problem must not kill the run
                log.exception(f"problem {p['problem_id']} failed: {e}")
                continue
            correct = answers_match(traj.final_answer, p.get("gold_answer", ""))
            rec = make_record(p, traj, correct)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            records.append(rec)
            log.info(
                f"[{p['problem_id']}] ans={traj.final_answer} "
                f"gold={p.get('gold_answer')} correct={correct} "
                f"stop={traj.stop_reason} steps={traj.n_steps} "
                f"calls={traj.n_model_calls} {time.time()-t0:.0f}s"
            )

    metrics = summarize(records)
    metrics["method"] = "consensus_decoding"
    metrics["dataset"] = args.dataset
    metrics["prompt_style"] = args.prompt_style
    metrics["decoder_config"] = {k: dad.get(k) for k in
        ["branch_k", "commit_tau", "adjudicators", "lookahead_steps",
         "step_max_tokens", "max_steps"]}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("=" * 60)
    log.info(f"accuracy: {metrics.get('accuracy')}  "
             f"({metrics.get('n_correct')}/{metrics.get('n_problems')})")
    log.info(f"mean model calls/problem: {metrics.get('mean_model_calls'):.1f}")
    log.info(f"decision mix: {metrics.get('decision_distribution')}")
    log.info(f"stop reasons: {metrics.get('stop_reason_distribution')}")
    log.info(f"DPO pairs collected: "
             f"{metrics['training_signal_totals']['dpo_pairs']}")
    log.info(f"wrote {traj_path} and {metrics_path}")


if __name__ == "__main__":
    main()