"""
run_disagreement_mining.py
==========================

Mine GENUINE, DENSE disagreements on AIME 2024 with Qwen3-8B and write them to
JSONL, plus a calibration summary that answers the question the whole method
hinges on:

    When the model's answer distribution genuinely splits and then collapses,
    does it collapse onto the CORRECT answer?

That number decides which paper you are writing:
  * high  -> teacher-free consensus carries real signal; the disagreement
             contexts are worth distilling on.
  * low   -> consensus != truth; fall back to OPSD-with-ground-truth.

The gold answer is used ONLY to LABEL records for this calibration. It is never
used to decide whether a fork is genuine, so the extraction stays teacher-free.

Usage
-----
    python run_disagreement_mining.py \
        --config configs/dad_config_qwen3_8b.yaml \
        --dataset aime_2024 \
        --n_problems 5 \
        --out_dir runs/disagreement_mine_smoke

    # full run
    python run_disagreement_mining.py \
        --config configs/dad_config_qwen3_8b.yaml \
        --dataset aime_2024 --n_problems -1 \
        --out_dir runs/disagreement_mine_full \
        --k_rollouts 8 --rollout_max_tokens 8192

File placement: put this at the repo root next to run_consensus.py; put
disagreement_miner.py in src/dad/ next to consensus_decoder.py.
"""

import argparse
import json
import os
import time
import traceback


# --------------------------------------------------------------------------- #
# Backend adapter: wraps the project's HFBackend.generate_text into the single
# `complete(...)` method the miner expects. (HFBackend already batches via
# num_return_sequences, so this is essentially a rename.)
# --------------------------------------------------------------------------- #
class MinerBackendAdapter:
    def __init__(self, hf_backend):
        self.hf = hf_backend

    def complete(self, prefix, n, max_new_tokens, do_sample=True,
                 temperature=None):
        return self.hf.generate_text(prefix, n=n, max_new_tokens=max_new_tokens,
                                     do_sample=do_sample, temperature=temperature)


def load_model_and_backend(cfg):
    """Heavy imports kept local so --help etc. don't require torch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.dad.consensus_decoder import HFBackend

    model_name = cfg["model"]["name"]
    device = cfg["model"]["device"]
    print(f"[load] {model_name} -> {device}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device)
    model.eval()
    hf = HFBackend(model, tok, cfg)
    return MinerBackendAdapter(hf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="aime_2024")
    ap.add_argument("--n_problems", type=int, default=5)
    ap.add_argument("--prompt_style", choices=["stepwise", "default"],
                    default="default",
                    help="stepwise => blank-line-delimited steps (cleaner "
                         "anchors); default => natural Qwen3 thinking.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", action="store_true")
    # miner knobs (override MinerConfig defaults)
    ap.add_argument("--k_rollouts", type=int, default=6)
    ap.add_argument("--rollout_max_tokens", type=int, default=4096)
    ap.add_argument("--rollout_temperature", type=float, default=0.7)
    ap.add_argument("--max_anchors", type=int, default=10)
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--min_distinct_before", type=int, default=2)
    ap.add_argument("--collapse_frac", type=float, default=0.75)
    ap.add_argument("--max_blank_frac", type=float, default=0.5)
    args = ap.parse_args()

    import yaml
    from src.data.dataset import (get_inference_dataset, format_prompt,
                                  format_stepwise_prompt)
    from src.dad.disagreement_miner import mine_problem, MinerConfig

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("dataset", {})["name"] = args.dataset
    model_name = cfg["model"]["name"]

    miner_cfg = MinerConfig(
        k_rollouts=args.k_rollouts,
        rollout_max_tokens=args.rollout_max_tokens,
        rollout_temperature=args.rollout_temperature,
        max_anchors=args.max_anchors,
        min_support=args.min_support,
        min_distinct_before=args.min_distinct_before,
        collapse_frac=args.collapse_frac,
        max_blank_frac=args.max_blank_frac,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    forks_path = os.path.join(args.out_dir, "genuine_disagreements.jsonl")
    tel_path = os.path.join(args.out_dir, "telemetry.jsonl")
    metrics_path = os.path.join(args.out_dir, "calibration.json")

    done = set()
    if args.resume and os.path.exists(tel_path):
        with open(tel_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["problem_id"])
                except Exception:
                    pass
        print(f"[resume] skipping {len(done)} already-mined problems")

    problems = get_inference_dataset(cfg)
    if args.n_problems > 0:
        problems = problems[:args.n_problems]

    backend = load_model_and_backend(cfg)
    fmt = (format_stepwise_prompt if args.prompt_style == "stepwise"
           else format_prompt)

    f_forks = open(forks_path, "a")
    f_tel = open(tel_path, "a")

    # running calibration tallies
    n_forks = 0
    n_resolved_correct = 0
    n_gold_in_contention = 0

    t0 = time.time()
    for prob in problems:
        pid = prob.get("id", prob.get("problem_id", problems.index(prob)))
        if pid in done:
            continue
        gold = str(prob.get("answer", prob.get("gold_answer", "")))
        prompt = fmt(prob, model_name)
        try:
            forks, tel = mine_problem(backend, pid, prompt, miner_cfg,
                                      gold_answer=gold)
        except Exception as e:
            tel = {"problem_id": pid, "error": str(e),
                   "trace": traceback.format_exc()[-1500:]}
            forks = []
            print(f"[err] problem {pid}: {e}")

        for r in forks:
            f_forks.write(json.dumps(r.as_dict()) + "\n")
            n_forks += 1
            if r.resolved_correctly:
                n_resolved_correct += 1
            if r.gold_was_in_contention:
                n_gold_in_contention += 1
        f_forks.flush()
        f_tel.write(json.dumps(tel) + "\n")
        f_tel.flush()

        gf = tel.get("n_genuine_forks", 0)
        print(f"[mine] pid={pid} gold={gold} genuine_forks={gf} "
              f"(running: {n_resolved_correct}/{n_forks} resolved->correct)")

    f_forks.close()
    f_tel.close()

    # --- the number that decides the project ------------------------------- #
    calib = {
        "n_problems_mined": len(problems) - len(done),
        "n_genuine_forks": n_forks,
        "resolved_correct_rate": (n_resolved_correct / n_forks) if n_forks else None,
        "gold_in_contention_rate": (n_gold_in_contention / n_forks) if n_forks else None,
        "wall_time_sec": round(time.time() - t0, 1),
        "interpretation": (
            "resolved_correct_rate is P(consensus collapses onto the gold "
            "answer | a genuine fork was detected). High => teacher-free "
            "consensus is a usable training signal. Low => consensus != truth; "
            "use OPSD with ground-truth instead."),
    }
    with open(metrics_path, "w") as f:
        json.dump(calib, f, indent=2)
    print("\n=== CALIBRATION ===")
    print(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()