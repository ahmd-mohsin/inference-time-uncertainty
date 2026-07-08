# ============================================================================
# Recursive self-distillation flywheel  (docs/RL.md — the "phenomenon" experiment).
#
# QUESTION: iterated tail self-distillation as a DYNAMICAL SYSTEM. Run
#   harvest -> SFT -> eval  repeatedly, feeding each round's model into the next,
# and measure whether the model's reasoning SUPPORT keeps expanding, plateaus, or
# collapses. This is Component B run in a pure loop with NO GRPO between rounds, so
# any change in pass@k is attributable to self-distillation alone.
#
# Per round r we log (metrics.jsonl, one line per round):
#   - pass@k curve on the FULL eval set (the headline: does the curve rise each round?)
#   - pass@k on the HARD subset (where expansion is possible; sharper signal)
#   - harvest yield (# distinct correct tail rollouts) — the "fuel" available to distill
#   - problem-label migration: stuck->hard->solved counts (support frontier moving)
#   - per-problem pass@k deltas vs round 0 (which problems became reachable)
#   - novelty of the harvested set (are we distilling diverse or redundant solutions?)
#
# The paper-defining figure: pass@k vs round (one curve per round), + a convergence
# panel (mean pass@k and hard-set pass@k vs round -> fixed point or collapse).
#
# Design notes:
#   * Each round is independent processes (harvest vLLM, SFT trainer, eval vLLM) so GPU
#     memory is released between stages (see model_utils merge + the launcher's stop_vllm
#     pattern). We run stages as subprocesses and free the GPU between them.
#   * Fully resumable: a round is skipped if its round dir already has the eval json, so
#     an instance death mid-flywheel resumes at the next unfinished round.
#   * model lineage: round r SFTs FROM round r-1's model (its merged full dir), so the
#     distribution compounds. Round 0's base is the CLI --init model (base, or oursABC).
# ============================================================================

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sh(cmd, log_path=None):
    """Run a shell command, streaming to log_path if given. Returns exit code."""
    print(f">> {cmd}", flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"\n>> {cmd}\n"); f.flush()
            return subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return subprocess.call(cmd, shell=True)


def free_gpu(min_free_mib=2000, tries=40):
    """Kill leftover vLLM/EngineCore and wait for GPU0 memory to release (same pattern as
    the training launcher's stop_vllm — a harvest/eval vLLM must not start on a full GPU)."""
    subprocess.call("pkill -9 -f 'trl vllm' 2>/dev/null; pkill -9 -f vllm 2>/dev/null; "
                    "pkill -9 -f EngineCore 2>/dev/null", shell=True)
    for pid in _gpu_pids():
        subprocess.call(f"kill -9 {pid} 2>/dev/null", shell=True)
    for _ in range(tries):
        used = _gpu0_mib()
        if used is not None and used < min_free_mib:
            return True
        time.sleep(3)
    return False


def _gpu_pids():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null",
            shell=True, text=True)
        return [p.strip() for p in out.split("\n") if p.strip()]
    except Exception:
        return []


def _gpu0_mib():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null",
            shell=True, text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def load_hard_ids(difficulty_json):
    """problem_ids labeled 'hard' in the difficulty file (the support-expansion target set)."""
    if not difficulty_json or not os.path.exists(difficulty_json):
        return None
    d = json.load(open(difficulty_json))
    return sorted(x["problem_id"] for x in d.get("per_problem", []) if x.get("label") == "hard")


def passk_from_eval(eval_json, subset_ids=None):
    """Compute the pass@k curve from an evaluate_passk output json, optionally on a subset of
    problem_ids. Recomputes from per_problem so we can slice to the hard set."""
    d = json.load(open(eval_json))
    per = d["per_problem"]
    if subset_ids is not None:
        sset = set(subset_ids)
        per = [p for p in per if p["problem_id"] in sset]
    if not per:
        return {}
    ks = sorted(int(k) for k in per[0]["pass_at_k"].keys())
    return {k: float(sum(p["pass_at_k"][str(k)] for p in per) / len(per)) for k in ks}


def per_problem_passk(eval_json):
    d = json.load(open(eval_json))
    return {p["problem_id"]: {int(k): v for k, v in p["pass_at_k"].items()}
            for p in d["per_problem"]}


def relabel(eval_json, k_hi, solved_thresh=0.5):
    """Re-label every problem from a round's eval: stuck (pass@k_hi==0), solved (pass@1>=thresh),
    else hard. Lets us track the support FRONTIER migrating across rounds."""
    d = json.load(open(eval_json))
    counts = {"solved": 0, "hard": 0, "stuck": 0}
    labels = {}
    for p in d["per_problem"]:
        pk = p["pass_at_k"]
        p1 = pk.get("1", 0.0)
        phi = pk.get(str(k_hi), max(pk.values()))
        lab = "stuck" if phi == 0.0 else ("solved" if p1 >= solved_thresh else "hard")
        labels[p["problem_id"]] = lab; counts[lab] += 1
    return counts, labels


def harvest_novelty(harvest_jsonl):
    """Mean pairwise novelty of the harvested completions (are we distilling diverse paths?).
    Reuses the same TF-IDF novelty as Component A."""
    if not os.path.exists(harvest_jsonl):
        return None
    texts = [json.loads(l)["completion"] for l in open(harvest_jsonl) if l.strip()]
    if len(texts) < 2:
        return 0.0
    try:
        from rl_training.semantic import embed_texts, pairwise_novelty
        emb = embed_texts(texts)
        return float(pairwise_novelty(emb).mean())
    except Exception:
        return None


def run_flywheel(a):
    root = Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.jsonl"
    hard_ids = load_hard_ids(a.difficulty_json)
    py = sys.executable
    cur_model = a.init_model          # round r SFTs from cur_model; round 0 = base/oursABC
    done_rounds = _completed_rounds(metrics_path)
    print(f">> flywheel: {a.rounds} rounds, init={a.init_model}, hard_ids={hard_ids}", flush=True)

    # Round 0 = evaluate the STARTING model (no SFT yet) -> the baseline of the dynamical system.
    for r in range(0, a.rounds + 1):
        rdir = root / f"round{r}"; rdir.mkdir(exist_ok=True)
        eval_json = rdir / "passk_eval.json"
        harvest_jsonl = rdir / "harvest.jsonl"
        rlog = rdir / "round.log"

        if r in done_rounds and eval_json.exists():
            print(f">> round {r} already complete — skipping (resume)", flush=True)
            # keep model lineage going even when skipping
            sft_out = rdir / "sft_model"
            cur_model = str(sft_out) if (sft_out / "adapter_model.safetensors").exists() else cur_model
            continue

        # ---- (1) round r>0: SFT on the PREVIOUS round's harvest, producing this round's model ----
        if r > 0:
            prev_harvest = root / f"round{r-1}" / "harvest.jsonl"
            sft_out = rdir / "sft_model"
            if os.path.getsize(prev_harvest) if prev_harvest.exists() else 0:
                free_gpu()
                rc = sh(f"CUDA_VISIBLE_DEVICES={a.gpu} HF_HUB_OFFLINE=1 {py} -m rl_training.harvest "
                        f"--mode sft --model-path '{cur_model}' --out-jsonl '{prev_harvest}' "
                        f"--output-dir '{sft_out}' --epochs {a.sft_epochs} --lr {a.sft_lr}", rlog)
                if rc == 0 and (sft_out / "adapter_model.safetensors").exists():
                    cur_model = str(sft_out)
                    print(f">> round {r}: SFT done -> {cur_model}", flush=True)
                else:
                    print(f">> round {r}: SFT failed (rc={rc}); carrying previous model forward", flush=True)
            else:
                print(f">> round {r}: previous harvest empty -> flywheel has run dry (no fuel). "
                      f"Recording and continuing.", flush=True)

        # ---- (2) eval pass@k on the current model (the round's measurement) ----
        if not eval_json.exists():
            free_gpu()
            rc = sh(f"CUDA_VISIBLE_DEVICES={a.gpu} HF_HUB_OFFLINE=1 {py} -m rl_training.evaluate_passk "
                    f"--model-path '{cur_model}' --dataset {a.dataset} --n-samples {a.n_samples} "
                    f"--n-problems -1 --max-new-tokens {a.max_new_tokens} --tensor-parallel-size 1 "
                    f"--output-dir '{rdir}' --tag eval", rlog)
            # evaluate_passk writes passk_eval.json in rdir
            if not eval_json.exists():
                print(f">> round {r}: EVAL FAILED (rc={rc}); aborting flywheel", flush=True); return

        # ---- (3) harvest tail on hard problems from the current model (fuel for round r+1) ----
        if not harvest_jsonl.exists():
            free_gpu()
            sh(f"CUDA_VISIBLE_DEVICES={a.gpu} HF_HUB_OFFLINE=1 {py} -m rl_training.harvest "
               f"--mode harvest --model-path '{cur_model}' --dataset {a.dataset} "
               f"{'--difficulty-json ' + a.difficulty_json if a.difficulty_json else ''} "
               f"--k {a.harvest_k} --max-keep {a.max_keep} --max-new-tokens {a.max_new_tokens} "
               f"--tensor-parallel-size 1 --out-jsonl '{harvest_jsonl}'", rlog)

        # ---- (4) compute + record this round's metrics ----
        full = passk_from_eval(eval_json)
        hard = passk_from_eval(eval_json, hard_ids) if hard_ids else {}
        counts, _ = relabel(eval_json, k_hi=max(full) if full else 32)
        n_harvest = sum(1 for l in open(harvest_jsonl) if l.strip()) if harvest_jsonl.exists() else 0
        rec = {
            "round": r, "model": cur_model,
            "passk_full": full, "passk_hard": hard,
            "label_counts": counts,
            "harvest_yield": n_harvest,
            "harvest_novelty": harvest_novelty(str(harvest_jsonl)),
            "mean_passk_full": float(sum(full.values()) / len(full)) if full else None,
            "mean_passk_hard": float(sum(hard.values()) / len(hard)) if hard else None,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f">> round {r} METRICS: full pass@k={full} | hard={hard} | "
              f"harvest={n_harvest} | labels={counts}", flush=True)

    print(">> flywheel complete. metrics -> " + str(metrics_path), flush=True)


def _completed_rounds(metrics_path):
    if not os.path.exists(metrics_path):
        return set()
    return {json.loads(l)["round"] for l in open(metrics_path) if l.strip()}


def main():
    ap = argparse.ArgumentParser(description="Recursive self-distillation flywheel")
    ap.add_argument("--init-model", required=True,
                    help="starting model: base HF id, or a checkpoint dir (e.g. oursABC final)")
    ap.add_argument("--rounds", type=int, default=5, help="number of harvest->SFT rounds")
    ap.add_argument("--dataset", default="aime_all")
    ap.add_argument("--difficulty-json", default="", help="hard-set labels (for hard-subset pass@k)")
    ap.add_argument("--output-dir", default="rl_training/runs/flywheel")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--harvest-k", type=int, default=64)
    ap.add_argument("--max-keep", type=int, default=4, help="distinct correct rollouts kept/problem")
    ap.add_argument("--n-samples", type=int, default=32, help="eval samples/problem")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--sft-lr", type=float, default=1e-6)
    a = ap.parse_args()
    run_flywheel(a)


if __name__ == "__main__":
    main()
