"""
Disagreement Characterization Study

Runs greedy, Maj@M, and DAD on a dataset, recording per-problem:
  - D^(r): number of disputed claims at each round
  - pass@k at each round (was correct answer in ANY sample?)
  - pass@1 for greedy, Maj@M, and DAD
  - answer distributions per round

Then bins by initial disagreement D^(1) and computes the table
the professor requested.

Usage:
    python run_disagreement_characterization.py \
        --config configs/dad_config_qwen3.yaml \
        --dataset amc --n_problems -1 \
        --output_dir results/characterization

Output:
    results/characterization/<model>_<dataset>_char_raw.json
    results/characterization/<model>_<dataset>_char_table.json
    results/characterization/<model>_<dataset>_char_summary.txt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

import torch
import yaml
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _normalize_for_voting(answer):
    """Normalize answer for majority voting comparison."""
    if answer is None or answer == "":
        return ""
    from src.data.dataset import normalize_answer
    n = normalize_answer(answer)
    n = n.lstrip("0") or "0"
    try:
        val = float(n)
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.8f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        pass
    return n


def run_greedy_single(model, tokenizer, prompt, cfg):
    """Run greedy decoding on a single problem, return (answer, text, tokens)."""
    from src.data.dataset import extract_boxed_answer, extract_numeric_answer

    device = cfg["model"]["device"]
    max_tokens = cfg["model"].get("max_new_tokens", 2048)

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    )["input_ids"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, prompt_ids.shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    answer = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""

    del out
    torch.cuda.empty_cache()
    return answer, gen_text, len(gen_ids)


def run_sampling_single(model, tokenizer, prompt, cfg, n_samples=8):
    """Run n_samples with temperature, return list of (answer, text, tokens)."""
    from src.data.dataset import extract_boxed_answer, extract_numeric_answer

    device = cfg["model"]["device"]
    dad_cfg = cfg.get("dad", {})
    max_tokens = dad_cfg.get("max_gen_tokens", 2048)
    temperature = dad_cfg.get("temperature", 0.7)
    top_p = dad_cfg.get("top_p", 0.95)

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    )["input_ids"].to(device)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out[0, prompt_ids.shape[1]:].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            ans = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
            samples.append({"answer": ans, "text": gen_text, "tokens": len(gen_ids)})
            del out
            torch.cuda.empty_cache()

    return samples


def majority_vote(samples):
    """Return majority answer from list of sample dicts."""
    counts = defaultdict(int)
    for s in samples:
        norm = _normalize_for_voting(s["answer"])
        counts[norm] += 1
    if not counts:
        return ""
    best = max(counts, key=counts.get)
    # Find original (unnormalized) answer
    for s in samples:
        if _normalize_for_voting(s["answer"]) == best:
            return s["answer"]
    return best


def check_pass_at_k(samples, gold, answers_match_fn):
    """Check if correct answer appears in any sample."""
    for s in samples:
        if answers_match_fn(s["answer"], gold):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Disagreement Characterization Study")
    parser.add_argument("--config", default="configs/dad_config_qwen3.yaml")
    parser.add_argument("--dataset", default="amc")
    parser.add_argument("--n_problems", type=int, default=-1)
    parser.add_argument("--output_dir", default="results/characterization")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing raw JSON file")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["name"] = args.dataset
    cfg["dataset"]["n_problems"] = args.n_problems

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = cfg["model"].get("name", "model").split("/")[-1]
    prefix = f"{model_short}_{args.dataset}"

    logger.info("=" * 60)
    logger.info("Disagreement Characterization Study")
    logger.info("=" * 60)
    logger.info(f"  Model:    {cfg['model']['name']}")
    logger.info(f"  Dataset:  {args.dataset}")
    logger.info(f"  N:        {args.n_problems}")
    logger.info(f"  M:        {cfg['dad']['m_samples']}")
    logger.info(f"  R:        {cfg['dad']['max_rounds']}")

    # Load model and data
    from src.data.model_loader import ModelLoader
    from src.data.dataset import (
        get_inference_dataset, format_prompt, answers_match,
        extract_boxed_answer, extract_numeric_answer,
    )

    model, tokenizer = ModelLoader(cfg).load()

    if hasattr(model, 'config'):
        old_max_pos = getattr(model.config, 'max_position_embeddings', None)
        if old_max_pos and old_max_pos < 32768:
            model.config.max_position_embeddings = 32768
            logger.info(f"  Fixed max_position_embeddings: {old_max_pos} -> 32768")

    problems = get_inference_dataset(cfg)
    logger.info(f"Loaded {len(problems)} problems")

    # Import DAD generator — we need a modified version that exposes per-round data
    from src.dad.dad_generator import DADGenerator
    generator = DADGenerator(model, tokenizer, cfg)

    # Resume support
    raw_path = output_dir / f"{prefix}_char_raw.json"
    all_data = []
    done_ids = set()
    if args.resume and raw_path.exists():
        with open(raw_path) as f:
            all_data = json.load(f)
        done_ids = {d["problem_id"] for d in all_data}
        logger.info(f"Resuming: {len(done_ids)} problems already done")

    todo = [p for p in problems if p["problem_id"] not in done_ids]

    n_correct_greedy = sum(1 for d in all_data if d.get("greedy_correct", False))
    n_correct_maj = sum(1 for d in all_data if d.get("maj_correct", False))
    n_correct_dad = sum(1 for d in all_data if d.get("dad_correct", False))

    pbar = tqdm(todo, desc="Characterization", unit="prob")

    for prob_idx, problem in enumerate(pbar):
        prompt = format_prompt(problem, cfg["model"]["name"])
        gold = problem["gold_answer"]
        t0 = time.time()

        try:
            # ── Step 1: Greedy ────────────────────────────────────
            greedy_ans, greedy_text, greedy_tokens = run_greedy_single(
                model, tokenizer, prompt, cfg
            )
            greedy_correct = answers_match(greedy_ans, gold)
            if greedy_correct:
                n_correct_greedy += 1

            # ── Step 2: DAD with per-round logging ────────────────
            prompt_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096,
            )["input_ids"].to(cfg["model"]["device"])

            gen = generator.generate(prompt_ids, problem_text=problem["question"])

            dad_answer = gen.extracted_answer
            dad_correct = answers_match(dad_answer, gold)
            if dad_correct:
                n_correct_dad += 1

            # Extract per-round disagreement data from the generator
            # The generator stores per-round info in its internal state
            per_round_data = []
            n_rounds = gen.n_rounds

            # Get per-round disputed/agreed counts and answer distributions
            # from the disagreement maps stored during generation
            if hasattr(gen, 'per_round_disagreement_maps'):
                for r, dm in enumerate(gen.per_round_disagreement_maps):
                    per_round_data.append({
                        "round": r + 1,
                        "n_agreed": dm.get("n_agreed", 0),
                        "n_disputed": dm.get("n_disputed", 0),
                        "answer_entropy": dm.get("answer_entropy", 0),
                        "confidence": dm.get("confidence", 0),
                        "answer_dist": dm.get("answer_distribution", {}),
                    })
            else:
                # Fallback: use what we have from the final disagreement map
                per_round_data.append({
                    "round": n_rounds,
                    "n_agreed": gen.disagreement_map.get("n_agreed", 0),
                    "n_disputed": gen.disagreement_map.get("n_disputed", 0),
                    "answer_entropy": gen.disagreement_map.get("answer_entropy", 0),
                    "confidence": gen.disagreement_map.get("confidence", 0),
                    "answer_dist": gen.disagreement_map.get("answer_distribution", {}),
                })

            # Get D^(1) — disputed claims at round 1
            if per_round_data:
                d1_disputed = per_round_data[0].get("n_disputed", 0)
            else:
                d1_disputed = gen.disagreement_map.get("n_disputed", 0)

            # ── Step 3: Maj@M from round-1 samples ────────────────
            # The DAD generator's round-1 samples ARE the M independent samples
            # We can reconstruct Maj@M from the round-1 answer distribution
            r1_dist = {}
            if per_round_data:
                r1_dist = per_round_data[0].get("answer_dist", {})
            elif len(gen.answer_entropy_per_round) > 0:
                r1_dist = gen.disagreement_map.get("answer_distribution", {})

            # Majority vote from round-1 distribution
            if r1_dist:
                maj_answer_norm = max(r1_dist, key=r1_dist.get) if r1_dist else ""
                maj_correct = answers_match(maj_answer_norm, gold)
            else:
                maj_correct = False
                maj_answer_norm = ""

            if maj_correct:
                n_correct_maj += 1

            # ── Step 4: Pass@k computation ────────────────────────
            # Check if correct answer appears in ANY round-1 sample
            # We approximate this from the answer distribution
            pass_at_k_r1 = False
            for ans_key in r1_dist:
                if answers_match(ans_key, gold):
                    pass_at_k_r1 = True
                    break

            # Check if correct answer appears in final answer distribution
            final_dist = gen.disagreement_map.get("answer_distribution", {})
            pass_at_k_final = False
            for ans_key in final_dist:
                if answers_match(ans_key, gold):
                    pass_at_k_final = True
                    break

            # Also check the DAD answer itself
            if dad_correct:
                pass_at_k_final = True

            wall_time = time.time() - t0

            entry = {
                "problem_id": problem["problem_id"],
                "gold_answer": gold,
                "source": problem.get("source", ""),
                "level": problem.get("level", ""),
                "problem_type": problem.get("problem_type", ""),
                # Greedy
                "greedy_answer": greedy_ans,
                "greedy_correct": greedy_correct,
                # Majority vote (from round-1 samples)
                "maj_answer": maj_answer_norm,
                "maj_correct": maj_correct,
                # DAD
                "dad_answer": dad_answer,
                "dad_correct": dad_correct,
                "dad_n_rounds": n_rounds,
                "dad_total_tokens": gen.total_tokens,
                # Disagreement characterization
                "d1_disputed": d1_disputed,
                "d1_agreed": per_round_data[0].get("n_agreed", 0) if per_round_data else 0,
                "entropy_per_round": gen.answer_entropy_per_round,
                "confidence_per_round": gen.confidence_per_round,
                "per_round_data": per_round_data,
                # Pass@k
                "pass_at_k_r1": pass_at_k_r1,
                "pass_at_k_final": pass_at_k_final,
                # Timing
                "wall_time": wall_time,
            }

        except Exception as e:
            logger.warning(f"Problem {problem['problem_id']} failed: {e}")
            import traceback; traceback.print_exc()
            entry = {
                "problem_id": problem["problem_id"],
                "gold_answer": gold,
                "source": problem.get("source", ""),
                "level": problem.get("level", ""),
                "problem_type": problem.get("problem_type", ""),
                "greedy_answer": "", "greedy_correct": False,
                "maj_answer": "", "maj_correct": False,
                "dad_answer": "", "dad_correct": False,
                "dad_n_rounds": 0, "dad_total_tokens": 0,
                "d1_disputed": -1, "d1_agreed": 0,
                "entropy_per_round": [], "confidence_per_round": [],
                "per_round_data": [],
                "pass_at_k_r1": False, "pass_at_k_final": False,
                "wall_time": time.time() - t0,
                "error": str(e),
            }

        all_data.append(entry)

        # Incremental save
        with open(raw_path, "w") as f:
            json.dump(all_data, f, indent=2, default=str)

        total_done = len(all_data)
        pbar.set_postfix(
            g=f"{n_correct_greedy}/{total_done}",
            m=f"{n_correct_maj}/{total_done}",
            d=f"{n_correct_dad}/{total_done}",
            D1=entry.get("d1_disputed", "?"),
        )

        torch.cuda.empty_cache()

    pbar.close()

    # ── Compute binned table ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Computing disagreement characterization table...")

    # Define bins for D^(1)
    bins = [
        ("0", lambda d: d == 0),
        ("1--3", lambda d: 1 <= d <= 3),
        ("4--7", lambda d: 4 <= d <= 7),
        ("8--15", lambda d: 8 <= d <= 15),
        ("16+", lambda d: d >= 16),
    ]

    table_rows = []
    for bin_label, bin_fn in bins:
        bin_problems = [e for e in all_data if bin_fn(e.get("d1_disputed", -1))]
        n_bin = len(bin_problems)
        if n_bin == 0:
            table_rows.append({
                "bin": bin_label, "n": 0,
                "greedy_acc": 0, "maj_acc": 0, "dad_acc": 0,
                "delta_dad_maj": 0,
                "pass_k_r1": 0, "pass_k_final": 0,
            })
            continue

        greedy_acc = sum(1 for e in bin_problems if e["greedy_correct"]) / n_bin
        maj_acc = sum(1 for e in bin_problems if e["maj_correct"]) / n_bin
        dad_acc = sum(1 for e in bin_problems if e["dad_correct"]) / n_bin
        pass_k_r1 = sum(1 for e in bin_problems if e["pass_at_k_r1"]) / n_bin
        pass_k_final = sum(1 for e in bin_problems if e["pass_at_k_final"]) / n_bin

        table_rows.append({
            "bin": bin_label,
            "n": n_bin,
            "greedy_acc": round(greedy_acc * 100, 1),
            "maj_acc": round(maj_acc * 100, 1),
            "dad_acc": round(dad_acc * 100, 1),
            "delta_dad_maj": round((dad_acc - maj_acc) * 100, 1),
            "pass_k_r1": round(pass_k_r1 * 100, 1),
            "pass_k_final": round(pass_k_final * 100, 1),
        })

    # Save table
    table_path = output_dir / f"{prefix}_char_table.json"
    with open(table_path, "w") as f:
        json.dump(table_rows, f, indent=2)

    # ── Print results ────────────────────────────────────────────
    total = len(all_data)
    logger.info(f"\nOverall accuracy ({total} problems):")
    logger.info(f"  Greedy:  {sum(1 for e in all_data if e['greedy_correct'])}/{total} "
                f"({sum(1 for e in all_data if e['greedy_correct'])/total*100:.1f}%)")
    logger.info(f"  Maj@M:   {sum(1 for e in all_data if e['maj_correct'])}/{total} "
                f"({sum(1 for e in all_data if e['maj_correct'])/total*100:.1f}%)")
    logger.info(f"  DAD:     {sum(1 for e in all_data if e['dad_correct'])}/{total} "
                f"({sum(1 for e in all_data if e['dad_correct'])/total*100:.1f}%)")

    logger.info(f"\n{'D^(1)':<8} {'n':>4} {'Greedy':>8} {'Maj@M':>8} {'DAD':>8} "
                f"{'Δ(DAD-Maj)':>11} {'pass@k R1':>10} {'pass@k Rf':>10}")
    logger.info("-" * 75)
    for row in table_rows:
        logger.info(
            f"{row['bin']:<8} {row['n']:>4} {row['greedy_acc']:>7.1f}% "
            f"{row['maj_acc']:>7.1f}% {row['dad_acc']:>7.1f}% "
            f"{row['delta_dad_maj']:>+10.1f}  {row['pass_k_r1']:>9.1f}% "
            f"{row['pass_k_final']:>9.1f}%"
        )

    # ── Summary text ─────────────────────────────────────────────
    summary_path = output_dir / f"{prefix}_char_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Disagreement Characterization: {cfg['model']['name']} on {args.dataset}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total problems: {total}\n")
        f.write(f"Greedy: {sum(1 for e in all_data if e['greedy_correct'])}/{total}\n")
        f.write(f"Maj@M:  {sum(1 for e in all_data if e['maj_correct'])}/{total}\n")
        f.write(f"DAD:    {sum(1 for e in all_data if e['dad_correct'])}/{total}\n\n")
        f.write(f"{'D^(1)':<8} {'n':>4} {'Greedy':>8} {'Maj@M':>8} {'DAD':>8} "
                f"{'Delta':>8} {'p@k_R1':>8} {'p@k_Rf':>8}\n")
        f.write("-" * 65 + "\n")
        for row in table_rows:
            f.write(f"{row['bin']:<8} {row['n']:>4} {row['greedy_acc']:>7.1f}% "
                    f"{row['maj_acc']:>7.1f}% {row['dad_acc']:>7.1f}% "
                    f"{row['delta_dad_maj']:>+7.1f}  {row['pass_k_r1']:>7.1f}% "
                    f"{row['pass_k_final']:>7.1f}%\n")

    # ── Per-problem scatter data for plotting ────────────────────
    scatter_data = []
    for e in all_data:
        if e.get("d1_disputed", -1) >= 0:
            scatter_data.append({
                "d1": e["d1_disputed"],
                "delta": 1 if e["dad_correct"] else 0,
                "maj_correct": 1 if e["maj_correct"] else 0,
                "greedy_correct": 1 if e["greedy_correct"] else 0,
            })

    scatter_path = output_dir / f"{prefix}_char_scatter.json"
    with open(scatter_path, "w") as f:
        json.dump(scatter_data, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Raw data:    {raw_path}")
    logger.info(f"Table:       {table_path}")
    logger.info(f"Summary:     {summary_path}")
    logger.info(f"Scatter:     {scatter_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()