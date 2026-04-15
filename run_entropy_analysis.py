"""
Entropy Inversion Analysis & Scatter Plot Generator

This script:
1. Loads a model and a dataset (default: MATH-500, 50 problems)
2. For each problem, generates M=8 samples
3. Computes:
   - Per-token entropy (mean across positions) for each sample
   - Answer-level entropy H_ans across the M samples
   - Whether greedy is correct
   - Whether Maj@8 is correct
   - Whether DAD is correct (runs DAD pipeline)
4. Saves raw data to JSONL
5. Generates the entropy inversion scatter plot

Usage:
    python run_entropy_analysis.py --config configs/dad_config.yaml --dataset math500 --n_problems 50

Output:
    results/entropy_analysis.jsonl   — raw per-problem data
    results/entropy_inversion.pdf    — the scatter plot
    results/entropy_inversion.png    — PNG version
"""

import argparse
import json
import logging
import sys
import time
import math
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def compute_token_entropy(model, tokenizer, prompt_ids, gen_ids, device):
    """Compute mean per-token entropy of a generated sequence.
    
    For each generated token t, compute H(t) = -sum(p * log2(p)) over the vocab,
    then return the mean across all generated positions.
    
    This measures how "confident" the model was at each step.
    Low entropy = model was very sure about each token.
    """
    full_ids = torch.cat([prompt_ids, gen_ids.unsqueeze(0)], dim=1).to(device)
    
    with torch.no_grad():
        outputs = model(full_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
    
    # We only care about positions corresponding to generated tokens
    # The logit at position i predicts token i+1
    prompt_len = prompt_ids.shape[1]
    gen_len = len(gen_ids)
    
    if gen_len == 0:
        return 0.0
    
    # Logits that predicted the generated tokens: positions [prompt_len-1, prompt_len, ..., prompt_len+gen_len-2]
    # These predict tokens at positions [prompt_len, prompt_len+1, ..., prompt_len+gen_len-1]
    start = prompt_len - 1
    end = prompt_len + gen_len - 1
    
    if end > logits.shape[0]:
        end = logits.shape[0]
    
    gen_logits = logits[start:end]  # (gen_len, vocab_size)
    
    # Compute entropy at each position
    probs = torch.softmax(gen_logits.float(), dim=-1)
    log_probs = torch.log2(probs + 1e-12)
    entropies = -(probs * log_probs).sum(dim=-1)  # (gen_len,)
    
    mean_entropy = entropies.mean().item()
    
    # Also return max and min for analysis
    return {
        "mean": mean_entropy,
        "median": entropies.median().item(),
        "max": entropies.max().item(),
        "min": entropies.min().item(),
        "std": entropies.std().item(),
    }


def compute_answer_entropy(answers):
    """Compute entropy of the answer distribution."""
    counts = Counter(answers)
    total = len(answers)
    if total == 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return ent


def normalize_answer_for_voting(answer):
    """Normalize answer for majority voting comparison."""
    if not answer:
        return ""
    answer = str(answer).strip()
    answer = answer.replace(",", "").replace("\\,", "")
    answer = answer.strip("$").strip()
    # Try float-to-int conversion
    try:
        val = float(answer)
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        pass
    return answer.lower().strip()


def main():
    parser = argparse.ArgumentParser(description="Entropy Inversion Analysis")
    parser.add_argument("--config", default="configs/dad_config.yaml")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--m_samples", type=int, default=8)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--skip_dad", action="store_true", help="Skip DAD runs (faster, just compute entropy)")
    parser.add_argument("--skip_token_entropy", action="store_true", help="Skip per-token entropy (much faster)")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["name"] = args.dataset
    cfg["dataset"]["n_problems"] = args.n_problems
    cfg["dad"]["m_samples"] = args.m_samples

    logger.info("=" * 60)
    logger.info("Entropy Inversion Analysis")
    logger.info("=" * 60)
    logger.info(f"  Dataset:    {args.dataset}")
    logger.info(f"  N problems: {args.n_problems}")
    logger.info(f"  M samples:  {args.m_samples}")
    logger.info(f"  Skip DAD:   {args.skip_dad}")
    logger.info(f"  Skip token entropy: {args.skip_token_entropy}")

    # ── Load model and data ───────────────────────────────────────
    from src.data.model_loader import ModelLoader
    from src.data.dataset import (
        get_inference_dataset, format_prompt, extract_boxed_answer,
        extract_numeric_answer, answers_match, normalize_answer,
    )

    model, tokenizer = ModelLoader(cfg).load()
    device = cfg["model"]["device"]
    problems = get_inference_dataset(cfg)
    logger.info(f"Loaded {len(problems)} problems")

    # Ensure max_position_embeddings is set correctly
    if hasattr(model, 'config') and model.config.max_position_embeddings < 32768:
        model.config.max_position_embeddings = 32768

    max_gen_tokens = cfg["dad"].get("max_gen_tokens", 2048)
    temperature = cfg["dad"].get("temperature", 0.7)
    top_p = cfg["dad"].get("top_p", 0.95)

    # ── Optionally load DAD generator ─────────────────────────────
    dad_generator = None
    if not args.skip_dad:
        from src.dad.dad_generator import DADGenerator
        dad_generator = DADGenerator(model, tokenizer, cfg)

    # ── Run analysis ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "entropy_analysis.jsonl"
    
    results = []

    for prob_idx, problem in enumerate(problems):
        t0 = time.time()
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(device)

        # ── Step 1: Greedy baseline ───────────────────────────────
        with torch.no_grad():
            greedy_out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
            )
        greedy_gen_ids = greedy_out[0, prompt_ids.shape[1]:]
        greedy_text = tokenizer.decode(greedy_gen_ids, skip_special_tokens=True)
        greedy_answer = extract_boxed_answer(greedy_text) or extract_numeric_answer(greedy_text) or ""
        greedy_correct = answers_match(greedy_answer, problem["gold_answer"])

        # Token entropy for greedy
        greedy_token_entropy = None
        if not args.skip_token_entropy and len(greedy_gen_ids) > 0:
            try:
                greedy_token_entropy = compute_token_entropy(
                    model, tokenizer, prompt_ids, greedy_gen_ids, device
                )
            except Exception as e:
                logger.warning(f"Token entropy failed for problem {prob_idx}: {e}")
        
        del greedy_out
        torch.cuda.empty_cache()

        # ── Step 2: Sample M solutions ────────────────────────────
        sample_answers = []
        sample_token_entropies = []
        
        with torch.no_grad():
            for s_idx in range(args.m_samples):
                out = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=max_gen_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                )
                gen_ids = out[0, prompt_ids.shape[1]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                ans = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
                sample_answers.append(ans)

                # Token entropy per sample
                if not args.skip_token_entropy and len(gen_ids) > 0:
                    try:
                        te = compute_token_entropy(model, tokenizer, prompt_ids, gen_ids, device)
                        sample_token_entropies.append(te["mean"])
                    except Exception:
                        sample_token_entropies.append(None)
                
                del out
                torch.cuda.empty_cache()

        # ── Step 3: Compute answer-level metrics ──────────────────
        normalized_answers = [normalize_answer_for_voting(a) for a in sample_answers]
        answer_counts = Counter(normalized_answers)
        majority_answer_norm = answer_counts.most_common(1)[0][0] if answer_counts else ""
        
        # Find the original (un-normalized) version of the majority answer
        majority_answer = ""
        for a, n in zip(sample_answers, normalized_answers):
            if n == majority_answer_norm:
                majority_answer = a
                break
        
        majority_correct = answers_match(majority_answer, problem["gold_answer"])
        answer_entropy = compute_answer_entropy(normalized_answers)
        majority_fraction = answer_counts.most_common(1)[0][1] / len(normalized_answers) if answer_counts else 0

        # ── Step 4: Run DAD (if enabled) ──────────────────────────
        dad_correct = None
        dad_answer = None
        dad_rounds = None
        if dad_generator is not None:
            try:
                dad_result = dad_generator.generate(prompt_ids, problem_text=problem["question"])
                dad_answer = dad_result.extracted_answer
                dad_correct = answers_match(dad_answer, problem["gold_answer"])
                dad_rounds = dad_result.n_rounds
            except Exception as e:
                logger.warning(f"DAD failed for problem {prob_idx}: {e}")
                dad_correct = False
            torch.cuda.empty_cache()

        # ── Step 5: Record results ────────────────────────────────
        wall_time = time.time() - t0

        result = {
            "problem_id": problem["problem_id"],
            "gold_answer": problem["gold_answer"],
            "level": problem.get("level", ""),
            "problem_type": problem.get("problem_type", ""),
            # Greedy
            "greedy_answer": greedy_answer,
            "greedy_correct": greedy_correct,
            "greedy_token_entropy": greedy_token_entropy,
            # Sampling
            "sample_answers": sample_answers,
            "answer_distribution": dict(answer_counts),
            "answer_entropy": answer_entropy,
            "majority_answer": majority_answer,
            "majority_correct": majority_correct,
            "majority_fraction": majority_fraction,
            "n_distinct_answers": len(answer_counts),
            # Token-level entropy stats across samples
            "mean_sample_token_entropy": float(np.mean([x for x in sample_token_entropies if x is not None])) if sample_token_entropies and any(x is not None for x in sample_token_entropies) else None,
            # DAD
            "dad_answer": dad_answer,
            "dad_correct": dad_correct,
            "dad_rounds": dad_rounds,
            # Meta
            "wall_time_sec": wall_time,
        }
        results.append(result)

        # Write incrementally
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Log progress
        greedy_acc = sum(1 for r in results if r["greedy_correct"]) / len(results)
        maj_acc = sum(1 for r in results if r["majority_correct"]) / len(results)
        dad_acc_str = ""
        if dad_correct is not None:
            dad_acc = sum(1 for r in results if r["dad_correct"]) / len(results)
            dad_acc_str = f" DAD={dad_acc:.3f}"

        logger.info(
            f"[{prob_idx+1}/{len(problems)}] "
            f"H_ans={answer_entropy:.2f} "
            f"maj_frac={majority_fraction:.0%} "
            f"greedy={'Y' if greedy_correct else 'N'} "
            f"maj={'Y' if majority_correct else 'N'} "
            f"{'dad=Y' if dad_correct else ('dad=N' if dad_correct is not None else '')} "
            f"| Acc: greedy={greedy_acc:.3f} maj={maj_acc:.3f}{dad_acc_str} "
            f"| {wall_time:.1f}s"
        )

    # ── Summary ───────────────────────────────────────────────────
    logger.info("=" * 60)
    n = len(results)
    n_greedy = sum(1 for r in results if r["greedy_correct"])
    n_maj = sum(1 for r in results if r["majority_correct"])
    logger.info(f"Greedy:  {n_greedy}/{n} = {n_greedy/n:.4f}")
    logger.info(f"Maj@{args.m_samples}:  {n_maj}/{n} = {n_maj/n:.4f}")
    if not args.skip_dad:
        n_dad = sum(1 for r in results if r["dad_correct"])
        logger.info(f"DAD:     {n_dad}/{n} = {n_dad/n:.4f}")
    
    # Entropy inversion stats
    zero_entropy = [r for r in results if r["answer_entropy"] == 0.0]
    if zero_entropy:
        n_ze = len(zero_entropy)
        n_ze_greedy_wrong = sum(1 for r in zero_entropy if not r["greedy_correct"])
        n_ze_maj_wrong = sum(1 for r in zero_entropy if not r["majority_correct"])
        logger.info(f"\nEntropy Inversion (H_ans=0): {n_ze} problems")
        logger.info(f"  Greedy wrong:   {n_ze_greedy_wrong}/{n_ze}")
        logger.info(f"  Maj@{args.m_samples} wrong:  {n_ze_maj_wrong}/{n_ze}")
        if not args.skip_dad:
            n_ze_dad_correct = sum(1 for r in zero_entropy if r["dad_correct"])
            logger.info(f"  DAD correct:    {n_ze_dad_correct}/{n_ze}")
            logger.info(f"  -> DAD recovers {n_ze_dad_correct} problems from entropy inversion regime")

    logger.info(f"\nResults saved to {results_path}")
    logger.info("Run: python plot_entropy_inversion.py to generate the figure")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()