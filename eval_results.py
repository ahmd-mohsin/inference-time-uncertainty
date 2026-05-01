"""
Evaluate accuracy from a JSONL results file.

Usage:
    # Quick check (use pre-computed 'correct' field)
    python eval_results.py results.jsonl

    # Re-evaluate with current answers_match (in case matching was updated)
    python eval_results.py results.jsonl --rematch

    # Show wrong answers
    python eval_results.py results.jsonl --show_wrong

    # Show all answers
    python eval_results.py results.jsonl --show_all
"""

import argparse
import json
import sys


def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy from JSONL results")
    parser.add_argument("file", help="Path to JSONL results file")
    parser.add_argument("--rematch", action="store_true",
                        help="Re-evaluate correctness using current answers_match")
    parser.add_argument("--show_wrong", action="store_true",
                        help="Show problems answered incorrectly")
    parser.add_argument("--show_all", action="store_true",
                        help="Show all problems with pred vs gold")
    args = parser.parse_args()

    results = load_jsonl(args.file)
    if not results:
        print(f"No results found in {args.file}")
        sys.exit(1)

    # Optionally re-match with current answers_match
    if args.rematch:
        try:
            from src.data.dataset import answers_match
            for r in results:
                r["correct"] = answers_match(r.get("extracted_answer"), r.get("gold_answer", ""))
            print("[Re-matched with current answers_match]\n")
        except ImportError:
            print("Warning: Could not import answers_match, using stored 'correct' field")

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    acc = n_correct / n if n > 0 else 0

    method = results[0].get("method", "unknown")
    source = results[0].get("source", "unknown")

    print(f"File:     {args.file}")
    print(f"Method:   {method}")
    print(f"Source:   {source}")
    print(f"Problems: {n}")
    print(f"Correct:  {n_correct}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print()

    # Per-level breakdown
    by_level = {}
    for r in results:
        level = r.get("level", "")
        if level:
            if level not in by_level:
                by_level[level] = {"correct": 0, "total": 0}
            by_level[level]["total"] += 1
            if r.get("correct"):
                by_level[level]["correct"] += 1

    if by_level:
        print("Per-level breakdown:")
        print(f"  {'Level':<15} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print(f"  {'-'*40}")
        for level in sorted(by_level.keys()):
            d = by_level[level]
            lacc = d["correct"] / d["total"] if d["total"] > 0 else 0
            print(f"  {level:<15} {d['correct']:>8} {d['total']:>8} {lacc:>8.3f}")
        print()

    # DAD-specific stats
    if method == "dad":
        rounds = [r.get("n_rounds", 0) for r in results]
        tokens = [r.get("total_tokens", 0) for r in results]
        if rounds:
            print(f"DAD Stats:")
            print(f"  Avg rounds:  {sum(rounds)/len(rounds):.2f}")
            print(f"  Avg tokens:  {sum(tokens)/len(tokens):.0f}")
            print()

    # Show wrong or all
    if args.show_wrong or args.show_all:
        print("=" * 70)
        for r in results:
            if args.show_all or (args.show_wrong and not r.get("correct")):
                pid = r.get("problem_id", "?")
                pred = r.get("extracted_answer", "")
                gold = r.get("gold_answer", "")
                correct = "✓" if r.get("correct") else "✗"
                print(f"  [{correct}] Problem {pid}: pred={pred!r}  gold={gold!r}")
        print("=" * 70)


if __name__ == "__main__":
    main()