import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

def load_all_metrics(output_dir: str, dataset_filter: str = "") -> list[dict]:
    metrics = []
    for p in Path(output_dir).glob("metrics_*.json"):
        with open(p) as f:
            m = json.load(f)
        if dataset_filter and m.get("dataset", "") != dataset_filter:
            continue
        m["_path"] = str(p)
        metrics.append(m)
    return metrics

def load_all_predictions(output_dir: str, dataset_filter: str = "") -> dict[str, list[dict]]:
    import jsonlines
    results = {}
    for p in Path(output_dir).glob("predictions_*.jsonl"):
        with jsonlines.open(str(p)) as reader:
            preds = list(reader)
        if not preds:
            continue
        ds = preds[0].get("source", "")
        if dataset_filter and dataset_filter not in ds:
            continue
        key = p.stem.replace("predictions_", "")
        results[key] = preds
    return results

def print_main_table(metrics: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("TABLE 1: Main Results")
    print("=" * 80)

    header = f"{'Method':<35} {'N':>4} {'λ':>5} {'Acc':>7} {'Tokens':>8} {'Time/q':>7} {'PRM':>6} {'Div':>6}"
    print(header)
    print("-" * 80)

    def sort_key(m):
        mode = m.get("mode", "")
        n = m.get("n_solutions", 0)
        lam = m.get("lambda_diversity", 0)
        order = {"greedy": 0, "sampling_vote": 1, "rebase": 2, "rmi_tree": 3}
        return (order.get(mode, 99), lam, n)

    for m in sorted(metrics, key=sort_key):
        mode = m.get("mode", "?")
        n = m.get("n_solutions", 1)
        lam = m.get("lambda_diversity", 0)
        agg = m.get("aggregation", "")

        if mode == "greedy":
            name = "Greedy"
        elif mode == "sampling_vote":
            name = f"Sampling+{agg.replace('_', ' ').title()}"
        elif mode == "rebase":
            name = "REBASE (λ=0)"
        elif mode == "rmi_tree":
            name = f"RMI-Tree (λ={lam})"
        else:
            name = mode

        print(
            f"{name:<35} {n:>4} {lam:>5.2f} "
            f"{m.get('accuracy', 0):>7.4f} "
            f"{m.get('mean_tokens', 0):>8.0f} "
            f"{m.get('mean_wall_time', 0):>7.1f} "
            f"{m.get('mean_prm_reward', 0):>6.3f} "
            f"{m.get('mean_diversity', 0):>6.3f}"
        )

    print("=" * 80)

def print_lambda_ablation(metrics: list[dict]) -> None:
    rmi_runs = [m for m in metrics if m.get("mode") == "rmi_tree"]
    if not rmi_runs:
        return

    print("\n" + "=" * 60)
    print("TABLE 2: λ Ablation (RMI-Tree)")
    print("=" * 60)

    header = f"{'λ':>5} {'N':>4} {'Accuracy':>9} {'Mean Div':>9} {'Mean PRM':>9}"
    print(header)
    print("-" * 60)

    for m in sorted(rmi_runs, key=lambda x: (x.get("lambda_diversity", 0), x.get("n_solutions", 0))):
        print(
            f"{m.get('lambda_diversity', 0):>5.2f} "
            f"{m.get('n_solutions', 0):>4} "
            f"{m.get('accuracy', 0):>9.4f} "
            f"{m.get('mean_diversity', 0):>9.4f} "
            f"{m.get('mean_prm_reward', 0):>9.4f}"
        )

    print("=" * 60)

def print_compute_tradeoff(metrics: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("FIGURE DATA: Compute-Accuracy Tradeoff")
    print("=" * 60)

    header = f"{'Method':<30} {'N':>4} {'Tokens':>8} {'Accuracy':>9}"
    print(header)
    print("-" * 60)

    for m in sorted(metrics, key=lambda x: x.get("mean_tokens", 0)):
        mode = m.get("mode", "?")
        lam = m.get("lambda_diversity", 0)
        if mode == "rmi_tree":
            name = f"RMI-Tree (λ={lam})"
        elif mode == "rebase":
            name = "REBASE"
        elif mode == "sampling_vote":
            name = f"Sampling+{m.get('aggregation', 'vote')}"
        else:
            name = mode

        print(
            f"{name:<30} "
            f"{m.get('n_solutions', 1):>4} "
            f"{m.get('mean_tokens', 0):>8.0f} "
            f"{m.get('accuracy', 0):>9.4f}"
        )

    print("=" * 60)

def analyze_diversity_impact(predictions: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS: Diversity Impact (Paired)")
    print("=" * 60)

    rebase_runs = {k: v for k, v in predictions.items() if "rebase" in k.lower()}
    rmi_runs = {k: v for k, v in predictions.items() if "rmi_tree" in k.lower()}

    for rk, rpreds in rebase_runs.items():
        for mk, mpreds in rmi_runs.items():
            if len(rpreds) != len(mpreds):
                continue

            r_by_id = {p["problem_id"]: p for p in rpreds}
            m_by_id = {p["problem_id"]: p for p in mpreds}
            common = set(r_by_id) & set(m_by_id)

            if len(common) < 5:
                continue

            rebase_only = 0
            rmi_only = 0
            both_correct = 0
            both_wrong = 0

            for pid in common:
                rc = r_by_id[pid].get("correct", False)
                mc = m_by_id[pid].get("correct", False)
                if rc and mc:
                    both_correct += 1
                elif rc and not mc:
                    rebase_only += 1
                elif not rc and mc:
                    rmi_only += 1
                else:
                    both_wrong += 1

            n = len(common)
            print(f"\n  {rk} vs {mk}  ({n} problems)")
            print(f"    Both correct:   {both_correct:>4} ({both_correct/n:.1%})")
            print(f"    REBASE only:    {rebase_only:>4} ({rebase_only/n:.1%})")
            print(f"    RMI-Tree only:  {rmi_only:>4} ({rmi_only/n:.1%})")
            print(f"    Both wrong:     {both_wrong:>4} ({both_wrong/n:.1%})")
            print(f"    Net improvement: +{rmi_only - rebase_only} problems")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="aime_2024")
    p.add_argument("--output_dir", default="data/inference_outputs")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)

    metrics = load_all_metrics(args.output_dir, args.dataset)
    logger.info(f"Found {len(metrics)} metrics files for dataset={args.dataset}")

    if not metrics:
        logger.warning("No metrics files found. Run experiments first.")
        return

    print_main_table(metrics)
    print_lambda_ablation(metrics)
    print_compute_tradeoff(metrics)

    predictions = load_all_predictions(args.output_dir, args.dataset)
    if predictions:
        analyze_diversity_impact(predictions)

if __name__ == "__main__":
    main()