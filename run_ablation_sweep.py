import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def build_cmd(
    mode: str,
    dataset: str,
    n_problems: int,
    n_solutions: int,
    lambda_div: float,
    aggregation: str,
    config: str = "configs/rmi_search_config.yaml",
    extra_args: list = None,
) -> list[str]:
    cmd = [
        sys.executable, "run_rmi_search.py",
        "--config", config,
        "--mode", mode,
        "--dataset", dataset,
        "--n_solutions", str(n_solutions),
        "--lambda_div", str(lambda_div),
        "--aggregation", aggregation,
    ]
    if n_problems > 0:
        cmd.extend(["--n_problems", str(n_problems)])
    if extra_args:
        cmd.extend(extra_args)
    return cmd

def main():
    p = argparse.ArgumentParser(description="RMI-Tree Ablation Sweep")
    p.add_argument("--dataset", default="aime_2024")
    p.add_argument("--n_problems", type=int, default=-1)
    p.add_argument("--config", default="configs/rmi_search_config.yaml")
    p.add_argument("--quick", action="store_true", help="Only run N=8 for fast iteration")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    n_values = [8] if args.quick else [8, 16, 32, 64]

    experiments = []

    experiments.append({
        "name": "greedy",
        "cmd": [
            sys.executable, "run_inference.py",
            "--mode", "greedy",
            "--dataset", args.dataset,
            "--config", "configs/inference_config.yaml",
        ],
    })

    for n in n_values:
        for agg in ["majority_vote", "weighted_vote"]:
            experiments.append({
                "name": f"sampling_{agg}_n{n}",
                "cmd": build_cmd(
                    mode="sampling_vote",
                    dataset=args.dataset,
                    n_problems=args.n_problems,
                    n_solutions=n,
                    lambda_div=0.0,
                    aggregation=agg,
                    config=args.config,
                ),
            })

    for n in n_values:
        experiments.append({
            "name": f"rebase_n{n}",
            "cmd": build_cmd(
                mode="rebase",
                dataset=args.dataset,
                n_problems=args.n_problems,
                n_solutions=n,
                lambda_div=0.0,
                aggregation="weighted_vote",
                config=args.config,
            ),
        })

    for lam in [0.25, 0.5, 1.0]:
        for n in n_values:
            experiments.append({
                "name": f"rmi_tree_lam{lam}_n{n}",
                "cmd": build_cmd(
                    mode="rmi_tree",
                    dataset=args.dataset,
                    n_problems=args.n_problems,
                    n_solutions=n,
                    lambda_div=lam,
                    aggregation="weighted_vote",
                    config=args.config,
                ),
            })

    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"N values: {n_values}")
    logger.info("")

    results_log = []

    for i, exp in enumerate(experiments):
        logger.info(f"[{i+1}/{len(experiments)}] {exp['name']}")
        logger.info(f"  CMD: {' '.join(exp['cmd'])}")

        if args.dry_run:
            logger.info("  (dry run — skipped)")
            continue

        try:
            result = subprocess.run(
                exp["cmd"],
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if result.returncode != 0:
                logger.warning(f"  FAILED (rc={result.returncode})")
                logger.warning(f"  stderr: {result.stderr[-500:]}")
            else:
                for line in result.stdout.split("\n"):
                    if "Accuracy" in line and ":" in line:
                        logger.info(f"  {line.strip()}")
                        break

            results_log.append({
                "name": exp["name"],
                "returncode": result.returncode,
                "cmd": " ".join(exp["cmd"]),
            })

        except subprocess.TimeoutExpired:
            logger.warning(f"  TIMEOUT after 7200s")
            results_log.append({
                "name": exp["name"],
                "returncode": -1,
                "cmd": " ".join(exp["cmd"]),
                "error": "timeout",
            })
        except Exception as e:
            logger.warning(f"  ERROR: {e}")
            results_log.append({
                "name": exp["name"],
                "returncode": -1,
                "cmd": " ".join(exp["cmd"]),
                "error": str(e),
            })

    log_path = f"data/inference_outputs/ablation_log_{args.dataset}.json"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    logger.info(f"\nExperiment log → {log_path}")

if __name__ == "__main__":
    main()