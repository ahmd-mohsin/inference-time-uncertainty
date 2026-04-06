import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ABLATION_MODES = ["greedy", "entropy_only", "prompt_only", "digte"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _run_mode(mode: str, cfg: dict, config_path: str) -> dict:
    short = cfg["model"]["short_name"]
    cmd = [
        sys.executable, "run_inference.py",
        "--config", config_path,
        "--mode", mode,
        "--model", cfg["model"]["name"],
        "--model_short_name", short,
        "--dataset", cfg["dataset"]["name"],
    ]
    n = cfg["dataset"].get("n_problems", -1)
    if n > 0:
        cmd += ["--n_problems", str(n)]

    logger.info(f"\n{'='*60}\nRunning ablation: {mode}\n{'='*60}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error(f"Mode {mode} failed with return code {result.returncode}")
        return {}

    metrics_path = (
        cfg["output"]["metrics_file"]
        .replace("{model_short_name}", short)
        .replace("{decoding_mode}", mode)
    )
    if not Path(metrics_path).exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return {}

    with open(metrics_path) as f:
        return json.load(f)


def print_table(results: dict[str, dict]) -> None:
    from src.evaluation.compute_matched import ComputeMatchedEvaluator
    rows = []
    for mode, m in results.items():
        rows.append({
            "method": mode,
            "n": m.get("n_problems", 0),
            "accuracy": m.get("accuracy", 0.0),
            "mean_tokens": m.get("mean_tokens", 0.0),
            "mean_expansion_tokens": m.get("mean_expansion_tokens", 0.0),
            "entropy_trigger_rate": m.get("mean_entropy_trigger_rate", 0.0),
            "expansion_trigger_rate": m.get("mean_expansion_trigger_rate", 0.0),
            "mean_wall_time": m.get("mean_wall_time", 0.0),
        })
    rows.sort(key=lambda x: -x["accuracy"])
    ComputeMatchedEvaluator({}).print_ablation_table(rows)

    if "greedy" in results and "digte" in results:
        delta_acc = results["digte"]["accuracy"] - results["greedy"]["accuracy"]
        delta_tok = (
            (results["digte"]["mean_tokens"] - results["greedy"]["mean_tokens"])
            / max(1.0, results["greedy"]["mean_tokens"]) * 100
        )
        logger.info(f"\nDIGTE vs Greedy:")
        logger.info(f"  Accuracy delta  : {delta_acc:+.4f} ({delta_acc*100:+.1f}pp)")
        logger.info(f"  Token overhead  : {delta_tok:+.1f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIGTE Ablation Runner")
    p.add_argument("--config", default="configs/inference_config.yaml")
    p.add_argument("--modes", nargs="+", default=ABLATION_MODES)
    p.add_argument("--model_short_name", default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model_short_name:
        cfg["model"]["short_name"] = args.model_short_name

    logger.info("DIGTE — Ablation Study")
    logger.info(f"  modes   : {args.modes}")
    logger.info(f"  dataset : {cfg['dataset']['name']}")

    all_results: dict[str, dict] = {}
    for mode in args.modes:
        m = _run_mode(mode, cfg, args.config)
        if m:
            all_results[mode] = m
            logger.info(f"  {mode}: acc={m.get('accuracy', 0):.4f}")

    if all_results:
        print_table(all_results)
        short = cfg["model"]["short_name"]
        out_path = Path(cfg["output"]["dir"]) / f"ablation_summary_{short}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved ablation summary to {out_path}")
    else:
        logger.warning("No results collected")


if __name__ == "__main__":
    main()