import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml
import torch

logger = logging.getLogger(__name__)


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level_str = log_cfg.get("level", "INFO")
    level = getattr(logging, level_str, logging.INFO)
    log_file = log_cfg.get("log_file", "logs/calibration.log")
    log_file = log_file.replace("{model_short_name}", cfg["model"]["short_name"])
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        force=True,
    )


def _resolve_paths(cfg: dict) -> dict:
    short = cfg["model"]["short_name"]
    cfg["output"]["token_data_file"] = cfg["output"]["token_data_file"].replace(
        "{model_short_name}", short
    )
    cfg["logging"]["log_file"] = cfg["logging"]["log_file"].replace(
        "{model_short_name}", short
    )
    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def run_phase1(cfg: dict, skip_collection: bool = False) -> dict:
    from src.data.model_loader import ModelLoader
    from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
    from src.uncertainty.disagreement import SemanticDisagreementDetector
    from src.calibration.token_collector import TokenDataCollector
    from src.calibration.disagreement_collector import DisagreementDataCollector
    from src.calibration.threshold_optimizer import ThresholdOptimizer
    from src.calibration.analysis import CalibrationAnalyzer

    t_start = time.time()
    cal_cfg = cfg["calibration"]

    token_data_path = cfg["output"]["token_data_file"]
    disagreement_data_path = token_data_path.replace(".jsonl", "_disagreement.jsonl")
    thresholds_path = cfg["output"]["thresholds_file"]

    loader = ModelLoader(cfg)
    model, tokenizer = loader.load()

    zone_classifier = SemanticLoadZoneClassifier.from_config(cfg)
    optimizer = ThresholdOptimizer(cfg)
    analyzer = CalibrationAnalyzer(cfg)

    if not skip_collection:
        logger.info("=" * 60)
        logger.info("PHASE 1a — Collecting token data (entropy + step labels)")
        logger.info("=" * 60)
        collector = TokenDataCollector(
            model=model,
            tokenizer=tokenizer,
            zone_classifier=zone_classifier,
            cfg=cfg,
        )
        collector.collect(save_path=token_data_path)
    else:
        logger.info(f"Skipping 1a — loading from {token_data_path}")

    logger.info("=" * 60)
    logger.info("PHASE 1b — Optimizing tau_e (entropy threshold)")
    logger.info("=" * 60)
    token_records = optimizer.load_token_records(token_data_path)
    entropy_result = optimizer.optimize_entropy_threshold(token_records)
    tau_e = entropy_result["tau_e"]

    if not skip_collection:
        logger.info("=" * 60)
        logger.info("PHASE 1c — Collecting disagreement data at entropy-triggered positions")
        logger.info("=" * 60)
        detector = SemanticDisagreementDetector(
            k_continuations=cal_cfg["k_continuations"],
            continuation_length=cal_cfg["continuation_length"],
            temperature=cal_cfg["temperature"],
            disagreement_threshold=0.0,
        )
        dis_collector = DisagreementDataCollector(
            model=model,
            tokenizer=tokenizer,
            zone_classifier=zone_classifier,
            detector=detector,
            tau_e=tau_e,
            cfg=cfg,
        )
        dis_collector.collect(save_path=disagreement_data_path)
    else:
        logger.info(f"Skipping 1c — loading from {disagreement_data_path}")

    logger.info("=" * 60)
    logger.info("PHASE 1d — Optimizing tau_d (disagreement threshold)")
    logger.info("=" * 60)
    disagreement_records = optimizer.load_disagreement_records(disagreement_data_path)
    disagreement_result = optimizer.optimize_disagreement_threshold(disagreement_records)
    tau_d = disagreement_result["tau_d"]

    logger.info("=" * 60)
    logger.info("PHASE 1e — Signal comparison and plots")
    logger.info("=" * 60)
    signal_comparison = optimizer.compare_signals(token_records)
    entropy_stats = analyzer.plot_entropy_distribution(token_records, tau_e)
    disagreement_stats = analyzer.plot_disagreement_distribution(disagreement_records, tau_d)
    analyzer.plot_signal_comparison_roc(token_records)

    metadata = {
        **entropy_result,
        **disagreement_result,
        "signal_comparison": signal_comparison,
        "n_token_records": len(token_records),
        "n_disagreement_records": len(disagreement_records),
        "calibration_time_sec": round(time.time() - t_start, 1),
    }

    optimizer.save_thresholds(
        model_short_name=cfg["model"]["short_name"],
        tau_e=tau_e,
        tau_d=tau_d,
        metadata=metadata,
        path=thresholds_path,
    )

    analyzer.print_summary(
        tau_e=tau_e,
        tau_d=tau_d,
        entropy_stats=entropy_stats,
        disagreement_stats=disagreement_stats,
        signal_comparison=signal_comparison,
    )

    logger.info(f"Phase 1 complete in {time.time() - t_start:.1f}s")
    return {"tau_e": tau_e, "tau_d": tau_d, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIGTE — Phase 1: Threshold Calibration")
    p.add_argument("--config", default="configs/calibration_config.yaml")
    p.add_argument("--model", default=None, help="HuggingFace model name")
    p.add_argument("--model_short_name", default=None)
    p.add_argument("--n_problems", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--metric", choices=["auroc", "f1", "precision"], default=None)
    p.add_argument(
        "--skip_collection",
        action="store_true",
        help="Skip 1a and 1c data collection; load existing JSONL files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.model_short_name:
        cfg["model"]["short_name"] = args.model_short_name
    if args.n_problems is not None:
        cfg["calibration"]["n_problems"] = args.n_problems
    if args.seed is not None:
        cfg["calibration"]["seed"] = args.seed
    if args.metric:
        cfg["calibration"]["metric"] = args.metric

    cfg = _resolve_paths(cfg)
    setup_logging(cfg)

    logger.info("DIGTE — Phase 1: Threshold Calibration")
    logger.info(f"  config            : {args.config}")
    logger.info(f"  model             : {cfg['model']['name']}")
    logger.info(f"  model_short_name  : {cfg['model']['short_name']}")
    logger.info(f"  dataset           : {cfg['calibration']['dataset']} / {cfg['calibration']['split']}")
    logger.info(f"  n_problems        : {cfg['calibration']['n_problems']}")
    logger.info(f"  metric            : {cfg['calibration']['metric']}")
    logger.info(f"  skip_collection   : {args.skip_collection}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU               : {props.name}  {props.total_memory / 1e9:.1f} GB")
    else:
        logger.warning("  No CUDA GPU found — running on CPU (very slow)")

    result = run_phase1(cfg, skip_collection=args.skip_collection)
    logger.info(
        f"Done.  tau_e={result['tau_e']:.4f}  tau_d={result['tau_d']:.4f}"
    )


if __name__ == "__main__":
    main()