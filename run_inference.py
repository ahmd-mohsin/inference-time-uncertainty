import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

MODES = ["digte", "greedy", "beam", "adadec", "entropy_only", "prompt_only", "ptcs", "egmi"]


def setup_logging(cfg: dict, mode: str) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
    short = cfg["model"]["short_name"]
    log_file = (
        log_cfg.get("log_file", "logs/inference_{model_short_name}_{decoding_mode}.log")
        .replace("{model_short_name}", short)
        .replace("{decoding_mode}", mode)
    )
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
        force=True,
    )


def _resolve_output_paths(cfg: dict, mode: str) -> dict:
    short = cfg["model"]["short_name"]
    for key in ["predictions_file", "metrics_file", "detail_log_file"]:
        if key in cfg["output"]:
            cfg["output"][key] = (
                cfg["output"][key]
                .replace("{model_short_name}", short)
                .replace("{decoding_mode}", mode)
            )
    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def _load_thresholds(cfg: dict) -> tuple[float, float]:
    path = cfg["decoding"]["thresholds_file"]
    short = cfg["model"]["short_name"]
    if not Path(path).exists():
        logger.warning(f"No thresholds file at {path} — using defaults tau_e=1.5 tau_d=0.3")
        return 1.5, 0.3
    with open(path) as f:
        all_t = json.load(f)
    if short not in all_t:
        logger.warning(f"No entry for '{short}' in thresholds — using defaults")
        return 1.5, 0.3
    return float(all_t[short]["tau_e"]), float(all_t[short]["tau_d"])


def _build_generator(mode: str, model, tokenizer, tau_e: float, tau_d: float, cfg: dict, log_detail: bool):
    from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
    from src.uncertainty.entropy_filter import EntropyPreFilter
    from src.uncertainty.disagreement import SemanticDisagreementDetector

    dec = cfg.get("decoding", {})
    zone = SemanticLoadZoneClassifier.from_config(cfg)

    if mode == "digte":
        from src.inference.digte_generator import DIGTEGenerator
        return DIGTEGenerator(
            model=model, tokenizer=tokenizer,
            zone_classifier=zone,
            entropy_filter=EntropyPreFilter(
                threshold=tau_e,
                min_tokens_before_trigger=dec.get("min_tokens_before_trigger", 20),
            ),
            disagreement_detector=SemanticDisagreementDetector(
                k_continuations=dec.get("k_continuations", 3),
                continuation_length=dec.get("continuation_length", 12),
                temperature=dec.get("continuation_temperature", 0.8),
                disagreement_threshold=tau_d,
            ),
            cfg=cfg,
            log_detail=log_detail,
        )

    if mode == "greedy":
        from src.baselines.greedy import GreedyGenerator
        return GreedyGenerator(model, tokenizer, cfg)

    if mode == "beam":
        from src.baselines.beam_search import BeamSearchGenerator
        return BeamSearchGenerator(model, tokenizer, cfg, beam_width=dec.get("beam_width", 3))

    if mode == "adadec":
        from src.baselines.adadec_math import AdaDecGenerator
        return AdaDecGenerator(
            model=model, tokenizer=tokenizer,
            zone_classifier=zone, tau_e=tau_e, cfg=cfg,
            lookahead_length=dec.get("continuation_length", 5),
            lookahead_beam_size=dec.get("k_continuations", 3),
        )

    if mode == "entropy_only":
        from src.baselines.entropy_only_expansion import EntropyOnlyExpansionGenerator
        return EntropyOnlyExpansionGenerator(
            model=model, tokenizer=tokenizer,
            zone_classifier=zone, tau_e=tau_e, cfg=cfg,
        )

    if mode == "egmi":
        from src.inference.egmi_generator import EGMIGenerator
        from src.uncertainty.mixture_injector import EntropyGatedMixtureInjector
        import json as _json
        from pathlib import Path as _Path
        dec_cfg = cfg.get("decoding", {})
        short = cfg["model"]["short_name"]
        tau_low = dec_cfg.get("egmi_tau_low", None)
        if tau_low is None:
            tpath = dec_cfg.get("thresholds_file", "data/calibration_outputs/learned_thresholds.json")
            if _Path(tpath).exists():
                with open(tpath) as _f:
                    _t = _json.load(_f)
                tau_low = float(_t.get(short, {}).get("tau_low_egmi", 0.3))
            else:
                tau_low = 0.3
        logger.info(f"EGMI tau_low={tau_low:.5f}  window={dec_cfg.get('egmi_strategy_window', 40)}  k={dec_cfg.get('egmi_top_k', 2)}")
        injector = EntropyGatedMixtureInjector(
            model=model,
            low_entropy_threshold=tau_low,
            strategy_window=dec_cfg.get("egmi_strategy_window", 40),
            top_k=dec_cfg.get("egmi_top_k", 2),
            device=cfg["model"]["device"],
        )
        return EGMIGenerator(
            model=model,
            tokenizer=tokenizer,
            injector=injector,
            cfg=cfg,
            log_detail=log_detail,
        )

    if mode == "ptcs":
        from src.inference.ptcs_generator import PTCSGenerator
        from src.uncertainty.pretokencommitment import PreTokenCommitmentDetector
        dec_cfg = cfg.get("decoding", {})
        detector = PreTokenCommitmentDetector(
            k_probes=dec_cfg.get("ptcs_k_probes", 5),
            noise_std=dec_cfg.get("ptcs_noise_std", 0.01),
            divergence_threshold=dec_cfg.get("ptcs_divergence_threshold", 0.05),
            device=cfg["model"]["device"],
        )
        return PTCSGenerator(
            model=model,
            tokenizer=tokenizer,
            detector=detector,
            cfg=cfg,
            log_detail=log_detail,
        )

    if mode == "prompt_only":
        from src.baselines.prompt_only import PromptOnlyGenerator
        return PromptOnlyGenerator(
            model=model, tokenizer=tokenizer,
            zone_classifier=zone,
            disagreement_detector=SemanticDisagreementDetector(
                k_continuations=dec.get("k_continuations", 3),
                continuation_length=dec.get("continuation_length", 12),
                temperature=dec.get("continuation_temperature", 0.8),
                disagreement_threshold=tau_d,
            ),
            tau_e=tau_e, tau_d=tau_d, cfg=cfg,
        )

    raise ValueError(f"Unknown mode: {mode}")


def _gen_to_stats(result, mode: str) -> dict:
    base = {
        "generated_text": result.generated_text,
        "total_tokens": result.total_tokens,
        "wall_time_sec": result.wall_time_sec,
        "decoding_mode": mode,
        "n_entropy_triggers": result.n_entropy_triggers,
        "n_expansion_triggers": result.n_expansion_triggers,
        "total_expansion_tokens": result.total_expansion_tokens,
        "trigger_rate_entropy": result.trigger_rate_entropy,
        "trigger_rate_expansion": result.trigger_rate_expansion,
    }
    if mode == "digte":
        base["trace"] = [vars(s) for s in result.trace]
    if mode == "egmi":
        base["n_injections"] = result.n_injections
        base["injection_positions"] = result.injection_positions
        base["mean_injection_entropy"] = result.mean_injection_entropy
    if mode == "ptcs":
        base["commitment_divergence"] = result.commitment_divergence
        base["is_unstable"] = result.is_unstable
        base["steered"] = result.steered
        base["selected_probe_idx"] = result.selected_probe_idx
        base["probe_wall_time_sec"] = result.probe_wall_time_sec
    return base


def run_inference(cfg: dict, mode: str) -> None:
    import jsonlines
    from src.data.dataset import get_inference_dataset, format_prompt
    from src.data.model_loader import ModelLoader
    from src.evaluation.math_eval import score_prediction
    from src.evaluation.metrics import MetricsAggregator, ProblemResult

    log_detail = cfg["output"].get("log_detail", False)

    model, tokenizer = ModelLoader(cfg).load()
    tau_e, tau_d = _load_thresholds(cfg)
    logger.info(f"Thresholds: tau_e={tau_e:.4f}  tau_d={tau_d:.4f}")

    generator = _build_generator(mode, model, tokenizer, tau_e, tau_d, cfg, log_detail)

    problems = get_inference_dataset(cfg)
    logger.info(f"Running {len(problems)} problems  mode={mode}")

    aggregator = MetricsAggregator(decoding_mode=mode, cfg=cfg)
    pred_path = cfg["output"]["predictions_file"]
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(pred_path, mode="w") as writer:
        for i, problem in enumerate(problems):
            prompt = format_prompt(problem, cfg["model"]["name"])
            prompt_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            )["input_ids"].to(cfg["model"]["device"])

            try:
                gen = generator.generate(prompt_ids)
            except Exception as e:
                logger.warning(f"Problem {problem['problem_id']} failed: {e}")
                continue

            score = score_prediction(gen.generated_text, problem["gold_answer"])
            stats = _gen_to_stats(gen, mode)

            writer.write({
                "problem_id": problem["problem_id"],
                "source": problem.get("source", ""),
                "question": problem["question"],
                "gold_answer": problem["gold_answer"],
                **stats,
                **score,
            })

            aggregator.add(ProblemResult(
                problem_id=problem["problem_id"],
                source=problem.get("source", ""),
                question=problem["question"],
                gold_answer=problem["gold_answer"],
                predicted_answer=score["extracted_answer"],
                correct=score["correct"],
                total_tokens=stats["total_tokens"],
                n_entropy_triggers=stats["n_entropy_triggers"],
                n_expansion_triggers=stats["n_expansion_triggers"],
                total_expansion_tokens=stats["total_expansion_tokens"],
                trigger_rate_entropy=stats["trigger_rate_entropy"],
                trigger_rate_expansion=stats["trigger_rate_expansion"],
                wall_time_sec=stats["wall_time_sec"],
                decoding_mode=mode,
                has_boxed=score.get("has_boxed", False),
                level=problem.get("level", ""),
                problem_type=problem.get("problem_type", ""),
            ))

            if (i + 1) % 50 == 0:
                acc = sum(r.correct for r in aggregator.results) / len(aggregator.results)
                logger.info(f"  [{i+1}/{len(problems)}] acc={acc:.4f}")

    metrics = aggregator.compute()
    aggregator.print_summary(metrics)
    aggregator.save_metrics(metrics, cfg["output"]["metrics_file"])
    logger.info(f"Predictions → {pred_path}")
    logger.info(f"Metrics     → {cfg['output']['metrics_file']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIGTE Phase 2 — Inference")
    p.add_argument("--config", default="configs/inference_config.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--model_short_name", default=None)
    p.add_argument("--mode", default="digte", choices=MODES)
    p.add_argument("--dataset", default=None)
    p.add_argument("--n_problems", type=int, default=None)
    p.add_argument("--log_detail", action="store_true")
    p.add_argument("--thresholds_file", default=None)
    p.add_argument("--tau_e", type=float, default=None)
    p.add_argument("--tau_d", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.model_short_name:
        cfg["model"]["short_name"] = args.model_short_name
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.n_problems is not None:
        cfg["dataset"]["n_problems"] = args.n_problems
    if args.log_detail:
        cfg["output"]["log_detail"] = True
    if args.thresholds_file:
        cfg["decoding"]["thresholds_file"] = args.thresholds_file

    if args.tau_e is not None or args.tau_d is not None:
        short = cfg["model"]["short_name"]
        path = cfg["decoding"]["thresholds_file"]
        existing = {}
        if Path(path).exists():
            with open(path) as f:
                existing = json.load(f)
        existing.setdefault(short, {})
        if args.tau_e is not None:
            existing[short]["tau_e"] = args.tau_e
        if args.tau_d is not None:
            existing[short]["tau_d"] = args.tau_d
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    cfg = _resolve_output_paths(cfg, args.mode)
    setup_logging(cfg, args.mode)

    logger.info("DIGTE — Phase 2: Inference")
    logger.info(f"  mode    : {args.mode}")
    logger.info(f"  model   : {cfg['model']['name']}")
    logger.info(f"  dataset : {cfg['dataset']['name']}")
    logger.info(f"  n_probs : {cfg['dataset'].get('n_problems', -1)}")

    if torch.cuda.is_available():
        logger.info(f"  GPU     : {torch.cuda.get_device_name(0)}")

    run_inference(cfg, args.mode)


if __name__ == "__main__":
    main()