import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _load_jsonlines(path: str) -> list[dict]:
    import jsonlines
    with jsonlines.open(path) as reader:
        return list(reader)


def validate(
    token_records: list[dict],
    disagreement_records: list[dict],
    tau_e: float,
) -> dict:
    from sklearn.metrics import roc_auc_score

    zone = [r for r in token_records if r["in_semantic_zone"]]
    if not zone:
        return {"error": "no zone records"}

    labels_zone = np.array([int(not r["is_correct_step"]) for r in zone])
    if labels_zone.sum() == 0 or labels_zone.sum() == len(labels_zone):
        return {"error": "degenerate labels in zone"}

    results: dict = {}

    for name, scores in {
        "entropy_auroc": np.array([r["entropy"] for r in zone]),
        "neg_margin_auroc": np.array([-r["logit_margin"] for r in zone]),
        "neg_top1_auroc": np.array([-r["top1_prob"] for r in zone]),
    }.items():
        try:
            results[name] = float(roc_auc_score(labels_zone, scores))
        except Exception:
            results[name] = 0.5

    triggered = [r for r in disagreement_records if r.get("entropy_triggered", False)]
    if triggered:
        d_scores = np.array([r["disagreement_score"] for r in triggered])
        d_labels = np.array([int(not r["is_correct_step"]) for r in triggered])
        if 0 < d_labels.sum() < len(d_labels):
            try:
                results["disagreement_auroc"] = float(roc_auc_score(d_labels, d_scores))
            except Exception:
                results["disagreement_auroc"] = 0.5

            combined = []
            d_lookup = {(r["problem_id"], r["position"]): r for r in triggered}
            for rec in zone:
                key = (rec["problem_id"], rec["position"])
                if key in d_lookup:
                    combined.append({
                        "score": rec["entropy"] * d_lookup[key]["disagreement_score"],
                        "label": int(not rec["is_correct_step"]),
                    })
            if combined:
                cs = np.array([c["score"] for c in combined])
                cl = np.array([c["label"] for c in combined])
                if 0 < cl.sum() < len(cl):
                    try:
                        results["combined_auroc"] = float(roc_auc_score(cl, cs))
                    except Exception:
                        results["combined_auroc"] = 0.5

    d_auroc = results.get("disagreement_auroc", 0.0)
    results["claim1_supported"] = (
        d_auroc > results.get("entropy_auroc", 0)
        and d_auroc > results.get("neg_margin_auroc", 0)
        and d_auroc > results.get("neg_top1_auroc", 0)
    )

    sep = "=" * 60
    logger.info(sep)
    logger.info("CLAIM 1 VALIDATION")
    logger.info("Claim: d_t predicts step errors better than entropy,")
    logger.info("       logit margin, and top-1 confidence at zone positions.")
    logger.info(sep)
    logger.info(f"  entropy_auroc      : {results.get('entropy_auroc', 'N/A'):.4f}")
    logger.info(f"  neg_margin_auroc   : {results.get('neg_margin_auroc', 'N/A'):.4f}")
    logger.info(f"  neg_top1_auroc     : {results.get('neg_top1_auroc', 'N/A'):.4f}")
    logger.info(f"  disagreement_auroc : {results.get('disagreement_auroc', 'N/A'):.4f}")
    logger.info(f"  combined_auroc     : {results.get('combined_auroc', 'N/A'):.4f}")
    logger.info(f"  Claim 1 supported  : {results['claim1_supported']}")
    logger.info(sep)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Claim 1")
    p.add_argument("--config", default="configs/calibration_config.yaml")
    p.add_argument("--model_short_name", default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    short = args.model_short_name or cfg["model"]["short_name"]
    thresholds_path = cfg["output"]["thresholds_file"]

    if not Path(thresholds_path).exists():
        logger.error(f"Run calibrate.py first — {thresholds_path} not found")
        sys.exit(1)

    with open(thresholds_path) as f:
        all_t = json.load(f)

    tau_e = all_t.get(short, {}).get("tau_e", 1.5)
    token_path = cfg["output"]["token_data_file"].replace("{model_short_name}", short)
    dis_path = token_path.replace(".jsonl", "_disagreement.jsonl")

    if not Path(token_path).exists():
        logger.error(f"Token data not found: {token_path}")
        sys.exit(1)

    token_records = _load_jsonlines(token_path)
    disagreement_records = _load_jsonlines(dis_path) if Path(dis_path).exists() else []

    results = validate(token_records, disagreement_records, tau_e)

    out_path = Path(cfg["output"]["dir"]) / f"claim1_validation_{short}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()