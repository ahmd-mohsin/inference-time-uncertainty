# End-to-end pipeline: sample → embed → distance → persistence → ceiling detection.
import json
import logging
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np

from topological_persistence.config import ExperimentConfig
from topological_persistence.sampler import get_sampler, Chain
from topological_persistence.embeddings import embed_chains
from topological_persistence.distances import compute_distance_matrix
from topological_persistence.persistence import compute_topological_signature
from topological_persistence.ceiling_detector import detect_ceiling, compare_topologies
from topological_persistence.conditioning import build_disagreement_workspace, build_conditioned_prompt

logger = logging.getLogger(__name__)


def format_problem_prompt(problem: dict, model_name: str) -> str:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.dataset import format_prompt
    return format_prompt(problem, model_name)


def run_single_problem(problem: dict, cfg: ExperimentConfig) -> dict:
    t_start = time.time()
    sampler = get_sampler(cfg.sampling)

    prompt = format_problem_prompt(problem, cfg.sampling.model_name)
    logger.info(f"Sampling {cfg.sampling.n_chains} IID chains for problem {problem.get('problem_id', '?')}")
    chains_iid = sampler.sample_chains(prompt, cfg.sampling.n_chains)

    embedding_data_iid = embed_chains(chains_iid, cfg.embedding)
    D_iid = compute_distance_matrix(embedding_data_iid, cfg.topology.distance_metric)
    sig_iid = compute_topological_signature(D_iid, cfg.topology.max_homology_dim, cfg.topology.n_radii)

    sig_cond = None
    chains_cond = None
    if cfg.conditioned_chains:
        workspace = build_disagreement_workspace(chains_iid)
        cond_prompt = build_conditioned_prompt(
            problem["question"], workspace, cfg.sampling.model_name
        )
        logger.info(f"Sampling {cfg.n_conditioned_chains} conditioned chains")
        chains_cond = sampler.sample_chains(cond_prompt, cfg.n_conditioned_chains)

        embedding_data_cond = embed_chains(chains_cond, cfg.embedding)
        D_cond = compute_distance_matrix(embedding_data_cond, cfg.topology.distance_metric)
        sig_cond = compute_topological_signature(D_cond, cfg.topology.max_homology_dim, cfg.topology.n_radii)

    signal = detect_ceiling(sig_iid, sig_cond)

    comparison = None
    if sig_cond is not None:
        comparison = compare_topologies(sig_iid, sig_cond)

    result = {
        "problem_id": problem.get("problem_id", -1),
        "question": problem.get("question", "")[:200],
        "gold_answer": problem.get("gold_answer", ""),
        "n_chains_iid": len(chains_iid),
        "n_chains_conditioned": len(chains_cond) if chains_cond else 0,
        "answers_iid": [c.answer for c in chains_iid],
        "answers_conditioned": [c.answer for c in chains_cond] if chains_cond else [],
        "signal": asdict(signal),
        "comparison": comparison,
        "distance_matrix_iid": D_iid.tolist(),
        "wall_time_sec": time.time() - t_start,
    }
    return result


def run_experiment(cfg: ExperimentConfig) -> list[dict]:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.dataset import get_inference_dataset

    dataset_cfg = {"dataset": {"name": cfg.dataset, "split": "test", "n_problems": cfg.n_problems, "seed": cfg.seed}}
    problems = get_inference_dataset(dataset_cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, problem in enumerate(problems):
        logger.info(f"[{i+1}/{len(problems)}] Problem {problem.get('problem_id', i)}")
        try:
            result = run_single_problem(problem, cfg)
            results.append(result)

            with open(output_dir / f"problem_{problem.get('problem_id', i)}.json", "w") as f:
                json.dump(result, f, indent=2)

            logger.info(
                f"  Verdict: {result['signal']['verdict']} "
                f"(p={result['signal']['ceiling_probability']:.2f}, "
                f"H1={result['signal']['h1_n_features']})"
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"problem_id": problem.get("problem_id", i), "error": str(e)})

    summary = {
        "n_problems": len(results),
        "n_ceiling": sum(1 for r in results if r.get("signal", {}).get("verdict") == "CEILING_REACHED"),
        "n_scalable": sum(1 for r in results if r.get("signal", {}).get("verdict") == "SCALABLE"),
        "n_uncertain": sum(1 for r in results if r.get("signal", {}).get("verdict") == "UNCERTAIN"),
        "n_errors": sum(1 for r in results if "error" in r),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. Ceiling={summary['n_ceiling']}, Scalable={summary['n_scalable']}, "
                f"Uncertain={summary['n_uncertain']}, Errors={summary['n_errors']}")
    return results
