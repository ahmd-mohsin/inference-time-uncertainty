# Entry point for running the topological persistence ceiling detection experiment.
import argparse
import logging

from topological_persistence.config import load_config
from topological_persistence.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--n-problems", type=int, default=None)
    parser.add_argument("--n-chains", type=int, default=None)
    parser.add_argument("--representation", type=str, choices=["point", "curve", "steps"], default=None)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)

    if args.model:
        cfg.sampling.model_name = args.model
    if args.dataset:
        cfg.dataset = args.dataset
    if args.n_problems:
        cfg.n_problems = args.n_problems
    if args.n_chains:
        cfg.sampling.n_chains = args.n_chains
        cfg.n_conditioned_chains = args.n_chains
    if args.representation:
        cfg.embedding.representation = args.representation
    if args.no_vllm:
        cfg.sampling.use_vllm = False
    if args.output_dir:
        cfg.output_dir = args.output_dir

    run_experiment(cfg)


if __name__ == "__main__":
    main()
