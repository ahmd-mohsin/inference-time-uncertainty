# Robust two-phase runner: persistent vLLM generation, then single HF hidden-state pass.
#
# Avoids the per-problem vLLM<->HF swapping that crashes TP=8 engines after many reinits.
#   Phase A: load vLLM ONCE (TP=8), generate IID + conditioned chains for ALL problems,
#            save raw text/answers to chains_raw.json.
#   Phase B: tear down vLLM, load HF ONCE, extract subsampled hidden states for every chain.
#   Phase C: per-problem topology + ceiling detection -> problem_*.json (same schema as run.py).
#
# Usage:
#   python -m topological_persistence.run_robust --model Qwen/Qwen3-32B --dataset aime_2026 \
#       --n-problems 30 --n-chains 8 --output-dir data/topological_outputs_aime2026

import argparse
import json
import logging
import re
import sys
import os
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topological_persistence.config import load_config
from topological_persistence.sampler import Chain
from topological_persistence.embeddings import embed_chains, ncd_distance_matrix
from topological_persistence.distances import compute_distance_matrix
from topological_persistence.persistence import compute_topological_signature
from topological_persistence.ceiling_detector import detect_ceiling, compare_topologies
from topological_persistence.conditioning import build_disagreement_workspace, build_conditioned_prompt
from topological_persistence.pipeline import format_problem_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HIDDEN_SUBSAMPLE = 32


def _extract_answer(text: str) -> str:
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m[-1].strip() if m else ""


def phase_a_generate(cfg, problems, out_dir):
    """One persistent vLLM generates IID + conditioned chains for all problems."""
    from vllm import LLM, SamplingParams
    logger.info(f"PHASE A: loading vLLM TP={cfg.sampling.tensor_parallel_size} (once)")
    llm = LLM(
        model=cfg.sampling.model_name, dtype=cfg.sampling.dtype,
        tensor_parallel_size=cfg.sampling.tensor_parallel_size,
        trust_remote_code=True, max_model_len=16384 + 512,
        enable_prefix_caching=True, gpu_memory_utilization=0.9,
    )

    def gen(prompt, n):
        params = SamplingParams(n=n, max_tokens=cfg.sampling.max_new_tokens,
                                temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
                                stop=["<|im_end|>", "<|endoftext|>"])
        out = llm.generate([prompt], params)[0]
        chains = []
        for o in out.outputs:
            chains.append({"text": o.text, "answer": _extract_answer(o.text),
                           "truncated": o.finish_reason == "length",
                           "n_tokens": len(o.token_ids) if o.token_ids else 0})
        return chains

    raw_path = Path(out_dir) / "chains_raw.json"
    raw = {}
    if raw_path.exists():
        raw = json.load(open(raw_path))

    for i, problem in enumerate(problems):
        pid = str(problem.get("problem_id", i))
        if pid in raw:
            logger.info(f"  [{i+1}/{len(problems)}] problem {pid} already generated, skip")
            continue
        logger.info(f"  [{i+1}/{len(problems)}] problem {pid}: generating IID...")
        prompt = format_problem_prompt(problem, cfg.sampling.model_name)
        iid = gen(prompt, cfg.sampling.n_chains)

        iid_chains = [Chain(text=c["text"], answer=c["answer"], n_tokens=c["n_tokens"],
                            truncated=c["truncated"]) for c in iid]
        workspace = build_disagreement_workspace(iid_chains)
        cond_prompt = build_conditioned_prompt(problem["question"], workspace, cfg.sampling.model_name)
        logger.info(f"  [{i+1}/{len(problems)}] problem {pid}: generating conditioned...")
        cond = gen(cond_prompt, cfg.n_conditioned_chains)

        raw[pid] = {"question": problem["question"], "gold_answer": problem.get("gold_answer", ""),
                    "iid": iid, "cond": cond}
        json.dump(raw, open(raw_path, "w"))
        logger.info(f"    saved raw chains for problem {pid}")

    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()
    logger.info("PHASE A complete, vLLM unloaded")
    return raw_path


def phase_b_hidden_states(cfg, raw_path, out_dir):
    """One HF load extracts subsampled hidden states for every saved chain."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("PHASE B: loading HF model (once) for hidden states")
    tok = AutoTokenizer.from_pretrained(cfg.sampling.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.sampling.model_name, torch_dtype=getattr(torch, cfg.sampling.dtype),
        device_map="auto", trust_remote_code=True)
    model.eval()

    raw = json.load(open(raw_path))
    hidden_path = Path(out_dir) / "hidden_states.npz"
    store = {}

    @torch.no_grad()
    def extract(prompt, generation):
        full = prompt + generation
        ids = tok(full, return_tensors="pt", truncation=True, max_length=16384 + 512)["input_ids"].to(model.device)
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, plen:, :].cpu().float().numpy()
        torch.cuda.empty_cache()
        if h.shape[0] > 0:
            idx = list(range(0, h.shape[0], HIDDEN_SUBSAMPLE)) or [0]
            return h[idx]
        return np.zeros((1, model.config.hidden_size), dtype=np.float32)

    for pid, d in raw.items():
        prompt = format_problem_prompt({"question": d["question"]}, cfg.sampling.model_name)
        for tag, chains in [("iid", d["iid"]), ("cond", d["cond"])]:
            for j, c in enumerate(chains):
                key = f"{pid}_{tag}_{j}"
                if key in store:
                    continue
                store[key] = extract(prompt, c["text"])
        logger.info(f"  extracted hidden states for problem {pid}")
        np.savez_compressed(hidden_path, **store)

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    logger.info("PHASE B complete, HF unloaded")
    return hidden_path


def phase_c_topology(cfg, raw_path, hidden_path, out_dir):
    """Per-problem topology + ceiling detection, writes problem_*.json."""
    logger.info("PHASE C: computing topology + ceiling detection")
    raw = json.load(open(raw_path))
    H = np.load(hidden_path)

    results = []
    for pid, d in raw.items():
        def build_chains(tag, chains):
            out = []
            for j, c in enumerate(chains):
                hs = H[f"{pid}_{tag}_{j}"]
                out.append(Chain(text=c["text"], answer=c["answer"], n_tokens=c["n_tokens"],
                                 truncated=c["truncated"], hidden_states=hs))
            return out

        chains_iid = build_chains("iid", d["iid"])
        chains_cond = build_chains("cond", d["cond"])

        emb_iid = embed_chains(chains_iid, cfg.embedding)
        D_iid = compute_distance_matrix(emb_iid, cfg.topology.distance_metric)
        sig_iid = compute_topological_signature(D_iid, cfg.topology.max_homology_dim, cfg.topology.n_radii)

        emb_cond = embed_chains(chains_cond, cfg.embedding)
        D_cond = compute_distance_matrix(emb_cond, cfg.topology.distance_metric)
        sig_cond = compute_topological_signature(D_cond, cfg.topology.max_homology_dim, cfg.topology.n_radii)

        signal = detect_ceiling(sig_iid, sig_cond)
        comparison = compare_topologies(sig_iid, sig_cond)

        D_ncd = ncd_distance_matrix(chains_iid)
        sig_ncd = compute_topological_signature(D_ncd, cfg.topology.max_homology_dim, cfg.topology.n_radii)
        signal_ncd = detect_ceiling(sig_ncd)
        iu = np.triu_indices_from(D_ncd, k=1)

        result = {
            "problem_id": int(pid), "question": d["question"][:200],
            "gold_answer": d["gold_answer"],
            "n_chains_iid": len(chains_iid), "n_chains_conditioned": len(chains_cond),
            "answers_iid": [c.answer for c in chains_iid],
            "answers_conditioned": [c.answer for c in chains_cond],
            "signal": asdict(signal),
            "signal_ncd": {"mean_ncd": float(D_ncd[iu].mean()), "min_ncd": float(D_ncd[iu].min()),
                           "max_ncd": float(D_ncd[iu].max()), "h1_features_ncd": signal_ncd.h1_n_features,
                           "verdict_ncd": signal_ncd.verdict},
            "comparison": comparison,
            "distance_matrix_iid": D_iid.tolist(), "distance_matrix_ncd": D_ncd.tolist(),
        }
        with open(Path(out_dir) / f"problem_{pid}.json", "w") as f:
            json.dump(result, f, indent=2)
        results.append(result)
        logger.info(f"  problem {pid}: verdict={signal.verdict} H1={signal.h1_n_features}")

    summary = {"n_problems": len(results),
               "n_ceiling": sum(1 for r in results if r["signal"]["verdict"] == "CEILING_REACHED"),
               "n_scalable": sum(1 for r in results if r["signal"]["verdict"] == "SCALABLE"),
               "n_uncertain": sum(1 for r in results if r["signal"]["verdict"] == "UNCERTAIN")}
    json.dump(summary, open(Path(out_dir) / "summary.json", "w"), indent=2)
    logger.info(f"PHASE C complete: {summary}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--dataset", default="aime_2026")
    p.add_argument("--n-problems", type=int, default=30)
    p.add_argument("--n-chains", type=int, default=8)
    p.add_argument("--output-dir", default="data/topological_outputs_aime2026")
    args = p.parse_args()

    cfg = load_config()
    cfg.sampling.model_name = args.model
    cfg.sampling.use_vllm = True
    cfg.dataset = args.dataset
    cfg.n_problems = args.n_problems
    cfg.sampling.n_chains = args.n_chains
    cfg.n_conditioned_chains = args.n_chains
    cfg.output_dir = args.output_dir

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from src.data.dataset import get_inference_dataset
    problems = get_inference_dataset({"dataset": {"name": cfg.dataset, "split": "test",
                                                   "n_problems": cfg.n_problems, "seed": cfg.seed}})

    raw_path = phase_a_generate(cfg, problems, args.output_dir)
    hidden_path = phase_b_hidden_states(cfg, raw_path, args.output_dir)
    phase_c_topology(cfg, raw_path, hidden_path, args.output_dir)
    logger.info("ALL PHASES COMPLETE")


if __name__ == "__main__":
    main()
