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
from topological_persistence.ceiling_detector import detect_ceiling, detect_ceiling_v2, compare_topologies
from topological_persistence.conditioning import build_disagreement_workspace, build_conditioned_prompt
from topological_persistence.pipeline import format_problem_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HIDDEN_SUBSAMPLE = 32


class _NpJSONEncoder(json.JSONEncoder):
    """JSON encoder that tolerates numpy scalars/arrays (np.bool_, np.float32, etc.).

    Guards against the per-problem dump crashing on a stray numpy type leaking out of
    the topology/spectral code (which already cost shard 7 problems 15 & 23 once).
    """
    def default(self, o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _extract_answer(text: str) -> str:
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m[-1].strip() if m else ""


def _suffix(shard_index, num_shards):
    """Per-shard filename suffix; empty string for the non-sharded (single-GPU) path."""
    return f"_shard{shard_index}" if num_shards > 1 else ""


def phase_a_generate(cfg, problems, out_dir, shard_index=0, num_shards=1):
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

    raw_path = Path(out_dir) / f"chains_raw{_suffix(shard_index, num_shards)}.json"
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


def phase_b_hidden_states(cfg, raw_path, out_dir, shard_index=0, num_shards=1):
    """One HF load extracts subsampled hidden states for every saved chain."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("PHASE B: loading HF model (once) for hidden states")
    tok = AutoTokenizer.from_pretrained(cfg.sampling.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.sampling.model_name, torch_dtype=getattr(torch, cfg.sampling.dtype),
        device_map="auto", trust_remote_code=True)
    model.eval()

    raw = json.load(open(raw_path))
    hidden_path = Path(out_dir) / f"hidden_states{_suffix(shard_index, num_shards)}.npz"
    store = {}

    @torch.no_grad()
    def extract(prompt, generation):
        """Returns (last_layer_token_seq, multilayer_pooled).

        last_layer_token_seq: subsampled per-token states from the FINAL layer
            (unchanged schema -> topology/effective-rank consumers keep working).
        multilayer_pooled: shape (3, hidden) = mean-pooled generation states at the
            mid / three-quarter / last layers. Probing literature: mid-layers encode
            'what the model knows' better than the output layer, which is committed to
            the (here often wrong) emitted token. Cheap: 3 vectors, not 3 token-seqs.
        """
        full = prompt + generation
        ids = tok(full, return_tensors="pt", truncation=True, max_length=16384 + 512)["input_ids"].to(model.device)
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple len = n_layers+1 (idx 0 = embeddings)
        L = len(hs) - 1
        layer_idx = sorted({L // 2, (3 * L) // 4, L})  # mid, 3/4, last
        last = hs[-1][0, plen:, :].cpu().float().numpy()
        pooled = np.stack([hs[li][0, plen:, :].mean(axis=0).cpu().float().numpy()
                           if (hs[li].shape[1] - plen) > 0 else np.zeros(hs[li].shape[-1], dtype=np.float32)
                           for li in layer_idx])
        torch.cuda.empty_cache()
        if last.shape[0] > 0:
            idx = list(range(0, last.shape[0], HIDDEN_SUBSAMPLE)) or [0]
            return last[idx], pooled
        return np.zeros((1, model.config.hidden_size), dtype=np.float32), pooled

    for pid, d in raw.items():
        prompt = format_problem_prompt({"question": d["question"]}, cfg.sampling.model_name)
        for tag, chains in [("iid", d["iid"]), ("cond", d["cond"])]:
            for j, c in enumerate(chains):
                key = f"{pid}_{tag}_{j}"
                if key in store:
                    continue
                last_seq, pooled = extract(prompt, c["text"])
                store[key] = last_seq               # unchanged consumers
                store[f"{key}__ml"] = pooled        # new: 3-layer pooled vectors
        logger.info(f"  extracted hidden states for problem {pid}")
        np.savez_compressed(hidden_path, **store)

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    logger.info("PHASE B complete, HF unloaded")
    return hidden_path


def phase_c_topology(cfg, raw_path, hidden_path, out_dir, write_summary=True):
    """Per-problem topology + ceiling detection, writes problem_*.json.

    Per-problem JSON filenames key off the unique problem_id, so multiple shards write
    to the same out_dir without collision. The aggregate summary.json is written only
    when write_summary=True (the final/merge invocation), so a shard doesn't overwrite
    it with a partial count.
    """
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

        # Primary verdict: answer-distribution + spectral signals (point embeddings).
        # H1 topology is computed above for reference/plots only.
        points_iid = emb_iid.get("points")
        points_cond = emb_cond.get("points")
        signal = detect_ceiling_v2(
            answers_iid=[c.answer for c in chains_iid],
            points_iid=points_iid, points_cond=points_cond, sig_iid=sig_iid,
        )
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
            json.dump(result, f, indent=2, cls=_NpJSONEncoder)
        results.append(result)
        logger.info(f"  problem {pid}: verdict={signal.verdict} "
                    f"ent={signal.answer_entropy:.2f} uniq={signal.n_unique_answers} "
                    f"eRank={signal.effective_rank:.2f} sgain={signal.spectral_gain} "
                    f"(H1={signal.h1_n_features}, ref)")

    if write_summary:
        write_summary_from_dir(out_dir)
    logger.info(f"PHASE C complete: {len(results)} problems this shard")


def write_summary_from_dir(out_dir):
    """Aggregate summary.json from ALL problem_*.json in out_dir (across shards)."""
    all_results = []
    for p in sorted(Path(out_dir).glob("problem_*.json")):
        with open(p) as f:
            all_results.append(json.load(f))
    summary = {"n_problems": len(all_results),
               "n_ceiling": sum(1 for r in all_results if r["signal"]["verdict"] == "CEILING_REACHED"),
               "n_scalable": sum(1 for r in all_results if r["signal"]["verdict"] == "SCALABLE"),
               "n_uncertain": sum(1 for r in all_results if r["signal"]["verdict"] == "UNCERTAIN")}
    json.dump(summary, open(Path(out_dir) / "summary.json", "w"), indent=2)
    logger.info(f"Summary written: {summary}")
    return summary


def merge_hidden_states(out_dir, num_shards):
    """Merge per-shard hidden_states_shard*.npz into a single hidden_states.npz."""
    merged = {}
    for s in range(num_shards):
        shard_path = Path(out_dir) / f"hidden_states_shard{s}.npz"
        if shard_path.exists():
            with np.load(shard_path) as z:
                for k in z.files:
                    merged[k] = z[k]
    if merged:
        np.savez_compressed(Path(out_dir) / "hidden_states.npz", **merged)
        logger.info(f"Merged {len(merged)} hidden-state arrays from {num_shards} shards")


def merge_chains_raw(out_dir, num_shards):
    """Merge per-shard chains_raw_shard*.json into a single chains_raw.json."""
    merged = {}
    for s in range(num_shards):
        shard_path = Path(out_dir) / f"chains_raw_shard{s}.json"
        if shard_path.exists():
            merged.update(json.load(open(shard_path)))
    if merged:
        json.dump(merged, open(Path(out_dir) / "chains_raw.json", "w"))
        logger.info(f"Merged chains_raw for {len(merged)} problems from {num_shards} shards")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--dataset", default="aime_2026")
    p.add_argument("--n-problems", type=int, default=30)
    p.add_argument("--n-chains", type=int, default=8)
    p.add_argument("--output-dir", default="data/topological_outputs_aime2026")
    # Data-parallel sharding: launch N copies pinned to different GPUs via
    # CUDA_VISIBLE_DEVICES, each with --shard-index i --num-shards N. Problems are
    # round-robin assigned to shards. After all shards finish, run --merge-only once
    # to combine chains_raw / hidden_states and write the aggregate summary.
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--merge-only", action="store_true",
                   help="Skip generation; merge shard outputs and write summary.json")
    p.add_argument("--phase-c-only", action="store_true",
                   help="Recompute Phase C (topology+JSON) from an existing shard's "
                        "chains_raw/hidden_states. No GPU generation. Use to recover from "
                        "a Phase-C crash without regenerating chains.")
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

    if args.merge_only:
        logger.info(f"MERGE-ONLY: combining {args.num_shards} shards in {args.output_dir}")
        merge_chains_raw(args.output_dir, args.num_shards)
        merge_hidden_states(args.output_dir, args.num_shards)
        write_summary_from_dir(args.output_dir)
        logger.info("MERGE COMPLETE")
        return

    if args.phase_c_only:
        raw_path = Path(args.output_dir) / f"chains_raw{_suffix(args.shard_index, args.num_shards)}.json"
        hidden_path = Path(args.output_dir) / f"hidden_states{_suffix(args.shard_index, args.num_shards)}.npz"
        logger.info(f"PHASE-C-ONLY: recomputing topology/JSON from {raw_path.name}, {hidden_path.name}")
        phase_c_topology(cfg, raw_path, hidden_path, args.output_dir, write_summary=False)
        logger.info("PHASE-C-ONLY complete")
        return

    from src.data.dataset import get_inference_dataset
    problems = get_inference_dataset({"dataset": {"name": cfg.dataset, "split": "test",
                                                   "n_problems": cfg.n_problems, "seed": cfg.seed}})

    # round-robin shard assignment (keeps load balanced across GPUs)
    if args.num_shards > 1:
        problems = [pr for k, pr in enumerate(problems) if k % args.num_shards == args.shard_index]
        logger.info(f"SHARD {args.shard_index}/{args.num_shards}: {len(problems)} problems")

    raw_path = phase_a_generate(cfg, problems, args.output_dir, args.shard_index, args.num_shards)
    hidden_path = phase_b_hidden_states(cfg, raw_path, args.output_dir, args.shard_index, args.num_shards)
    # each shard writes its own problem_*.json; summary is written by --merge-only
    phase_c_topology(cfg, raw_path, hidden_path, args.output_dir,
                     write_summary=(args.num_shards == 1))
    logger.info(f"SHARD {args.shard_index} ALL PHASES COMPLETE")


if __name__ == "__main__":
    main()
