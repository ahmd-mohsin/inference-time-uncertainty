#!/usr/bin/env bash
#
# run_parallel.sh — Data-parallel pipeline across all 8 GPUs.
#
# An 8B model fits on one 40GB A100, so instead of tensor-parallel (TP=8, slow + crash-prone)
# we run 8 INDEPENDENT vLLM replicas, one pinned per GPU via CUDA_VISIBLE_DEVICES. Problems
# are round-robin sharded across replicas. After all shards finish we merge and analyze.
#
# Stages:
#   1. Generate  — 8 shards in parallel (each shard: vLLM gen + HF hidden states + topology)
#   2. Merge     — combine chains_raw/hidden_states, write summary.json
#   3. Validate  — 8 shards in parallel (64 chains/problem), then merge validation.json
#   4. Spectral re-analysis + plots
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

DIR="data/topological_outputs_aime2026_qwen8b"
MODEL="Qwen/Qwen3-8B"
NPROB=30
NCHAINS=8
NVAL=64
NGPU=8

mkdir -p "$DIR" ~/logs

echo "===== STAGE 1: Generate ($NGPU shards in parallel) ====="
pids=()
for s in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$s python -m topological_persistence.run_robust \
        --model "$MODEL" --dataset aime_2026 --n-problems "$NPROB" \
        --n-chains "$NCHAINS" --output-dir "$DIR" \
        --shard-index "$s" --num-shards "$NGPU" \
        > ~/logs/gen_shard${s}.log 2>&1 &
    pids+=($!)
    echo "  launched gen shard $s on GPU $s (pid ${pids[-1]})"
done
echo "  waiting for ${#pids[@]} generation shards..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "Stage 1 done (fail=$fail)"

echo "===== STAGE 2: Merge generation shards ====="
python -m topological_persistence.run_robust \
    --output-dir "$DIR" --num-shards "$NGPU" --merge-only \
    > ~/logs/merge_gen.log 2>&1
echo "Stage 2 done (exit $?)"

echo "===== STAGE 3: Validate ($NGPU shards in parallel, $NVAL chains/problem) ====="
pids=()
for s in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$s python -m topological_persistence.analyze_results \
        --results-dir "$DIR" --validate --n-validation "$NVAL" \
        --dataset aime_2026 --model "$MODEL" \
        --shard-index "$s" --num-shards "$NGPU" \
        > ~/logs/val_shard${s}.log 2>&1 &
    pids+=($!)
    echo "  launched val shard $s on GPU $s (pid ${pids[-1]})"
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
python -m topological_persistence.analyze_results \
    --results-dir "$DIR" --num-shards "$NGPU" --merge-only \
    > ~/logs/merge_val.log 2>&1
echo "Stage 3 done (exit $?)"

echo "===== STAGE 4: Spectral re-analysis + plots ====="
python -m topological_persistence.spectral_reanalysis \
    --data-dir "$DIR" > ~/logs/spectral.log 2>&1
python -m topological_persistence.plot_metrics \
    --results-dir "$DIR" --out-dir "$DIR/figures" > ~/logs/plot.log 2>&1
echo "Stage 4 done (exit $?)"

echo "===== ALL STAGES COMPLETE ====="
touch ~/PARALLEL_DONE_8B
