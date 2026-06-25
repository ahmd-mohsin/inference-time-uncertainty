#!/usr/bin/env bash
#
# run_gap.sh — Verification-Generation gap study, 8-way data-parallel.
# 8 independent vLLM replicas (one per GPU), problems round-robin sharded, then merge+analyze.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

MODEL="${1:-Qwen/Qwen3-4B}"
DIR="data/verification_gap_qwen4b"
DATASET="aime_all"
NPROB=90
NCHAINS=16
NGPU=8
mkdir -p "$DIR" ~/logs

echo "===== STAGE 1: Generate + Verify ($NGPU shards, model=$MODEL) ====="
pids=()
for s in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$s python -m verification_gap.run_gap \
    --model "$MODEL" --dataset "$DATASET" --n-problems "$NPROB" --n-chains "$NCHAINS" \
    --output-dir "$DIR" --shard-index "$s" --num-shards "$NGPU" \
    > ~/logs/gap_shard${s}.log 2>&1 &
  pids+=($!); echo "  shard $s -> GPU $s (pid ${pids[-1]})"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "Stage 1 done (fail=$fail)"

echo "===== STAGE 2: Merge + analyze ====="
python -m verification_gap.run_gap --output-dir "$DIR" --num-shards "$NGPU" --merge-only > ~/logs/gap_merge.log 2>&1
python -m verification_gap.analyze_gap --data-dir "$DIR" > ~/logs/gap_analysis.log 2>&1
echo "Stage 2 done (exit $?)"

echo "===== GAP STUDY COMPLETE ====="
touch ~/GAP_DONE
