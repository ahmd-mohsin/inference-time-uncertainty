#!/usr/bin/env bash
#
# run_scaled.sh — Scaled experiment on 8 A100s. Changes vs the 30-problem run, each tied
# to a finding from that run:
#   * 90 problems (AIME 2024+2025+2026) — prior AUCs rested on 5 scaling positives; 3x
#     problems -> ~15 positives -> trustworthy stats.
#   * multi-layer hidden states (mid/3q/last) saved in Phase B (run_robust change).
#   * Stage 5 probe_analysis: multi-layer eff-rank (D2) + internal-correctness probe (D12).
#   * answer-entropy stays computed but is NO LONGER trusted as the verdict (AUC 0.49).
# 8-way data parallel: ~11 problems/GPU.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

DIR="data/topological_outputs_aime_all_qwen8b"
MODEL="Qwen/Qwen3-8B"
DATASET="aime_all"
NPROB=90
NCHAINS=8
NVAL=64
NGPU=8
mkdir -p "$DIR" ~/logs

echo "===== STAGE 1: Generate ($NGPU shards, $NPROB problems) ====="
pids=()
for s in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$s python -m topological_persistence.run_robust \
    --model "$MODEL" --dataset "$DATASET" --n-problems "$NPROB" \
    --n-chains "$NCHAINS" --output-dir "$DIR" --shard-index "$s" --num-shards "$NGPU" \
    > ~/logs/s_gen_shard${s}.log 2>&1 &
  pids+=($!); echo "  gen shard $s -> GPU $s (pid ${pids[-1]})"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "Stage 1 done (fail=$fail)"

echo "===== STAGE 2: Merge ====="
python -m topological_persistence.run_robust --output-dir "$DIR" --num-shards "$NGPU" --merge-only > ~/logs/s_merge.log 2>&1
echo "Stage 2 done (exit $?)"

echo "===== STAGE 3: Validate ($NGPU shards, $NVAL chains) ====="
pids=()
for s in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$s python -m topological_persistence.analyze_results \
    --results-dir "$DIR" --validate --n-validation "$NVAL" \
    --dataset "$DATASET" --model "$MODEL" --n-problems "$NPROB" \
    --shard-index "$s" --num-shards "$NGPU" \
    > ~/logs/s_val_shard${s}.log 2>&1 &
  pids+=($!); echo "  val shard $s -> GPU $s (pid ${pids[-1]})"
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
python -m topological_persistence.analyze_results --results-dir "$DIR" --num-shards "$NGPU" --merge-only > ~/logs/s_merge_val.log 2>&1
echo "Stage 3 done (val fail=$fail)"

echo "===== STAGE 4: Spectral re-analysis + plots ====="
python -m topological_persistence.spectral_reanalysis --data-dir "$DIR" > ~/logs/s_spectral.log 2>&1
python -m topological_persistence.plot_metrics --results-dir "$DIR" --out-dir "$DIR/figures" > ~/logs/s_plot.log 2>&1

echo "===== STAGE 5: Probe + multi-layer analysis (D12 + D2) ====="
python -m topological_persistence.probe_analysis --data-dir "$DIR" > ~/logs/s_probe.log 2>&1
echo "Stage 5 done (exit $?)"

echo "===== SCALED RUN COMPLETE ====="
touch ~/SCALED_DONE
