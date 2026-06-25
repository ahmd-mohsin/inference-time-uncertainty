#!/usr/bin/env bash
#
# recover_validation.sh — Re-run ONLY validation (+spectral+probe) for the 90-problem run.
#
# Bug: analyze_results used cfg.n_problems (default 30) instead of the run's 90, so
# validation covered only the first 30 (all AIME-2024) -> validation.json had 30, and
# spectral/probe ran on the wrong subset. Generation/hidden-states for all 90 are intact,
# so we only redo Stages 3-5 with --n-problems 90.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

DIR="data/topological_outputs_aime_all_qwen8b"
MODEL="Qwen/Qwen3-8B"
DATASET="aime_all"
NPROB=90
NVAL=64
NGPU=8

# clear stale (30-problem) validation outputs so merge is clean
rm -f $DIR/validation.json $DIR/validation_shard*.json
echo "cleared stale validation files"

echo "===== Validate ($NGPU shards, $NVAL chains, $NPROB problems) ====="
pids=()
for s in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$s python -m topological_persistence.analyze_results \
    --results-dir "$DIR" --validate --n-validation "$NVAL" \
    --dataset "$DATASET" --model "$MODEL" --n-problems "$NPROB" \
    --shard-index "$s" --num-shards "$NGPU" \
    > ~/logs/r_val_shard${s}.log 2>&1 &
  pids+=($!); echo "  val shard $s -> GPU $s (pid ${pids[-1]})"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
python -m topological_persistence.analyze_results --results-dir "$DIR" --num-shards "$NGPU" --merge-only > ~/logs/r_merge_val.log 2>&1
echo "Validation done (fail=$fail)"

echo "===== Spectral + probe ====="
python -m topological_persistence.spectral_reanalysis --data-dir "$DIR" > ~/logs/r_spectral.log 2>&1
python -m topological_persistence.probe_analysis --data-dir "$DIR" > ~/logs/r_probe.log 2>&1
echo "Spectral+probe done"

echo "===== RECOVERY VALIDATION COMPLETE ====="
touch ~/RECOVER_VAL_DONE
