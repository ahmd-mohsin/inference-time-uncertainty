#!/usr/bin/env bash
#
# recover_run.sh — Recover the Qwen3-8B run after the Phase-C numpy-bool crash.
#
# What happened: shard 7's Phase C crashed writing problem JSON (np.bool_ not
# serializable) -> problems 15 & 23 had no JSON -> merge/validation/spectral all failed.
# Fix is in code (bool() + _NpJSONEncoder + load_results skips bad files). Shard 7's
# chains_raw + hidden_states are intact, so we recover WITHOUT regenerating chains.
#
# Steps: (1) re-run shard 7 Phase C only [CPU]  (2) merge  (3) full validation [8 GPU]
#        (4) spectral + plots.
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

DIR="data/topological_outputs_aime2026_qwen8b"
MODEL="Qwen/Qwen3-8B"
NVAL=64
NGPU=8

echo "===== STEP 1: Phase-C recovery for shard 7 (problems 7,15,23) — CPU ====="
python -m topological_persistence.run_robust \
    --output-dir "$DIR" --shard-index 7 --num-shards "$NGPU" --phase-c-only \
    > ~/logs/recover_phasec.log 2>&1
echo "Step 1 done (exit $?); problem jsons now: $(ls $DIR/problem_*.json 2>/dev/null | wc -l)"

echo "===== STEP 2: Merge generation shards ====="
python -m topological_persistence.run_robust \
    --output-dir "$DIR" --num-shards "$NGPU" --merge-only \
    > ~/logs/recover_merge.log 2>&1
echo "Step 2 done (exit $?)"

echo "===== STEP 3: Validate ($NGPU shards in parallel, $NVAL chains/problem) ====="
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
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
python -m topological_persistence.analyze_results \
    --results-dir "$DIR" --num-shards "$NGPU" --merge-only \
    > ~/logs/recover_merge_val.log 2>&1
echo "Step 3 done (val fail=$fail)"

echo "===== STEP 4: Spectral re-analysis + plots ====="
python -m topological_persistence.spectral_reanalysis --data-dir "$DIR" > ~/logs/spectral.log 2>&1
python -m topological_persistence.plot_metrics --results-dir "$DIR" --out-dir "$DIR/figures" > ~/logs/plot.log 2>&1
echo "Step 4 done (exit $?)"

echo "===== RECOVERY COMPLETE ====="
touch ~/RECOVER_DONE_8B
