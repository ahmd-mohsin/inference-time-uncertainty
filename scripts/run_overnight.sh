#!/usr/bin/env bash
#
# run_overnight.sh — Unattended end-to-end: generate AIME2026, validate both datasets, plot.
# Runs ON the Greenland instance. Each stage logs to its own file; failures don't abort the rest.
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

DIR2024="data/topological_outputs"
DIR2026="data/topological_outputs_aime2026"
MODEL="Qwen/Qwen3-32B"
NVAL=64

echo "===== STAGE 1: Generate AIME 2026 (30 problems) ====="
python -m topological_persistence.run \
    --model "$MODEL" --dataset aime_2026 --n-problems 30 \
    --n-chains 8 --representation curve --verbose \
    --output-dir "$DIR2026" > ~/gen_2026.log 2>&1
echo "Stage 1 done (exit $?)"

echo "===== STAGE 2: Validate AIME 2024 (64 chains/problem) ====="
python -m topological_persistence.analyze_results \
    --results-dir "$DIR2024" --validate --n-validation "$NVAL" \
    --dataset aime_2024 --model "$MODEL" > ~/validate_2024.log 2>&1
echo "Stage 2 done (exit $?)"

echo "===== STAGE 3: Validate AIME 2026 (64 chains/problem) ====="
python -m topological_persistence.analyze_results \
    --results-dir "$DIR2026" --validate --n-validation "$NVAL" \
    --dataset aime_2026 --model "$MODEL" > ~/validate_2026.log 2>&1
echo "Stage 3 done (exit $?)"

echo "===== STAGE 4: Plots ====="
python -m topological_persistence.plot_metrics \
    --results-dir "$DIR2024" --out-dir "$DIR2024/figures" > ~/plot_2024.log 2>&1
python -m topological_persistence.plot_metrics \
    --results-dir "$DIR2026" --out-dir "$DIR2026/figures" > ~/plot_2026.log 2>&1
echo "Stage 4 done (exit $?)"

echo "===== ALL STAGES COMPLETE ====="
touch ~/OVERNIGHT_DONE
