#!/usr/bin/env bash
#
# run_overnight.sh — Unattended: generate AIME2026 (robust two-phase), validate both, plot.
# Robust runner uses ONE persistent vLLM + ONE HF load (no per-problem TP reinit crashes).
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

DIR2024="data/topological_outputs"
DIR2026="data/topological_outputs_aime2026"
MODEL="Qwen/Qwen3-32B"
NVAL=64

echo "===== STAGE 1: Generate AIME 2026 (robust two-phase) ====="
python -m topological_persistence.run_robust \
    --model "$MODEL" --dataset aime_2026 --n-problems 30 \
    --n-chains 8 --output-dir "$DIR2026" > ~/gen_2026.log 2>&1
echo "Stage 1 done (exit $?)"

echo "===== STAGE 2: Validate AIME 2026 (64 chains/problem) ====="
python -m topological_persistence.analyze_results \
    --results-dir "$DIR2026" --validate --n-validation "$NVAL" \
    --dataset aime_2026 --model "$MODEL" > ~/validate_2026.log 2>&1
echo "Stage 2 done (exit $?)"

echo "===== STAGE 3: Plots ====="
python -m topological_persistence.plot_metrics \
    --results-dir "$DIR2026" --out-dir "$DIR2026/figures" > ~/plot_2026.log 2>&1
python -m topological_persistence.plot_metrics \
    --results-dir "$DIR2024" --out-dir "$DIR2024/figures" > ~/plot_2024.log 2>&1
echo "Stage 3 done (exit $?)"

echo "===== ALL STAGES COMPLETE ====="
touch ~/OVERNIGHT_DONE
