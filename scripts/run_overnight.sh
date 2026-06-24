#!/usr/bin/env bash
#
# run_overnight.sh — Unattended: generate AIME2026 with Qwen3-8B (single GPU), validate,
# re-analyze with the new spectral + answer-distribution signals, plot.
#
# Methodology change (see topological_persistence/RESULTS.md): the H1 persistent-homology
# verdict was non-predictive (AUC 0.33). The primary signal is now answer-distribution
# diversity + spectral effective rank (detect_ceiling_v2). Qwen3-8B is used because it has
# real headroom on AIME, making `actually_scales` measurable (Qwen3-32B was saturated).
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

DIR2026="data/topological_outputs_aime2026_qwen8b"
MODEL="Qwen/Qwen3-8B"
NVAL=64

echo "===== STAGE 1: Generate AIME 2026 (Qwen3-8B, robust two-phase) ====="
python -m topological_persistence.run_robust \
    --model "$MODEL" --dataset aime_2026 --n-problems 30 \
    --n-chains 8 --output-dir "$DIR2026" > ~/gen_2026_8b.log 2>&1
echo "Stage 1 done (exit $?)"

echo "===== STAGE 2: Validate AIME 2026 (64 chains/problem) ====="
python -m topological_persistence.analyze_results \
    --results-dir "$DIR2026" --validate --n-validation "$NVAL" \
    --dataset aime_2026 --model "$MODEL" > ~/validate_2026_8b.log 2>&1
echo "Stage 2 done (exit $?)"

echo "===== STAGE 3: Spectral re-analysis (signal vs ground truth) ====="
python -m topological_persistence.spectral_reanalysis \
    --data-dir "$DIR2026" > ~/spectral_2026_8b.log 2>&1
echo "Stage 3 done (exit $?)"

echo "===== STAGE 4: Plots ====="
python -m topological_persistence.plot_metrics \
    --results-dir "$DIR2026" --out-dir "$DIR2026/figures" > ~/plot_2026_8b.log 2>&1
echo "Stage 4 done (exit $?)"

echo "===== ALL STAGES COMPLETE ====="
touch ~/OVERNIGHT_DONE_8B
