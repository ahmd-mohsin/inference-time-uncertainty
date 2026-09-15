#!/bin/bash
# Cluster A: COMP placeholder rows — Phi-3.5 (4th family) + RVP-from-base ablation (Coder-1.5B/3B).
cd $HOME/inference-time-uncertainty
echo "[matrix_A] start $(date -u)"
# 4th family: Phi-3.5-mini on CompDAG (mid 7->9)
env BASE=microsoft/Phi-3.5-mini-instruct FAM=phi TRD=7 OODD=9 bash rl_training/rvp_scripts/rvp_family.sh
# RVP-from-base ablation (skip RFT coverage stage): does coverage-first matter?
env BASE=Qwen/Qwen2.5-Coder-1.5B-Instruct FAM=ab_c15 TRD=7 OODD=9 SKIP_RFT=1 bash rl_training/rvp_scripts/rvp_family.sh
env BASE=Qwen/Qwen2.5-Coder-3B-Instruct   FAM=ab_c3  TRD=7 OODD=9 SKIP_RFT=1 bash rl_training/rvp_scripts/rvp_family.sh
echo "[matrix_A] done $(date -u)"
