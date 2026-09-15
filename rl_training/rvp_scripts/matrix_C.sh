#!/bin/bash
# Cluster C: size-contrast hard math (Math-1.5B fits single-GPU DPO) + deepseek-math-7B (family) + mech-interp.
cd $HOME/inference-time-uncertainty
CLUSTER=C setsid bash rl_training/rvp_scripts/dispatch_workers.sh >$HOME/gu/logs/dispatch_C.log 2>&1 &
echo "[matrix_C] start $(date -u)"
# size contrast: Math-1.5B on MATH-500 (single-GPU DPO fits; no sharding needed)
env BASE=Qwen/Qwen2.5-Math-1.5B-Instruct EVAL=math500 TAG=mhC_q15m_m500 DPO_BSZ=1 DPO_MAXLEN=640 bash rl_training/rvp_scripts/math_hard.sh
# 2nd family bigger-model, sharded: deepseek-math-7b on MATH-500
export ACC_CFG=rl_training/accelerate_zero3_offload.yaml DPO_MAXLEN=512 NSEED=3
env BASE=deepseek-ai/deepseek-math-7b-instruct EVAL=math500 TAG=mhC_dsm_m500 bash rl_training/rvp_scripts/math_hard_shard.sh
# mechanistic-interpretability: full base->GRPO->RFT->RVP margin trajectory (GSM8K cell)
env BASE=Qwen/Qwen2.5-1.5B-Instruct bash rl_training/rvp_scripts/mech_full.sh 2>/dev/null || echo "mech_full skipped"
echo "[matrix_C] done $(date -u)"
