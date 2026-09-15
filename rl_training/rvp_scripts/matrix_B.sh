#!/bin/bash
# Cluster B: bigger-model HARD MATH (sharded ZeRO-3 full-param DPO) — MATH-500, then AMC.
cd $HOME/inference-time-uncertainty
CLUSTER=B setsid bash rl_training/rvp_scripts/dispatch_workers.sh >$HOME/gu/logs/dispatch_B.log 2>&1 &
echo "[matrix_B] start $(date -u)"
export ACC_CFG=rl_training/accelerate_zero3_offload.yaml DPO_MAXLEN=512 NSEED=3
env BASE=Qwen/Qwen2.5-Math-7B EVAL=math500 TAG=mhB_q7m_m500 bash rl_training/rvp_scripts/math_hard_shard.sh
env BASE=Qwen/Qwen2.5-Math-7B EVAL=amc     TAG=mhB_q7m_amc  bash rl_training/rvp_scripts/math_hard_shard.sh
echo "[matrix_B] done $(date -u)"
