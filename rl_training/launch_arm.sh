#!/usr/bin/env bash
# Relaunchable RL arm. Usage: launch_arm.sh <RVAR> <out_suffix> [steps] [seed]
# RVAR in {binary,fraction,residual,cert_residual}. Repo+model+data on nvme. Clears GPUs first (free node only).
set -uo pipefail
export HOME=/home/greenland-user REPO=/tmp/instance_storage/gu/repo
NV=/tmp/instance_storage/gu
RVAR="${1:-cert_residual}"; SUF="${2:-a}"; STEPS="${3:-150}"; SEED="${4:-42}"
export RVAR SEED
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
sleep 3
cd $REPO
RVAR="$RVAR" SEED="$SEED" bash rl_training/go_repair_grpo.sh $NV/m_qc $NV/repair_data/repair_rep_a.jsonl $NV/repair_grpo_${RVAR}_${SUF} $STEPS
