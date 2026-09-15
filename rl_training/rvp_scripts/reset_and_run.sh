#!/bin/bash
# Detached GPU reset + hard-math flywheel launch. Run under `setsid nohup` so that
# killing GPU procs (shared-PID namespace) does NOT cut the ssh shell that started it.
# env: BASE EVAL TAG [BANK DPO_BSZ]
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
mkdir -p $HOME/gu/logs
# 1) hard-clear all rl/vLLM GPU processes, loop until GPUs are truly free
for i in 1 2 3 4 5 6; do
  pkill -9 -f 'python3 -m rl_training' 2>/dev/null      # NOT '-f rl_training' (that matches this script's own path -> self-kill)
  pkill -9 -f 'accelerate.commands.launch' 2>/dev/null
  pkill -9 -f 'VLLM' 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
  sleep 6
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  echo "[reset] attempt $i gpuprocs=$n" >> $HOME/gu/logs/${TAG}_reset.log
  [ "$n" -eq 0 ] && break
done
# 2) launch the (resumable) flywheel on clean GPUs
FLYWHEEL="${FLYWHEEL:-math_hard.sh}"
setsid nohup env BASE="$BASE" BANK="${BANK:-math_full}" EVAL="$EVAL" TAG="$TAG" DPO_BSZ="${DPO_BSZ:-1}" \
  DPO_MAXLEN="${DPO_MAXLEN:-768}" MAXTOK="${MAXTOK:-1024}" MAXLEN="${MAXLEN:-3072}" NSEED="${NSEED:-3}" ACC_CFG="${ACC_CFG:-rl_training/accelerate_zero3.yaml}" \
  GEN_GPU_MEM="${GEN_GPU_MEM:-0.55}" EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.45}" VLLM_TP="${VLLM_TP:-1}" CONSOLIDATE="${CONSOLIDATE:-0}" \
  bash rl_training/rvp_scripts/$FLYWHEEL > $HOME/gu/logs/${TAG}_run.log 2>&1 &
echo "[reset] launched $TAG pid $!" >> $HOME/gu/logs/${TAG}_reset.log
