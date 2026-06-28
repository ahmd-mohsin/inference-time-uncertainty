#!/usr/bin/env bash
# rl_reset_launch.sh — RUNS ON A NODE. Hard-resets the GPUs (kills any train/vLLM/EngineCore
# zombie holding memory), waits until GPU0 is free, then launches one arm DETACHED via setsid
# so it survives the parent SSH session dropping (shared-namespace pods drop SSH on pkill).
#
# Usage: bash scripts/rl_reset_launch.sh <arm> <model> <dataset> [difficulty_json]
set -uo pipefail
ARM="${1:?arm}"; MODEL="${2:-Qwen/Qwen3-8B}"; DATASET="${3:-aime_all}"; DIFF="${4:-}"
cd ~/inference-time-uncertainty 2>/dev/null || cd ~/inference-time-uncertainity

echo ">> resetting node for arm=$ARM"
pkill -9 -f rl_experiment 2>/dev/null
pkill -9 -f train_grpo   2>/dev/null
pkill -9 -f accelerate   2>/dev/null
pkill -9 -f torch.distributed 2>/dev/null
pkill -9 -f vllm 2>/dev/null
pkill -9 -f EngineCore 2>/dev/null
pkill -9 -f smoke_zero2 2>/dev/null
# kill anything still holding a GPU by pid (EngineCore names are missed by pkill -f vllm)
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null
done

# wait up to ~2min for GPU0 to drop below 2GB
for _ in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${used:-99999}" -lt 2000 ] 2>/dev/null && break
  sleep 3
done
echo ">> GPU0 now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"

# launch the arm detached; logs to ~/logs/rl_<arm>.log
mkdir -p ~/logs
setsid bash scripts/rl_experiment.sh "$ARM" "$MODEL" "$DATASET" "$DIFF" \
  > ~/logs/rl_${ARM}.log 2>&1 < /dev/null &
echo ">> launched arm=$ARM pid=$! (detached, log ~/logs/rl_${ARM}.log)"
