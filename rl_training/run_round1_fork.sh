#!/usr/bin/env bash
# CORE round-1 fork experiment on ONE node (8 GPUs): full-parameter GRPO, one arm.
#   arm=grpo      : plain GRPO (--no-novelty), the standard round-1
#   arm=floor     : GRPO + support-floor constraint (--support-ratchet), coverage-preserving round-1
# Server-mode vLLM on GPU0 (proven recipe), full-FT ZeRO-3 training on GPU1-7.
# Usage: bash run_round1_fork.sh <grpo|floor> [steps]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
PY=/usr/bin/python3
mkdir -p $HOME/logs

ARM="${1:?arm: grpo|floor}"
STEPS="${2:-400}"
MODEL="Qwen/Qwen2.5-Math-7B"
DATASET="olympiad_bench"
DIFF="$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json"
BANK="$HOME/inference-time-uncertainty/rl_training/runs/bank/oly_fragile_bank.jsonl"
RUN="$HOME/inference-time-uncertainty/rl_training/runs/r1_${ARM}"
mkdir -p "$RUN"
VLLM_GPUS="0"; TRAIN_GPUS="1,2,3,4,5,6,7"; NTRAIN=7

echo ">> [vLLM] server on GPU0"
CUDA_VISIBLE_DEVICES=$VLLM_GPUS HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$MODEL" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.85 --port 8000 \
  > $HOME/logs/r1_${ARM}_vllm.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo ">> vLLM UP"; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo ">> vLLM FAILED"; tail -20 $HOME/logs/r1_${ARM}_vllm.log; exit 1; }

# arm-specific flags
EXTRA=""
if [ "$ARM" = "floor" ]; then
  EXTRA="--support-ratchet --ratchet-bank $BANK --ratchet-alpha 0.5 --ratchet-mu 0.5"
fi

echo ">> [train] full-FT GRPO arm=$ARM steps=$STEPS $EXTRA"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes $NTRAIN \
  --main_process_ip 127.0.0.1 --main_process_port 29501 --rdzv_backend c10d \
  -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$MODEL" --dataset "$DATASET" --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations 4 --max-completion-length 3072 \
  --output-dir "$RUN" $EXTRA \
  > $HOME/logs/r1_${ARM}_train.log 2>&1
echo ">> [train] exit $?"
pkill -9 -f "trl.scripts.vllm_serve" 2>/dev/null || true
echo ">> DONE arm=$ARM -> $RUN"
