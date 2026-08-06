#!/usr/bin/env bash
# ROUND-2 continued RL: identical plain GRPO on a round-1 fork checkpoint.
#   fork=grpo  -> continue from r1_grpo/checkpoint-400   (standard round-1)
#   fork=floor -> continue from r1_floor/checkpoint-400  (coverage-preserved round-1)
# NO floor in round 2 (the floor only acted in round 1 as the ratchet). Round 2 tests whether the
# PRESERVED COVERAGE from round 1 raises the continued-RL ceiling.
# Server-mode vLLM GPU0 + full-FT ZeRO-3 GPU1-7. Usage: bash run_round2.sh <grpo|floor> [steps]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
PY=/usr/bin/python3
mkdir -p $HOME/logs

FORK="${1:?fork: grpo|floor}"
STEPS="${2:-400}"
CKPT="$HOME/inference-time-uncertainty/rl_training/runs/r1_${FORK}/checkpoint-400"
DIFF="$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json"
RUN="$HOME/inference-time-uncertainty/rl_training/runs/r2_from_${FORK}"
mkdir -p "$RUN"
VLLM_GPUS="0"; TRAIN_GPUS="1,2,3,4,5,6,7"; NTRAIN=7

echo ">> [vLLM] server on GPU0 from round-1 $FORK checkpoint"
CUDA_VISIBLE_DEVICES=$VLLM_GPUS HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$CKPT" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > $HOME/logs/r2_${FORK}_vllm.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo ">> vLLM UP"; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo ">> vLLM FAILED"; tail -20 $HOME/logs/r2_${FORK}_vllm.log; exit 1; }

echo ">> [train] round-2 plain GRPO from $FORK fork, $STEPS steps (no floor)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes $NTRAIN \
  --main_process_ip 127.0.0.1 --main_process_port 29501 --rdzv_backend c10d \
  -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$CKPT" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations 4 --max-completion-length 2560 \
  --output-dir "$RUN" \
  > $HOME/logs/r2_${FORK}_train.log 2>&1
echo ">> [train] exit $?"
pkill -9 -f "trl.scripts.vllm_serve" 2>/dev/null || true
echo ">> DONE round2 fork=$FORK -> $RUN"
