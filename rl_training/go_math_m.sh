#!/usr/bin/env bash
# ARM A (baseline) — plain GRPO on GSM8K, parameterized by model. Usage: go_math_m.sh <seed> <MODEL> <mtag>
# vLLM server on GPU0 + accelerate ZeRO-2 GRPO on GPU1-7 (server mode). One node (8 GPU) per model.
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
SEED="${1:-0}"; MODEL="${2:-Qwen/Qwen2.5-3B}"; MTAG="${3:-q3b}"
GU=/tmp/instance_storage/gu; LOGS=$GU/logs; OUT=$GU/grpo_${MTAG}_s$SEED; PORT=${PORT:-8000}
echo "[go_math_m $MTAG s$SEED] MODEL=$MODEL vLLM-serve GPU0 :$PORT $(date)"
CUDA_VISIBLE_DEVICES=0 nohup $PY -m trl.scripts.vllm_serve --model "$MODEL" --port $PORT --gpu_memory_utilization 0.9 \
  > "$LOGS/gm_vllm_${MTAG}_s$SEED.log" 2>&1 &
VLLM_PID=$!
# wait for server up (max ~8 min for big-model load)
for t in $(seq 1 96); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "[go_math_m] vLLM up after ${t}x5s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "[go_math_m] vLLM DIED — see gm_vllm log"; exit 1; }
  sleep 5
done
echo "[go_math_m $MTAG s$SEED] launch GRPO GPU1-7 $(date)"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero2.yaml --num_processes 7 \
  -m rl_training.train_grpo --model "$MODEL" --dataset gsm8k --reward-mode math --no-novelty \
  --vllm-mode server --num-generations 8 --num-train-steps 400 --max-completion-length 1024 \
  --save-steps 100 --save-total-limit 8 --output-dir "$OUT" --seed $SEED \
  > "$LOGS/gm_train_${MTAG}_s$SEED.log" 2>&1
echo "[go_math_m $MTAG s$SEED] DONE rc=$? -> $OUT $(date)"; kill $VLLM_PID 2>/dev/null
