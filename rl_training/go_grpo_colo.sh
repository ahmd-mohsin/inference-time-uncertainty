#!/usr/bin/env bash
# ARM A (baseline) via COLOCATE vLLM — single-GPU, no server/ZeRO rendezvous (robust). Usage: go_grpo_colo.sh <seed> <MODEL> <mtag> <GPU>
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim VLLM_ALLOW_INSECURE_SERIALIZATION=1
PY=/usr/bin/python3
SEED="${1:-0}"; MODEL="${2:-Qwen/Qwen2.5-3B}"; MTAG="${3:-q3b}"; GPU="${4:-0}"
OUT=/tmp/instance_storage/gu/grpo_${MTAG}_s$SEED
echo "[go_grpo_colo $MTAG s$SEED] MODEL=$MODEL colocate GPU$GPU $(date)"
CUDA_VISIBLE_DEVICES=$GPU VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.35} $PY -m rl_training.train_grpo --model "$MODEL" \
  --dataset gsm8k --reward-mode math --no-novelty --vllm-mode colocate --num-generations 8 --num-train-steps 400 \
  --max-completion-length 1024 --save-steps 100 --save-total-limit 6 --output-dir "$OUT" --seed $SEED
echo "[go_grpo_colo $MTAG s$SEED] DONE rc=$? -> $OUT $(date)"
