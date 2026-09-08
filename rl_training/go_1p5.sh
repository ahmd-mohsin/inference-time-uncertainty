#!/usr/bin/env bash
# ONE 1.5B GRPO job on ONE GPU: colocate vLLM in-process (no server, no ZeRO-3, no accelerate/c10d
# rendezvous). This is the reliable parallel primitive — run 8 of these across the 8 GPUs independently.
# Usage: bash go_1p5.sh <gpu_id> <src_model> <tag> <dataset> <steps> [extra args...]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASHINFER HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
GPU="$1"; SRC="$2"; TAG="$3"; DATASET="$4"; STEPS="$5"; shift 5; EXTRA="$*"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
RUN=$NV/j_$TAG; mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"

# single-process, single-GPU, colocate vLLM. HF_HUB_OFFLINE=0 so dataset loads online (no prewarm).
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER \
  $PY -m rl_training.train_grpo --no-lora --no-novelty --vllm-mode colocate \
  --model "$SRC" --dataset "$DATASET" --num-train-steps "$STEPS" \
  --num-generations 8 --max-completion-length 2048 --output-dir "$RUN" \
  --save-steps 10 --save-total-limit 100 $EXTRA \
  > "$LOGS/j_${TAG}.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"
echo "j $TAG (gpu $GPU) done rc=$RC -> $RUN"
