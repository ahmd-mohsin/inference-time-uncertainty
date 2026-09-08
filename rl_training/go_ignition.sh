#!/usr/bin/env bash
# H6 SFT-IGNITION: does a tiny SFT "seed" unlock GRPO's OOD transfer? Usage: go_ignition.sh <SFT_STEPS> <GPU>
# Stage 1: SFT the base for N steps on the verified bank (places a little mass). Stage 2: continue with
# plain GRPO 150 steps (sharpen). Eval OOD separately. N=0 == pure GRPO (~0.30); large N == SFT->GRPO (~0.43).
# Prediction: a SHARP KNEE at small N — a few SFT steps flip RL from ~0 transfer to SFT-level transfer.
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim VLLM_GPU_MEM_UTIL=0.35
PY=/usr/bin/python3
N="${1:-5}"; GPU="${2:-0}"; MODEL=Qwen/Qwen2.5-3B
GU=/tmp/instance_storage/gu; LOGS=$GU/logs; BANK=$GU/h2_bank_q3b.jsonl
SEED_DIR=$GU/ign_sft${N}; OUT=$GU/ign_sft${N}_grpo150; MP=$((31400 + GPU * 13))
echo "[ignition N=$N GPU=$GPU MP=$MP] $(date)"
# Stage 1: SFT N steps from base (skip if N=0 -> pure GRPO from base)
if [ "$N" -gt 0 ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -m rl_training.sft_train --data "$BANK" --model "$MODEL" \
    --out "$SEED_DIR" --max-steps "$N" --save-steps "$N" --bsz 8 > "$LOGS/ign_sft${N}.log" 2>&1
  INIT="--init-adapter $SEED_DIR"
else
  INIT=""
fi
# Stage 2: GRPO 150 steps (from the SFT seed, or from base if N=0)
CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$MP $PY -m rl_training.train_grpo --model "$MODEL" $INIT \
  --dataset gsm8k --reward-mode math --no-novelty --vllm-mode colocate --num-generations 8 \
  --num-train-steps 150 --max-completion-length 1024 --output-dir "$OUT" > "$LOGS/ign_grpo${N}.log" 2>&1
echo "[ignition N=$N] DONE rc=$? -> $OUT $(date)"
