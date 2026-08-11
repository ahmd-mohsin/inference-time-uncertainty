#!/usr/bin/env bash
# ROUND-2 resume-aware direct launcher (on-worker). Same as go_r2_direct.sh but FIRST pulls the
# latest round-2 checkpoint from HF into RUN and passes --resume-from, so a node death costs
# <=1 checkpoint-interval (10 steps) instead of restarting at 0. Uses flat local r1_${FORK}_ckpt
# as the base model for vLLM. Usage: bash go_r2_resume.sh <grpo|floor>
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
FORK="$1"
BASE=$HOME/inference-time-uncertainty/rl_training/runs/r1_${FORK}_ckpt
RUN=$HOME/inference-time-uncertainty/rl_training/runs/r2_from_${FORK}
R2REPO=muahmed7338/cov-r2-from-${FORK}-7b
DIFF=$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"
[ -f "$BASE/config.json" ] || { echo "FATAL: $BASE not flat"; exit 1; }

# --- resume: pull latest r2 checkpoint from HF (only if not already local) ---
$PY rl_training/hf_ckpt_daemon.py resume --run-dir "$RUN" --repo "$R2REPO" 2>&1 | tail -2
RESUME=""; LAST=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)
if [ -n "$LAST" ] && [ -f "$RUN/checkpoint-$LAST/trainer_state.json" ]; then
  RESUME="--resume-from $RUN/checkpoint-$LAST"; echo ">> r2 $FORK resuming from checkpoint-$LAST"
else
  echo ">> r2 $FORK no resumable checkpoint; starting fresh"
fi

setsid nohup $PY rl_training/hf_ckpt_daemon.py watch --run-dir "$RUN" --repo "$R2REPO" --every 45 > $HOME/logs/r2_${FORK}_hfpush.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 > $HOME/logs/r2_${FORK}_vllm.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -15 $HOME/logs/r2_${FORK}_vllm.log; exit 1; }
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty --model "$BASE" --dataset olympiad_bench --difficulty-json "$DIFF" --num-train-steps 100 --num-generations 4 --max-completion-length 2560 --output-dir "$RUN" $RESUME > $HOME/logs/r2_${FORK}_train.log 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null; echo "r2$FORK done rc=$RC"
