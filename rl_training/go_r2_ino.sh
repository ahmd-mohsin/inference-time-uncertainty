#!/usr/bin/env bash
# ROUND-2 continued-RL for the Instruct x Omni-MATH cell (the top-Delta diversity cell, gap +9.51).
# Fork design (identical to paper R4): round-1 forks already exist as the diversity arms —
#   plain arm  = ctrl fork  (FORK=grpo),  expSR arm = floor fork (FORK=floor).
# This runs the IDENTICAL unconstrained round-2 GRPO from each fork, then we eval pass@k.
# Reuses go_r2_nvme.sh's hard-won discipline: ALL big writes on nvme (never home -> pod eviction ->
# cluster death), hf daemon --reap for death-proof + flat disk, resume-aware from HF.
# Usage: bash go_r2_ino.sh <grpo|floor>
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
FORK="${1:?grpo|floor}"
case "$FORK" in grpo) SRC=plain;; floor) SRC=expSR;; *) echo "FORK must be grpo|floor"; exit 1;; esac

NV=/tmp/instance_storage/gu
sudo mkdir -p "$NV" 2>/dev/null; sudo chown greenland-user:greenland-users "$NV" 2>/dev/null
mkdir -p "$NV/logs"; LOGS="$NV/logs"
CELL=$NV/cell_qwen25_7b_instruct_omni_math_hard
BASE=$CELL/$SRC                               # round-1 fork = round-2 base (flat full model on nvme)
R1REPO=muahmed7338/cov-r1-ino-${FORK}-7b
RUN=$NV/r2_ino_from_${FORK}
R2REPO=muahmed7338/cov-r2-ino-${FORK}-7b
DIFF=$CELL/difficulty.json
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"

# --- ensure the round-1 fork base is present locally; else pull it from HF (fast node uplink) ---
if [ ! -f "$BASE/config.json" ]; then
  echo ">> $FORK base not local; downloading $R1REPO -> $BASE"
  mkdir -p "$BASE"
  $PY -c "import os; from huggingface_hub import snapshot_download; snapshot_download('$R1REPO', repo_type='model', local_dir='$BASE', token=os.environ['HF_TOKEN'])" 2>&1 | tail -2
fi
[ -f "$BASE/config.json" ] || { echo "FATAL: $BASE missing config.json"; exit 1; }
[ -f "$DIFF" ] || { echo "FATAL: $DIFF (omni difficulty) missing"; exit 1; }

# --- resume: pull latest RESUMABLE r2 checkpoint from HF into RUN ---
$PY rl_training/hf_ckpt_daemon.py resume --run-dir "$RUN" --repo "$R2REPO" 2>&1 | tail -2
RESUME=""; LAST=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)
if [ -n "$LAST" ] && [ -f "$RUN/checkpoint-$LAST/trainer_state.json" ]; then
  RESUME="--resume-from $RUN/checkpoint-$LAST"; echo ">> r2 $FORK resuming from checkpoint-$LAST"
else
  echo ">> r2 $FORK starting fresh"
fi

# --- prewarm Omni-MATH once (online) so the 7 accelerate workers read a hot cache offline ---
echo ">> [prewarm] caching Omni-MATH"
$PY -c "import os;from datasets import load_dataset;load_dataset('KbsdJames/Omni-MATH',token=os.environ.get('HF_TOKEN'));print('PREWARM_OK')" 2>&1 | tail -1

# --- hf push daemon WITH --reap (keeps nvme flat, death-proofs each ckpt) ---
setsid nohup $PY rl_training/hf_ckpt_daemon.py watch --run-dir "$RUN" --repo "$R2REPO" --every 45 --reap \
  > "$LOGS/r2_ino_${FORK}_hfpush.log" 2>&1 &

# --- server-mode vLLM on GPU0 ---
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/r2_ino_${FORK}_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -15 "$LOGS/r2_ino_${FORK}_vllm.log"; exit 1; }

# --- identical round-2 GRPO (plain, no floor) on GPU1-7, ZeRO-3, 100 steps ---
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$BASE" --dataset omni_math_hard --difficulty-json "$DIFF" \
  --num-train-steps 100 --num-generations 4 --max-completion-length 2560 --output-dir "$RUN" $RESUME \
  > "$LOGS/r2_ino_${FORK}_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
echo "r2 ino $FORK done rc=$RC"
