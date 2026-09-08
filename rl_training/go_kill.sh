#!/usr/bin/env bash
# CSPO experiment — KILL phase. Plain GRPO from BASE on the fragile band, checkpointing every 10 steps
# and KEEPING the whole trajectory, so we can offline-certify each checkpoint's SAMPLED capability
# recoverability p_hat(q) and locate the Gp<1 extinction boundary (Gp sweep) + pick killed checkpoints
# for the kill->rescue branch. Death-proofs each checkpoint to HF (keeps local too).
# Usage: bash go_kill.sh [steps]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
STEPS="${1:-150}"; SEED="${2:-0}"; G="${3:-8}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
BASE=$NV/base_qwen25math7b
RUN=$NV/e8_kill_g${G}_s${SEED}
DIFF=$NV/difficulty_olympiad_7b.json
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"

# --- self-setup (safe on any fresh node) -------------------------------------------------
getent hosts "$(hostname)" >/dev/null 2>&1 || sudo bash -c "echo \"127.0.0.1 $(hostname)\" >> /etc/hosts" || true
$PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
# prewarm dataset (offline train needs it cached)
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 $PY -c "from datasets import load_dataset as L; L('math-ai/olympiadbench')" >/dev/null 2>&1 || true
# base model (sharded -> reliable snapshot)
if [ ! -f "$BASE/config.json" ]; then
  echo ">> downloading base Qwen2.5-Math-7B"
  HF_HUB_OFFLINE=0 $PY - "$BASE" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Math-7B", local_dir=sys.argv[1],
  allow_patterns=["*.json","*.txt","*.model","tokenizer*","vocab*","merges*","model-*.safetensors","model.safetensors"])
print("base ready")
PY
fi
[ -f "$BASE/config.json" ] || { echo "FATAL: base missing"; exit 1; }
[ -f "$DIFF" ] || { echo "FATAL: difficulty $DIFF missing (scp it)"; exit 1; }

# death-proof: push every checkpoint to HF, KEEP local (no --reap) so we can certify each offline
setsid nohup $PY rl_training/hf_ckpt_daemon.py watch --run-dir "$RUN" --repo muahmed7338/cspo-kill-g${G}-s${SEED}-7b --every 60 \
  > "$LOGS/kill_g${G}_s${SEED}_daemon.log" 2>&1 & disown

# vLLM server GPU0
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/kill_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -15 "$LOGS/kill_vllm.log"; exit 1; }

# ZeRO-3 full-FT plain GRPO on GPU1-7, save every 10 steps, keep all
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$BASE" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations "$G" --max-completion-length 2560 --output-dir "$RUN" \
  --save-steps 10 --save-total-limit 100 \
  > "$LOGS/kill_g${G}_s${SEED}_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
echo "kill done rc=$RC steps=$STEPS G=$G seed=$SEED -> $RUN (checkpoints every 10 for the Gp sweep)"
