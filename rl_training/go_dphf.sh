#!/usr/bin/env bash
# DPH-F baseline (forward-KL rehearsal) — interventional arm from BASE, identical recipe to the floor
# arm (go_e8_arm.sh) EXCEPT the penalty is mass-covering NLL on the bank (--dph-forward-kl) instead of
# the one-sided floor. Isolates penalty SHAPE. 150 steps, full-FT, vLLM-server + ZeRO-3.
# Usage: bash go_dphf.sh [steps] [mu]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
STEPS="${1:-150}"; MU="${2:-0.02}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
BASE=$NV/base_qwen25math7b
RUN=$NV/e8_dphf
DIFF=$NV/difficulty_olympiad_7b.json
BANK=$NV/ratchet_bank_e8.jsonl
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"

# 0) base model (sharded -> snapshot downloads reliably)
if [ ! -f "$BASE/config.json" ]; then
  echo ">> downloading Qwen2.5-Math-7B base"
  $PY - "$BASE" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Math-7B", local_dir=sys.argv[1],
  allow_patterns=["*.json","*.txt","*.model","tokenizer*","vocab*","merges*","model-*.safetensors","model.safetensors"])
print("base ready")
PY
fi
[ -f "$BASE/config.json" ] || { echo "FATAL: base missing"; exit 1; }
[ -f "$BANK" ] || { echo "FATAL: bank $BANK missing (scp it)"; exit 1; }
[ -f "$DIFF" ] || { echo "FATAL: difficulty $DIFF missing (scp it)"; exit 1; }

# 1) vLLM server on GPU0
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/dphf_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -15 "$LOGS/dphf_vllm.log"; exit 1; }

# 2) ZeRO-3 full-FT GRPO on GPU1-7 with DPH-F forward-KL rehearsal
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$BASE" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations 8 --max-completion-length 2560 --output-dir "$RUN" \
  --dph-forward-kl --ratchet-bank "$BANK" --ratchet-mu "$MU" \
  > "$LOGS/dphf_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
echo "dphf done rc=$RC steps=$STEPS mu=$MU -> $RUN"
