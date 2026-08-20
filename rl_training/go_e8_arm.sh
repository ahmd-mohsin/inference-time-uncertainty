#!/usr/bin/env bash
# E8-interventional: short full-FT GRPO from BASE, one arm. arm=plain (control) | floor (GRPO+support
# floor). Same data/steps for both -> the ONLY difference is the off-policy floor. Measure fragile-mode
# mass before(base)/after via teacher-forced logprob on the bank. Usage: bash go_e8_arm.sh <plain|floor> [steps]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
PY=/usr/bin/python3
ARM="${1:?plain|floor}"; STEPS="${2:-25}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
BASE=$NV/base_qwen25math7b
RUN=$NV/e8_${ARM}
DIFF=$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json
BANK=$NV/ratchet_bank_e8.jsonl
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"
[ -f "$BASE/config.json" ] || { echo "FATAL: base model missing"; exit 1; }

EXTRA=""
[ "$ARM" = "floor" ] && EXTRA="--support-ratchet --ratchet-bank $BANK --ratchet-alpha 0.5 --ratchet-mu 0.5"

# vLLM server on GPU0 (base model)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/e8_${ARM}_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -15 "$LOGS/e8_${ARM}_vllm.log"; exit 1; }

# ZeRO-3 full-FT GRPO on GPU1-7 (prewarmed dataset -> offline ok)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$BASE" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations 8 --max-completion-length 2560 --output-dir "$RUN" $EXTRA \
  > "$LOGS/e8_${ARM}_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
echo "e8 $ARM done rc=$RC steps=$STEPS -> $RUN"
