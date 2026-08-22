#!/usr/bin/env bash
# E6 baseline arm: full-FT GRPO from base with a baseline-specific modifier, SAME data/steps as the
# E8 plain/floor arms (fair 3+-way comparison). Usage: bash go_e6_baseline.sh <name> <steps> "<extra flags>"
#   globalkl : "--beta 0.04"                    (standard KL-to-base coverage baseline)
#   ucpo     : "--ucpo ..."  (needs impl)       (uniformity among correct rollouts)
#   pba      : "--pba ..."   (needs impl)       (per-problem base anchoring)
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
PY=/usr/bin/python3
NAME="${1:?name}"; STEPS="${2:-150}"; EXTRA="${3:-}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
BASE=$NV/base_qwen25math7b
RUN=$NV/e6_${NAME}
DIFF=$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"
[ -f "$BASE/config.json" ] || { echo "FATAL: base model missing"; exit 1; }

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/e6_${NAME}_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -15 "$LOGS/e6_${NAME}_vllm.log"; exit 1; }

# UCPO baseline = correctness-gated intra-group diversity reward (enable novelty); all others plain (--no-novelty)
NOV="--no-novelty"; [ "$NAME" = "ucpo" ] && NOV="--novelty-lambda 0.5"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora $NOV \
  --model "$BASE" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations 8 --max-completion-length 2560 --output-dir "$RUN" $EXTRA \
  > "$LOGS/e6_${NAME}_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
echo "e6 $NAME done rc=$RC steps=$STEPS -> $RUN"
