#!/usr/bin/env bash
# Self-repair GRPO via the PROVEN train_grpo.py (correct reward signature). vLLM server GPU0 +
# accelerate ZeRO-2 GRPO GPU1-7. Usage: bash go_repair_grpo.sh <model_dir> <repair_jsonl> <out> [steps]
set -uo pipefail; export HOME=/home/greenland-user; cd ${REPO:-/tmp/instance_storage/gu/repo}
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=FLASHINFER HF_HUB_DISABLE_XET=1
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3; MODEL="$1"; DATA="$2"; OUT="$3"; STEPS="${4:-200}"
# LoRA by default (reliable, fits ZeRO-2); set FULLFT=1 for full fine-tuning (needs more memory).
LORA_FLAG=""; [ "${FULLFT:-0}" = "1" ] && LORA_FLAG="--no-lora"
# reward variant: binary | fraction | residual | cert_residual (default). Passed to code_repair_reward.
RVAR="${RVAR:-cert_residual}"
SEED="${SEED:-42}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS" "$OUT"
getent hosts "$(hostname)" >/dev/null 2>&1 || sudo bash -c "echo \"127.0.0.1 $(hostname)\" >> /etc/hosts" || true
[ -f "$MODEL/config.json" ] || { echo FATAL no model; exit 1; }; [ -f "$DATA" ] || { echo FATAL no data; exit 1; }
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 setsid nohup $PY -m trl.scripts.vllm_serve --model "$MODEL" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.85 --port 8000 > "$LOGS/rgrpo_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 REPAIR_REWARD_VARIANT="$RVAR" $PY -m accelerate.commands.launch --config_file rl_training/accelerate_zero2.yaml --num_processes 7 --main_process_ip 127.0.0.1 --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo $LORA_FLAG --reward-mode code --no-novelty --vllm-mode server --model "$MODEL" --dataset "repair:$DATA" --num-train-steps "$STEPS" --seed "$SEED" --num-generations 8 --max-completion-length 1024 --output-dir "$OUT" --save-steps 40 --save-total-limit 3 > "$LOGS/rgrpo_train.log" 2>&1
RC=$?; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null; [ "$RC" = 0 ] && touch "$OUT/RGRPO_DONE"; echo "repair-grpo(train_grpo) rc=$RC -> $OUT"
