#!/usr/bin/env bash
# Reasoning-Optionality ADAPTATION-SHOCK. Take a source checkpoint (different mode-collapse level,
# ~matched current accuracy) and RL-ADAPT it on a SHIFT distribution; the per-step training reward IS
# the adaptation curve. Thesis: more mode-collapsed sources adapt SLOWER (reward rises slower) even
# though pre-shift pass@k is similar. Compare reward-vs-step across sources spanning the collapse
# spectrum {base, DPH-F-preserved, G4, G8, G14 kills}.
# Usage: bash go_adapt.sh <src_model_dir> <tag> <dataset> [steps]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
SRC="$1"; TAG="$2"; DATASET="$3"; STEPS="${4:-80}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
RUN=$NV/adapt_${TAG}
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"
# self-setup (no inline dataset prewarm — it hangs; train loads datasets ONLINE instead)
getent hosts "$(hostname)" >/dev/null 2>&1 || sudo bash -c "echo \"127.0.0.1 $(hostname)\" >> /etc/hosts" || true
[ -f "$SRC/config.json" ] || { echo "FATAL: src $SRC missing"; exit 1; }

# self-clear GPUs (stranded compute-apps from prior certs/kills block the new vLLM alloc)
pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 6

# vLLM server on the SOURCE (GPU0)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$SRC" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/adapt_${TAG}_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done

# ZeRO-3 GRPO adaptation on the SHIFT dataset (no difficulty-json -> full/self-filtered band).
# logging_steps=10 -> reward logged; per-step reward = adaptation curve. save every 20 for optional certs.
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$SRC" --dataset "$DATASET" \
  --num-train-steps "$STEPS" --num-generations 8 --max-completion-length 2560 --output-dir "$RUN" \
  --save-steps 20 --save-total-limit 100 \
  > "$LOGS/adapt_${TAG}_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
echo "adapt $TAG done rc=$RC -> $RUN (reward-vs-step in adapt_${TAG}_train.log)"
