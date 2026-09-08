#!/usr/bin/env bash
# CSPO kill->RESCUE. Branch from a KILLED checkpoint (capabilities driven below Gp~1) and run one arm:
#   plain  = continued on-policy GRPO (control)            -> should NOT revive extinct capabilities
#   rescue = GRPO + off-policy reservoir push (forward-KL NLL on verified base-correct traces)
#            -> should revive capabilities on-policy can no longer reach (Prop 3/4)
# Identical steps/data; the only difference is the off-policy rescue term. Certify p_hat before/after.
# Usage: bash go_rescue.sh <killed_ckpt_dir> <plain|rescue> [steps] [mu]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
CKPT="$1"; ARM="$2"; STEPS="${3:-60}"; MU="${4:-0.05}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
DIFF=$NV/difficulty_olympiad_7b.json
BANK=$NV/ratchet_bank_e8.jsonl      # verified base-correct reservoir B_q
RUN=$NV/rescue_${ARM}
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"
# self-setup
getent hosts "$(hostname)" >/dev/null 2>&1 || sudo bash -c "echo \"127.0.0.1 $(hostname)\" >> /etc/hosts" || true
$PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 $PY -c "from datasets import load_dataset as L; L('math-ai/olympiadbench')" >/dev/null 2>&1 || true
[ -f "$CKPT/config.json" ] || { echo "FATAL: killed ckpt $CKPT missing"; exit 1; }
[ -f "$DIFF" ] || { echo "FATAL: diff missing"; exit 1; }

EXTRA=""
[ "$ARM" = "rescue" ] && { [ -f "$BANK" ] || { echo "FATAL: rescue needs bank $BANK"; exit 1; }; \
  EXTRA="--dph-forward-kl --ratchet-bank $BANK --ratchet-mu $MU"; }

# vLLM server on the KILLED checkpoint (GPU0)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$CKPT" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/rescue_${ARM}_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done

# ZeRO-3 GRPO from the killed checkpoint, save every 10 steps (to certify recovery trajectory)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$CKPT" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations 8 --max-completion-length 2560 --output-dir "$RUN" \
  --save-steps 10 --save-total-limit 100 $EXTRA \
  > "$LOGS/rescue_${ARM}_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null
echo "rescue $ARM done rc=$RC -> $RUN"
