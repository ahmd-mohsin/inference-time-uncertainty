#!/usr/bin/env bash
# ROUND-2 resume-aware launcher — ALL big writes on NVME (/tmp/instance_storage), NOT home.
# WHY: home (/home/greenland-user) lives on the pod overlay = ephemeral-storage, limit 500Gi.
# Full-FT 7B ckpts (~95GB each, every 10 steps) filled it in ~1.5h -> kubelet evicted the pod ->
# gang schedule (minAvailable:3) killed the WHOLE cluster. The nvme mount is 6.9TB and is NOT
# counted against the ephemeral limit. So: base model, HF cache, and RUN dir all go on nvme;
# the hf daemon runs with --reap (deletes local ckpts after HF push, keeps newest) so disk stays flat.
# Usage: bash go_r2_nvme.sh <grpo|floor>
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
FORK="$1"

# --- NVME scratch (6.9TB, not ephemeral-metered) ---
NV=/tmp/instance_storage/gu
sudo mkdir -p "$NV" 2>/dev/null; sudo chown greenland-user:greenland-users "$NV" 2>/dev/null
# NOTE: do NOT override HF_HOME/HF_HUB_CACHE. Everything big we write is the run dir (RUN, on nvme,
# passed explicitly) and the base model (BASE, on nvme, passed via --model). The HF caches in
# ~/.cache stay tiny (dataset is 8MB; no model is pulled through the hub — vLLM/trainer read the
# explicit local $BASE path). Redirecting HF_HUB_CACHE to empty nvme + HF_HUB_OFFLINE=1 made the
# olympiadbench load miss its cache -> ConnectionError(OfflineModeIsEnabled). Leaving caches at
# default keeps the offline dataset load working; disk stays flat because RUN (the only big writer)
# is on nvme and the daemon --reaps it.
mkdir -p "$NV/logs"
LOGS="$NV/logs"

BASE="$NV/r1_${FORK}_ckpt"                    # flat r1 fork base model (on nvme)
RUN="$NV/r2_from_${FORK}"                     # round-2 output dir (on nvme)
R2REPO=muahmed7338/cov-r2-from-${FORK}-7b
DIFF=$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json
mkdir -p "$RUN"; rm -f "$RUN/TRAIN_DONE"
[ -f "$BASE/config.json" ] || { echo "FATAL: $BASE not flat (run prep_r1_base_nvme first)"; exit 1; }

# --- resume: pull latest RESUMABLE r2 checkpoint from HF into RUN ---
$PY rl_training/hf_ckpt_daemon.py resume --run-dir "$RUN" --repo "$R2REPO" 2>&1 | tail -2
RESUME=""; LAST=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)
if [ -n "$LAST" ] && [ -f "$RUN/checkpoint-$LAST/trainer_state.json" ]; then
  RESUME="--resume-from $RUN/checkpoint-$LAST"; echo ">> r2 $FORK resuming from checkpoint-$LAST"
else
  echo ">> r2 $FORK no resumable checkpoint; starting fresh"
fi

# --- pre-warm the olympiadbench dataset ONCE (single process, online) so the 7 accelerate workers
# read a hot cache. 7 workers hitting the Hub simultaneously unauthenticated get rate-limited and
# `datasets` reports it as OfflineModeIsEnabled -> ConnectionError. One warm call fixes it. ---
echo ">> [prewarm] caching olympiad_bench dataset (single process, online)"
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 $PY -c "import sys,os; sys.path.insert(0,os.getcwd()); from src.data.dataset import get_inference_dataset; d=get_inference_dataset({'dataset':{'name':'olympiad_bench','split':'test','n_problems':-1,'seed':42}}); print('prewarm ok', len(d))" 2>&1 | tail -2

# --- hf push daemon WITH --reap (keeps nvme flat) ---
setsid nohup $PY rl_training/hf_ckpt_daemon.py watch --run-dir "$RUN" --repo "$R2REPO" --every 45 --reap \
  > "$LOGS/r2_${FORK}_hfpush.log" 2>&1 &

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > "$LOGS/r2_${FORK}_vllm.log" 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo VLLMUP; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -15 "$LOGS/r2_${FORK}_vllm.log"; exit 1; }

# OFFLINE for train — the prewarm above cached the dataset; 7 workers reading a HOT cache offline
# avoids both Hub rate-limiting (concurrent unauth requests -> OfflineModeIsEnabled) and concurrent
# cache-write races. --model is an explicit local nvme path so no model is pulled.
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
  --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$BASE" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps 100 --num-generations 4 --max-completion-length 2560 --output-dir "$RUN" $RESUME \
  > "$LOGS/r2_${FORK}_train.log" 2>&1
RC=$?; [ "$RC" = 0 ] && touch "$RUN/TRAIN_DONE"; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null; echo "r2$FORK done rc=$RC"
