#!/usr/bin/env bash
# ROUND-2 continued RL from a round-1 fork checkpoint — DEATH-PROOF via HF Hub.
#   fork=grpo  -> start from muahmed7338/cov-r1-grpo-7b  latest checkpoint (standard round-1)
#   fork=floor -> start from muahmed7338/cov-r1-floor-7b latest checkpoint (coverage-preserved)
# NO floor in round 2 (tests whether preserved coverage raises the continued-RL ceiling).
# Round-2 checkpoints push to muahmed7338/cov-r2-from-<fork>-7b and resume from there on node death.
# Requires HF_TOKEN. Usage: bash run_round2.sh <grpo|floor> [steps]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASHINFER
PY=/usr/bin/python3
mkdir -p $HOME/logs
[ -z "${HF_TOKEN:-}" ] && { echo "ERR: HF_TOKEN not set"; exit 1; }

FORK="${1:?fork: grpo|floor}"
STEPS="${2:-400}"
R1REPO="muahmed7338/cov-r1-${FORK}-7b"
R2REPO="muahmed7338/cov-r2-from-${FORK}-7b"
DIFF="$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json"
BASE="$HOME/inference-time-uncertainty/rl_training/runs/r1_${FORK}_ckpt"   # downloaded r1 fork
RUN="$HOME/inference-time-uncertainty/rl_training/runs/r2_from_${FORK}"
mkdir -p "$BASE" "$RUN"; rm -f "$RUN/TRAIN_DONE"
VLLM_GPUS="0"; TRAIN_GPUS="1,2,3,4,5,6,7"; NTRAIN=7

# 1) fetch the round-1 fork's LATEST checkpoint from HF -> flat dir BASE (the round-2 starting model)
echo ">> [r1-fetch] download latest $R1REPO -> $BASE"
$PY - <<PY
import os
from huggingface_hub import HfApi, snapshot_download
tok=os.environ["HF_TOKEN"]; api=HfApi(token=tok)
f=list(api.list_repo_files("$R1REPO", repo_type="model"))
cks=sorted({int(x.split('-')[1].split('/')[0]) for x in f if x.startswith('checkpoint-')})
assert cks, "no r1 checkpoint in $R1REPO"
ck=f"checkpoint-{cks[-1]}"; print("latest r1:", ck)
snapshot_download("$R1REPO", repo_type="model", allow_patterns=f"{ck}/*", local_dir="$BASE/dl", token=tok)
import shutil, glob
src=os.path.join("$BASE","dl",ck)
for fn in os.listdir(src):
    shutil.move(os.path.join(src,fn), os.path.join("$BASE",fn))
print("r1 fork model ready at $BASE")
PY

# 2) if round-2 was already partway (prior node), resume its latest checkpoint from HF
echo ">> [r2-resume] pull latest round-2 checkpoint from $R2REPO if any"
$PY rl_training/hf_ckpt_daemon.py resume --run-dir "$RUN" --repo "$R2REPO" 2>&1 | tail -2
RESUME=""; LAST=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)
[ -n "$LAST" ] && RESUME="--resume-from $RUN/checkpoint-$LAST" && echo ">> r2 resuming from checkpoint-$LAST"

# 3) start the round-2 checkpoint pusher
setsid nohup $PY rl_training/hf_ckpt_daemon.py watch --run-dir "$RUN" --repo "$R2REPO" --every 45 \
  > $HOME/logs/r2_${FORK}_hfpush.log 2>&1 &

echo ">> [vLLM] server on GPU0 from r1 fork model"
CUDA_VISIBLE_DEVICES=$VLLM_GPUS HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
  --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
  > $HOME/logs/r2_${FORK}_vllm.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && { echo ">> vLLM UP"; break; }; sleep 3; done
curl -s localhost:8000/health >/dev/null 2>&1 || { echo ">> vLLM FAILED"; tail -20 $HOME/logs/r2_${FORK}_vllm.log; exit 1; }

echo ">> [train] round-2 plain GRPO from $FORK fork, $STEPS steps $RESUME (no floor)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
  --config_file rl_training/accelerate_zero3.yaml --num_processes $NTRAIN \
  --main_process_ip 127.0.0.1 --main_process_port 29501 --rdzv_backend c10d \
  -m rl_training.train_grpo --no-lora --no-novelty \
  --model "$BASE" --dataset olympiad_bench --difficulty-json "$DIFF" \
  --num-train-steps "$STEPS" --num-generations 4 --max-completion-length 2560 \
  --output-dir "$RUN" $RESUME \
  > $HOME/logs/r2_${FORK}_train.log 2>&1
RC=$?
echo ">> [train] exit $RC"
[ "$RC" = "0" ] && touch "$RUN/TRAIN_DONE"
pkill -9 -f "trl.scripts.vllm_serve" 2>/dev/null || true
echo ">> DONE round2 fork=$FORK rc=$RC -> $RUN (repo $R2REPO)"
