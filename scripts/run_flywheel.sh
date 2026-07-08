#!/usr/bin/env bash
# Recursive self-distillation flywheel — run on ONE GPU (e.g. an idle worker node or after
# oursABC finishes). Iterates harvest->SFT->eval from --init-model for N rounds, logging pass@k
# per round. Resumable: re-run to continue from the last completed round.
#   Usage: bash scripts/run_flywheel.sh <init_model> <difficulty.json> [rounds]
source ~/miniconda3/etc/profile.d/conda.sh; conda activate topo
cd ~/inference-time-uncertainty
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT="${1:?init model path/id}"; DIFF="${2:-rl_training/runs/difficulty.json}"; ROUNDS="${3:-5}"
mkdir -p ~/logs
setsid python -m rl_training.flywheel --init-model "$INIT" --difficulty-json "$DIFF" \
  --rounds "$ROUNDS" --dataset aime_all --harvest-k 64 --max-keep 4 --n-samples 32 \
  --max-new-tokens 8192 --gpu 0 --output-dir rl_training/runs/flywheel \
  > ~/logs/flywheel.log 2>&1 < /dev/null &
echo ">> flywheel launched pid=$! (init=$INIT, rounds=$ROUNDS) -> ~/logs/flywheel.log"
