#!/usr/bin/env bash
# Stratified pass@k (downstream go/no-go) on ONE node, 8-GPU data-parallel + merge, on a LOCAL model.
# Usage: bash go_sp.sh <MODEL_DIR> <TAG> <DATASET> [MAXPROB] [NSAMPLES]
set -uo pipefail
export HOME=/home/greenland-user; cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
D="$1"; TAG="$2"; DATASET="$3"; MAXPROB="${4:-150}"; NS="${5:-32}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
[ -f "$D/config.json" ] || { echo "FATAL: model not at $D"; exit 1; }
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.stratified_passk --model-path "$D" --dataset "$DATASET" \
    --tag "$TAG" --shard-index $g --num-shards 8 --max-problems "$MAXPROB" --n-samples "$NS" \
    --output-dir "$OUT" > "$LOGS/sp_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.stratified_passk --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/sp_${TAG}_merge.log"
touch "$OUT/SP_${TAG}_DONE"; echo ">> SP $TAG COMPLETE"
