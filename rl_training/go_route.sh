#!/usr/bin/env bash
# Routing measure (exp #1) on ONE node, 8-GPU data-parallel over problems + merge. Teacher-forced
# logp of the strategy-prefix seeds (no generation). Uses an already-fetched local model dir.
# Usage: bash go_route.sh <MODEL_DIR> <TAG> <DATASET> [MAXPROB]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=0
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
D="$1"; TAG="$2"; DATASET="$3"; MAXPROB="${4:-150}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
[ -f "$D/config.json" ] || { echo "FATAL: model not at $D"; exit 1; }
N=8
for g in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.route_logprob \
    --model-path "$D" --dataset "$DATASET" --tag "$TAG" \
    --shard-index $g --num-shards $N --max-problems "$MAXPROB" --output-dir "$OUT" \
    > "$LOGS/route_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.route_logprob --merge --tag "$TAG" --num-shards $N --output-dir "$OUT" \
  2>&1 | tee "$LOGS/route_${TAG}_merge.log"
touch "$OUT/ROUTE_${TAG}_DONE"
echo ">> ROUTE $TAG COMPLETE -> $OUT/route_${TAG}.json"
