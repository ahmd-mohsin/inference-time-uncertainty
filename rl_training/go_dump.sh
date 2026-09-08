#!/usr/bin/env bash
# Generate self-repair GRPO training data (dump_repair_data), 8-GPU DP + merge. Model = local cached dir.
# Usage: bash go_dump.sh <dir> <TAG> <BENCH> [NPROBE]
set -uo pipefail
export HOME=/home/greenland-user; cd ${REPO:-/tmp/instance_storage/gu/repo}
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3; D="$1"; TAG="$2"; BENCH="$3"; NPROBE="${4:-3}"
NV=/tmp/instance_storage/gu; OUT=$NV/repair_data; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
[ -f "$D/config.json" ] || { echo "FATAL no model $D"; exit 1; }
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.dump_repair_data --model-path "$D" --bench "$BENCH" --tag "$TAG" \
    --shard-index $g --num-shards 8 --n-probe "$NPROBE" --output-dir "$OUT" ${KEEPCODE:+--keep-code} > "$LOGS/dump_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.dump_repair_data --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/dump_${TAG}_merge.log"
touch "$OUT/DUMP_${TAG}_DONE"; echo ">> DUMP $TAG COMPLETE"
