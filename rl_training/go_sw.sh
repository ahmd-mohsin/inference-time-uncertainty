#!/usr/bin/env bash
# CRPO causal matrix (DE_P / IE_E / CSR) — prefetch model once, then 8-GPU shard fan-out + merge.
# Usage: bash go_cm.sh <MODEL_ID_OR_DIR> <TAG> [BENCH] [NPOOL] [K]
set -uo pipefail
export HOME=/home/greenland-user; cd ${REPO:-/tmp/instance_storage/gu/repo}
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3; MODEL="$1"; TAG="$2"; BENCH="${3:-mbpp}"; NPOOL="${4:-10}"; K="${5:-8}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
echo ">> prefetch $MODEL"
$PY - "$MODEL" <<'PY' 2>&1 | tail -3
import sys, os
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(sys.argv[1], token=os.environ.get("HF_TOKEN"))
    print("PREFETCH_OK", p)
except Exception as e:
    print("PREFETCH_WARN", e)
PY
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.swaps --model-path "$MODEL" --bench "$BENCH" --tag "$TAG" \
    --shard-index $g --num-shards 8 --n-pool "$NPOOL" --k "$K" --output-dir "$OUT" > "$LOGS/sw_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.swaps --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/sw_${TAG}_merge.log"
touch "$OUT/SW_${TAG}_DONE"; echo ">> CM $TAG COMPLETE"
