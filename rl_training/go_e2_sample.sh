#!/usr/bin/env bash
# E2 launcher: download base model once, then 8-GPU sharded base-solution sampling on the fragile band.
# Writes bank_e2_shard{0..7}.jsonl on nvme. Usage: bash go_e2_sample.sh [N] [dataset] [subset]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$(cat $HOME/.hf_token 2>/dev/null || true)
PY=/usr/bin/python3
N="${1:-256}"; DS="${2:-olympiad_bench}"; SUBSET="${3:-hard}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
DIFF=$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json
BASE_LOCAL=$NV/base_qwen25math7b
MODEL="Qwen/Qwen2.5-Math-7B"

# 1) download base ONCE to a local dir (avoid 8 simultaneous downloads)
if [ ! -f "$BASE_LOCAL/config.json" ]; then
  echo ">> [e2] downloading base $MODEL -> $BASE_LOCAL"
  $PY - <<PY
import os; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", local_dir="$BASE_LOCAL",
                  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","vocab*","merges*"])
print("base ready")
PY
fi
[ -f "$BASE_LOCAL/config.json" ] || { echo ">> FATAL: base model not downloaded"; exit 1; }

# 2) fan out 8 shards, one per GPU
echo ">> [e2] launching 8 shards, N=$N, dataset=$DS, subset=$SUBSET"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s HF_HUB_OFFLINE=1 setsid nohup $PY rl_training/sample_base_solutions.py \
    --model "$BASE_LOCAL" --n "$N" --dataset "$DS" --difficulty-json "$DIFF" --subset "$SUBSET" \
    --num-shards 8 --shard-index $s --out "$NV/bank_e2_${DS}_shard${s}.jsonl" \
    > "$LOGS/e2_sample_shard${s}.log" 2>&1 &
done
echo ">> [e2] 8 shards launched. Merge: cat $NV/bank_e2_${DS}_shard*.jsonl > $NV/bank_e2_${DS}.jsonl"
