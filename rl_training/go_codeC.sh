#!/usr/bin/env bash
# CODE arm-C: harvest verified MBPP-train code (8-shard) -> LoRA-SFT. Usage: go_codeC.sh <seed> <MODEL> <mtag> [GENGPUS]
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim
PY=/usr/bin/python3
SEED="${1:-0}"; MODEL="${2:-Qwen/Qwen2.5-7B}"; MTAG="${3:-code_q7b}"; GG="${4:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra G <<< "$GG"; NSH=${#G[@]}
GU=/tmp/instance_storage/gu; LOGS=$GU/logs; DATA=$GU/sft_data/codedata_${MTAG}.jsonl; OUT=$GU/sft_${MTAG}_s$SEED
echo "[go_codeC $MTAG] harvest MBPP-train $(date)"
i=0; for gpu in "${G[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu $PY -m rl_training.code_gen_verified --model-path "$MODEL" --n 400 --k 4 \
    --tag $MTAG --shard-index $i --num-shards $NSH > "$LOGS/codegen_${MTAG}_$i.log" 2>&1 &
  i=$((i+1)); done
wait
$PY -m rl_training.code_gen_verified --tag $MTAG --num-shards $NSH --merge > "$LOGS/codegen_${MTAG}_merge.log" 2>&1
echo "[go_codeC $MTAG] SFT data=$(wc -l <$DATA 2>/dev/null) traces $(date)"
CUDA_VISIBLE_DEVICES=${G[0]} $PY -m rl_training.sft_train --model "$MODEL" --data "$DATA" --out "$OUT" \
  --max-steps 800 --save-steps 400 --seed $SEED --bsz ${BSZ:-8} > "$LOGS/sft_${MTAG}.log" 2>&1
echo "[go_codeC $MTAG] DONE rc=$? -> $OUT $(date)"
