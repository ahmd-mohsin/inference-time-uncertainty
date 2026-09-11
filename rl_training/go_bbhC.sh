#!/usr/bin/env bash
# BBH (logic) arm-C: harvest verified BBH-train traces (8-shard) -> LoRA-SFT. go_bbhC.sh <seed> <MODEL> <mtag> [GENGPUS]
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim
PY=/usr/bin/python3
SEED="${1:-0}"; MODEL="${2:-Qwen/Qwen2.5-7B}"; MTAG="${3:-bbh_q7b}"; GG="${4:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra G <<< "$GG"; NSH=${#G[@]}
GU=/tmp/instance_storage/gu; L=$GU/logs; DATA=$GU/sft_data/bbhdata_${MTAG}.jsonl; OUT=$GU/sft_${MTAG}_s$SEED
echo "[go_bbhC $MTAG] harvest BBH-train $(date)"
i=0; for gpu in "${G[@]}"; do CUDA_VISIBLE_DEVICES=$gpu $PY -m rl_training.bbh_gen --model-path "$MODEL" --k 4 --tag $MTAG --shard-index $i --num-shards $NSH >"$L/bbhgen_${MTAG}_$i.log" 2>&1 & i=$((i+1)); done; wait
$PY -m rl_training.bbh_gen --tag $MTAG --num-shards $NSH --merge >"$L/bbhgen_${MTAG}_merge.log" 2>&1
echo "[go_bbhC $MTAG] SFT data=$(wc -l <$DATA 2>/dev/null) $(date)"
CUDA_VISIBLE_DEVICES=${G[0]} $PY -m rl_training.sft_train --model "$MODEL" --data "$DATA" --out "$OUT" --max-steps 600 --save-steps 300 --seed $SEED --bsz ${BSZ:-8} >"$L/sft_${MTAG}.log" 2>&1
echo "[go_bbhC $MTAG] DONE rc=$? -> $OUT $(date)"
