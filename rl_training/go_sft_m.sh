#!/usr/bin/env bash
# ARM C (proposed method) — parameterized by model. Usage: go_sft_m.sh <seed> <MODEL> <mtag> [GENGPUS]
# Harvest verifier-correct GSM8K traces (sharded gen) then LoRA-SFT on them (single-GPU). Multi-family safe
# (chat template via tokenizer). GENGPUS = comma list for gen shards (default 0-7); SFT runs on first GPU.
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim TP=${TP:-1}
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
SEED="${1:-0}"; MODEL="${2:-Qwen/Qwen2.5-3B}"; MTAG="${3:-q3b}"; GG="${4:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra G <<< "$GG"; NSH=${#G[@]}
GU=/tmp/instance_storage/gu; LOGS=$GU/logs; DATA=$GU/sft_data/sftdata_sft_${MTAG}_s$SEED.jsonl
OUT=$GU/sft_${MTAG}_s$SEED; TAG=sft_${MTAG}_s$SEED
echo "[go_sft_m $MTAG s$SEED] MODEL=$MODEL gen shards=$NSH on GPUs $GG $(date)"
i=0; for gpu in "${G[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu $PY -m rl_training.gen_verified --model-path "$MODEL" --n 900 --k 4 --dataset ${DS:-gsm8k} \
    --tag $TAG --shard-index $i --num-shards $NSH > "$LOGS/genv_${MTAG}_s${SEED}_$i.log" 2>&1 &
  i=$((i+1)); done
wait
$PY -m rl_training.gen_verified --tag $TAG --num-shards $NSH --merge > "$LOGS/genv_${MTAG}_s${SEED}_merge.log" 2>&1
echo "[go_sft_m $MTAG s$SEED] SFT phase data=$(wc -l <$DATA 2>/dev/null) traj $(date)"
CUDA_VISIBLE_DEVICES=${G[0]} $PY -m rl_training.sft_train --model "$MODEL" --data "$DATA" --out "$OUT" \
  --max-steps 1200 --save-steps 400 --seed $SEED --bsz ${BSZ:-8} > "$LOGS/sft_train_${MTAG}_s$SEED.log" 2>&1
echo "[go_sft_m $MTAG s$SEED] DONE rc=$? -> $OUT $(date)"
