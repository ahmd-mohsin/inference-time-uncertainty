#!/usr/bin/env bash
set -uo pipefail; export HOME=/home/greenland-user; cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3; D="$1"; TAG="$2"; DIFF="${3:-MEDIUM}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
[ -f "$D/config.json" ] || { echo "FATAL no model $D"; exit 1; }
for g in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.taco_cert --model-path "$D" --difficulty "$DIFF" --tag "$TAG" --shard-index $g --num-shards 8 --max-problems 400 --k 6 --output-dir "$OUT" > "$LOGS/tacocert_${TAG}_s${g}.log" 2>&1 & done
wait
$PY -m rl_training.taco_cert --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/tacocert_${TAG}_merge.log"
touch "$OUT/TACOCERT_${TAG}_DONE"; echo ">> TACOCERT $TAG COMPLETE"
