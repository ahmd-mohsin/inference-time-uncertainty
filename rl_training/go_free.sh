#!/usr/bin/env bash
# Free-generation routing (behavioral rho via LLM-judge) on ONE node, 8-GPU data-parallel + merge.
# Usage: bash go_free.sh <MODEL_DIR> <TAG> <DATASET> [MAXPROB] [NSAMPLES]
set -uo pipefail
export HOME=/home/greenland-user; cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH VLLM_ATTENTION_BACKEND=FLASHINFER HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0
export EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
D="$1"; TAG="$2"; DATASET="$3"; MAXPROB="${4:-150}"; NS="${5:-64}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
[ -f "$D/config.json" ] || { echo "FATAL: model not at $D"; exit 1; }
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.free_route --model-path "$D" --dataset "$DATASET" \
    --tag "$TAG" --shard-index $g --num-shards 8 --max-problems "$MAXPROB" --n-samples "$NS" \
    --output-dir "$OUT" > "$LOGS/free_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.free_route --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/free_${TAG}_merge.log"
touch "$OUT/FREE_${TAG}_DONE"; echo ">> FREE $TAG COMPLETE -> $OUT/free_${TAG}.json"
