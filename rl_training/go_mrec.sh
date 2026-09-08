#!/usr/bin/env bash
set -uo pipefail; export HOME=/home/greenland-user; cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=0 EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3; D=$1; TAG=$2; DS=$3; NDEF=${4:-8}; NREC=${5:-12}
OUT=/tmp/instance_storage/gu/eval_out; LOGS=/tmp/instance_storage/gu/logs; mkdir -p $OUT $LOGS
[ -f $D/config.json ]||{ echo FATAL no model $D; exit 1; }
for g in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.math_recover --model-path $D --dataset $DS --tag $TAG --shard-index $g --num-shards 8 --n-def $NDEF --n-rec $NREC --output-dir $OUT >$LOGS/mrec_${TAG}_s${g}.log 2>&1 & done
wait
$PY -m rl_training.math_recover --merge --tag $TAG --num-shards 8 --output-dir $OUT 2>&1|tee $LOGS/mrec_${TAG}_merge.log
touch $OUT/MREC_${TAG}_DONE; echo ">> MREC $TAG COMPLETE"
