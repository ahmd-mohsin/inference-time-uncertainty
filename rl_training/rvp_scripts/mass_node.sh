#!/bin/bash
# Run probability-mass accounting for every present final-24 cell that has base/rft/rvp/rft2 in ~/gu/f24_*.
# One cell per GPU, detached, S3-synced. Env: BASE (base HF model), PREFIX (default f24_m15).
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1
cd $HOME/inference-time-uncertainty
BASE=${BASE:?}; PREFIX=${PREFIX:-f24_m15}; G=$HOME/gu; L=$G/logs; mkdir -p $L
gpu=0
for V in $G/${PREFIX}_*; do
  tag=$(basename $V)
  [ -f $V/rvp/merged_full/config.json ] || continue
  [ -s $V/pairs.jsonl ] || continue
  models="base=$BASE,rft=$V/rft/merged_full,rvp=$V/rvp/merged_full"
  [ -f $V/rft2/merged_full/config.json ] && models="$models,rft2=$V/rft2/merged_full"
  echo "== mass-accounting $tag on GPU $gpu =="
  setsid nohup env CUDA_VISIBLE_DEVICES=$gpu bash -c "python3 -m rl_training.mass_accounting --models '$models' --data $V/pairs.jsonl --n 80 --out $V/mass_${tag}.json > $L/${tag}_mass.log 2>&1; python3 rl_training/rvp_scripts/s3_sync.py $tag >/dev/null 2>&1" >$L/disp_mass_${tag}.log 2>&1 &
  gpu=$(( (gpu+1) % 8 )); sleep 5
done
echo "[mass_node] dispatched on $(date -u)"
