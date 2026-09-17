#!/bin/bash
# Wait for a cell's RVP checkpoint + pairs AND the node to go idle (post-eval, 0 GPU procs), then
# run the mechanistic-interpretability panel (base vs rvp) with no training contention. S3-sync.
# env: TAG BASE
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1
cd $HOME/inference-time-uncertainty
TAG=${TAG:?}; BASE=${BASE:?}; V=$HOME/gu/$TAG
for i in $(seq 1 300); do   # up to ~5h
  if [ -f $V/rvp_s1/config.json ] && [ -s $V/pairs.jsonl ] && [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l)" -eq 0 ]; then
    git pull --rebase 2>&1|tail -1
    CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.mech_interp --base "$BASE" --rvp $V/rvp_s1 --data $V/pairs.jsonl --n 80 --out $V/mech_$TAG.json > $HOME/gu/logs/mech_$TAG.log 2>&1
    python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1
    echo "MECH_DONE $TAG $(date -u)" >> $HOME/gu/logs/mech_$TAG.log
    break
  fi
  sleep 60
done
