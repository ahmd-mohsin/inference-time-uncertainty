#!/bin/bash
# Seed-variance for CIs on ONE existing cell, from its own RFT init + pairs, at the paper's beta=0.1.
# Runs extra seeds (default 2,3) so, with the matrix seed-1 point, we get a 3-seed mean +/- CI.
# env: CELL (existing tag w/ rft/merged_full + pairs.jsonl) BASE EVAL [NEVAL SEEDS]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
CELL=${CELL:?}; BASE=${BASE:?}; EVAL=${EVAL:?}; NEVAL=${NEVAL:-200}; SEEDS=${SEEDS:-2 3}
SRC=$HOME/gu/$CELL; INIT=$SRC/rft/merged_full
[ -f $INIT/config.json ] || { echo "[seedci] $CELL: no rft/merged_full — abort"; exit 1; }
[ -s $SRC/pairs.jsonl ]   || { echo "[seedci] $CELL: no pairs.jsonl — abort"; exit 1; }
for s in $SEEDS; do
  T=seed_${CELL}_s${s}; V=$HOME/gu/$T; mkdir -p $V; cp -n $SRC/pairs.jsonl $V/pairs.jsonl
  echo "[seedci] $CELL seed=$s -> $T $(date -u)"
  env INIT="$INIT" BASE="$BASE" BETA=0.1 SEED="$s" EVAL="$EVAL" TAG="$T" NEVAL="$NEVAL" MAXTOK=1024 \
    bash rl_training/rvp_scripts/beta_point.sh
done
echo "[seedci] $CELL done $(date -u)"
