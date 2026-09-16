#!/bin/bash
# Sequential beta dose-response on ONE existing cell, initialised from ITS OWN RFT checkpoint
# (the correct RVP init) and reusing ITS OWN pairs — no cross-node copy. For each beta: DPO from
# rft/merged_full -> eval base+rvp pass@1/cov (MAXTOK=1024, matches matrix) -> teacher-forced margin.
# env: CELL (existing tag with rft/merged_full + pairs.jsonl) BASE EVAL [NEVAL BETAS]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
CELL=${CELL:?}; BASE=${BASE:?}; EVAL=${EVAL:?}; NEVAL=${NEVAL:-200}; BETAS=${BETAS:-0.03 0.2 0.5}
SRC=$HOME/gu/$CELL; INIT=$SRC/rft/merged_full
[ -f $INIT/config.json ] || { echo "[bsweep] $CELL: no rft/merged_full — abort"; exit 1; }
[ -s $SRC/pairs.jsonl ]   || { echo "[bsweep] $CELL: no pairs.jsonl — abort"; exit 1; }
for b in $BETAS; do
  bt=$(echo $b | tr -d .); T=bsw_${CELL}_b${bt}; V=$HOME/gu/$T; mkdir -p $V
  cp -n $SRC/pairs.jsonl $V/pairs.jsonl
  echo "[bsweep] $CELL beta=$b -> $T $(date -u)"
  env INIT="$INIT" BASE="$BASE" BETA="$b" EVAL="$EVAL" TAG="$T" NEVAL="$NEVAL" MAXTOK=1024 \
    bash rl_training/rvp_scripts/beta_point.sh
done
echo "[bsweep] $CELL sweep done $(date -u)"
