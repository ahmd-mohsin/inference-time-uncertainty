#!/bin/bash
# Generic WORKER matrix. If WEVAL set -> single-GPU math_hard cell (<=3B fits); else comp rvp_family cell.
cd $HOME/inference-time-uncertainty
echo "[matrix_W] $WFAM start $(date -u)"
if [ -n "$WEVAL" ]; then
  env BASE="$WBASE" EVAL="$WEVAL" TAG="wm_${WFAM}" DPO_BSZ=1 DPO_MAXLEN=640 bash rl_training/rvp_scripts/math_hard.sh
else
  env BASE="$WBASE" FAM="$WFAM" TRD="${WTRD:-7}" OODD="${WOODD:-9}" bash rl_training/rvp_scripts/rvp_family.sh
fi
echo "[matrix_W] $WFAM done $(date -u)"
