#!/bin/bash
# Generic WORKER matrix: one comp rvp_family cell defined by env WBASE/WFAM/WTRD/WOODD.
cd $HOME/inference-time-uncertainty
echo "[matrix_W] $WFAM start $(date -u)"
env BASE="$WBASE" FAM="$WFAM" TRD="${WTRD:-7}" OODD="${WOODD:-9}" bash rl_training/rvp_scripts/rvp_family.sh
echo "[matrix_W] $WFAM done $(date -u)"
