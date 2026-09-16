#!/bin/bash
# Fresh-node self-driver: clone repo -> bootstrap deps (if needed) -> nvtx/deepspeed -> run $RUNCMD.
# Chains so a bare pytorch-base pod goes from nothing to running a flywheel unattended.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1
cd $HOME
[ -d inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git
cd inference-time-uncertainty
mkdir -p $HOME/gu/logs
# bootstrap heavy deps only if vllm missing (survives restarts that keep nvme but wipe home)
python3 -c "import vllm" 2>/dev/null || bash rl_training/queue/FULL_BOOTSTRAP.sh >$HOME/bootstrap.log 2>&1
pip install --break-system-packages -q -U nvtx 2>/dev/null
pip install --break-system-packages -q math_verify jsonlines deepspeed 2>/dev/null
git pull --rebase 2>&1 | tail -1
if [ -n "$CELL_BASE" ]; then
  # single sharded bigger-model math cell (per-pod), reuses the validated ZeRO-3 recipe
  echo "[boot] deps ready $(date -u); launching CELL $CELL_TAG ($CELL_BASE / $CELL_EVAL)" >> $HOME/gu/logs/boot.log
  env FLYWHEEL=math_hard_shard.sh BASE="$CELL_BASE" EVAL="$CELL_EVAL" TAG="$CELL_TAG" \
    NSEED="${NSEED:-3}" ACC_CFG="${ACC_CFG:-rl_training/accelerate_zero3_offload.yaml}" \
    DPO_MAXLEN="${DPO_MAXLEN:-512}" CONSOLIDATE=0 \
    SKIP_RFT="${SKIP_RFT:-0}" DPO_BETA="${DPO_BETA:-0.1}" DPO_STEPS="${DPO_STEPS:-300}" \
    NEVAL="${NEVAL:-400}" EVAL_CONC="${EVAL_CONC:-1}" MAXLEN="${MAXLEN:-3072}" \
    GEN_GPU_MEM="${GEN_GPU_MEM:-0.55}" EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.45}" VLLM_TP="${VLLM_TP:-1}" \
    bash rl_training/rvp_scripts/reset_and_run.sh >> $HOME/gu/logs/cell_${CELL_TAG}.log 2>&1
else
  echo "[boot] deps ready $(date -u); launching matrix_${CLUSTER}" >> $HOME/gu/logs/boot.log
  bash rl_training/rvp_scripts/matrix_${CLUSTER}.sh >> $HOME/gu/logs/matrix_${CLUSTER}.log 2>&1
fi
echo "[boot] done rc=$? $(date -u)" >> $HOME/gu/logs/boot.log
