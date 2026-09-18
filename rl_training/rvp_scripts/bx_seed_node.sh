#!/bin/bash
# Launch additional base-model seed cells (Qwen2.5-Math-1.5B) to tighten the RVP CIs, one GPU each,
# detached + per-stage S3-synced, then the bx_followon watcher (frontier/ReST/mass per cell).
# Env: BASE (base HF model), PREFIX (default bx_m15), SEEDS (space-separated, e.g. "7 8"). Fills up to 8 GPUs.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/inference-time-uncertainty
BASE=${BASE:?}; PREFIX=${PREFIX:-bx_m15}; SEEDS=${SEEDS:?}; G=$HOME/gu; L=$G/logs; mkdir -p $L
declare -A NV=( [math500]=200 [olympiad_bench]=150 [omni_math]=200 [gsm8k]=200 [amc]=60 [deepmath]=200 )
declare -A DAB=( [math500]=m500 [olympiad_bench]=ol [omni_math]=om [gsm8k]=gsm [amc]=amc [deepmath]=dm )
gpu=0
for s in $SEEDS; do
  for ev in math500 gsm8k olympiad_bench omni_math amc deepmath; do
    [ $gpu -ge 8 ] && break 2
    tag=${PREFIX}_${DAB[$ev]}_s${s}
    [ -f $G/$tag/RES.md ] && grep -q CELL1GPU_DONE $G/$tag/RES.md 2>/dev/null && { echo "$tag done, skip"; continue; }
    echo "== $tag on GPU $gpu =="
    setsid nohup env GPU=$gpu BASE=$BASE EVAL=$ev NEVAL=${NV[$ev]} SEED=$s TAG=$tag GEN_GPU_MEM=0.45 bash rl_training/rvp_scripts/math_cell_1gpu.sh >$L/cell_${tag}.log 2>&1 &
    gpu=$((gpu+1)); sleep 8
  done
done
# followon watcher: runs frontier + ReST + mass per cell as its RVP lands
[ -f rl_training/rvp_scripts/bx_followon.sh ] && setsid nohup env BASE=$BASE PREFIX=$PREFIX bash rl_training/rvp_scripts/bx_followon.sh >$L/followon_seed.log 2>&1 &
echo "[bx_seed_node] launched $gpu cells + followon on $(date -u)"
