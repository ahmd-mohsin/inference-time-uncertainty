#!/bin/bash
# 5-hour award-push wave, run ON a node that already has final-24 RVP checkpoints in ~/gu/f24_*.
#  FRONTIER (P0.1): re-eval base & rvp saving per-sample answers -> self-consistency / best-of-n
#    frontier ("why not best-of-n?"). Fast (~20 min/cell).
#  ITERRFT (P0.3): round-2 rejection-sampling FT (ReST/STaR) from the RFT model -> eval; the
#    "just do more RFT" baseline (expect it to plateau while RVP's margin step wins).
# Dispatcher discovers one present cell per dataset and re-invokes itself per-cell, detached,
# one GPU each, per-stage S3-synced. Env: BASE (base HF model), PREFIX (tag glob, default f24_m15).
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/inference-time-uncertainty
BASE=${BASE:?}; PREFIX=${PREFIX:-f24_m15}; G=$HOME/gu; L=$G/logs; mkdir -p $L
export MAXLEN=${MAXLEN:-3072} MAXTOK=${MAXTOK:-1024}
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); [ -n "$L13" ] && export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
declare -A NV=( [math500]=200 [olympiad_bench]=150 [omni_math]=200 [gsm8k]=200 [amc]=60 [deepmath]=200 )
declare -A DAB=( [math500]=m500 [olympiad_bench]=ol [omni_math]=om [gsm8k]=gsm [amc]=amc [deepmath]=dm )

if [ -n "$MODE" ]; then   # ---- per-cell worker ----
  export CUDA_VISIBLE_DEVICES=${GPU:?} GEN_GPU_MEM=${GEN_GPU_MEM:-0.45}
  V=$G/${TAG:?}; nv=${NV[$EVAL]:-200}
  if [ "$MODE" = frontier ]; then
    [ -f $V/rvp/merged_full/config.json ] || { echo "[frontier] $TAG no rvp ckpt"; exit 0; }
    SAVE_SAMPLES=1 python3 -m rl_training.math_rvp --mode eval --model $BASE --dataset $EVAL --split test --n $nv --k 16 --out $V/ev_base_ss.json >$L/${TAG}_ss_base.log 2>&1
    SAVE_SAMPLES=1 python3 -m rl_training.math_rvp --mode eval --model $V/rvp/merged_full --dataset $EVAL --split test --n $nv --k 16 --out $V/ev_rvp_ss.json >$L/${TAG}_ss_rvp.log 2>&1
    python3 -m rl_training.rvp_scripts.selectbench --base $V/ev_base_ss.json --rvp $V/ev_rvp_ss.json --tag $TAG --out $V/frontier_${TAG}.json >$L/${TAG}_frontier.log 2>&1
    python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
    echo "[frontier] $TAG DONE"; cat $L/${TAG}_frontier.log
  elif [ "$MODE" = iterrft ]; then
    [ -f $V/rft/merged_full/config.json ] || { echo "[iterrft] $TAG no rft ckpt"; exit 0; }
    [ -s $V/bank2.jsonl ] || python3 -m rl_training.math_rvp --mode bank --model $V/rft/merged_full --dataset math_full --split train --n 500 --k 8 --out $V/bank2.jsonl >$L/${TAG}_bank2.log 2>&1
    cat $V/bank.jsonl $V/bank2.jsonl > $V/bank12.jsonl 2>/dev/null
    python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
    [ -d $V/rft2/merged_full ] || { python3 -m rl_training.sft_train --model $BASE --data $V/bank12.jsonl --out $V/rft2 --seed 1 --max-steps 600 --bsz 8 >$L/${TAG}_rft2.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rft2')" >>$L/${TAG}_rft2.log 2>&1; }
    [ -f $V/rft2/merged_full/config.json ] || { echo "[iterrft] $TAG rft2 FAILED"; exit 1; }
    [ -s $V/ev_rft2.json ] || python3 -m rl_training.math_rvp --mode eval --model $V/rft2/merged_full --dataset $EVAL --split test --n $nv --k 16 --out $V/ev_rft2.json >$L/${TAG}_ev_rft2.log 2>&1
    python3 -c "import json;d=json.load(open('$V/ev_rft2.json'));print('[iterrft] $TAG rft2 pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))"
    python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
  fi
  exit 0
fi

# ---- dispatcher ----
gpu=0
for ev in math500 olympiad_bench omni_math gsm8k amc deepmath; do
  tag=""; for s in 1 2 3 4; do t=${PREFIX}_${DAB[$ev]}_s${s}; [ -d $G/$t/rvp/merged_full ] && { tag=$t; break; }; done
  [ -z "$tag" ] && { echo "no ckpt for $ev, skip"; continue; }
  g2=$(( (gpu+1) % 8 ))
  echo "== $tag: frontier GPU$gpu, iterrft GPU$g2 =="
  setsid nohup env MODE=frontier GPU=$gpu EVAL=$ev TAG=$tag BASE=$BASE bash rl_training/rvp_scripts/run_5h.sh >$L/disp_front_${tag}.log 2>&1 &
  setsid nohup env MODE=iterrft  GPU=$g2  EVAL=$ev TAG=$tag BASE=$BASE bash rl_training/rvp_scripts/run_5h.sh >$L/disp_iter_${tag}.log 2>&1 &
  gpu=$(( (gpu+2) % 8 )); sleep 8
done
echo "[run_5h] dispatched frontier+iterrft on $(date -u)"
