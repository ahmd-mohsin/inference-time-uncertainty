#!/bin/bash
# DATA-EFFICIENCY curve: given an RFT init + full verified pairs, train RVP on head-N subsets and
# eval each -> pass@1 vs #pairs. Answers "how few self-labeled pairs does RVP need?". Reuses
# dpo_train + math_rvp. env: INIT (rft/merged_full) BASE EVAL TAG PAIRS(full pairs.jsonl) [SIZES NEVAL]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_SKIP_CUDA_CHECK=1
export MAXLEN=${MAXLEN:-3072} DPO_MAXLEN=${DPO_MAXLEN:-768}
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
pip install --break-system-packages --quiet -U nvtx 2>/dev/null
ACC=rl_training/accelerate_zero3_offload.yaml
INIT=${INIT:?}; BASE=${BASE:?}; EVAL=${EVAL:?}; TAG=${TAG:?}; PAIRS=${PAIRS:?}; NEVAL=${NEVAL:-200}; SIZES=${SIZES:-25 50 100 400 800}
V=$HOME/gu/$TAG; mkdir -p $V; R=$V/RES.md; : >$R
echo "# DATA-EFF init=$INIT eval=$EVAL full_pairs=$(wc -l <$PAIRS) $(date -u)" >>$R
[ -s $V/ev_base.json ] || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode eval --model $BASE --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_base.json >$V/evb.log 2>&1
B=$(python3 -c "import json;print('%.4f'%json.load(open('$V/ev_base.json'))['pass1'])")
echo "base pass1=$B" >>$R
for N in $SIZES; do
  D=$V/n$N; mkdir -p $D; head -n $N $PAIRS > $D/pairs.jsonl
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 5
  [ -f $D/rvp/config.json ] || CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DPO_FULL=1 python3 -m accelerate.commands.launch --config_file $ACC --num_processes 8 --main_process_ip 127.0.0.1 --main_process_port $((29500+RANDOM%2000)) \
    -m rl_training.dpo_train --full --model $INIT --data $D/pairs.jsonl --out $D/rvp --seed 1 --beta ${DPO_BETA:-0.1} --max-steps ${DPO_STEPS:-300} --bsz 1 >$D/dpo.log 2>&1
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 8
  [ -s $D/ev.json ] || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode eval --model $D/rvp --dataset $EVAL --split test --n $NEVAL --k 16 --out $D/ev.json >$D/ev.log 2>&1
  python3 -c "import json;d=json.load(open('$D/ev.json'));print('npairs=$N rvp_pass1=%.4f delta=%+.4f'%(d['pass1'],d['pass1']-$B))" >>$R 2>>$R
  python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
done
echo "DATAEFF_DONE $(date -u)" >>$R; cat $R
