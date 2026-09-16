#!/bin/bash
# One point of the beta dose-response curve (Prop 2 validation): given a base model and a
# PRE-PLACED pairs.jsonl at ~/gu/$TAG/pairs.jsonl, run full-param ZeRO-3 DPO at a single BETA,
# then eval pass@1/coverage (base + rvp) and the teacher-forced margin. Reuses existing pairs
# (no bank/RFT/pair-gen) so it is fast. env: BASE BETA EVAL TAG [NEVAL DPO_STEPS]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_SKIP_CUDA_CHECK=1
export MAXLEN=${MAXLEN:-3072} DPO_MAXLEN=${DPO_MAXLEN:-768}
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
pip install --break-system-packages --quiet -U nvtx 2>/dev/null
ACC=rl_training/accelerate_zero3_offload.yaml
BASE=${BASE:-Qwen/Qwen2.5-Math-7B}; BETA=${BETA:-0.1}; EVAL=${EVAL:-math500}; TAG=${TAG:-bsw}; NEVAL=${NEVAL:-200}
V=$HOME/gu/$TAG; mkdir -p $V; R=$V/RES.md; : >$R
echo "# BETA-POINT base=$BASE beta=$BETA eval=$EVAL $(date -u)" >>$R
[ -s $V/pairs.jsonl ] || { echo "NO pairs.jsonl at $V — abort" >>$R; cat $R; exit 1; }
echo "pairs=$(wc -l <$V/pairs.jsonl)" >>$R
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 5
# DPO at this beta (full-param, 8-GPU sharded)
[ -f $V/rvp/config.json ] || CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DPO_FULL=1 python3 -m accelerate.commands.launch --config_file $ACC --num_processes 8 --main_process_ip 127.0.0.1 --main_process_port $((29500+RANDOM%2000)) \
  -m rl_training.dpo_train --full --model $BASE --data $V/pairs.jsonl --out $V/rvp --seed 1 --beta $BETA --max-steps ${DPO_STEPS:-300} --bsz 1 >$V/dpo.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 8
# eval base + rvp, and teacher-forced margin of rvp
[ -s $V/ev_base.json ] || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode eval --model $BASE --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_base.json >$V/ev_base.log 2>&1
[ -s $V/ev_rvp.json ]  || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode eval --model $V/rvp --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_rvp.json >$V/ev_rvp.log 2>&1
CUDA_VISIBLE_DEVICES=0 DPO_MAXLEN=768 python3 -m rl_training.math_rvp --mode margin --model $V/rvp --data $V/pairs.jsonl --n 200 --out $V/margin.json >$V/margin.log 2>&1
python3 -c "import json;b=json.load(open('$V/ev_base.json'));e=json.load(open('$V/ev_rvp.json'));m=json.load(open('$V/margin.json'));print('beta=$BETA base_p1=%.4f rvp_p1=%.4f rvp_cov=%.4f margin=%.4f delta=%+.4f'%(b['pass1'],e['pass1'],e['coverage_passk'],m['margin'],e['pass1']-b['pass1']))" >>$R 2>>$R
echo "BETA_POINT_DONE $(date -u)" >>$R; cat $R
SYNC_CKPT=0 python3 rl_training/rvp_scripts/s3_sync.py $TAG 2>&1|tail -1
