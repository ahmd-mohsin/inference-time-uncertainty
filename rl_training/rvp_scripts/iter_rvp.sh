#!/bin/bash
# ITERATIVE RVP (self-distillation round): given an existing round-1 RVP checkpoint, regenerate
# verified y+/y- pairs FROM IT, then run RVP DPO FROM IT again -> round-2 checkpoint. Tests whether
# the selection margin compounds across rounds or plateaus. env: SRC (round-1 rvp dir) BASE(for base eval ref)
# EVAL TAG [NEVAL DPO_STEPS]. Reuses math_rvp + dpo_train; full-param ZeRO-3.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_SKIP_CUDA_CHECK=1
export MAXLEN=${MAXLEN:-3072} MAXTOK=${MAXTOK:-1024} DPO_MAXLEN=${DPO_MAXLEN:-768}
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
pip install --break-system-packages --quiet -U nvtx 2>/dev/null
ACC=rl_training/accelerate_zero3_offload.yaml
SRC=${SRC:?}; BASE=${BASE:?}; EVAL=${EVAL:?}; TAG=${TAG:?}; NEVAL=${NEVAL:-200}; BANK=${BANK:-math_full}
V=$HOME/gu/$TAG; mkdir -p $V; R=$V/RES.md; : >$R
echo "# ITER-RVP round2 src=$SRC eval=$EVAL $(date -u)" >>$R
[ -f $SRC/config.json ] || { echo "no round-1 rvp at $SRC — abort" >>$R; cat $R; exit 1; }
# 1) regenerate verified pairs FROM the round-1 RVP model (its own current y+/y-)
[ -s $V/pairs.jsonl ] || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.55 python3 -m rl_training.math_rvp --mode pairs --model $SRC --dataset $BANK --split train --n 800 --k 12 --max-pairs-per 2 --out $V/pairs.jsonl >$V/pairs.log 2>&1
echo "pairs=$(wc -l <$V/pairs.jsonl)" >>$R
python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 6
# 2) RVP DPO round-2 FROM the round-1 checkpoint
[ -f $V/rvp/config.json ] || CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DPO_FULL=1 python3 -m accelerate.commands.launch --config_file $ACC --num_processes 8 --main_process_ip 127.0.0.1 --main_process_port $((29500+RANDOM%2000)) \
  -m rl_training.dpo_train --full --model $SRC --data $V/pairs.jsonl --out $V/rvp --seed 1 --beta ${DPO_BETA:-0.1} --max-steps ${DPO_STEPS:-300} --bsz 1 >$V/dpo.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 8
# 3) eval base / round-1 / round-2
[ -s $V/ev_base.json ] || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode eval --model $BASE --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_base.json >$V/evb.log 2>&1
[ -s $V/ev_r1.json ]   || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode eval --model $SRC --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_r1.json >$V/ev1.log 2>&1
[ -s $V/ev_r2.json ]   || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode eval --model $V/rvp --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_r2.json >$V/ev2.log 2>&1
python3 -c "import json;b=json.load(open('$V/ev_base.json'));r1=json.load(open('$V/ev_r1.json'));r2=json.load(open('$V/ev_r2.json'));print('base=%.4f r1(rvp)=%.4f r2(iter)=%.4f  r2-r1=%+.4f'%(b['pass1'],r1['pass1'],r2['pass1'],r2['pass1']-r1['pass1']))" >>$R 2>>$R
echo "ITER_DONE $(date -u)" >>$R; cat $R
python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
