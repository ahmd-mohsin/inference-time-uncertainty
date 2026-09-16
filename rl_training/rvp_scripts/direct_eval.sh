#!/bin/bash
# Minimal robust eval of trained cell: base / rft / rvp_s1, ONE vLLM at a time (no consolidation, no concurrency).
# env: TAG BASE EVAL [NEVAL MAXLEN GEN_GPU_MEM]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty
V=$HOME/gu/$TAG; L=$HOME/gu/logs; R=$V/RES_direct.md; : >$R
NEVAL=${NEVAL:-150}; MAXLEN=${MAXLEN:-2048}; MEM=${GEN_GPU_MEM:-0.55}
echo "# DIRECT eval $TAG base=$BASE eval=$EVAL $(date -u)" >>$R
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; sleep 6
declare -A M=( [base]="$BASE" [rft]="$V/rft/merged_full" [rvp]="$V/rvp_s1" )
for arm in base rft rvp; do
  [ "$arm" = base ] || [ -f "${M[$arm]}/config.json" ] || { echo "$arm MISSING" >>$R; continue; }
  CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=$MEM MAXLEN=$MAXLEN python3 -m rl_training.math_rvp --mode eval --model "${M[$arm]}" --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_$arm.json >$L/${TAG}_direct_$arm.log 2>&1
  python3 -c "import json;d=json.load(open('$V/ev_$arm.json'));print('$arm pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; sleep 4
done
echo "DIRECT_DONE $(date -u)" >>$R; cat $R
python3 rl_training/rvp_scripts/s3_sync.py $TAG 2>&1|tail -1
