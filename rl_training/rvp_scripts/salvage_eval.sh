#!/bin/bash
# Direct consolidate + eval of already-trained sharded RVP seeds (bypass flywheel re-training).
# env: BASE EVAL TAG [NEVAL KE]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty
B=${BASE:-Qwen/Qwen2.5-Math-7B}; EVAL=${EVAL:-aime}; TAG=${TAG:-mh_q7m_aime}; NEVAL=${NEVAL:-400}; KE=${KE:-16}
V=$HOME/gu/$TAG; L=$HOME/gu/logs; R=$V/RES_salvage.md; : >$R
RFT=$V/rft/merged_full
# stop any running flywheel + clear GPUs
pkill -9 -f 'python3 -m rl_training' 2>/dev/null; pkill -9 -f accelerate.commands.launch 2>/dev/null; pkill -9 -f VLLM 2>/dev/null
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; sleep 6
# consolidate each trained RVP seed -> canonical HF for vLLM
for d in rvp_s1 rvp_s2 rvp_s3 rvp_s4 rvp_s5; do
  [ -f $V/$d/config.json ] || continue
  [ -f $V/$d/.consolidated ] && continue
  CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch,shutil,os
from transformers import AutoModelForCausalLM, AutoTokenizer
d='$V/$d'
m=AutoModelForCausalLM.from_pretrained(d,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True); m.save_pretrained(d,safe_serialization=True)
AutoTokenizer.from_pretrained('$RFT').save_pretrained(d)
gc='$RFT/generation_config.json'
if os.path.exists(gc): shutil.copy(gc,d+'/generation_config.json')
shutil.rmtree(os.path.join(d,'checkpoint-300'),ignore_errors=True)
open(d+'/.consolidated','w').close(); print('consolidated',d)" >>$L/${TAG}_salvage_consolidate.log 2>&1
done
# hard-clear GPUs AGAIN after consolidation (it leaves a 7B resident on GPU0 -> vLLM memory-profiling assert)
pkill -9 -f 'from_pretrained' 2>/dev/null; pkill -9 -f 'python3 -c' 2>/dev/null
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; sleep 8
# eval base/rft/xrft + rvp seeds across GPUs
declare -A E=( [base]=$B [rft]=$RFT [xrft]=$V/xrft/merged_full )
for s in 1 2 3 4 5; do [ -f $V/rvp_s$s/config.json ] && E[rvp_s$s]=$V/rvp_s$s; done
g=0; for tag in "${!E[@]}"; do
  [ "$tag" = base ] || [ -e "${E[$tag]}/config.json" ] || continue
  CUDA_VISIBLE_DEVICES=$((g%8)) GEN_GPU_MEM=0.45 setsid nohup python3 -m rl_training.math_rvp --mode eval --model "${E[$tag]}" --dataset $EVAL --split test --n $NEVAL --k $KE --out $V/ev_$tag.json >$L/${TAG}_salvage_ev_$tag.log 2>&1 &
  g=$((g+1)); sleep 2; done
wait
for tag in base rft xrft rvp_s1 rvp_s2 rvp_s3 rvp_s4 rvp_s5; do python3 -c "import json;d=json.load(open('$V/ev_$tag.json'));print('$tag pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "SALVAGE_DONE $(date -u)" >>$R; cat $R