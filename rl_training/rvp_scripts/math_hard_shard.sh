#!/bin/bash
# Harder-math RVP for BIG models via ZeRO-3 sharded FULL-PARAM DPO (7B/14B/32B).
# RFT/xrft stay single-GPU LoRA; RVP + shuf are full-param DPO sharded across all 8 GPUs
# (LoRA+ZeRO-3 crashes -> full-param). RVP seeds run SEQUENTIALLY (each uses 8 GPUs).
# env: BASE EVAL TAG [BANK NBANK NEVAL KB KP KE NSEED ACC_CFG DPO_MAXLEN]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAXLEN=${MAXLEN:-3072} MAXTOK=${MAXTOK:-1024} DPO_MAXLEN=${DPO_MAXLEN:-768} DS_SKIP_CUDA_CHECK=1
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1; pip install --break-system-packages --quiet jsonlines math_verify deepspeed 2>/dev/null
pip install --break-system-packages --quiet -U nvtx 2>/dev/null  # deepspeed needs nvtx.get_domain() (pytorch-base ships too-old nvtx)
B=${BASE:-Qwen/Qwen2.5-Math-7B}; BANK=${BANK:-math_full}; EVAL=${EVAL:-math500}; TAG=${TAG:-mhs}
NBANK=${NBANK:-800}; NEVAL=${NEVAL:-400}; KB=${KB:-8}; KP=${KP:-12}; KE=${KE:-16}; NSEED=${NSEED:-3}
ACC=${ACC_CFG:-rl_training/accelerate_zero3.yaml}
G=$HOME/gu; L=$G/logs; V=$G/$TAG; mkdir -p $V $L; R=$V/RES.md; : >$R
echo "# MATH-HARD-SHARD RVP base=$B bank=$BANK eval=$EVAL zero3-full $(date -u)" >>$R
# 1) bank
[ -s $V/bank.jsonl ] || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.55 python3 -m rl_training.math_rvp --mode bank --model $B --dataset $BANK --split train --n $NBANK --k $KB --out $V/bank.jsonl >$L/${TAG}_bank.log 2>&1
echo "bank=$(wc -l <$V/bank.jsonl)" >>$R
# 2) RFT (single-GPU LoRA) -> merged full model
[ -d $V/rft/merged_full ] || { CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $B --data $V/bank.jsonl --out $V/rft --seed 1 --max-steps 300 --bsz 8 >$L/${TAG}_rft.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rft')" >>$L/${TAG}_rft.log 2>&1; }
RFT=$V/rft/merged_full
# 3) pairs + shuffled control
[ -s $V/pairs.jsonl ] || CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.55 python3 -m rl_training.math_rvp --mode pairs --model $RFT --dataset $BANK --split train --n $NBANK --k $KP --max-pairs-per 2 --out $V/pairs.jsonl >$L/${TAG}_pairs.log 2>&1 &
[ -s $V/shuf.jsonl ] || CUDA_VISIBLE_DEVICES=1 GEN_GPU_MEM=0.55 python3 -m rl_training.math_rvp --mode pairs --model $RFT --dataset $BANK --split train --n $NBANK --k $KP --max-pairs-per 2 --shuffle --out $V/shuf.jsonl >$L/${TAG}_shufgen.log 2>&1 &
wait
python3 -c "import json;seen=set();f=open('$V/pos.jsonl','w')
[f.write(json.dumps({'prompt':r['prompt'],'completion':r['chosen']})+'\n') for r in (json.loads(l) for l in open('$V/pairs.jsonl')) if not (r['chosen'] in seen or seen.add(r['chosen']))]"
echo "pairs=$(wc -l <$V/pairs.jsonl) pos=$(wc -l <$V/pos.jsonl)" >>$R
# free ALL lingering vLLM/GPU procs before sharded training (loop until GPUs truly clear)
export DPO_FULL=1
for i in 1 2 3 4 5; do
  pkill -9 -f 'math_rvp --mode' 2>/dev/null; pkill -9 -f 'VLLM' 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
  sleep 6; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l)
  echo "[shard] pre-DPO clear attempt $i gpuprocs=$n" >>$R; [ "$n" -eq 0 ] && break
done
# 4a) RVP seeds: SEQUENTIAL full-param ZeRO-3 DPO across all 8 GPUs
for s in $(seq 1 $NSEED); do
  [ -f $V/rvp_s$s/config.json ] && continue
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DPO_FULL=1 python3 -m accelerate.commands.launch --config_file $ACC --num_processes 8 --main_process_ip 127.0.0.1 \
    -m rl_training.dpo_train --full --model $RFT --data $V/pairs.jsonl --out $V/rvp_s$s --seed $s --max-steps 300 --bsz 1 >$L/${TAG}_rvp$s.log 2>&1
done
# 4b) shuffled-pair control (sharded full-param DPO)
[ -f $V/shuf/config.json ] || CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DPO_FULL=1 python3 -m accelerate.commands.launch --config_file $ACC --num_processes 8 --main_process_ip 127.0.0.1 \
    -m rl_training.dpo_train --full --model $RFT --data $V/shuf.jsonl --out $V/shuf --seed 1 --max-steps 300 --bsz 1 >$L/${TAG}_shuf.log 2>&1
# 4c) xrft positive-only (single-GPU LoRA)
[ -d $V/xrft/merged_full ] || { CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $RFT --data $V/pos.jsonl --out $V/xrft --seed 1 --max-steps 300 --bsz 8 >$L/${TAG}_xrft.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/xrft')" >>$L/${TAG}_merge.log 2>&1; }
# 4d) CONSOLIDATE full-param (ZeRO-3) DPO outputs into canonical HF checkpoints
# (DeepSpeed-saved dirs don't always load in vLLM; reload+resave + copy a known-good tokenizer)
for d in rvp_s1 rvp_s2 rvp_s3 rvp_s4 rvp_s5 shuf; do
  [ -f $V/$d/config.json ] || continue
  [ -f $V/$d/.consolidated ] && continue
  CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch,shutil,os
from transformers import AutoModelForCausalLM, AutoTokenizer
d='$V/$d'
m=AutoModelForCausalLM.from_pretrained(d,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
m.save_pretrained(d,safe_serialization=True)
AutoTokenizer.from_pretrained('$RFT').save_pretrained(d)
gc='$RFT/generation_config.json'
if os.path.exists(gc): shutil.copy(gc, d+'/generation_config.json')
shutil.rmtree(os.path.join(d,'checkpoint-300'),ignore_errors=True)
open(d+'/.consolidated','w').close()
print('consolidated',d)" >>$L/${TAG}_consolidate.log 2>&1
done
# 5) eval pass@1 on hard OOD set. RVP/shuf are full dirs; xrft/rft use merged_full.
declare -A E=( [base]=$B [rft]=$RFT [xrft]=$V/xrft/merged_full [shuf]=$V/shuf )
for s in $(seq 1 $NSEED); do E[rvp_s$s]=$V/rvp_s$s; done
ARMS="base rft $(for s in $(seq 1 $NSEED); do echo -n rvp_s$s' '; done) xrft shuf"
g=0; for tag in $ARMS; do
  [ "$tag" = base ] || [ -e "${E[$tag]}/config.json" ] || [ -d "${E[$tag]}" ] || continue
  [ -s $V/ev_$tag.json ] && continue
  CUDA_VISIBLE_DEVICES=$((g%8)) GEN_GPU_MEM=0.45 setsid nohup python3 -m rl_training.math_rvp --mode eval --model "${E[$tag]}" --dataset $EVAL --split test --n $NEVAL --k $KE --out $V/ev_$tag.json >$L/${TAG}_ev_$tag.log 2>&1 &
  g=$((g+1)); sleep 2; done
wait
for tag in $ARMS; do python3 -c "import json;d=json.load(open('$V/ev_$tag.json'));print('$tag pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "MATH_HARD_SHARD_DONE $(date -u)" >>$R
cat $R