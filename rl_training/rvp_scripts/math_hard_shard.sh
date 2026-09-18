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
# 14B/32B single-GPU LoRA SFT OOMs at bsz 8 (28GB model) -> auto-lower RFT/xrft batch, else no merged_full -> pairs=0
RFT_BSZ=${RFT_BSZ:-$(echo "$B"|grep -qiE '14b|32b' && echo 1 || echo 8)}
ACC=${ACC_CFG:-rl_training/accelerate_zero3.yaml}
G=$HOME/gu; L=$G/logs; V=$G/$TAG; mkdir -p $V $L; R=$V/RES.md; : >$R
echo "# MATH-HARD-SHARD RVP base=$B bank=$BANK eval=$EVAL zero3-full $(date -u)" >>$R
# SKIP_RFT=1 (coverage-preserving mode for strong/big bases): base already has coverage, RFT collapses it
# -> skip bank+RFT, do RVP DIRECTLY from base (pairs from base's own correct/incorrect).
if [ "${SKIP_RFT:-0}" = 1 ]; then
  echo "SKIP_RFT=1: RVP-from-base (no RFT; preserve base coverage)" >>$R
  RFT=$B
else
  # 1) bank
  if [ ! -s $V/bank.jsonl ]; then   # DATA-PARALLEL bank-gen across all 8 GPUs (was 1) -> ~8x less idle
    for g in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$g GEN_GPU_MEM=${GEN_GPU_MEM:-0.55} python3 -m rl_training.math_rvp --mode bank --model $B --dataset $BANK --split train --n $NBANK --k $KB --num-shards 8 --shard-index $g --out $V/bank.s$g.jsonl >$L/${TAG}_bank$g.log 2>&1 & sleep 3; done
    wait; cat $V/bank.s*.jsonl > $V/bank.jsonl 2>/dev/null; rm -f $V/bank.s*.jsonl
  fi
  echo "bank=$(wc -l <$V/bank.jsonl)" >>$R
  SYNC_CKPT=1 python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
  # 2) RFT (single-GPU LoRA) -> merged full model
  [ -d $V/rft/merged_full ] || { CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $B --data $V/bank.jsonl --out $V/rft --seed 1 --max-steps 300 --bsz $RFT_BSZ >$L/${TAG}_rft.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rft')" >>$L/${TAG}_rft.log 2>&1; }
  RFT=$V/rft/merged_full
  SYNC_CKPT=1 python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
fi
# 3) pairs + shuffled control. Gen is TP-aware: 14B (28GB bf16) does NOT fit single 40GB card
# with KV cache -> vLLM EngineCore OOMs at gen start. VLLM_TP>1 splits weights across GPUs.
# pairs uses GPUs [0..TP-1], shuf uses [TP..2TP-1] so both run in parallel without collision.
# DATA-PARALLEL pair-gen: pairs sharded across GPUs 0-3, shuf across GPUs 4-7 -> all 8 busy, concurrent
if [ ! -s $V/pairs.jsonl ]; then
  for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s GEN_GPU_MEM=${GEN_GPU_MEM:-0.55} python3 -m rl_training.math_rvp --mode pairs --model $RFT --dataset $BANK --split train --n $NBANK --k $KP --max-pairs-per 2 ${HARDNEG:+--hard-neg} --num-shards 4 --shard-index $s --out $V/pairs.s$s.jsonl >$L/${TAG}_pairs$s.log 2>&1 & sleep 3; done
fi
if [ ! -s $V/shuf.jsonl ]; then
  for s in 0 1 2 3; do g=$((s+4)); CUDA_VISIBLE_DEVICES=$g GEN_GPU_MEM=${GEN_GPU_MEM:-0.55} python3 -m rl_training.math_rvp --mode pairs --model $RFT --dataset $BANK --split train --n $NBANK --k $KP --max-pairs-per 2 --shuffle --num-shards 4 --shard-index $s --out $V/shuf.s$s.jsonl >$L/${TAG}_shuf$s.log 2>&1 & sleep 3; done
fi
wait
[ -s $V/pairs.jsonl ] || { cat $V/pairs.s*.jsonl > $V/pairs.jsonl 2>/dev/null; rm -f $V/pairs.s*.jsonl; }
[ -s $V/shuf.jsonl ]  || { cat $V/shuf.s*.jsonl  > $V/shuf.jsonl  2>/dev/null; rm -f $V/shuf.s*.jsonl; }
python3 -c "import json;seen=set();f=open('$V/pos.jsonl','w')
[f.write(json.dumps({'prompt':r['prompt'],'completion':r['chosen']})+'\n') for r in (json.loads(l) for l in open('$V/pairs.jsonl')) if not (r['chosen'] in seen or seen.add(r['chosen']))]"
echo "pairs=$(wc -l <$V/pairs.jsonl) pos=$(wc -l <$V/pos.jsonl)" >>$R
# EARLY S3 sync of bank+pairs (jsonl) — pods have died during/after bank-gen (~1h in) losing all
# work; syncing the generated data here means a re-run can reuse it instead of regenerating.
SYNC_CKPT=1 python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
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
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DPO_FULL=1 python3 -m accelerate.commands.launch --config_file $ACC --num_processes 8 --main_process_ip 127.0.0.1 --main_process_port $((29500+RANDOM%2000)) \
    -m rl_training.dpo_train --full --model $RFT --data $V/pairs.jsonl --out $V/rvp_s$s --seed $s --beta ${DPO_BETA:-0.1} --max-steps ${DPO_STEPS:-300} --bsz 1 >$L/${TAG}_rvp$s.log 2>&1
  SYNC_CKPT=1 python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
done
# 4b) shuffled-pair control (sharded full-param DPO)
[ -f $V/shuf/config.json ] || CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DPO_FULL=1 python3 -m accelerate.commands.launch --config_file $ACC --num_processes 8 --main_process_ip 127.0.0.1 --main_process_port $((29500+RANDOM%2000)) \
    -m rl_training.dpo_train --full --model $RFT --data $V/shuf.jsonl --out $V/shuf --seed 1 --beta ${DPO_BETA:-0.1} --max-steps ${DPO_STEPS:-300} --bsz 1 >$L/${TAG}_shuf.log 2>&1
SYNC_CKPT=1 python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
# 4c) xrft positive-only (single-GPU LoRA)
[ -d $V/xrft/merged_full ] || { CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $RFT --data $V/pos.jsonl --out $V/xrft --seed 1 --max-steps 300 --bsz $RFT_BSZ >$L/${TAG}_xrft.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/xrft')" >>$L/${TAG}_merge.log 2>&1; }
SYNC_CKPT=1 python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
# 4d) CONSOLIDATE — DISABLED by default. The full-param save (zero3_save_16bit_model + tok.save_pretrained)
# is already a complete HF dir that vLLM loads; reloading 7B x5 to re-save was UNNECESSARY and OOM-killed
# the pods (host-memory spike -> B & C died here). Only the pre-eval GPU-clear (below) was actually needed.
# Set CONSOLIDATE=1 to force it (CPU-safe reload) if a raw save ever fails to load.
if [ "${CONSOLIDATE:-0}" = 1 ]; then
for d in rvp_s1 rvp_s2 rvp_s3 rvp_s4 rvp_s5 shuf; do
  [ -f $V/$d/config.json ] || continue
  [ -f $V/$d/.consolidated ] && continue
  python3 -c "
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
fi
# DEATH-PROOF: sync trained checkpoints to S3 BEFORE eval. The eval stage is where pods keep
# dying (host saturation at the DPO->multi-engine-eval transition killed A-workers, C-main, B-main).
# Syncing here means a pod death during eval no longer loses the rvp/shuf/xrft checkpoints.
SYNC_CKPT=1 python3 rl_training/rvp_scripts/s3_sync.py $TAG 2>&1 | tail -1
# hard-clear GPUs before eval (frees any DPO/consolidation residue -> avoids vLLM memory-profiling assert)
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; sleep 8
# 5) eval pass@1 on hard OOD set. RVP/shuf are full dirs; xrft/rft use merged_full.
declare -A E=( [base]=$B [rft]=$RFT [xrft]=$V/xrft/merged_full [shuf]=$V/shuf )
for s in $(seq 1 $NSEED); do E[rvp_s$s]=$V/rvp_s$s; done
ARMS="base rft $(for s in $(seq 1 $NSEED); do echo -n rvp_s$s' '; done) xrft shuf"
# SERIALIZE eval FULLY (EVAL_CONC default 1): even 2 concurrent vLLM engines saturate host RAM at
# the DPO->eval transition and kill the pod (3x observed). One engine at a time + per-arm S3 sync.
CONC=${EVAL_CONC:-1}; g=0; run=0
for tag in $ARMS; do
  [ "$tag" = base ] || [ -e "${E[$tag]}/config.json" ] || [ -d "${E[$tag]}" ] || continue
  [ -s $V/ev_$tag.json ] && continue
  CUDA_VISIBLE_DEVICES=$((g%8)) GEN_GPU_MEM=${EVAL_GPU_MEM:-0.45} setsid nohup python3 -m rl_training.math_rvp --mode eval --model "${E[$tag]}" --dataset $EVAL --split test --n $NEVAL --k $KE --out $V/ev_$tag.json >$L/${TAG}_ev_$tag.log 2>&1 &
  g=$((g+1)); run=$((run+1)); [ $((run % CONC)) -eq 0 ] && { wait; python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1; }
  sleep 2
done
wait
for tag in $ARMS; do python3 -c "import json;d=json.load(open('$V/ev_$tag.json'));print('$tag pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "MATH_HARD_SHARD_DONE $(date -u)" >>$R
cat $R
SYNC_CKPT=${SYNC_CKPT:-1} python3 rl_training/rvp_scripts/s3_sync.py $TAG 2>&1|tail -1