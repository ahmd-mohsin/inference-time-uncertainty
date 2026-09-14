#!/bin/bash
# Harder-math RVP flywheel for BIGGER models. Env:
#   BASE   = math-specialized base (e.g. Qwen/Qwen2.5-Math-7B)
#   BANK   = train pool for bank/pairs (default math_full = MATH train, model has coverage)
#   EVAL   = hard OOD eval set (math500 | aime | amc | omni_math | olympiad_bench)
#   TAG    = output dir suffix ; NBANK/NEVAL/KB/KE optional
# base->RFT->{RVP s1..s5, xrft, shuf}->pass@1 eval on EVAL. GPU0-7 on one node.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAXLEN=${MAXLEN:-3072} MAXTOK=${MAXTOK:-1024}
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1; pip install --break-system-packages --quiet jsonlines math_verify 2>/dev/null
B=${BASE:-Qwen/Qwen2.5-Math-7B}; BANK=${BANK:-math_full}; EVAL=${EVAL:-math500}; TAG=${TAG:-mh}
NBANK=${NBANK:-800}; NEVAL=${NEVAL:-400}; KB=${KB:-8}; KP=${KP:-12}; KE=${KE:-16}
G=$HOME/gu; L=$G/logs; V=$G/$TAG; mkdir -p $V $L; R=$V/RES.md; : >$R
echo "# MATH-HARD RVP base=$B bank=$BANK eval=$EVAL $(date -u)" >>$R
# 1) bank (verified-correct from train pool)
CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.55 python3 -m rl_training.math_rvp --mode bank --model $B --dataset $BANK --split train --n $NBANK --k $KB --out $V/bank.jsonl >$L/${TAG}_bank.log 2>&1
echo "bank=$(wc -l <$V/bank.jsonl)" >>$R
# 2) RFT (coverage stage)
CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $B --data $V/bank.jsonl --out $V/rft --seed 1 --max-steps 300 --bsz 8 >$L/${TAG}_rft.log 2>&1
python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rft')" >>$L/${TAG}_rft.log 2>&1
RFT=$V/rft/merged_full
# 3) preference pairs (from RFT ckpt) + shuffled control
CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.55 python3 -m rl_training.math_rvp --mode pairs --model $RFT --dataset $BANK --split train --n $NBANK --k $KP --max-pairs-per 2 --out $V/pairs.jsonl >$L/${TAG}_pairs.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 GEN_GPU_MEM=0.55 python3 -m rl_training.math_rvp --mode pairs --model $RFT --dataset $BANK --split train --n $NBANK --k $KP --max-pairs-per 2 --shuffle --out $V/shuf.jsonl >$L/${TAG}_shufgen.log 2>&1 &
wait
python3 -c "import json;seen=set();f=open('$V/pos.jsonl','w')
[f.write(json.dumps({'prompt':r['prompt'],'completion':r['chosen']})+'\n') for r in (json.loads(l) for l in open('$V/pairs.jsonl')) if not (r['chosen'] in seen or seen.add(r['chosen']))]"
echo "pairs=$(wc -l <$V/pairs.jsonl) pos=$(wc -l <$V/pos.jsonl)" >>$R
# 4) RVP (5 seeds) + xrft (pos-only) + shuf (label ctrl), all matched-budget from RFT
for s in 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$((s-1)) setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s$s --seed $s --max-steps 300 >$L/${TAG}_rvp$s.log 2>&1 & done
CUDA_VISIBLE_DEVICES=5 setsid nohup python3 -m rl_training.sft_train --model $RFT --data $V/pos.jsonl --out $V/xrft --seed 1 --max-steps 300 --bsz 8 >$L/${TAG}_xrft.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/shuf.jsonl --out $V/shuf --seed 1 --max-steps 300 >$L/${TAG}_shuf.log 2>&1 &
wait
for d in rvp_s1 rvp_s2 rvp_s3 rvp_s4 rvp_s5 xrft shuf; do python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/$d')" >>$L/${TAG}_merge.log 2>&1; done
# 5) eval pass@1 on the HARD OOD set
declare -A E=( [base]=$B [rft]=$RFT [rvp_s1]=$V/rvp_s1/merged_full [rvp_s2]=$V/rvp_s2/merged_full [rvp_s3]=$V/rvp_s3/merged_full [rvp_s4]=$V/rvp_s4/merged_full [rvp_s5]=$V/rvp_s5/merged_full [xrft]=$V/xrft/merged_full [shuf]=$V/shuf/merged_full )
g=0; for tag in base rft rvp_s1 rvp_s2 rvp_s3 rvp_s4 rvp_s5 xrft shuf; do
  CUDA_VISIBLE_DEVICES=$((g%8)) GEN_GPU_MEM=0.45 setsid nohup python3 -m rl_training.math_rvp --mode eval --model "${E[$tag]}" --dataset $EVAL --split test --n $NEVAL --k $KE --out $V/ev_$tag.json >$L/${TAG}_ev_$tag.log 2>&1 &
  g=$((g+1)); sleep 2; done
wait
for tag in base rft rvp_s1 rvp_s2 rvp_s3 rvp_s4 rvp_s5 xrft shuf; do python3 -c "import json;d=json.load(open('$V/ev_$tag.json'));print('$tag pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "MATH_HARD_DONE $(date -u)" >>$R
cat $R