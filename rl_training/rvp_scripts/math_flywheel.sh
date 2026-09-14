#!/bin/bash
# GSM8K RVP flywheel (env BASE): base->RFT->RVP pairs->{RVP,xrft,shuf}->pass@1 eval on GSM8K test. GPU0-5.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
G=$HOME/gu; L=$G/logs; V=$G/mrvp; mkdir -p $V; R=$V/RES.md; : >$R; B=${BASE:-Qwen/Qwen2.5-1.5B-Instruct}
echo "# MATH-RVP (gsm8k) base=$B $(date -u)" >>$R
CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode bank --model $B --split train --n 600 --k 8 --out $V/bank.jsonl >$L/mrvp_bank.log 2>&1
echo "bank=$(wc -l <$V/bank.jsonl)" >>$R
CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $B --data $V/bank.jsonl --out $V/rft --seed 1 --max-steps 250 --bsz 8 >$L/mrvp_rft.log 2>&1
python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rft')" >>$L/mrvp_rft.log 2>&1
RFT=$V/rft/merged_full
CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode pairs --model $RFT --split train --n 600 --k 12 --max-pairs-per 2 --out $V/pairs.jsonl >$L/mrvp_pairs.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 GEN_GPU_MEM=0.5 python3 -m rl_training.math_rvp --mode pairs --model $RFT --split train --n 600 --k 12 --max-pairs-per 2 --shuffle --out $V/shuf.jsonl >$L/mrvp_shufgen.log 2>&1 &
wait
python3 -c "import json;seen=set();f=open('$V/pos.jsonl','w')
[f.write(json.dumps({'prompt':r['prompt'],'completion':r['chosen']})+'\n') for r in (json.loads(l) for l in open('$V/pairs.jsonl')) if not (r['chosen'] in seen or seen.add(r['chosen']))]"
echo "pairs=$(wc -l <$V/pairs.jsonl) pos=$(wc -l <$V/pos.jsonl)" >>$R
CUDA_VISIBLE_DEVICES=0 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s1 --seed 1 --max-steps 250 >$L/mrvp_rvp1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s2 --seed 2 --max-steps 250 >$L/mrvp_rvp2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup python3 -m rl_training.sft_train --model $RFT --data $V/pos.jsonl --out $V/xrft --seed 1 --max-steps 250 --bsz 8 >$L/mrvp_xrft.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/shuf.jsonl --out $V/shuf --seed 1 --max-steps 250 >$L/mrvp_shuf.log 2>&1 &
wait
for d in rvp_s1 rvp_s2 xrft shuf; do python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/$d')" >>$L/mrvp_merge.log 2>&1; done
declare -A E=( [base]=$B [rft]=$RFT [rvp_s1]=$V/rvp_s1/merged_full [rvp_s2]=$V/rvp_s2/merged_full [xrft]=$V/xrft/merged_full [shuf]=$V/shuf/merged_full )
g=0; for tag in base rft rvp_s1 rvp_s2 xrft shuf; do
  CUDA_VISIBLE_DEVICES=$g GEN_GPU_MEM=0.45 setsid nohup python3 -m rl_training.math_rvp --mode eval --model "${E[$tag]}" --dataset gsm8k --split test --n 400 --k 8 --out $V/ev_$tag.json >$L/mrvp_ev_$tag.log 2>&1 &
  g=$((g+1)); sleep 2; done
wait
for tag in base rft rvp_s1 rvp_s2 xrft shuf; do python3 -c "import json;d=json.load(open('$V/ev_$tag.json'));print('$tag pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "MATH_RVP_DONE $(date -u)" >>$R
