#!/bin/bash
# Full RVP flywheel for a 2nd family from scratch (env: BASE, FAM). pools->RFT->RVP pairs->{RVP,xrft,shuf}->pass@1 eval.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
G=$HOME/gu; L=$G/logs; V=$G/rvp_${FAM}; mkdir -p $V $G/comp_data $L; R=$V/RES.md; : >$R
echo "# RVP family=$FAM base=$BASE $(date -u)" >>$R
TRD=${TRD:-7}; OODD=${OODD:-9}; TR=$G/comp_data/w_tr_d${TRD}.jsonl; OOD=$G/comp_data/w_ood_d${OODD}.jsonl
[ -f $TR ]  || python3 -m rl_training.comp_tasks --emit C --n 400 --depth ${TRD} --seed0 $((TRD*1000)) --out $TR >>$L/rvpfam_setup.log 2>&1
[ -f $OOD ] || python3 -m rl_training.comp_tasks --emit C --n 300 --depth ${OODD} --seed0 $((OODD*1000+700)) --out $OOD >>$L/rvpfam_setup.log 2>&1
# 1) RFT: harvest verified bank from base, sft_train base->RFT
CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.comp_gen --model $BASE --pool $TR --k 8 --n 400 --temperature 1.0 --out $V/wbank.jsonl >$L/rvpfam_harv.log 2>&1
echo "bank=$(wc -l <$V/wbank.jsonl)" >>$R
CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $BASE --data $V/wbank.jsonl --out $V/rft --seed 1 --max-steps 250 --bsz 8 >$L/rvpfam_rft.log 2>&1
python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rft')" >>$L/rvpfam_rft.log 2>&1
RFT=$V/rft/merged_full
# 2) RVP pairs from RFT + positives + shuffled
CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.rvp_gen --model $RFT --pool $TR --n 400 --k 12 --max-pairs-per 2 --out $V/pairs.jsonl >$L/rvpfam_pairs.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 GEN_GPU_MEM=0.5 python3 -m rl_training.rvp_gen --model $RFT --pool $TR --n 400 --k 12 --max-pairs-per 2 --shuffle --out $V/shuf.jsonl >$L/rvpfam_shufgen.log 2>&1 &
wait
python3 -c "import json;seen=set();f=open('$V/pos.jsonl','w')
[f.write(json.dumps({'prompt':r['prompt'],'completion':r['chosen']})+'\n') for r in (json.loads(l) for l in open('$V/pairs.jsonl')) if not (r['chosen'] in seen or seen.add(r['chosen']))]"
echo "pairs=$(wc -l <$V/pairs.jsonl) pos=$(wc -l <$V/pos.jsonl)" >>$R
# 3) arms from RFT
CUDA_VISIBLE_DEVICES=0 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s1 --seed 1 --bsz ${DPO_BSZ:-4} --max-steps 250 >$L/rvpfam_rvp1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s2 --seed 2 --bsz ${DPO_BSZ:-4} --max-steps 250 >$L/rvpfam_rvp2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup python3 -m rl_training.sft_train --model $RFT --data $V/pos.jsonl --out $V/xrft_s1 --seed 1 --max-steps 250 --bsz 8 >$L/rvpfam_xrft.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/shuf.jsonl --out $V/shuf_s1 --seed 1 --bsz ${DPO_BSZ:-4} --max-steps 250 >$L/rvpfam_shuf.log 2>&1 &
wait
for d in rvp_s1 rvp_s2 xrft_s1 shuf_s1; do python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/$d')" >>$L/rvpfam_merge.log 2>&1; done
# 4) pass@1 eval: base, rft, arms
declare -A E=( [base]=$BASE [rft]=$RFT [rvp_s1]=$V/rvp_s1/merged_full [rvp_s2]=$V/rvp_s2/merged_full [xrft]=$V/xrft_s1/merged_full [shuf]=$V/shuf_s1/merged_full )
gpu=0; for tag in base rft rvp_s1 rvp_s2 xrft shuf; do
  CUDA_VISIBLE_DEVICES=$gpu EVAL_GPU_MEM=0.45 setsid nohup python3 -m rl_training.comp_eval --model "${E[$tag]}" --pool $OOD --n 200 --k 16 --seed 1 --temperature 0.8 --tag fam_$tag --out-dir $V/ev >$L/rvpfam_ev_$tag.log 2>&1 &
  gpu=$((gpu+1)); sleep 2; done
wait
for tag in base rft rvp_s1 rvp_s2 xrft shuf; do python3 -c "import json;d=json.load(open('$V/ev/comp_fam_$tag.json'));print('$tag pass1=%.4f cov16=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "RVPFAM_DONE $FAM $(date -u)" >>$R
