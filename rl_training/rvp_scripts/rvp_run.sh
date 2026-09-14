#!/bin/bash
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty; G=$HOME/gu; L=$G/logs; V=$G/rvp; mkdir -p $V; R=$V/RVPRES.md; : >$R
RFT=$G/m_c3hard_rft_s1/merged_full; TR=$(ls $G/comp_data/w_tr_d12.jsonl|head -1); OOD=$(ls $G/comp_data/w_ood_d14.jsonl|head -1)
echo "# RVP  rft=$RFT tr=$TR ood=$OOD $(date -u)" >>$R
# 1) generate verified pairs (GPU0) + shuffled control (GPU1)
CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.5 python3 -m rl_training.rvp_gen --model $RFT --pool $TR --n 400 --k 12 --max-pairs-per 2 --out $V/pairs.jsonl >$L/rvp_gen.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 GEN_GPU_MEM=0.5 python3 -m rl_training.rvp_gen --model $RFT --pool $TR --n 400 --k 12 --max-pairs-per 2 --shuffle --out $V/shuf.jsonl >$L/rvp_shufgen.log 2>&1 &
wait
echo "pairs=$(wc -l <$V/pairs.jsonl) shuf=$(wc -l <$V/shuf.jsonl)" >>$R
# positives-only bank for extra-RFT arm (chosen completions)
python3 -c "import json;seen=set();f=open('$V/pos.jsonl','w')
[f.write(json.dumps({'prompt':r['prompt'],'completion':r['chosen']})+'\n') for r in (json.loads(l) for l in open('$V/pairs.jsonl')) if not (r['chosen'] in seen or seen.add(r['chosen']))]"
echo "pos=$(wc -l <$V/pos.jsonl)" >>$R
# 2) train arms: RVP seeds1,2 (GPU0,1); shuffled (GPU2); extra-RFT (GPU3)
CUDA_VISIBLE_DEVICES=0 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s1 --seed 1 --max-steps 250 >$L/rvp_tr_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s2 --seed 2 --max-steps 250 >$L/rvp_tr_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/shuf.jsonl --out $V/shuf_s1 --seed 1 --max-steps 250 >$L/rvp_shuf.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup python3 -m rl_training.sft_train --model $RFT --data $V/pos.jsonl --out $V/xrft_s1 --seed 1 --max-steps 250 --bsz 8 >$L/rvp_xrft.log 2>&1 &
wait
# 3) merge + pass@1 eval (GPU0-3) vs RFT baseline
for d in rvp_s1 rvp_s2 shuf_s1 xrft_s1; do python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/$d')" >>$L/rvp_merge.log 2>&1; done
i=0; for tag in rvp_s1 rvp_s2 shuf_s1 xrft_s1; do
  CUDA_VISIBLE_DEVICES=$i EVAL_GPU_MEM=0.45 setsid nohup python3 -m rl_training.comp_eval --model $V/$tag/merged_full --pool $OOD --n 200 --k 16 --seed 1 --temperature 0.8 --tag rvp_$tag --out-dir $V/ev >$L/rvp_ev_$tag.log 2>&1 &
  i=$((i+1)); sleep 2; done
# also re-eval RFT baseline for matched comparison
CUDA_VISIBLE_DEVICES=4 EVAL_GPU_MEM=0.45 setsid nohup python3 -m rl_training.comp_eval --model $RFT --pool $OOD --n 200 --k 16 --seed 1 --temperature 0.8 --tag rvp_rftbase --out-dir $V/ev >$L/rvp_ev_rft.log 2>&1 &
wait
for tag in rftbase rvp_s1 rvp_s2 shuf_s1 xrft_s1; do
  python3 -c "import json;d=json.load(open('$V/ev/comp_rvp_$tag.json'));print('$tag pass1=%.4f cov16=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null
done
echo "RVP_DONE $(date -u)" >>$R
