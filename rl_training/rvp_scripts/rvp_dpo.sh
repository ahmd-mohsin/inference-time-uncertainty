#!/bin/bash
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty; G=$HOME/gu; L=$G/logs; V=$G/rvp; R=$V/RVPRES2.md; : >$R
RFT=$G/m_c3hard_rft_s1/merged_full; OOD=$(ls $G/comp_data/w_ood_d14.jsonl|head -1)
echo "# RVP-DPO (fixed) $(date -u) pairs=$(wc -l <$V/pairs.jsonl) shuf=$(wc -l <$V/shuf.jsonl)" >>$R
CUDA_VISIBLE_DEVICES=0 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s1 --seed 1 --max-steps 250 >$L/rvp2_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s2 --seed 2 --max-steps 250 >$L/rvp2_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/shuf.jsonl --out $V/shuf_s1 --seed 1 --max-steps 250 >$L/rvp2_shuf.log 2>&1 &
wait
for d in rvp_s1 rvp_s2 shuf_s1; do python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/$d')" >>$L/rvp2_merge.log 2>&1; done
i=0; for tag in rvp_s1 rvp_s2 shuf_s1; do
  CUDA_VISIBLE_DEVICES=$i EVAL_GPU_MEM=0.45 setsid nohup python3 -m rl_training.comp_eval --model $V/$tag/merged_full --pool $OOD --n 200 --k 16 --seed 1 --temperature 0.8 --tag rvp2_$tag --out-dir $V/ev >$L/rvp2_ev_$tag.log 2>&1 &
  i=$((i+1)); sleep 2; done
wait
for tag in rvp_s1 rvp_s2 shuf_s1; do python3 -c "import json;d=json.load(open('$V/ev/comp_rvp2_$tag.json'));print('$tag pass1=%.4f cov16=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "RVP2_DONE $(date -u)" >>$R
