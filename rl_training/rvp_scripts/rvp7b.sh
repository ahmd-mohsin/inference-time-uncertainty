#!/bin/bash
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty; G=$HOME/gu; L=$G/logs; V=$G/rvp_q7mid; R=$V/RES7brvp.md; : >$R
RFT=$V/rft/merged_full; OOD=$(ls $G/comp_data/w_ood_d9.jsonl|head -1)
CUDA_VISIBLE_DEVICES=0 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s1 --seed 1 --bsz 1 --max-steps 250 >$L/rvp7b_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp_s2 --seed 2 --bsz 1 --max-steps 250 >$L/rvp7b_s2.log 2>&1 &
wait
for d in rvp_s1 rvp_s2; do python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/$d')" >>$L/rvp7b_merge.log 2>&1; done
i=0; for tag in rvp_s1 rvp_s2; do CUDA_VISIBLE_DEVICES=$i EVAL_GPU_MEM=0.45 setsid nohup python3 -m rl_training.comp_eval --model $V/$tag/merged_full --pool $OOD --n 200 --k 16 --seed 1 --temperature 0.8 --tag q7rvp_$tag --out-dir $V/ev >$L/rvp7b_ev_$tag.log 2>&1 & i=$((i+1)); done
wait
for tag in rvp_s1 rvp_s2; do python3 -c "import json;d=json.load(open('$V/ev/comp_q7rvp_$tag.json'));print('7B-$tag pass1=%.4f cov16=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null; done
echo "RVP7B_DONE $(date -u)" >>$R
