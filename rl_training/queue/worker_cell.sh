#!/bin/bash
# colocate mechanism cell: base|GRPO|RFT|VSF for MODEL at TRD->OODD depth. Writes each acc to RES_$TAG.md immediately. env: MODEL TAG TRD OODD GM
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$HOME/gu/shim
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty; git pull --rebase 2>/dev/null||true
G=$HOME/gu; L=$G/logs; R=$G/RES_$TAG.md; mkdir -p $L; : > $R; echo "# $TAG $(date -u)" >>$R
M=$MODEL; TR=$G/comp_data/w_tr_d${TRD}.jsonl; OOD=$G/comp_data/w_ood_d${OODD}.jsonl; GM=${GM:-0.30}
[ -f $TR ]  || python3 -m rl_training.comp_tasks --emit C --n 400 --depth $TRD --seed0 $((TRD*1000)) --out $TR 2>&1|tail -1
[ -f $OOD ] || python3 -m rl_training.comp_tasks --emit C --n 300 --depth $OODD --seed0 $((OODD*1000+700)) --out $OOD 2>&1|tail -1
ev(){ CUDA_VISIBLE_DEVICES=0 EVAL_ENFORCE_EAGER=1 python3 -m rl_training.comp_eval --model $1 --pool $OOD --k 4 --temperature 0.8 --out $G/wev_${TAG}_$2 >$L/wev_${TAG}_$2.log 2>&1; echo "$2 $(grep -oE acc=[0-9.]+ $L/wev_${TAG}_$2.log|tail -1)" >>$R; }
BANK=$G/wbank_$TAG.jsonl
CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.comp_gen --model $M --pool $TR --k 8 --n 400 --temperature 1.0 --out $BANK >$L/wharv_$TAG.log 2>&1
echo "bank=$(wc -l <$BANK)" >>$R
ev $M base
# RFT
CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.sft_train --model $M --data $BANK --out $G/w_${TAG}_rft --seed 1 --max-steps 250 --bsz 8 >$L/w_${TAG}_rft.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$G/w_${TAG}_rft')">>$L/w_${TAG}_rft.log 2>&1; ev $G/w_${TAG}_rft/merged_full rft
# GRPO
CUDA_VISIBLE_DEVICES=0 VLLM_GPU_MEM_UTIL=$GM MASTER_PORT=29800 python3 -m rl_training.train_grpo --model $M --dataset comp:$TR --reward-mode comp --no-novelty --vllm-mode colocate --num-generations 8 --num-train-steps 300 --max-completion-length 512 --seed 1 --output-dir $G/w_${TAG}_grpo >$L/w_${TAG}_grpo.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$G/w_${TAG}_grpo')">>$L/w_${TAG}_grpo.log 2>&1; ev $G/w_${TAG}_grpo/merged_full grpo
# VSF
CUDA_VISIBLE_DEVICES=0 VLLM_GPU_MEM_UTIL=$GM MASTER_PORT=29801 python3 -m rl_training.train_grpo --model $M --dataset comp:$TR --reward-mode comp --no-novelty --vllm-mode colocate --num-generations 8 --num-train-steps 300 --max-completion-length 512 --vsf-bank $BANK --vsf-lambda 1.0 --vsf-bsz 8 --seed 1 --output-dir $G/w_${TAG}_vsf >$L/w_${TAG}_vsf.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$G/w_${TAG}_vsf')">>$L/w_${TAG}_vsf.log 2>&1; ev $G/w_${TAG}_vsf/merged_full vsf
echo "CELL_DONE $TAG $(date -u)" >>$R
