#!/bin/bash
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER PYTHONPATH=$HOME/gu/shim
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty; G=$HOME/gu; OUT=$G/cct/reliab; mkdir -p $OUT
POOL=$(ls $G/comp_data/w_ood_d14.jsonl 2>/dev/null|head -1)
declare -A M=( [base]=Qwen/Qwen2.5-Coder-3B-Instruct [grpo1]=$G/m_c3hard_grpo_s1/merged_full [grpo2]=$G/m_c3hard_grpo_s2/merged_full [grpo3]=$G/m_c3hard_grpo_s3/merged_full [rft1]=$G/m_c3hard_rft_s1/merged_full [rft2]=$G/m_c3hard_rft_s2/merged_full [rft3]=$G/m_c3hard_rft_s3/merged_full [vsf1]=$G/m_c3hard_vsf_s1/merged_full )
gpu=0
for tag in base grpo1 grpo2 grpo3 rft1 rft2 rft3 vsf1; do
  CUDA_VISIBLE_DEVICES=$gpu EVAL_GPU_MEM=0.45 setsid nohup python3 -m rl_training.comp_eval --model "${M[$tag]}" --pool $POOL --n 200 --k 16 --seed 1 --temperature 0.8 --tag rl_$tag --out-dir $OUT >$G/logs/reliab_$tag.log 2>&1 &
  gpu=$((gpu+1)); sleep 2
done
echo "RELIAB_LAUNCHED 8 evals $(date -u)"
