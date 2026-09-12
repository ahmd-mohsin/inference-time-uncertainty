#!/usr/bin/env bash
# NODE-1 QUEUE (memo W0/E1/Pareto): 7B-hard row {base,GRPO,RFT,VSF,VSF-minus-PG}, eval each on OOD(d14) + in-dist(d12). $HOME paths.
set -uo pipefail
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled
cd $HOME/inference-time-uncertainty; git pull --rebase 2>/dev/null||true
PY=python3; G=$HOME/gu; L=$G/logs; R=$G/RESULTS_q1.md; mkdir -p $L
M=Qwen/Qwen2.5-Coder-7B-Instruct; TR=$G/comp_data/hard_d12tr.jsonl; OOD=$G/comp_data/hard_d14ood.jsonl; BANK=$G/q1_bank.jsonl
[ -f $TR ]  || $PY -m rl_training.comp_tasks --emit C --n 400 --depth 12 --seed0 12000 --out $TR 2>&1|tail -1
[ -f $OOD ] || $PY -m rl_training.comp_tasks --emit C --n 300 --depth 14 --seed0 14500 --out $OOD 2>&1|tail -1
killgpu(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null; sleep 8; }
rec(){ echo "$1" | tee -a $R; }
doeval(){ local dir=$1 tag=$2 pool=$3; killgpu; CUDA_VISIBLE_DEVICES=0 EVAL_ENFORCE_EAGER=1 $PY -m rl_training.comp_eval --model $dir --pool $pool --k 4 --temperature 0.8 --out $G/q1ev_$tag >$L/ev_$tag.log 2>&1; echo "$(grep -oE acc=[0-9.]+ $L/ev_$tag.log|tail -1)"; }
srv(){ local extra="$1" out="$2"; killgpu
  CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -m trl.scripts.vllm_serve --model "$M" --tensor_parallel_size 1 --max_model_len 2560 --gpu_memory_utilization 0.85 --port 8000 >$L/vllm_$(basename $out).log 2>&1 &
  for i in $(seq 1 200); do curl -s localhost:8000/health >/dev/null 2>&1 && break; sleep 3; done
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 $PY -m accelerate.commands.launch --config_file rl_training/accelerate_zero2.yaml --num_processes 7 \
    --main_process_ip 127.0.0.1 --main_process_port 29560 --rdzv_backend c10d -m rl_training.train_grpo --no-novelty \
    --model "$M" --dataset comp:$TR --reward-mode comp --num-train-steps 300 --num-generations 8 --max-completion-length 512 $extra --output-dir $out >$L/train_$(basename $out).log 2>&1
  pkill -9 -f trl.scripts.vllm_serve 2>/dev/null; sleep 6; $PY -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$out')" >$L/mrg_$(basename $out).log 2>&1; }
rec "# NODE1 7B-hard (W0/E1) $(date -u)"; rec "arm | OOD | in-dist"
# bank + base
[ -f $BANK ] || { killgpu; CUDA_VISIBLE_DEVICES=0 GEN_GPU_MEM=0.85 $PY -m rl_training.comp_gen --model $M --pool $TR --k 8 --n 400 --temperature 1.0 --out $BANK >$L/harvest.log 2>&1; }
rec "base | $(doeval $M base_ood $OOD) | $(doeval $M base_id $TR)"
# RFT (LoRA SFT on bank)
killgpu; CUDA_VISIBLE_DEVICES=0 $PY -m rl_training.sft_train --model $M --data $BANK --out $G/q1_rft --seed 1 --max-steps 250 --bsz 4 >$L/sft_rft.log 2>&1; $PY -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$G/q1_rft')">>$L/sft_rft.log 2>&1
rec "RFT | $(doeval $G/q1_rft/merged_full rft_ood $OOD) | $(doeval $G/q1_rft/merged_full rft_id $TR)"
# GRPO
srv "" $G/q1_grpo; rec "GRPO | $(doeval $G/q1_grpo/merged_full grpo_ood $OOD) | $(doeval $G/q1_grpo/merged_full grpo_id $TR)"
# VSF
srv "--vsf-bank $BANK --vsf-lambda 1.0 --vsf-bsz 8" $G/q1_vsf; rec "VSF | $(doeval $G/q1_vsf/merged_full vsf_ood $OOD) | $(doeval $G/q1_vsf/merged_full vsf_id $TR)"
# VSF-minus-PG (W0): replay-only online
srv "--vsf-bank $BANK --vsf-lambda 1.0 --vsf-bsz 8 --vsf-pg-weight 0" $G/q1_vsf0; rec "VSF-noPG | $(doeval $G/q1_vsf0/merged_full vsf0_ood $OOD) | $(doeval $G/q1_vsf0/merged_full vsf0_id $TR)"
rec "=== NODE1 QUEUE DONE $(date -u) ==="
