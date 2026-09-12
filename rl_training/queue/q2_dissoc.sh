#!/usr/bin/env bash
# NODE-2 QUEUE (memo X-series): erosion vs non-acquisition at 1.5B-hard, colocate. Arms: base, RFT, GRPO,
# X6 zeroneg, X3 KL-leash, X1 disjoint-replay(VSF w/ unrelated bank), X9 prior-support(RFT->GRPO). Eval OOD(d14)+in-dist(d12).
set -uo pipefail
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim
cd $HOME/inference-time-uncertainty; git pull --rebase 2>/dev/null||true
PY=python3; G=$HOME/gu; L=$G/logs; R=$G/RESULTS_q2.md; mkdir -p $L
M=Qwen/Qwen2.5-Coder-1.5B-Instruct; TR=$G/comp_data/hard_d12tr.jsonl; OOD=$G/comp_data/hard_d14ood.jsonl
DISJ=$G/comp_data/disjoint_d12.jsonl; BANK=$G/q2_bank.jsonl; DBANK=$G/q2_disjbank.jsonl
[ -f $TR ]  || $PY -m rl_training.comp_tasks --emit C --n 400 --depth 12 --seed0 12000 --out $TR 2>&1|tail -1
[ -f $OOD ] || $PY -m rl_training.comp_tasks --emit C --n 300 --depth 14 --seed0 14500 --out $OOD 2>&1|tail -1
[ -f $DISJ ] || $PY -m rl_training.comp_tasks --emit C --n 400 --depth 12 --seed0 88000 --out $DISJ 2>&1|tail -1   # disjoint prompts (diff seeds)
rec(){ echo "$1" | tee -a $R; }
ev(){ local dir=$1 g=$2 pool=$3 tag=$4; CUDA_VISIBLE_DEVICES=$g python3 -m rl_training.comp_eval --model $dir --pool $pool --k 4 --temperature 0.8 --out $G/q2ev_$tag >$L/q2ev_$tag.log 2>&1; grep -oE acc=[0-9.]+ $L/q2ev_$tag.log|tail -1; }
gtrain(){ local g=$1 out=$2 extra="$3"; CUDA_VISIBLE_DEVICES=$g VLLM_GPU_MEM_UTIL=0.30 MASTER_PORT=$((29700+g)) python3 -m rl_training.train_grpo --model $M --dataset comp:$TR --reward-mode comp --no-novelty --vllm-mode colocate --num-generations 8 --num-train-steps 300 --max-completion-length 512 --seed 1 $extra --output-dir $out >$L/q2_$(basename $out).log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$out')">>$L/q2_$(basename $out).log 2>&1; }
rec "# NODE2 1.5B-hard dissociation $(date -u)"; rec "arm | OOD | in-dist"
# harvest banks (GPU0): same-pool + disjoint-pool
CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.comp_gen --model $M --pool $TR --k 8 --n 400 --temperature 1.0 --out $BANK >$L/q2harvest.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 -m rl_training.comp_gen --model $M --pool $DISJ --k 8 --n 400 --temperature 1.0 --out $DBANK >$L/q2harvestd.log 2>&1
rec "base | $(ev $M 0 $OOD base_ood) | $(ev $M 0 $TR base_id)"
# RFT (needed for X9) on GPU1
CUDA_VISIBLE_DEVICES=1 python3 -m rl_training.sft_train --model $M --data $BANK --out $G/q2_rft --seed 1 --max-steps 250 --bsz 8 >$L/q2_rft.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$G/q2_rft')">>$L/q2_rft.log 2>&1
# parallel arms on GPUs 2-6
gtrain 2 $G/q2_grpo "" & gtrain 3 $G/q2_zeroneg "--adv-transform zeroneg" & gtrain 4 $G/q2_kl "--beta 0.2" & gtrain 5 $G/q2_disjvsf "--vsf-bank $DBANK --vsf-lambda 1.0 --vsf-bsz 8" & wait
# X9 prior-support: GRPO warm-started from RFT adapter (GPU2)
gtrain 2 $G/q2_priorsup "--init-adapter $G/q2_rft"
rec "RFT | $(ev $G/q2_rft/merged_full 1 $OOD rft_ood) | $(ev $G/q2_rft/merged_full 1 $TR rft_id)"
rec "GRPO | $(ev $G/q2_grpo/merged_full 2 $OOD grpo_ood) | $(ev $G/q2_grpo/merged_full 2 $TR grpo_id)"
rec "X6-zeroneg | $(ev $G/q2_zeroneg/merged_full 3 $OOD zn_ood) | $(ev $G/q2_zeroneg/merged_full 3 $TR zn_id)"
rec "X3-KL0.2 | $(ev $G/q2_kl/merged_full 4 $OOD kl_ood) | $(ev $G/q2_kl/merged_full 4 $TR kl_id)"
rec "X1-disjointVSF | $(ev $G/q2_disjvsf/merged_full 5 $OOD dj_ood) | $(ev $G/q2_disjvsf/merged_full 5 $TR dj_id)"
rec "X9-priorsupport | $(ev $G/q2_priorsup/merged_full 2 $OOD ps_ood) | $(ev $G/q2_priorsup/merged_full 2 $TR ps_id)"
rec "=== NODE2 QUEUE DONE $(date -u) ==="
