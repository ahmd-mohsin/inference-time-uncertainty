#!/usr/bin/env bash
# §55 STAGE-3 CONTRIBUTION: CONSOLIDATION-SCHEDULED RL vs baselines, matched TOTAL compute.
# Dynamic method: alternate [GRPO block] with a GAP DIAGNOSTIC (in-dist GSM8K vs OOD MATH-500 proxy);
# when gap>tau, insert an SFT CONSOLIDATION block on verified traces; return to RL. Leave when OOD stabilizes.
# Usage: go_conssched.sh <ARM> <SEED> <GPU>   ARM in {dyn, fixedSR, grpo, sft}
#   dyn     = consolidation-scheduled (the method): GRPO50 -> [gap? SFT50] -> GRPO50 -> [gap? SFT50] -> GRPO50
#   fixedSR = fixed SFT->RL baseline: SFT150 (from base) then GRPO150   (STRONGEST baseline)
#   grpo    = pure GRPO 300
#   sft     = pure SFT 300
# All ~300-step total budget. Eval MATH-500 after (separate). Bank = gu/h2_bank_q3b.jsonl (verified GSM8K).
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim VLLM_GPU_MEM_UTIL=0.35 EVAL_GPU_MEM=0.85
PY=/usr/bin/python3
ARM="${1:-dyn}"; SD="${2:-0}"; GPU="${3:-0}"; M=Qwen/Qwen2.5-3B
GU=/tmp/instance_storage/gu; BANK=$GU/h2_bank_q3b.jsonl; OUT=$GU/cs_${ARM}_s${SD}; L=$GU/logs
MP=$((34600 + GPU*13)); TAU=${TAU:-0.12}
export CUDA_VISIBLE_DEVICES=$GPU
echo "[conssched $ARM s$SD GPU$GPU] $(date)"
grpo(){ local init="$1" ds="$2" steps="$3" out="$4"; local ia=""; [ -n "$init" ] && ia="--init-adapter $init"
  MASTER_PORT=$MP $PY -m rl_training.train_grpo --model $M $ia --dataset $ds --reward-mode math --no-novelty \
    --vllm-mode colocate --num-generations 8 --num-train-steps $steps --max-completion-length 1024 \
    --output-dir $out >>$L/cs_${ARM}_s${SD}.log 2>&1; }
sft(){ local init="$1" steps="$2" out="$3"; local ia=""; [ -n "$init" ] && ia="--init-adapter $init"
  $PY -m rl_training.sft_train --data $BANK --model $M $ia --out $out --max-steps $steps --save-steps $steps --bsz 8 >>$L/cs_${ARM}_s${SD}.log 2>&1; }
gap(){ # in-dist(gsm8k test) - ood(math500) success on a small proxy; echo the gap
  $PY -m rl_training.panel_eval --model-path "$1" --panel test --n 80 --k 2 --tag cs_g_test_${ARM}_s${SD} >>$L/cs_${ARM}_s${SD}.log 2>&1
  $PY -m rl_training.panel_eval --model-path "$1" --panel math --n 80 --k 2 --tag cs_g_math_${ARM}_s${SD} >>$L/cs_${ARM}_s${SD}.log 2>&1
  $PY - <<PYEOF
import json
t=json.load(open("$GU/eval_out/pe_cs_g_test_${ARM}_s${SD}.json"))["mean_p"]
m=json.load(open("$GU/eval_out/pe_cs_g_math_${ARM}_s${SD}.json"))["mean_p"]
print(f"{t-m:.4f}")
PYEOF
}
case "$ARM" in
  grpo)    grpo "" gsm8k 300 $OUT ;;
  sft)     sft "" 300 $OUT ;;
  fixedSR) sft "" 150 ${OUT}_s; grpo ${OUT}_s gsm8k 150 $OUT ;;
  dyn)     # GRPO50 -> gap-gated SFT50 -> GRPO50 -> gap-gated SFT50 -> GRPO... (~300 total)
    cur=""; tot=0
    for blk in 1 2 3; do
      grpo "$cur" gsm8k 50 ${OUT}_g$blk; cur=${OUT}_g$blk; tot=$((tot+50))
      g=$(gap "$cur"); echo "[dyn s$SD] block$blk gap=$g" >>$L/cs_${ARM}_s${SD}.log
      awk "BEGIN{exit !($g > $TAU)}" && { sft "$cur" 50 ${OUT}_c$blk; cur=${OUT}_c$blk; tot=$((tot+50)); }
    done
    cp -r "$cur" "$OUT" 2>/dev/null || ln -s "$cur" "$OUT" ;;
esac
echo "[conssched $ARM s$SD] DONE rc=$? -> $OUT $(date)"
