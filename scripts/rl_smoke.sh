#!/usr/bin/env bash
#
# rl_smoke.sh — tiny end-to-end test on ONE node before any full run.
# Catches import/contract/OOM bugs cheaply (lesson learned: smoke-test before scale).
# 2 problems, 4 generations, 3 steps, short completions. Should finish in minutes.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

MODEL="${1:-Qwen/Qwen3-8B}"
echo "===== SMOKE 1/3: imports + reward unit test ====="
python - <<'PY'
from rl_training.rewards import correctness_reward, make_novelty_bonus, _content, _is_correct
# correctness
comps = [r"ans \boxed{42}", r"ans \boxed{7}"]
print("correctness:", correctness_reward(completions=comps, gold_answer=["42","42"]))
# novelty grouping (same prompt -> one group; both correct -> nonzero novelty if texts differ)
nb = make_novelty_bonus("sentence-transformers/all-MiniLM-L6-v2", lam=0.5)
prompts = ["P","P"]
comps2 = [r"Using algebra, \boxed{42}", r"By geometry instead, \boxed{42}"]
print("novelty:", nb(prompts=prompts, completions=comps2, gold_answer=["42","42"]))
print("SMOKE 1 OK")
PY

echo "===== SMOKE 2/3: GRPO 3 steps (LoRA + vLLM colocate, 1 GPU) ====="
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  -m rl_training.train_grpo --model "$MODEL" --dataset aime_all --n-problems 2 \
  --num-train-steps 3 --num-generations 4 --max-completion-length 1024 \
  --output-dir rl_training/runs/smoke 2>&1 | tail -25

echo "===== SMOKE 3/3: pass@k eval (8 samples, 2 problems) ====="
CUDA_VISIBLE_DEVICES=0 python -m rl_training.evaluate_passk --model-path "$MODEL" \
  --dataset aime_all --n-problems 2 --n-samples 8 --max-new-tokens 1024 \
  --output-dir rl_training/runs/smoke_eval --tag smoke 2>&1 | tail -8
echo "===== SMOKE COMPLETE ====="
