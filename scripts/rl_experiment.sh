#!/usr/bin/env bash
#
# rl_experiment.sh — Run ONE RL-expansion experiment arm on ONE node (8 GPUs).
# docs/RL.md. With 4 nodes, launch 4 arms in parallel (see rl_all_arms.sh):
#   arm=base   : no training; just eval the base model's pass@k curve (the control)
#   arm=grpo   : plain GRPO (novelty off)        -> should reproduce Yue crossover
#   arm=oursA  : GRPO + novelty reward (Comp A)
#   arm=oursAB : GRPO + novelty + off-policy harvest loop (Comp A+B)
#
# Usage: bash scripts/rl_experiment.sh <arm> <model> <dataset> [difficulty_json]
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

ARM="${1:?arm: base|grpo|oursA|oursAB}"
MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${3:-aime_all}"
DIFF="${4:-}"                              # difficulty.json for Component C (optional)
NPROB=-1
STEPS=500
NGEN=8
ACC=rl_training/accelerate_zero3.yaml
RUN=rl_training/runs/${ARM}
EVALDIR=rl_training/runs/eval
mkdir -p "$RUN" "$EVALDIR" ~/logs

# Pre-download the model ONCE (single process) so the 8 ZeRO-3 ranks don't race the HF
# cache and hit "missing shard" OSErrors. No-op if already cached.
echo ">> pre-fetching $MODEL into HF cache (avoids multi-rank download race)..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" 2>&1 | tail -1
# after the cache is complete, force offline so the 8 ranks never re-check the hub (race)
export HF_HUB_OFFLINE=1

run_eval () {  # $1 = model path/dir, $2 = tag
  python -m rl_training.evaluate_passk --model-path "$1" --dataset "$DATASET" \
    --n-samples 256 --n-problems "$NPROB" --output-dir "$EVALDIR" --tag "$2"
}

case "$ARM" in
  base)
    echo "===== ARM base: eval only ====="
    run_eval "$MODEL" base
    ;;
  grpo|oursA)
    # Clean ablation: grpo = standard GRPO on FULL data (no novelty, no C targeting) so it
    # is a true control for Yue's crossover. oursA = novelty (A) + hard-targeting (C).
    if [ "$ARM" = "grpo" ]; then NOV="--no-novelty"; USE_DIFF=""; else NOV="--novelty-lambda 0.5"; USE_DIFF="$DIFF"; fi
    echo "===== ARM $ARM: GRPO train ($STEPS steps; C=${USE_DIFF:-off}) ====="
    accelerate launch --config_file "$ACC" --num_processes 8 --num_machines 1 \
      -m rl_training.train_grpo --model "$MODEL" --dataset "$DATASET" \
      --n-problems "$NPROB" --num-train-steps "$STEPS" --num-generations "$NGEN" \
      --output-dir "$RUN" ${USE_DIFF:+--difficulty-json "$USE_DIFF"} $NOV
    run_eval "$RUN" "$ARM"
    ;;
  oursAB)
    echo "===== ARM oursAB: alternating GRPO + harvest (Comp A+B) ====="
    SEG=$((STEPS/4)); CUR="$MODEL"
    for r in 0 1 2 3; do
      echo "--- segment $r: GRPO $SEG steps ---"
      accelerate launch --config_file "$ACC" --num_processes 8 --num_machines 1 \
        -m rl_training.train_grpo --model "$CUR" --dataset "$DATASET" \
        --n-problems "$NPROB" --num-train-steps "$SEG" --num-generations "$NGEN" \
        --output-dir "$RUN/seg$r" ${DIFF:+--difficulty-json "$DIFF"} --novelty-lambda 0.5
      echo "--- segment $r: harvest tail + SFT ---"
      python -m rl_training.harvest --mode harvest --model-path "$RUN/seg$r" \
        --dataset "$DATASET" ${DIFF:+--difficulty-json "$DIFF"} --k 64 --max-keep 2 \
        --out-jsonl "$RUN/harvest$r.jsonl"
      python -m rl_training.harvest --mode sft --model-path "$RUN/seg$r" \
        --out-jsonl "$RUN/harvest$r.jsonl" --output-dir "$RUN/seg${r}_sft" --epochs 1
      CUR="$RUN/seg${r}_sft"
    done
    run_eval "$CUR" oursAB
    ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
echo "===== ARM $ARM DONE ====="
touch ~/RL_${ARM}_DONE
