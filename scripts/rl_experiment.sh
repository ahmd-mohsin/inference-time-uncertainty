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
MAXLEN=14336                               # generation budget inside 16k context window
ACC=rl_training/accelerate_zero3.yaml      # ZeRO-3 across the node's 8 GPUs (no offload needed)
RUN=rl_training/runs/${ARM}
EVALDIR=rl_training/runs/eval
mkdir -p "$RUN" "$EVALDIR" ~/logs
# fight long-context fragmentation OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pre-download the model ONCE (single process) so the 8 ZeRO-3 ranks don't race the HF
# cache and hit "missing shard" OSErrors. No-op if already cached.
echo ">> pre-fetching $MODEL into HF cache (avoids multi-rank download race)..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" 2>&1 | tail -1
# after the cache is complete, force offline so the 8 ranks never re-check the hub (race)
export HF_HUB_OFFLINE=1

run_eval () {  # $1 = model path/dir, $2 = tag
  python -m rl_training.evaluate_passk --model-path "$1" --dataset "$DATASET" \
    --n-samples 256 --n-problems "$NPROB" --max-new-tokens "$MAXLEN" \
    --output-dir "$EVALDIR" --tag "$2"
}

# ---- vLLM SERVER mode: dedicate 1 GPU to a generation server (TP=1), train on the other 7.
# 16k-context 8B training + generation does not fit COLOCATE on one 40GB card, so we split.
# TP=1 (single GPU) avoids the custom_all_reduce CUDA error that TP=2 hits in this container;
# an 8B model + 16k KV fits on one 40GB A100 at 0.9 util.
VLLM_GPUS="0"; TRAIN_GPUS="1,2,3,4,5,6,7"; NTRAIN=7; VLLM_PID=""
start_vllm () {  # $1 = model path/id
  echo ">> starting vLLM server on GPU $VLLM_GPUS (TP=1, 16k) ..."
  CUDA_VISIBLE_DEVICES=$VLLM_GPUS HF_HUB_OFFLINE=1 trl vllm-serve --model "$1" \
    --tensor_parallel_size 1 --max_model_len 16384 --gpu_memory_utilization 0.9 \
    --port 8000 > ~/logs/vllm_${ARM}.log 2>&1 &
  VLLM_PID=$!
  # wait until the server answers (up to ~10 min for load+graph capture)
  for _ in $(seq 1 120); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo ">> vLLM server up"; return 0; }
    sleep 5
  done
  echo ">> vLLM server FAILED to come up"; return 1
}
stop_vllm () { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap stop_vllm EXIT
# training uses TRAIN_GPUS (6 procs); vLLM is reached over HTTP (server mode in GRPOConfig)
train_launch () {  # passes through all args to accelerate/train_grpo on the training GPUs
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch --config_file "$ACC" \
    --num_processes $NTRAIN --num_machines 1 -m rl_training.train_grpo "$@"
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
    start_vllm "$MODEL" || exit 1
    train_launch --model "$MODEL" --dataset "$DATASET" \
      --n-problems "$NPROB" --num-train-steps "$STEPS" --num-generations "$NGEN" \
      --max-completion-length "$MAXLEN" \
      --output-dir "$RUN" ${USE_DIFF:+--difficulty-json "$USE_DIFF"} $NOV
    stop_vllm
    run_eval "$RUN" "$ARM"
    ;;
  oursAB)
    echo "===== ARM oursAB: alternating GRPO + harvest (Comp A+B) ====="
    SEG=$((STEPS/4)); CUR="$MODEL"
    for r in 0 1 2 3; do
      echo "--- segment $r: GRPO $SEG steps ---"
      start_vllm "$CUR" || exit 1
      train_launch --model "$CUR" --dataset "$DATASET" \
        --n-problems "$NPROB" --num-train-steps "$SEG" --num-generations "$NGEN" \
        --max-completion-length "$MAXLEN" \
        --output-dir "$RUN/seg$r" ${DIFF:+--difficulty-json "$DIFF"} --novelty-lambda 0.5
      stop_vllm
      echo "--- segment $r: harvest tail + SFT ---"
      python -m rl_training.harvest --mode harvest --model-path "$RUN/seg$r" \
        --dataset "$DATASET" ${DIFF:+--difficulty-json "$DIFF"} --k 64 --max-keep 2 \
        --max-new-tokens "$MAXLEN" --out-jsonl "$RUN/harvest$r.jsonl"
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
