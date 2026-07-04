#!/usr/bin/env bash
#
# rl_experiment_dp.sh — oursAB (A+B+C) on ONE 8-GPU node with DP-SCALED vLLM generation.
#
# Generation (not gradient compute) is GRPO's bottleneck at 14k-token rollouts. We attack it:
#   GPUs 0-3 : `trl vllm-serve --data_parallel_size 4` (4 replicas, TP=1 each) -> ~4x gen tput
#   GPUs 4-7 : 4 GRPO training ranks (ZeRO-2)
# grad_accum = 14 so effective batch = 4 ranks x bs1 x ga14 = 56 = oursA's (7 x 8). This keeps
# the A-vs-AB comparison clean (same optimization dynamics, just faster wall-clock).
#
# Runs the FULL oursAB arm: 4 segments, each GRPO+novelty -> harvest tail -> SFT (diversity
# injection) -> carry forward. Adapter->full-model merge between stages is handled in-code
# (rl_training/model_utils.merge_adapter_if_needed). Usage:
#   bash scripts/rl_experiment_dp.sh oursAB Qwen/Qwen3-8B aime_all <difficulty.json>
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

ARM="${1:?arm}"; MODEL="${2:?model}"; DATASET="${3:-aime_all}"; DIFF="${4:-}"
STEPS=500; NGEN=8; MAXLEN=14336
ACC=rl_training/accelerate_zero2.yaml
GA=14                                      # 4 train ranks x 14 = 56 = oursA effective batch
RUN="$PWD/rl_training/runs/${ARM}"; EVALDIR="$PWD/rl_training/runs/eval"
mkdir -p "$RUN" "$EVALDIR" ~/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">> pre-fetching $MODEL ..."; python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" 2>&1 | tail -1
python -c "from datasets import load_dataset; [load_dataset(d) for d in ['math-ai/aime24','math-ai/aime25','math-ai/aime26']]" 2>&1 | tail -1
export HF_HUB_OFFLINE=1

run_eval () { python -m rl_training.evaluate_passk --model-path "$1" --dataset "$DATASET" \
    --n-samples 32 --n-problems -1 --max-new-tokens 8192 --tensor-parallel-size 1 \
    --output-dir "$EVALDIR" --tag "$2"; }

VLLM_PID=""
start_vllm () {  # $1 = model path/id ; DP=4 on GPUs 0-3
  echo ">> starting vLLM DP=4 (GPUs 0-3, TP=1, 16k) model=$1 ..."
  CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 trl vllm-serve --model "$1" \
    --tensor_parallel_size 1 --data_parallel_size 4 --max_model_len 16384 \
    --gpu_memory_utilization 0.9 --port 8000 > ~/logs/vllm_${ARM}.log 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 180); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo ">> vLLM up (DP=4)"; return 0; }
    sleep 5
  done
  echo ">> vLLM FAILED to come up"; return 1
}
stop_vllm () { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null; pkill -9 -f "trl vllm-serve" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 4; }
trap stop_vllm EXIT

train_launch () {  # GRPO on GPUs 4-7 (4 ranks), grad_accum=GA to hold effective batch
  CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file "$ACC" \
    --num_processes 4 --num_machines 1 -m rl_training.train_grpo "$@" \
    --gradient-accumulation-steps "$GA"
}

# ---- FULL oursAB: 4 segments, GRPO+novelty -> harvest -> SFT -> carry forward ----
SEG=$((STEPS/4)); CUR="$MODEL"
for r in 0 1 2 3; do
  echo "===== segment $r: GRPO+novelty $SEG steps (model=$CUR) ====="
  start_vllm "$CUR" || exit 1
  if ! train_launch --model "$CUR" --dataset "$DATASET" \
    --n-problems -1 --num-train-steps "$SEG" --num-generations "$NGEN" \
    --max-completion-length "$MAXLEN" \
    --output-dir "$RUN/seg$r" ${DIFF:+--difficulty-json "$DIFF"} --novelty-lambda 0.5; then
    stop_vllm; echo "!! segment $r GRPO FAILED — aborting"; exit 1
  fi
  stop_vllm
  echo "===== segment $r: harvest tail + diversity-injection SFT ====="
  # harvest samples the current segment's model with its OWN DP-scaled vLLM (in-process, TP=1)
  python -m rl_training.harvest --mode harvest --model-path "$RUN/seg$r" \
    --dataset "$DATASET" ${DIFF:+--difficulty-json "$DIFF"} --k 64 --max-keep 2 \
    --max-new-tokens "$MAXLEN" --tensor-parallel-size 1 --out-jsonl "$RUN/harvest$r.jsonl"
  if [ -s "$RUN/harvest$r.jsonl" ]; then
    python -m rl_training.harvest --mode sft --model-path "$RUN/seg$r" \
      --out-jsonl "$RUN/harvest$r.jsonl" --output-dir "$RUN/seg${r}_sft" --epochs 1 \
      && CUR="$RUN/seg${r}_sft" || CUR="$RUN/seg$r"
  else
    echo "-- no harvest rollouts for seg$r; carrying GRPO checkpoint forward"
    CUR="$RUN/seg$r"
  fi
done
echo "===== oursAB training DONE (final model: $CUR) — running pass@k eval ====="
run_eval "$CUR" oursAB
echo "===== ARM oursAB DONE ====="
touch ~/RL_oursAB_DONE
