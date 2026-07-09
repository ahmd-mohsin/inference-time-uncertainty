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
STEPS="${RL_STEPS:-500}"
NGEN=8
MAXLEN="${RL_MAXLEN:-14336}"              # generation budget inside context window
# --- eval knobs (env-overridable; defaults preserve the 8B/AIME behavior) ---------------
# For the 1.5B/MATH-500 study we push to k=256, shorter solutions (4k), on the fast model.
EVAL_NSAMPLES="${RL_EVAL_NSAMPLES:-32}"
EVAL_MAXNEW="${RL_EVAL_MAXNEW:-8192}"
CTXLEN="${RL_CTXLEN:-16384}"              # vLLM max_model_len (1.5B is 4k-friendly)
# datasets to prefetch while online (AIME for 8B; math500 for the 1.5B study)
PREFETCH_DATASETS="${RL_PREFETCH:-math-ai/aime24,math-ai/aime25,math-ai/aime26}"
ACC=rl_training/accelerate_zero2.yaml      # ZeRO-2: shard optimizer+grads, keep params whole.
                                           # ZeRO-3 partitioned params and broke gradient-
                                           # checkpoint recompute (CheckpointError) with LoRA.
# ABSOLUTE paths: under HF_HUB_OFFLINE=1 a *relative* checkpoint dir (e.g. rl_training/runs/grpo)
# is misread by transformers/vLLM as a HF repo id -> "Invalid repository ID". Absolute dirs
# resolve as local model paths for both the trainer-resume and the pass@k eval.
RUN="$PWD/rl_training/runs/${ARM}"
EVALDIR="$PWD/rl_training/runs/eval"
mkdir -p "$RUN" "$EVALDIR" ~/logs
# fight long-context fragmentation OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pre-download the model ONCE (single process) so the 8 ZeRO-3 ranks don't race the HF
# cache and hit "missing shard" OSErrors. No-op if already cached.
echo ">> pre-fetching $MODEL into HF cache (avoids multi-rank download race)..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" 2>&1 | tail -1
# also pre-fetch the datasets (the load_dataset builder cache lives under ~/.cache/huggingface/
# datasets/, NOT hub/ — so a hub-only cache copy is insufficient and the offline ranks would
# hit 'OfflineModeIsEnabled'). Populate it once here while still online.
echo ">> pre-fetching datasets into cache ($PREFETCH_DATASETS)..."
python -c "from datasets import load_dataset; [load_dataset(d) for d in '$PREFETCH_DATASETS'.split(',') if d]" 2>&1 | tail -1
# after the cache is complete, force offline so the 8 ranks never re-check the hub (race)
export HF_HUB_OFFLINE=1

run_eval () {  # $1 = model path/dir, $2 = tag
  # 256 samples x 90 problems x 16k on 1 GPU = ~5 DAYS (measured). Make tractable.
  # NOTE: TP>1 fails in this container (EngineCore shm_broadcast init error — same root cause
  # as the custom_all_reduce issue that forced TP=1 for training), so we CANNOT parallelize
  # across GPUs. Instead cut the workload: 32 samples still gives pass@{1,2,4,8,16,32} (enough
  # to see the crossover), 8k tokens covers AIME solutions. ~32x less work than the original.
  python -m rl_training.evaluate_passk --model-path "$1" --dataset "$DATASET" \
    --n-samples "$EVAL_NSAMPLES" --n-problems "$NPROB" --max-new-tokens "$EVAL_MAXNEW" \
    --tensor-parallel-size 1 \
    --output-dir "$EVALDIR" --tag "$2"
}

# ---- vLLM SERVER mode: dedicate 1 GPU to a generation server (TP=1), train on the other 7.
# 16k-context 8B training + generation does not fit COLOCATE on one 40GB card, so we split.
# TP=1 (single GPU) avoids the custom_all_reduce CUDA error that TP=2 hits in this container;
# an 8B model + 16k KV fits on one 40GB A100 at 0.9 util.
VLLM_GPUS="0"; TRAIN_GPUS="1,2,3,4,5,6,7"; NTRAIN=7; VLLM_PID=""
start_vllm () {  # $1 = model path/id
  # vLLM cannot serve a bare LoRA adapter dir (e.g. a prior segment's seg${r}_sft output) — it
  # needs a full model with config.json. Merge adapter->base first if needed (idempotent).
  local M="$1"
  if [ -f "$M/adapter_config.json" ] && [ ! -f "$M/config.json" ]; then
    echo ">> vLLM model is a bare adapter; merging into base ..."
    M=$(CUDA_VISIBLE_DEVICES=$VLLM_GPUS python -c "from rl_training.model_utils import merge_adapter_if_needed as m; print(m('$M'))" 2>/dev/null | tail -1)
    echo ">> serving merged model: $M"
  fi
  echo ">> starting vLLM server on GPU $VLLM_GPUS (TP=1, 16k) ..."
  CUDA_VISIBLE_DEVICES=$VLLM_GPUS HF_HUB_OFFLINE=1 trl vllm-serve --model "$M" \
    --tensor_parallel_size 1 --max_model_len "$CTXLEN" --gpu_memory_utilization 0.9 \
    --port 8000 > ~/logs/vllm_${ARM}.log 2>&1 &
  VLLM_PID=$!
  # wait until the server answers (up to ~10 min for load+graph capture)
  for _ in $(seq 1 120); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo ">> vLLM server up"; return 0; }
    sleep 5
  done
  echo ">> vLLM server FAILED to come up"; return 1
}
stop_vllm () {
  # Kill the vLLM server AND its EngineCore worker(s), then WAIT for GPU0 memory to actually
  # release. Without the wait, the next stage (harvest's own vLLM) launches while GPU0 is still
  # occupied and OOMs ("Free memory on device cuda:0 ... less than desired GPU memory utilization").
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
  pkill -9 -f "trl vllm-serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null || true
  for _ in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    [ "${u:-99999}" -lt 2000 ] 2>/dev/null && break
    sleep 3
  done
  VLLM_PID=""
}
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
  grpo|oursA|oursAB_cont)
    # Clean ablation: grpo = standard GRPO on FULL data (no novelty, no C targeting) so it
    # is a true control for Yue's crossover. oursA = novelty (A) + hard-targeting (C).
    # oursAB_cont = oursAB continued as plain GRPO+novelty from its seg0 checkpoint (the
    # alternating harvest loop was too fragile under time pressure; this keeps Component A).
    if [ "$ARM" = "grpo" ]; then NOV="--no-novelty"; USE_DIFF=""; else NOV="--novelty-lambda 0.5"; USE_DIFF="$DIFF"; fi
    # AUTO-RESUME: if a checkpoint dir already exists (e.g. pulled from a dead instance),
    # resume from the highest-numbered one so a new instance continues instead of restarting.
    RESUME=""
    LASTCKPT=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LASTCKPT" ]; then
      RESUME="--resume-from $LASTCKPT"; echo ">> RESUMING from $LASTCKPT"
      # transformers reloads save_steps FROM the checkpoint's trainer_state on resume,
      # overriding our args ("X (from args) != Y (from trainer_state.json)") — so a stale
      # checkpoint silently keeps the OLD save cadence. Rewrite save_steps to match the config.
      # NOTE: only set keys that are valid TrainerState fields. save_total_limit is a
      # TrainingArguments field, NOT a TrainerState field — adding it makes TrainerState(**json)
      # raise "unexpected keyword argument 'save_total_limit'" on resume. Only touch save_steps.
      python - "$LASTCKPT/trainer_state.json" <<'PY' || true
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p)); d["save_steps"] = 10; d.pop("save_total_limit", None)
    json.dump(d, open(p, "w"), indent=2); print(">> patched save_steps=10 in", p)
except Exception as e:
    print(">> could not patch trainer_state:", e)
PY
    fi
    # WARM-START fallback: if no full checkpoint to --resume-from, but INIT_ADAPTER points to a
    # saved LoRA adapter dir (e.g. pulled from a dead instance, optimizer state lost), warm-start
    # the policy weights from it. Skipped if we already have a full checkpoint to resume.
    INIT=""
    if [ -z "$LASTCKPT" ] && [ -n "${INIT_ADAPTER:-}" ] && [ -f "${INIT_ADAPTER}/adapter_model.safetensors" ]; then
      INIT="--init-adapter $INIT_ADAPTER"; echo ">> WARM-START from adapter $INIT_ADAPTER"
    fi
    # EFFECTIVE-STEP accounting: warm-start resets the counter, so to hit a cumulative target
    # set EFFECTIVE_DONE to the effective steps already trained on prior instances. This run's
    # step budget is then (STEPS - EFFECTIVE_DONE) so cumulative lands on STEPS. (See
    # runs_pulled/EFFECTIVE_STEPS.md.) Only applied on a warm-start (fresh counter).
    if [ -n "$INIT" ] && [ -n "${EFFECTIVE_DONE:-}" ] && [ "${EFFECTIVE_DONE}" -gt 0 ] 2>/dev/null; then
      STEPS=$((STEPS - EFFECTIVE_DONE)); echo ">> EFFECTIVE_DONE=$EFFECTIVE_DONE -> this run trains $STEPS more steps to reach the cumulative target"
    fi
    echo "===== ARM $ARM: GRPO train ($STEPS steps; C=${USE_DIFF:-off}; resume=${LASTCKPT:-none}; warmstart=${INIT_ADAPTER:-none}; eff_done=${EFFECTIVE_DONE:-0}) ====="
    start_vllm "$MODEL" || exit 1
    if ! train_launch --model "$MODEL" --dataset "$DATASET" \
      --n-problems "$NPROB" --num-train-steps "$STEPS" --num-generations "$NGEN" \
      --max-completion-length "$MAXLEN" \
      --output-dir "$RUN" ${USE_DIFF:+--difficulty-json "$USE_DIFF"} $NOV $RESUME $INIT; then
      stop_vllm; echo "!! $ARM GRPO training FAILED"; exit 1
    fi
    stop_vllm
    run_eval "$RUN" "$ARM"
    ;;
  oursAB)
    echo "===== ARM oursAB: alternating GRPO + harvest (Comp A+B) ====="
    SEG=$((STEPS/4)); CUR="$MODEL"
    # RESUME support (added 2026-07-05): warm-start a partially-trained pipeline after an instance
    # death. SEG_START = first segment to (re)run (0-3); segments < SEG_START are assumed complete
    # and their seg${r}_sft (or seg${r}) dir is carried forward as CUR. INIT_ADAPTER warm-starts
    # the SEG_START segment's GRPO from a saved adapter (our mid-segment checkpoint). SEG_DONE =
    # steps already trained within SEG_START (so it trains SEG-SEG_DONE more). save_steps=10 in
    # config gives a resumable checkpoint every 10 steps within every segment.
    SEG_START="${SEG_START:-0}"; SEG_DONE="${SEG_DONE:-0}"
    if [ "$SEG_START" -gt 0 ]; then
      prev=$((SEG_START-1))
      CUR="$RUN/seg${prev}_sft"; [ -d "$CUR" ] || CUR="$RUN/seg${prev}"
      echo ">> RESUME: carrying forward completed segment $prev -> CUR=$CUR"
    fi
    for r in 0 1 2 3; do
      [ "$r" -lt "$SEG_START" ] && { echo "-- skipping completed segment $r"; continue; }
      THIS_SEG=$SEG; INIT_ARG=""
      if [ "$r" = "$SEG_START" ] && [ -n "${INIT_ADAPTER:-}" ] && [ -f "${INIT_ADAPTER}/adapter_model.safetensors" ]; then
        INIT_ARG="--init-adapter $INIT_ADAPTER"
        [ "${SEG_DONE:-0}" -gt 0 ] 2>/dev/null && THIS_SEG=$((SEG - SEG_DONE))
        echo ">> RESUME: warm-start segment $r from $INIT_ADAPTER; train $THIS_SEG more (SEG_DONE=$SEG_DONE)"
      fi
      echo "--- segment $r: GRPO $THIS_SEG steps ---"
      start_vllm "$CUR" || exit 1
      # FAIL-FAST: if a GRPO segment crashes, do NOT spin through the rest (the missing
      # seg${r} dir then gets misread as a HF repo id and every later segment dies too).
      if ! train_launch --model "$CUR" --dataset "$DATASET" \
        --n-problems "$NPROB" --num-train-steps "$THIS_SEG" --num-generations "$NGEN" \
        --max-completion-length "$MAXLEN" \
        --output-dir "$RUN/seg$r" ${DIFF:+--difficulty-json "$DIFF"} --novelty-lambda 0.5 $INIT_ARG; then
        stop_vllm; echo "!! segment $r GRPO FAILED — aborting oursAB"; exit 1
      fi
      stop_vllm
      echo "--- segment $r: harvest tail + SFT ---"
      # Pin harvest's vLLM to GPU0 (freed + verified idle by stop_vllm above). Explicit pinning
      # avoids it trying to spread across the just-vacated training GPUs.
      CUDA_VISIBLE_DEVICES=0 python -m rl_training.harvest --mode harvest --model-path "$RUN/seg$r" \
        --dataset "$DATASET" ${DIFF:+--difficulty-json "$DIFF"} --k 64 --max-keep 2 \
        --max-new-tokens "$MAXLEN" --out-jsonl "$RUN/harvest$r.jsonl"
      # if harvest produced no rollouts, skip SFT and carry the GRPO checkpoint forward
      if [ -s "$RUN/harvest$r.jsonl" ]; then
        python -m rl_training.harvest --mode sft --model-path "$RUN/seg$r" \
          --out-jsonl "$RUN/harvest$r.jsonl" --output-dir "$RUN/seg${r}_sft" --epochs 1 \
          && CUR="$RUN/seg${r}_sft" || CUR="$RUN/seg$r"
      else
        echo "-- no harvest rollouts for seg$r; carrying GRPO checkpoint forward"
        CUR="$RUN/seg$r"
      fi
    done
    run_eval "$CUR" oursAB
    ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
echo "===== ARM $ARM DONE ====="
touch ~/RL_${ARM}_DONE
