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
  grpo|grpo_long|oursA|oursC|oursAB_cont|expA|expC|expAC|oursA_rar|expM3|expM3A)
    # ABLATION MATRIX (methodology fixes #2, #3). Each arm toggles novelty (A) and hard-targeting (C):
    #   grpo       : standard GRPO, FULL data, no novelty, no C  -> control for Yue's crossover
    #   grpo_long  : grpo but RL_STEPS extended (compute-matched to oursABC's total updates) so a
    #                reviewer cannot attribute oursABC's coverage to "just more training".
    #   oursA      : novelty (A) + hard-targeting (C)            -> the sharpness arm
    #   oursC      : hard-targeting (C) ONLY, no novelty         -> isolates C (is targeting enough?)
    #   oursAB_cont: plain GRPO+novelty continued from a seg checkpoint (legacy)
    #   expA       : plain GRPO + FRAGILE-BAND CURRICULUM (oversample base-pass@1 in [.02,.30])
    #   expC       : plain GRPO + RARITY-WEIGHTED correctness reward
    #   expAC      : GRPO + curriculum + rarity (both new levers)
    EXTRA_FLAGS=""
    case "$ARM" in
      grpo|grpo_long) NOV="--no-novelty"; USE_DIFF="" ;;
      oursC)          NOV="--no-novelty"; USE_DIFF="$DIFF" ;;   # C only
      expA)           NOV="--no-novelty"; USE_DIFF="$DIFF"; EXTRA_FLAGS="--curriculum --frag-lo 0.02 --frag-hi 0.30 --frag-oversample 3" ;;
      expC)           NOV="--no-novelty"; USE_DIFF="";      EXTRA_FLAGS="--rarity-bonus --rarity-lambda 0.5" ;;
      expAC)          NOV="--no-novelty"; USE_DIFF="$DIFF"; EXTRA_FLAGS="--curriculum --frag-lo 0.02 --frag-hi 0.30 --frag-oversample 3 --rarity-bonus --rarity-lambda 0.5" ;;
      oursA_rar)      NOV="--novelty-lambda 0.5"; USE_DIFF="$DIFF"; EXTRA_FLAGS="--rarity-bonus --rarity-lambda 0.5" ;;  # HYBRID: novelty(A) + rarity(C)
      expM3)          NOV="--no-novelty"; USE_DIFF="";      EXTRA_FLAGS="--coverage-reward --coverage-lambda 1.0" ;;  # M3: GRPO + coverage-in-the-loop
      expM3A)         NOV="--no-novelty"; USE_DIFF="$DIFF"; EXTRA_FLAGS="--coverage-reward --coverage-lambda 1.0 --curriculum --frag-lo 0.02 --frag-hi 0.30 --frag-oversample 3" ;;  # M3 + fragile curriculum
      *)              NOV="--novelty-lambda 0.5"; USE_DIFF="$DIFF" ;;  # oursA / oursAB_cont: A(+C)
    esac
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
      --output-dir "$RUN" ${USE_DIFF:+--difficulty-json "$USE_DIFF"} $NOV ${EXTRA_FLAGS:-} $RESUME $INIT; then
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
        # PIN SFT to a single GPU. Without this it auto-spreads across all 8 GPUs via HF
        # device_map, and trl 1.7.0's _chunked_cross_entropy_loss crashes with "indices should
        # be on the same device as the indexed tensor (cuda:7)" (hidden[valid] cross-device) —
        # this silently failed EVERY segment's SFT, so Component B never distilled. Single-GPU
        # keeps hidden+valid co-located; 623 examples don't need model parallelism.
        CUDA_VISIBLE_DEVICES=0 python -m rl_training.harvest --mode sft --model-path "$RUN/seg$r" \
          --out-jsonl "$RUN/harvest$r.jsonl" --output-dir "$RUN/seg${r}_sft" --epochs 1 \
          && CUR="$RUN/seg${r}_sft" || CUR="$RUN/seg$r"
      else
        echo "-- no harvest rollouts for seg$r; carrying GRPO checkpoint forward"
        CUR="$RUN/seg$r"
      fi
    done
    run_eval "$CUR" oursAB
    ;;
  oursAB2)
    # IMPROVED A+B+C (fixes oursABC reverting pass@1 to base). Three levers vs oursAB:
    #  (1) SELECTIVE + CAPPED harvest: --max-pass-rate 0.25 (only distill the RARE tail, not
    #      easy problems that just reinforce the dominant mode) + --max-total 300 (gentler set).
    #  (2) LIGHTER SFT: SFT_LR (default 5e-7, was 1e-6) and SFT_EPOCHS (default 1) — restore
    #      coverage without flattening A's sharpened peak.
    #  (3) RE-SHARPEN: novelty-lambda ramps UP on post-harvest segments (LAMBDAS below), and the
    #      pipeline ends on a short GRPO+novelty re-sharpen (RESHARPEN steps) NOT on an SFT — so
    #      the last thing touching the model restores sharpness on the re-injected modes.
    echo "===== ARM oursAB2: selective harvest + light SFT + re-sharpen ====="
    SEG=$((STEPS/4)); CUR="$MODEL"
    SFT_LR="${SFT_LR:-5e-7}"; SFT_EPOCHS="${SFT_EPOCHS:-1}"
    MAXPASS="${HARVEST_MAX_PASS_RATE:-0.25}"; MAXTOT="${HARVEST_MAX_TOTAL:-300}"
    RESHARPEN="${RESHARPEN_STEPS:-40}"
    # per-segment novelty lambda (ramps up so post-harvest RL re-sharpens harder)
    LAMBDAS=(0.5 0.7 0.9 1.1)
    # RESUME (survives instance death): if a segment's output already exists on disk (pulled from a
    # dead instance + restored, or from an earlier run on this box), skip it and carry the completed
    # model forward. A segment is "done" if seg${r}_sft/ has an adapter (harvest+SFT finished) or
    # seg${r}/ has one (GRPO done, harvest yielded nothing). First incomplete segment resumes fresh.
    for r in 0 1 2 3; do
      if [ -f "$RUN/seg${r}_sft/adapter_model.safetensors" ]; then
        CUR="$RUN/seg${r}_sft"; echo ">> RESUME: segment $r already complete (seg${r}_sft) -> CUR=$CUR"; continue
      elif [ -f "$RUN/seg$r/adapter_model.safetensors" ] && [ -s "$RUN/harvest$r.jsonl" ] 2>/dev/null; then
        # GRPO done + harvest done but SFT maybe missing: fall through to (re)do SFT below only.
        :
      fi
      LAM=${LAMBDAS[$r]}
      # skip the GRPO re-train if seg$r GRPO adapter already exists (resume mid-segment)
      if [ -f "$RUN/seg$r/adapter_model.safetensors" ]; then
        echo ">> RESUME: segment $r GRPO already done, skipping to harvest/SFT"
      else
      echo "--- oursAB2 segment $r: GRPO $SEG steps (novelty-lambda=$LAM) ---"
      start_vllm "$CUR" || exit 1
      if ! train_launch --model "$CUR" --dataset "$DATASET" \
        --n-problems "$NPROB" --num-train-steps "$SEG" --num-generations "$NGEN" \
        --max-completion-length "$MAXLEN" \
        --output-dir "$RUN/seg$r" ${DIFF:+--difficulty-json "$DIFF"} --novelty-lambda "$LAM"; then
        stop_vllm; echo "!! oursAB2 segment $r GRPO FAILED"; exit 1
      fi
      stop_vllm
      fi   # end resume-skip of GRPO
      echo "--- oursAB2 segment $r: SELECTIVE harvest (max_pass_rate=$MAXPASS, max_total=$MAXTOT) + light SFT (lr=$SFT_LR) ---"
      # resume: skip harvest if we already have its output
      [ -s "$RUN/harvest$r.jsonl" ] || \
      CUDA_VISIBLE_DEVICES=0 python -m rl_training.harvest --mode harvest --model-path "$RUN/seg$r" \
        --dataset "$DATASET" ${DIFF:+--difficulty-json "$DIFF"} --k 64 --max-keep 2 \
        --max-pass-rate "$MAXPASS" --max-total "$MAXTOT" \
        --max-new-tokens "$MAXLEN" --out-jsonl "$RUN/harvest$r.jsonl"
      if [ -s "$RUN/harvest$r.jsonl" ]; then
        CUDA_VISIBLE_DEVICES=0 python -m rl_training.harvest --mode sft --model-path "$RUN/seg$r" \
          --out-jsonl "$RUN/harvest$r.jsonl" --output-dir "$RUN/seg${r}_sft" --epochs "$SFT_EPOCHS" --lr "$SFT_LR" \
          && CUR="$RUN/seg${r}_sft" || CUR="$RUN/seg$r"
      else
        echo "-- oursAB2 seg$r: no rare-tail rollouts; carrying GRPO checkpoint forward"; CUR="$RUN/seg$r"
      fi
    done
    # (3) FINAL RE-SHARPEN: a short GRPO+novelty pass so the pipeline does NOT end on a flattening
    # SFT. This is the key fix for pass@1 reverting to base.
    if [ -f "$RUN/resharpen/adapter_model.safetensors" ]; then
      CUR="$RUN/resharpen"; echo ">> RESUME: re-sharpen already done -> CUR=$CUR"
    elif [ "$RESHARPEN" -gt 0 ] 2>/dev/null; then
      echo "--- oursAB2 FINAL re-sharpen: GRPO $RESHARPEN steps (novelty-lambda=1.1) from $CUR ---"
      start_vllm "$CUR" || exit 1
      if train_launch --model "$CUR" --dataset "$DATASET" \
        --n-problems "$NPROB" --num-train-steps "$RESHARPEN" --num-generations "$NGEN" \
        --max-completion-length "$MAXLEN" \
        --output-dir "$RUN/resharpen" ${DIFF:+--difficulty-json "$DIFF"} --novelty-lambda 1.1; then
        CUR="$RUN/resharpen"
      else echo "!! re-sharpen failed; using pre-resharpen model"; fi
      stop_vllm
    fi
    run_eval "$CUR" oursAB2
    ;;
  oursB|sft_random)
    # ABLATION #3 (B-only) + CONTROL #2 (random-correct SFT). No GRPO — pure Component B from
    # BASE, so any pass@k change is attributable to harvest->SFT alone (isolates B from A/C/RL).
    #   oursB      : harvest the HARD-band tail from base, SFT on it (does B alone lift coverage?)
    #   sft_random : harvest correct rollouts from ALL problems (--all-problems), SFT on it. The
    #                control: if oursB helps but sft_random doesn't, the gain is the HARD TAIL,
    #                not generic SFT regularization / extra training data.
    ALLFLAG=""; [ "$ARM" = "sft_random" ] && ALLFLAG="--all-problems"
    echo "===== ARM $ARM: harvest($ARM) + SFT from BASE (no GRPO) ====="
    CUDA_VISIBLE_DEVICES=0 python -m rl_training.harvest --mode harvest --model-path "$MODEL" \
      --dataset "$DATASET" ${DIFF:+--difficulty-json "$DIFF"} --k 64 --max-keep 2 \
      --max-new-tokens "$MAXLEN" --out-jsonl "$RUN/harvest.jsonl" $ALLFLAG
    if [ -s "$RUN/harvest.jsonl" ]; then
      CUDA_VISIBLE_DEVICES=0 python -m rl_training.harvest --mode sft --model-path "$MODEL" \
        --out-jsonl "$RUN/harvest.jsonl" --output-dir "$RUN/sft" --epochs 1 \
        && CUR="$RUN/sft" || { echo "!! $ARM SFT failed"; exit 1; }
    else
      echo "!! $ARM: harvest produced no rollouts"; exit 1
    fi
    run_eval "$CUR" "$ARM"
    ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
echo "===== ARM $ARM DONE ====="
touch ~/RL_${ARM}_DONE
