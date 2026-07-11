#!/usr/bin/env bash
# run_eval_sharded.sh — RUNS ON ONE NODE. Data-parallel pass@k eval: split the problems across
# all local GPUs (one vLLM per GPU), then merge into passk_{tag}.json. pass@k is independent
# across problems and TP>1 crashes in this container, so this data-parallel split is the way to
# use all 8 GPUs — ~8x faster than a single-GPU eval (128k rollouts: ~2.5h -> ~20min).
#
# Usage: bash scripts/run_eval_sharded.sh <model_path_or_id> <tag> [dataset] [n_samples] [max_new]
# Env knobs (methodology fixes):
#   DIFF_JSON=<path>       restrict eval to a difficulty band (hard-band subset)
#   SUBSET_LABELS=hard     which labels to keep (default 'hard' when DIFF_JSON set)
#   EVAL_SEED=<int>        vLLM sampling seed for this replicate (multi-seed CIs)
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate digte
cd ~/inference-time-uncertainty
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${1:?model path/id}"; TAG="${2:?tag e.g. base|grpo|oursA|oursABC}"
DS="${3:-math500}"; NSAMP="${4:-256}"; MAXNEW="${5:-4096}"
NG="${NUM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)}"
DIFF_JSON="${DIFF_JSON:-}"; SUBSET_LABELS="${SUBSET_LABELS:-hard}"; EVAL_SEED="${EVAL_SEED:-}"; LEVEL="${LEVEL:-}"
# build the optional flag string once (empty unless a subset/seed/level is requested)
EXTRA=""
[ -n "$DIFF_JSON" ] && EXTRA="$EXTRA --difficulty-json $DIFF_JSON --subset-labels $SUBSET_LABELS"
[ -n "$EVAL_SEED" ] && EXTRA="$EXTRA --seed $EVAL_SEED"
[ -n "$LEVEL" ] && EXTRA="$EXTRA --level $LEVEL"
OUT="$PWD/rl_training/runs/eval"; mkdir -p "$OUT" ~/logs

# A bare LoRA adapter must be merged into a full model ONCE before the shards start (else 8
# processes race the same merge dir). Merge up-front on GPU0, then all shards load the merged dir.
echo ">> pre-merging adapter (if any) so shards share one merged model ..."
MERGED=$(CUDA_VISIBLE_DEVICES=0 python -c "from rl_training.model_utils import merge_adapter_if_needed as m; print(m('$MODEL'))" 2>/dev/null | tail -1)
[ -z "$MERGED" ] && MERGED="$MODEL"
echo ">> shards will load: $MERGED"

echo ">> launching $NG shards for tag=$TAG (n_samples=$NSAMP, dataset=$DS, extra='$EXTRA') ..."
pids=()
for s in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$s python -m rl_training.evaluate_passk \
    --model-path "$MERGED" --dataset "$DS" --n-problems -1 \
    --n-samples "$NSAMP" --max-new-tokens "$MAXNEW" --tensor-parallel-size 1 \
    --output-dir "$OUT" --tag "$TAG" --shard-index "$s" --num-shards "$NG" $EXTRA \
    > ~/logs/eval_${TAG}_shard${s}.log 2>&1 &
  pids+=($!)
  sleep 2   # stagger starts so the merge/cache reads don't collide
done

echo ">> waiting on ${#pids[@]} shards ..."
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "!! shard $i FAILED (see ~/logs/eval_${TAG}_shard${i}.log)"; fail=1; fi
done
[ "$fail" -ne 0 ] && { echo "!! at least one shard failed; NOT merging (would be a partial curve)"; exit 1; }

echo ">> all shards done; merging ..."
python -m rl_training.evaluate_passk --tag "$TAG" --output-dir "$OUT" \
  --num-shards "$NG" --merge --model-path "$MERGED"
echo ">> EVAL DONE for $TAG -> $OUT/passk_${TAG}.json"
touch ~/EVAL_${TAG}_DONE
