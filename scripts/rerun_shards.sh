#!/usr/bin/env bash
# rerun_shards.sh <model> <tag> <shard_idx_csv> <num_shards> [dataset] — RUNS ON A NODE.
# Re-run specific eval shard indices (e.g. after killing hung ones), then auto-merge when all
# num_shards partials exist. Lets you recover a sharded eval without redoing completed shards.
# DATASET is the 5th arg (default math500) — MUST match the original run or the merge mixes sets.
# Optional env: DIFF_JSON/SUBSET_LABELS (hard-band subset), EVAL_SEED (per-seed).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate digte
cd ~/inference-time-uncertainty
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${1:?model}"; TAG="${2:?tag}"; IDX="${3:?csv shard idx}"; NS="${4:?num_shards}"; DS="${5:-math500}"
OUT="$PWD/rl_training/runs/eval"; mkdir -p "$OUT" ~/logs
EXTRA=""
[ -n "${DIFF_JSON:-}" ] && EXTRA="$EXTRA --difficulty-json $DIFF_JSON --subset-labels ${SUBSET_LABELS:-hard}"
[ -n "${EVAL_SEED:-}" ] && EXTRA="$EXTRA --seed $EVAL_SEED"
pids=()
for s in ${IDX//,/ }; do
  CUDA_VISIBLE_DEVICES=$s python -m rl_training.evaluate_passk --model-path "$MODEL" --dataset "$DS" \
    --n-problems -1 --n-samples "${NSAMP:-256}" --max-new-tokens "${MAXNEW:-4096}" --tensor-parallel-size 1 \
    --output-dir "$OUT" --tag "$TAG" --shard-index "$s" --num-shards "$NS" $EXTRA \
    > ~/logs/eval_${TAG}_shard${s}.log 2>&1 &
  pids+=($!); sleep 2
done
for p in "${pids[@]}"; do wait "$p"; done
have=$(ls "$OUT"/passk_${TAG}.shard*-of-${NS}.json 2>/dev/null | wc -l)
if [ "$have" -eq "$NS" ]; then
  python -m rl_training.evaluate_passk --tag "$TAG" --output-dir "$OUT" --num-shards "$NS" --merge --model-path "$MODEL"
  touch ~/EVAL_${TAG}_DONE; echo "MERGED $TAG"
else echo "only $have/$NS shards present, not merging"; fi
