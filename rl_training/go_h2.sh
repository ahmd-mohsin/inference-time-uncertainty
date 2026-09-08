#!/usr/bin/env bash
# H1 (mass-placing axis) + H2 (GRPO-rescue) in ONE sweep. Usage: go_h2.sh <MODEL> <mtag> <MU> <GPU>
# GRPO + forward-KL NLL rehearsal on the model's OWN verified-correct traces (self-distillation),
# weight MU=beta_sd. beta_sd=0 == plain GRPO (sharpen); beta_sd->inf == pure SFT (mass-place); the
# sweep traces the axis. forward_kl_penalty = -policy_logp (pure NLL, ignores ref_logprob), so the
# bank only needs {prompt, completion, ref_logprob:0.0}. Harvest is the SAME verified set arm-C uses.
set -o pipefail
export HOME=/home/greenland-user; cd /tmp/instance_storage/gu/repo
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled PYTHONPATH=/tmp/instance_storage/gu/shim VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.35}
PY=/usr/bin/python3
MODEL="${1:-Qwen/Qwen2.5-3B}"; MTAG="${2:-q3b}"; MU="${3:-0.5}"; GPU="${4:-0}"
GU=/tmp/instance_storage/gu; LOGS=$GU/logs
RAW=$GU/h2_verified_${MTAG}.jsonl; BANK=$GU/h2_bank_${MTAG}.jsonl
OUT=$GU/h2_${MTAG}_mu${MU}; TAG=h2_${MTAG}_mu${MU}
MP=$(( 29600 + GPU * 11 ))   # distinct master port per GPU (one job/GPU on a node)
echo "[go_h2 $TAG] MODEL=$MODEL MU=$MU GPU=$GPU MASTER_PORT=$MP $(date)"

# 1) harvest verified traces ONCE per model (shared across the mu sweep) — reuse if present
if [ ! -s "$BANK" ]; then
  if [ ! -s "$RAW" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY -m rl_training.gen_verified --model-path "$MODEL" --n 900 --k 4 \
      --dataset ${DS:-gsm8k} --tag h2_${MTAG} --shard-index 0 --num-shards 1 > "$LOGS/h2_genv_${MTAG}.log" 2>&1
    # gen_verified writes sft_data/sftdata_h2_<mtag>.shard0-of-1.jsonl (or _s0) ; glob to RAW
    cp "$(ls $GU/sft_data/sftdata_h2_${MTAG}*.jsonl 2>/dev/null | head -1)" "$RAW" 2>/dev/null
  fi
  # convert verified {prompt/messages,completion} -> bank {prompt,completion,ref_logprob:0.0}
  $PY - "$RAW" "$BANK" <<'PYEOF'
import json,sys
inp,out=sys.argv[1],sys.argv[2]; n=0
with open(inp) as f, open(out,"w") as g:
    for line in f:
        try: r=json.loads(line)
        except: continue
        # sft_data schema is usually {"messages":[...]} or {"prompt","completion"}
        p=r.get("prompt"); c=r.get("completion")
        if p is None and "messages" in r:
            msgs=r["messages"]; p="".join(m["content"] for m in msgs if m["role"]!="assistant")
            c=next((m["content"] for m in msgs if m["role"]=="assistant"),"")
        if not p or not c: continue
        g.write(json.dumps({"prompt":p,"completion":c,"problem_id":n,"ref_logprob":0.0})+"\n"); n+=1
print(f"bank rows={n}")
PYEOF
fi
echo "[go_h2 $TAG] bank=$(wc -l <$BANK 2>/dev/null) traces"

# 2) GRPO + forward-KL self-distillation (mu=beta_sd) on the bank
CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$MP $PY -m rl_training.train_grpo --model "$MODEL" \
  --dataset ${DS:-gsm8k} --reward-mode math --no-novelty --vllm-mode colocate \
  --num-generations 8 --num-train-steps ${STEPS:-400} --max-completion-length 1024 --seed ${SEED:-0} \
  --dph-forward-kl --ratchet-bank "$BANK" --ratchet-mu "$MU" --ratchet-bank-batch ${BB:-2} \
  --output-dir "$OUT" > "$LOGS/$TAG.log" 2>&1
echo "[go_h2 $TAG] DONE rc=$? -> $OUT $(date)"
