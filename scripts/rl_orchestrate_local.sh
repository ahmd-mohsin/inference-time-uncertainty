#!/usr/bin/env bash
#
# rl_orchestrate_local.sh — RUNS ON LOCAL DEV (drives the 4-node cluster via the 1060 tunnel).
# Waits for env setup on all 4 nodes, pushes latest code, smoke-tests on main, and if the
# smoke passes, launches all 4 arms. Hourly result pull-back is a SEPARATE loop (rl_pull_loop).
set -uo pipefail
PORT=1060
SSHL="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -p $PORT greenland-user@localhost"
SSHW="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -p 2222"
MODEL="Qwen/Qwen3-8B"
DATASET="aime_all"
WORKERS="10.3.103.188 10.3.197.81 10.3.188.41"
LOCAL="/Users/cmohsinm/inference-time-uncertainty"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- 1. wait until env is IMPORTABLE on all 4 nodes (not process-watching: pgrep
#         self-matches the ssh command string, so we test the real signal — imports). ----
log "waiting for importable env (trl+vllm+torch) on 4 nodes..."
IMP="source ~/miniconda3/etc/profile.d/conda.sh && conda activate topo 2>/dev/null && python -c 'import trl,peft,sentence_transformers,vllm,torch; assert torch.cuda.device_count()==8' 2>/dev/null && echo READY || echo NOTYET"
while :; do
  M=$($SSHL "$IMP" 2>/dev/null | tail -1)
  WK=""; for ip in $WORKERS; do WK="$WK $ip:$($SSHL "$SSHW greenland-user@$ip \"$IMP\"" 2>/dev/null | tail -1)"; done
  echo "  main:$M$WK"
  [ "$M" = "READY" ] && ! echo "$WK" | grep -q NOTYET && break
  sleep 120
done
log "all envs importable"

# ---- 2. verify env health on main (imports) ----
log "verifying trl/peft/sentence-transformers/vllm on main..."
$SSHL "source ~/miniconda3/etc/profile.d/conda.sh && conda activate topo && python -c 'import trl,peft,sentence_transformers,vllm,torch; print(\"trl\",trl.__version__,\"torch gpus\",torch.cuda.device_count())'" 2>&1 | tail -3

# ---- 3. push latest code main + workers ----
log "pushing latest code to all nodes..."
rsync -az --delete --exclude=.git --exclude=__pycache__ --exclude=external --exclude='/data' \
  --exclude='rl_training/runs' --exclude='*.npz' --exclude='*.png' --exclude='*.safetensors' \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
  "$LOCAL/" greenland-user@localhost:inference-time-uncertainty/ 2>&1 | tail -1
$SSHL "cd ~/inference-time-uncertainty && chmod +x scripts/*.sh && for ip in $WORKERS; do rsync -az --delete --exclude=.git --exclude=__pycache__ --exclude=external --exclude=/data --exclude='rl_training/runs' --exclude='*.npz' -e \"$SSHW\" ./ greenland-user@\$ip:inference-time-uncertainty/ 2>/dev/null && $SSHW greenland-user@\$ip 'chmod +x ~/inference-time-uncertainty/scripts/*.sh'; done; echo code-distributed"

# ---- 4. smoke test on main (mandatory) ----
log "SMOKE TEST on main (Qwen3-8B, tiny)..."
$SSHL "cd ~/inference-time-uncertainty && bash scripts/rl_smoke.sh $MODEL > ~/logs/smoke.log 2>&1; echo SMOKE_EXIT=\$?"
SMOKE=$($SSHL "grep -c 'SMOKE COMPLETE' ~/logs/smoke.log 2>/dev/null" 2>/dev/null | tail -1)
$SSHL "tail -20 ~/logs/smoke.log" 2>&1 | tail -22
if [ "${SMOKE:-0}" -lt 1 ]; then
  log "SMOKE FAILED — NOT launching full run. Inspect ~/logs/smoke.log"
  exit 1
fi
log "smoke passed"

# ---- 5. launch all 4 arms ----
log "launching 4-arm study (base/grpo/oursA/oursAB)..."
$SSHL "cd ~/inference-time-uncertainty && nohup bash scripts/rl_all_arms.sh $MODEL $DATASET > ~/logs/orchestrate.log 2>&1 < /dev/null & echo ARMS_LAUNCHED \$!"
sleep 20
$SSHL "tail -15 ~/logs/orchestrate.log" 2>&1 | tail -16
log "DONE launching. Use rl_pull_loop for hourly result sync."
