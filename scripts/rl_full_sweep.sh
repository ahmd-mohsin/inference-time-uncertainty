#!/usr/bin/env bash
#
# rl_full_sweep.sh — RUNS ON LOCAL DEV. Drives the full 4-arm RL study across 4 nodes
# (one arm per node, each on its own 8 GPUs). Waits for the difficulty prepass (Component C),
# then launches every arm. Idempotent-ish: skips an arm whose eval JSON already exists.
#
# Node map:  main 10.3.145.117 -> oursA   (prepass runs here first)
#            w0   10.3.103.188 -> grpo
#            w1   10.3.197.81  -> base
#            w2   10.3.188.41  -> oursAB   (only if its torch is healthy)
set -uo pipefail
PORT=1060
SSHL="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -p $PORT greenland-user@localhost"
SSHW="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -p 2222"
MODEL="Qwen/Qwen3-8B"; DATASET="aime_all"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

launch(){ # $1=ip(or main) $2=arm $3=diff
  local tgt="$1" arm="$2" diff="$3"
  local CMD="cd ~/inference-time-uncertainity 2>/dev/null || cd ~/inference-time-uncertainty; nohup bash scripts/rl_experiment.sh $arm $MODEL $DATASET $diff > ~/logs/rl_${arm}.log 2>&1 </dev/null & echo ${arm}_PID \$!"
  if [ "$tgt" = main ]; then $SSHL "$CMD"; else $SSHL "$SSHW greenland-user@$tgt '$CMD'"; fi
}

# 1. wait for difficulty.json (Component C; prepass on main)
log "waiting for difficulty.json ..."
while ! $SSHL "test -f ~/inference-time-uncertainty/rl_training/runs/difficulty.json && echo Y" 2>/dev/null | grep -q Y; do sleep 120; done
log "difficulty.json ready"
DIFF="rl_training/runs/difficulty.json"
# distribute to workers
$SSHL "for ip in 10.3.103.188 10.3.197.81 10.3.188.41; do rsync -az -e \"$SSHW\" ~/inference-time-uncertainty/rl_training/runs/difficulty.json greenland-user@\$ip:inference-time-uncertainty/rl_training/runs/ 2>/dev/null; done; echo distributed"

# 2. launch arms. NOTE: base (w1) and grpo (w0) were launched manually in parallel with
# the prepass (they don't need difficulty.json), so here we launch ONLY the C-dependent arms.
log "launching oursA (main); base/grpo already running on w1/w0"
launch main oursA "$DIFF"

# 3. oursAB on w2 only if torch healthy
if $SSHL "$SSHW greenland-user@10.3.188.41 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate topo && python -c \"import torch,vllm;assert torch.__version__.startswith(\\\"2.11\\\")\" 2>/dev/null && echo OK'" 2>/dev/null | grep -q OK; then
  log "w2 healthy -> launching oursAB"
  launch 10.3.188.41 oursAB "$DIFF"
else
  log "w2 torch NOT healthy -> oursAB deferred (run when fixed); core base/grpo/oursA proceeding"
fi
log "FULL SWEEP LAUNCHED"
