#!/usr/bin/env bash
#
# rl_pull_loop.sh — RUNS ON LOCAL DEV. Every hour, gather all arms' results to local.
# Each worker first rsyncs its rl_training/runs JSON to main; then we pull main -> local.
# JSON-only (curves, summaries, difficulty, per-problem) — no checkpoints/safetensors.
set -uo pipefail
PORT=1060
SSHL="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -p $PORT greenland-user@localhost"
SSHW="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no -p 2222"
WORKERS="10.3.165.207 10.3.77.29 10.3.120.148"
LOCAL="/Users/cmohsinm/inference-time-uncertainty/rl_training/runs_pulled"
mkdir -p "$LOCAL"

while :; do
  TS=$(date +%Y%m%d_%H%M%S)
  echo "[$TS] gathering worker results -> main, then main -> local"
  # workers push their JSON results to main under runs_<ip>/
  $SSHL "cd ~/inference-time-uncertainty && for ip in $WORKERS; do mkdir -p rl_training/runs_from_\$ip; rsync -az --include='*/' --include='*.json' --include='*.jsonl' --exclude='*' -e \"$SSHW\" greenland-user@\$ip:inference-time-uncertainty/rl_training/runs/ rl_training/runs_from_\$ip/ 2>/dev/null; done; echo gathered" 2>&1 | tail -1
  # pull everything to local: JSON/JSONL results + LoRA adapters (small, resumable) + logs.
  # adapter_model.safetensors lets us RESUME on a fresh instance if this one dies.
  rsync -az --include='*/' --include='*.json' --include='*.jsonl' --include='*.log' \
    --include='adapter_model.safetensors' --include='adapter_config.json' --exclude='*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
    greenland-user@localhost:inference-time-uncertainty/rl_training/ "$LOCAL/" 2>&1 | tail -1
  echo "[$TS] pulled -> $LOCAL ; sleeping 1h"
  sleep 3600
done
