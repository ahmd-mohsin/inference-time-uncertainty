#!/usr/bin/env bash
# rl_pull_checkpoints.sh — RUNS ON LOCAL DEV. Called when an instance is about to die.
# Pulls the FULL latest resumable checkpoint dir for each arm (adapter + DeepSpeed optimizer
# state global_step*/ + rng_state + scheduler + trainer_state) to local, so the next instance
# can resume mid-run instead of restarting from base.
#
# The hourly rl_pull_loop.sh only grabs JSON+adapter (light, for monitoring); THIS grabs the
# heavy resumable state on demand.
set -uo pipefail
PORT=1060
SSHL="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PORT greenland-user@localhost"
SSHW="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no -p 2222"
RSYNC_E="ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
LOCAL="/Users/cmohsinm/inference-time-uncertainty/rl_training/runs_pulled"
# arm -> node IP (main reached via tunnel as localhost). EDIT IPs on a new instance.
MAIN_ARM="oursA"
declare -A WORKER_ARM=( [10.3.165.207]=grpo [10.3.120.148]=oursAB )   # base = eval only, no ckpt
mkdir -p "$LOCAL"

echo "[$(date +%H:%M:%S)] pulling latest resumable checkpoints to local"

# 1) workers stage their latest checkpoint-* into main under runs_from_<ip>/<arm>/
for ip in "${!WORKER_ARM[@]}"; do
  arm="${WORKER_ARM[$ip]}"
  $SSHL "$SSHW greenland-user@$ip 'cd ~/inference-time-uncertainty && LAST=\$(ls -d rl_training/runs/$arm/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); echo worker $ip $arm latest=\$LAST'"
  # pull worker latest ckpt -> main staging
  $SSHL "cd ~/inference-time-uncertainty && mkdir -p rl_training/ckpt_stage/$arm && LAST=\$($SSHW greenland-user@$ip 'cd ~/inference-time-uncertainty && ls -d rl_training/runs/$arm/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1') && [ -n \"\$LAST\" ] && rsync -a -e \"$SSHW\" greenland-user@$ip:inference-time-uncertainty/\$LAST/ rl_training/ckpt_stage/$arm/\$(basename \$LAST)/ && echo staged $arm \$LAST"
done

# 2) pull main's own arm latest ckpt into staging too
$SSHL "cd ~/inference-time-uncertainty && mkdir -p rl_training/ckpt_stage/$MAIN_ARM && LAST=\$(ls -d rl_training/runs/$MAIN_ARM/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1) && [ -n \"\$LAST\" ] && cp -r \$LAST rl_training/ckpt_stage/$MAIN_ARM/ 2>/dev/null; echo main $MAIN_ARM \$LAST"

# 3) pull all staged checkpoints main -> local
rsync -a -e "$RSYNC_E" greenland-user@localhost:inference-time-uncertainty/rl_training/ckpt_stage/ "$LOCAL/ckpt_stage/" 2>&1 | tail -2
echo "[$(date +%H:%M:%S)] checkpoints -> $LOCAL/ckpt_stage/"
du -sh "$LOCAL"/ckpt_stage/* 2>/dev/null
