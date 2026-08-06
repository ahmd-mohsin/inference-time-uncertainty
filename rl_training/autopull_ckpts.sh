#!/usr/bin/env bash
# Auto-pull round-1 fork checkpoints from workers to laptop the moment they appear.
# Runs on the LAPTOP, detached. Polls every 3min; pulls any checkpoint-N/model.safetensors
# not yet local. Self-reconnects the SSM tunnel. Kills itself after both final ckpts pulled.
set -uo pipefail
SSM=mi-02337d3d916d28624
LP=1066
LB=/Users/cmohsinm/inference-time-uncertainty/rl_training/runs_pulled/round2_forks
mkdir -p "$LB/grpo" "$LB/floor"
declare -A NODE=( [grpo]=10.3.153.2 [floor]=10.3.77.128 )
LOG=/tmp/autopull.log
echo "=== autopull start $(date -u +%H:%MZ) ===" > $LOG

recon () {
  pkill -f "localPortNumber.*$LP" 2>/dev/null; sleep 2
  nohup aws ssm start-session --target $SSM --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"2222\"],\"localPortNumber\":[\"$LP\"]}" \
    --profile greenland --region us-east-2 >/tmp/tun_ap.log 2>&1 &
  sleep 18
}
alive () { ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $LP greenland-user@localhost 'echo ok' 2>/dev/null | grep -q ok; }

for round in $(seq 1 200); do
  alive || recon
  for arm in grpo floor; do
    ip=${NODE[$arm]}
    # list checkpoint dirs on the worker that have a complete model.safetensors
    cks=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p $LP greenland-user@localhost \
      "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=15 -p 2222 greenland-user@$ip 'ls -d ~/inference-time-uncertainty/rl_training/runs/r1_${arm}/checkpoint-*/ 2>/dev/null'" 2>/dev/null | grep -oE 'checkpoint-[0-9]+' | sort -u)
    for ck in $cks; do
      [ -f "$LB/$arm/$ck/model.safetensors" ] && continue   # already have it
      mkdir -p "$LB/$arm/$ck"
      echo "$(date -u +%H:%MZ) pulling $arm/$ck" >> $LOG
      # two-hop: worker -> main /tmp -> laptop
      ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p $LP greenland-user@localhost \
        "rsync -a -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -p 2222' greenland-user@$ip:~/inference-time-uncertainty/rl_training/runs/r1_${arm}/$ck/ /tmp/ap_${arm}_${ck}/ >/dev/null 2>&1" 2>/dev/null
      rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $LP" \
        greenland-user@localhost:/tmp/ap_${arm}_${ck}/ "$LB/$arm/$ck/" >/dev/null 2>&1
      sz=$(ls -la "$LB/$arm/$ck/model.safetensors" 2>/dev/null | awk '{print $5}')
      echo "$(date -u +%H:%MZ) done $arm/$ck size=${sz:-FAIL}" >> $LOG
    done
  done
  # stop once both final checkpoint-400 are local
  if [ -f "$LB/grpo/checkpoint-400/model.safetensors" ] && [ -f "$LB/floor/checkpoint-400/model.safetensors" ]; then
    echo "$(date -u +%H:%MZ) BOTH final ckpts local — autopull done" >> $LOG; break
  fi
  sleep 180
done
