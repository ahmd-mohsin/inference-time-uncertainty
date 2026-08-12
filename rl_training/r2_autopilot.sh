#!/usr/bin/env bash
# ROUND-2 autopilot: keep both continued-RL arms alive to 100 steps, death-proof via HF.
# Runs on the LAPTOP, detached. Every ~4min: reconnect tunnel if needed, check each arm's
# training PID + step. If an arm's trainer is dead and it's not at 100/DONE, relaunch it via
# go_r2_direct.sh (which starts FRESH from the flat local r1_${fork}_ckpt; HF has the durable
# checkpoints for the resume path). Exits when BOTH arms have pushed checkpoint-100 to HF.
set -uo pipefail
SSM=mi-038a86af96d89bc6c
LP=1066
IP_grpo=10.3.85.133
IP_floor=10.3.202.183
NVLOG=/tmp/instance_storage/gu/logs   # logs live on nvme now
LOG=/tmp/r2_autopilot.log
HFTOK=$(cat /Users/cmohsinm/.hf_token 2>/dev/null || echo HF_TOKEN_REDACTED)
echo "=== r2 autopilot start $(date -u +%H:%MZ) ssm=$SSM ===" > $LOG

recon(){ pkill -f "localPortNumber.*$LP" 2>/dev/null; sleep 2
  nohup aws ssm start-session --target $SSM --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"2222\"],\"localPortNumber\":[\"$LP\"]}" \
    --profile greenland --region us-east-2 >/tmp/tun.log 2>&1 & sleep 20; }
alive(){ ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -p $LP greenland-user@localhost 'echo ok' 2>/dev/null|grep -q ok; }
# run a command on a worker (double-hop through main)
w(){ local ip=$1; shift; ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p $LP greenland-user@localhost \
  "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -p 2222 greenland-user@$ip \"$*\"" 2>/dev/null; }

hf_has100(){ /usr/bin/python3 - "$1" <<PY 2>/dev/null
import os,sys; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import HfApi
try:
  f=HfApi(token="$HFTOK").list_repo_files(sys.argv[1],repo_type="model")
  cks={int(x.split('-')[1].split('/')[0]) for x in f if x.startswith('checkpoint-')}
  sys.exit(0 if max(cks|{0})>=100 else 1)
except Exception: sys.exit(1)
PY
}

for round in $(seq 1 400); do
  alive || recon
  for fork in grpo floor; do
    eval "ip=\$IP_${fork}"
    if hf_has100 "muahmed7338/cov-r2-from-${fork}-7b"; then
      echo "$(date -u +%H:%MZ) $fork DONE (ckpt-100 on HF)" >> $LOG; continue; fi
    step=$(w $ip "tr '\r' '\n' < $NVLOG/r2_${fork}_train.log 2>/dev/null|grep -aoE '[0-9]+/100'|tail -1")
    trainup=$(w $ip "pgrep -f 'go_r2_nvme'|wc -l")
    trainup=${trainup:-0}
    echo "$(date -u +%H:%MZ) $fork step=${step:-?} driver=${trainup}" >> $LOG
    if [ "${trainup}" -lt 1 ]; then
      echo "$(date -u +%H:%MZ) $fork DRIVER DEAD -> relaunch" >> $LOG
      # kill GPU procs by PID only (pattern-pkill self-kills via shareProcessNamespace); no launch in same call
      w $ip "for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 \$p 2>/dev/null; done; sleep 4"
      w $ip "(setsid bash ~/go_r2_nvme.sh $fork >$NVLOG/r2_${fork}_driver.log 2>&1 </dev/null &); sleep 3; echo RELAUNCHED"
    fi
  done
  if hf_has100 muahmed7338/cov-r2-from-grpo-7b && hf_has100 muahmed7338/cov-r2-from-floor-7b; then
    echo "$(date -u +%H:%MZ) BOTH ARMS DONE (ckpt-100) — autopilot exit" >> $LOG; break; fi
  sleep 240
done
