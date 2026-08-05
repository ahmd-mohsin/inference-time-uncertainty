#!/usr/bin/env bash
# Comprehensive pre-death pull: all results from all 3 nodes -> laptop.
# Self-reconnects the SSM tunnel. Safe to run repeatedly (rsync is idempotent).
set -uo pipefail
LP=1066
MAIN=mi-07cc95b025e4709ed
WORKERS="10.3.213.46 10.3.217.27"
DEST=/Users/cmohsinm/inference-time-uncertainty/rl_training/runs_pulled/predeath_$(date +%H%M 2>/dev/null || echo pull)
mkdir -p "$DEST"

reconnect () {
  pkill -f "localPortNumber.*$LP" 2>/dev/null; sleep 2
  nohup aws ssm start-session --target $MAIN --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"2222\"],\"localPortNumber\":[\"$LP\"]}" \
    --profile greenland --region us-east-2 >/tmp/tun_pull.log 2>&1 &
  sleep 18
}
sshm () { ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p $LP greenland-user@localhost "$@"; }

reconnect
echo ">> [1/3] merge any complete Omni shards on main, then stage all eval JSONs to /tmp"
sshm 'cd ~/inference-time-uncertainty; n=$(ls rl_training/runs/eval/passk_*omni_cross*.shard*.json 2>/dev/null|wc -l);
  echo "omni shards present: $n/8";
  if [ "$n" = "8" ]; then
    python -m rl_training.evaluate_passk --merge --num-shards 4 --output-dir rl_training/runs/eval --tag base_omni_cross --model-path Qwen/Qwen2.5-Math-7B --dataset omni_math 2>&1|tail -1;
    python -m rl_training.evaluate_passk --merge --num-shards 4 --output-dir rl_training/runs/eval --tag oat_omni_cross  --model-path sail/Qwen2.5-Math-7B-Oat-Zero --dataset omni_math 2>&1|tail -1;
  fi'

echo ">> [2/3] pull MAIN eval JSONs + logs"
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $LP" \
  --include="passk_*.json" --include="*.json" --exclude="*" \
  greenland-user@localhost:~/inference-time-uncertainty/rl_training/runs/eval/ "$DEST/main_eval/" 2>/dev/null
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $LP" \
  --include="exp*.log" --exclude="*" \
  greenland-user@localhost:~/logs/ "$DEST/main_logs/" 2>/dev/null

echo ">> [3/3] pull WORKER eval JSONs (worker2 has the oly1024 shards)"
for ip in $WORKERS; do
  sshm "rsync -az -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -p 2222' --include='passk_*.json' --exclude='*' greenland-user@$ip:~/inference-time-uncertainty/rl_training/runs/eval/ /tmp/w_${ip//./_}/ 2>/dev/null; echo staged $ip"
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $LP" \
    greenland-user@localhost:/tmp/w_${ip//./_}/ "$DEST/worker_${ip//./_}/" 2>/dev/null
done

echo ">> DONE. Pulled to $DEST"
find "$DEST" -name "*.json" | wc -l | xargs echo "total json files pulled:"
ls -R "$DEST" 2>/dev/null | head -40
