#!/bin/bash
# Runs ON a main; bootstraps + launches a comp rvp_family cell on each of its 2 workers (48 GPU fill).
# Worker cells fill the CompDAG difficulty grid + extra families. env CLUSTER=A|B|C picks worker IPs+cells.
export HOME=/home/greenland-user
which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
case "$CLUSTER" in
  A) W=(10.2.219.184 10.2.38.212)
     S=("Qwen/Qwen2.5-Coder-1.5B-Instruct:c15hard:12:14" "Qwen/Qwen2.5-Coder-1.5B-Instruct:c15vh:16:18") ;;
  B) W=(10.2.185.207 10.2.70.5)
     S=("deepseek-ai/deepseek-coder-1.3b-instruct:dshard:12:14" "Qwen/Qwen2.5-7B-Instruct:q7Nmid:7:9") ;;
  C) W=(10.2.81.201 10.2.105.252)
     S=("Qwen/Qwen2.5-Coder-7B-Instruct:c7hard:12:14" "mistralai/Mistral-7B-Instruct-v0.3:mistral:7:9") ;;
  *) echo "[dispatch] unknown CLUSTER=$CLUSTER"; exit 0 ;;
esac
for i in 0 1; do
  ip=${W[$i]}; IFS=: read base fam trd oodd <<< "${S[$i]}"
  echo "[dispatch] -> worker $ip : $fam ($base d$trd->$oodd)"
  sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no \
    -o PreferredAuthentications=password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$ip \
    "cd ~ && [ -d inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git; cd inference-time-uncertainty && git pull --rebase 2>&1|tail -1; setsid nohup env CLUSTER=W WBASE=$base WFAM=$fam WTRD=$trd WOODD=$oodd bash rl_training/rvp_scripts/boot_and_run.sh >/tmp/bootW_$fam.log 2>&1 & echo worker-$fam-launched" 2>&1 | tail -2 &
done
wait
echo "[dispatch] done $(date -u)"
