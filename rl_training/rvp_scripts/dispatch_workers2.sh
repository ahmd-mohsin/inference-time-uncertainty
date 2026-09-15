#!/bin/bash
# Round-2 worker dispatch (re-fill idle 48 GPUs). All cells <=3B → single-GPU DPO fits.
# Accessible-hard math (Math-1.5B AMC/Olympiad) + 5th/6th families on math (general models have coverage
# there, unlike CompDAG) + extra comp grid. Spec = BASE:FAM:EVAL  (EVAL empty => comp rvp_family).
export HOME=/home/greenland-user
which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
case "$CLUSTER" in
  A) W=(10.2.219.184 10.2.38.212)
     S=("Qwen/Qwen2.5-Math-1.5B-Instruct:q15amc:amc" "Qwen/Qwen2.5-Math-1.5B-Instruct:q15olymp:olympiad_bench") ;;
  B) W=(10.2.185.207 10.2.70.5)
     S=("meta-llama/Llama-3.2-3B-Instruct:llama_m500:math500" "google/gemma-2-2b-it:gemma_m500:math500") ;;
  C) W=(10.2.81.201 10.2.105.252)
     S=("Qwen/Qwen2.5-1.5B-Instruct:q15N_gsm:gsm8k" "deepseek-ai/deepseek-coder-1.3b-instruct:dsvh::16:18") ;;
  *) echo "[dispatch2] unknown CLUSTER=$CLUSTER"; exit 0 ;;
esac
for i in 0 1; do
  ip=${W[$i]}; IFS=: read base fam ev trd oodd <<< "${S[$i]}"
  echo "[dispatch2] -> worker $ip : $fam base=$base eval=${ev:-comp}"
  sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no \
    -o PreferredAuthentications=password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$ip \
    "pkill -9 -f 'python3 -m rl_training' 2>/dev/null; pkill -9 -f accelerate.commands.launch 2>/dev/null; cd ~/inference-time-uncertainty 2>/dev/null && git pull --rebase 2>&1|tail -1; setsid nohup env CLUSTER=W WBASE=$base WFAM=$fam WEVAL=$ev WTRD=${trd:-7} WOODD=${oodd:-9} bash rl_training/rvp_scripts/boot_and_run.sh >/tmp/bootW2_$fam.log 2>&1 & echo w2-$fam-launched" 2>&1 | tail -2 &
done
wait
echo "[dispatch2] done $(date -u)"
