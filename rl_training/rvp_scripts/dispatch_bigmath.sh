#!/bin/bash
# Run ON a fresh main; launches one sharded bigger-model math cell on the main + each of its 2 workers.
# 3 sharded cells per cluster (each pod = 8-GPU ZeRO-3). env CLUSTER=Cnew|Dnew picks worker IPs + cell specs.
# Spec = HF_MODEL:TAG:EVAL  (EVAL in {amc, olympiad_bench, math500}).
export HOME=/home/greenland-user
which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
case "$CLUSTER" in
  Cnew) WK=(10.2.81.201 10.2.105.252)
        MAIN="Qwen/Qwen2.5-14B-Instruct:mh14_amc:amc"
        WS=("Qwen/Qwen2.5-14B-Instruct:mh14_olymp:olympiad_bench" "01-ai/Yi-1.5-9B-Chat:mh9_amc:amc") ;;
  Dnew) WK=(10.2.200.155 10.2.144.144)
        MAIN="01-ai/Yi-1.5-9B-Chat:mh9_olymp:olympiad_bench"
        WS=("Qwen/Qwen2.5-Math-7B:mh7q_olymp:olympiad_bench" "deepseek-ai/deepseek-math-7b-instruct:mh7d_olymp:olympiad_bench") ;;
  *) echo "unknown CLUSTER=$CLUSTER"; exit 0 ;;
esac
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1 | tail -1
IFS=: read b t e <<< "$MAIN"
setsid nohup env CELL_BASE="$b" CELL_EVAL="$e" CELL_TAG="$t" bash rl_training/rvp_scripts/boot_and_run.sh >/tmp/cell_$t.log 2>&1 &
echo "main-cell $t launched ($b / $e)"
for i in 0 1; do
  ip=${WK[$i]}; IFS=: read b t e <<< "${WS[$i]}"
  sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no \
    -o PreferredAuthentications=password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$ip \
    "cd ~ && [ -d inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git; cd inference-time-uncertainty && git pull --rebase 2>&1|tail -1; setsid nohup env CELL_BASE=$b CELL_EVAL=$e CELL_TAG=$t bash rl_training/rvp_scripts/boot_and_run.sh >/tmp/cell_$t.log 2>&1 & echo worker-cell-$t-launched" 2>&1 | tail -1 &
done
wait
echo "[dispatch_bigmath $CLUSTER] done $(date -u)"
