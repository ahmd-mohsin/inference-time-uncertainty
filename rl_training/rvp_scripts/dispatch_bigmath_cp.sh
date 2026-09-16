#!/bin/bash
# Coverage-PRESERVING bigger-model RVP: SKIP_RFT (RVP-from-base, keep base coverage) + gentle DPO (beta .3, 150 steps).
# One sharded cell per pod (main + 2 workers). env CLUSTER=Cnew|Dnew. Spec = HF_MODEL:TAG:EVAL:NEVAL
export HOME=/home/greenland-user
which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
COMMON="SKIP_RFT=1 DPO_BETA=0.3 DPO_STEPS=150 NSEED=1 ACC_CFG=rl_training/accelerate_zero3_offload.yaml DPO_MAXLEN=512 EVAL_CONC=2 FLYWHEEL=math_hard_shard.sh"
case "$CLUSTER" in
  Cnew) WK=(10.2.81.201 10.2.105.252)
        MAIN="Qwen/Qwen2.5-14B-Instruct:mh14cp_amc:amc:40"
        WS=("Qwen/Qwen2.5-14B-Instruct:mh14cp_olymp:olympiad_bench:150" "01-ai/Yi-1.5-9B-Chat:mh9cp_amc:amc:40") ;;
  Dnew) WK=(10.2.200.155 10.2.144.144)
        MAIN="01-ai/Yi-1.5-9B-Chat:mh9cp_olymp:olympiad_bench:150"
        WS=("Qwen/Qwen2.5-14B-Instruct:mh14cp_m500:math500:200" "01-ai/Yi-1.5-9B-Chat:mh9cp_m500:math500:200") ;;
  *) echo "unknown CLUSTER=$CLUSTER"; exit 0 ;;
esac
# 14B needs shorter gen ctx + higher single-GPU mem frac; 9B uses defaults
memenv() { echo "$1" | grep -qi '14b' && echo "MAXLEN=1536 GEN_GPU_MEM=0.9 EVAL_GPU_MEM=0.9" || echo "MAXLEN=2048 GEN_GPU_MEM=0.6 EVAL_GPU_MEM=0.55"; }
runcell() { # $1=host(localhost|ip)  spec=b:t:e:n
  local host=$1 b=$2 t=$3 e=$4 n=$5; local ME=$(memenv "$b")
  local CMD="cd ~/inference-time-uncertainty && git pull --rebase 2>&1|tail -1; pkill -9 -f 'rvp_scripts/math_hard_shard.sh'; pkill -9 -f direct_eval; pkill -9 -f accelerate.commands.launch; sleep 3; setsid nohup env $COMMON $ME BASE=$b EVAL=$e TAG=$t NEVAL=$n bash rl_training/rvp_scripts/reset_and_run.sh >/tmp/cp_$t.log 2>&1 & echo cp-$t-launched"
  if [ "$host" = localhost ]; then bash -lc "$CMD"; else
    sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$host "$CMD" </dev/null; fi
}
IFS=: read b t e n <<< "$MAIN"; runcell localhost "$b" "$t" "$e" "$n"
for i in 0 1; do IFS=: read b t e n <<< "${WS[$i]}"; runcell "${WK[$i]}" "$b" "$t" "$e" "$n" & done
wait
echo "[dispatch_bigmath_cp $CLUSTER] done $(date -u)"
