#!/bin/bash
# Sharded full-param 7B-Math breadth wave (the RELIABLE 7B path; 1-GPU LoRA-DPO hits a meta-tensor
# bug at 7B). One cell per node = 8-GPU sharded DPO. 9 cells = 3 new datasets x 3 clusters (3-fold
# spread via fresh banks). Keeps all 72 GPUs busy at the DPO stage; uses the death-proofed flywheel
# (EVAL_CONC=1, ckpt-sync-before-eval, 8-GPU data-parallel gen).
#   A(4210)=GSM8K, B(4211)=DeepMath, C(4212)=AMC ; base Qwen/Qwen2.5-Math-7B.
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)
WSSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222"
CELL_ENV="SKIP_RFT=0 DPO_BETA=0.1 DPO_STEPS=300 NSEED=1 DPO_MAXLEN=768 EVAL_CONC=1 CONSOLIDATE=0 ACC_CFG=rl_training/accelerate_zero3_offload.yaml MAXLEN=3072 GEN_GPU_MEM=0.55 EVAL_GPU_MEM=0.5 VLLM_TP=1"
node_b64() { # model tag eval neval
  cat <<EOS | base64 | tr -d '\n'
export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH
mkdir -p \$HOME/gu/logs
[ -d \$HOME/inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git \$HOME/inference-time-uncertainty
cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
pkill -9 -f math_cell_1gpu 2>/dev/null; pkill -9 -f math_hard_shard 2>/dev/null; pkill -9 -f 'math_rvp --mode' 2>/dev/null; pkill -9 -f accelerate.commands 2>/dev/null; pkill -9 -f dpo_train 2>/dev/null; pkill -9 -f sft_train 2>/dev/null; sleep 5
setsid nohup env CELL_BASE=$1 CELL_TAG=$2 CELL_EVAL=$3 NEVAL=$4 $CELL_ENV bash rl_training/rvp_scripts/boot_and_run.sh > \$HOME/gu/logs/cell_$2.log 2>&1 & echo launched-$2
EOS
}
launch() { local port=$1 w1=$2 w2=$3 tag=$4 ev=$5 nv=$6
  local B0=$(node_b64 Qwen/Qwen2.5-Math-7B s7_${tag}_n0 $ev $nv) B1=$(node_b64 Qwen/Qwen2.5-Math-7B s7_${tag}_n1 $ev $nv) B2=$(node_b64 Qwen/Qwen2.5-Math-7B s7_${tag}_n2 $ev $nv)
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "
    which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
    echo -n 'MAIN '; echo $B0 | base64 -d | bash
    echo -n 'W1 '; sshpass -p '' ssh $WSSH greenland-user@$w1 \"echo $B1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    echo -n 'W2 '; sshpass -p '' ssh $WSSH greenland-user@$w2 \"echo $B2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}
echo "=== A(4210) 7B x GSM8K ==="; launch 4210 10.2.88.179 10.2.140.204 gsm gsm8k 200
echo "=== B(4211) 7B x DeepMath ==="; launch 4211 10.2.110.65 10.2.68.155 dm deepmath 200
echo "=== C(4212) 7B x AMC ==="; launch 4212 10.2.80.244 10.2.226.121 amc amc 60
echo "[launch_sharded7b] 9 cells dispatched $(date -u)"
