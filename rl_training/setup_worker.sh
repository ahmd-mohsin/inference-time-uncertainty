#!/usr/bin/env bash
# Runs ON a main node. Sets up ONE worker over the fast internal net, then launches a job chain.
# Usage: bash setup_worker.sh <worker_ip> <model_basename> "<chain command>"
set -uo pipefail
export HOME=/home/greenland-user; NV=/tmp/instance_storage/gu
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222"
W="$1"; MODEL="$2"; CHAIN="$3"
# fresh worker nvme is root-owned -> create gu via empty-password sudo
ssh $SSHO greenland-user@$W "echo '' | sudo -S mkdir -p $NV 2>/dev/null; echo '' | sudo -S chown greenland-user:greenland-users $NV 2>/dev/null; mkdir -p $NV/repo $NV/repair_data $NV/logs $NV/eval_out"
rsync -az -e "ssh $SSHO" $NV/repo/ greenland-user@$W:$NV/repo/
rsync -az -e "ssh $SSHO" $NV/$MODEL/ greenland-user@$W:$NV/$MODEL/
rsync -az -e "ssh $SSHO" $NV/repair_data/ greenland-user@$W:$NV/repair_data/
[ -f $HOME/.hf_token ] && rsync -az -e "ssh $SSHO" $HOME/.hf_token greenland-user@$W:$HOME/.hf_token
ssh $SSHO greenland-user@$W "export HOME=/home/greenland-user; rm -f \$HOME/BOOTSTRAP_FAST_DONE; bash $NV/repo/rl_training/bootstrap_fast.sh > $NV/bootstrap.out 2>&1; nohup setsid bash -c '$CHAIN' > $NV/wchain.out 2>&1 </dev/null & disown; echo LAUNCHED_$W"
