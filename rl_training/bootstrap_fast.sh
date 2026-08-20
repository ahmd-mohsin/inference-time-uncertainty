#!/usr/bin/env bash
# Hardened bootstrap for pytorch-base-24.12 Greenland node — bakes in every fix we learned:
#   - public DNS (avoid k8s-DNS ndots poisoning)
#   - pip via PIP_CONFIG_FILE=/dev/null + explicit pypi index (base image pip.conf has a DEAD nvidia
#     extra-index + no-cache-dir -> installs never complete otherwise) + nvme cache
#   - flash_attn shadow (ABI mismatch)
# Writes ~/BOOTSTRAP_FAST_DONE on success. Run detached.
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME
LOG=$HOME/bootstrap_fast.log
exec > >(tee -a $LOG) 2>&1
echo "==== BOOTSTRAP_FAST START $(date -u +%H:%M:%SZ) on $(hostname) ===="

echo "== [1/5] public DNS =="
sudo bash -c 'printf "nameserver 8.8.8.8\nnameserver 1.1.1.1\n" > /etc/resolv.conf' || true
python3 -c "import socket; print('resolve:', socket.gethostbyname('pypi.org'))" || true

echo "== [2/5] clone repo =="
[ -d $HOME/inference-time-uncertainty/.git ] || \
  git clone --depth 1 https://github.com/ahmd-mohsin/inference-time-uncertainty.git $HOME/inference-time-uncertainty
cd $HOME/inference-time-uncertainty

echo "== [3/5] pip install (pip.conf BYPASS + nvme cache) =="
mkdir -p /tmp/instance_storage/gu/pipcache
PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=0 /usr/bin/python3 -m pip install --user \
  --no-warn-script-location --cache-dir /tmp/instance_storage/gu/pipcache \
  --index-url https://pypi.org/simple --retries 20 --timeout 60 \
  vllm==0.23.0 trl==1.7.0 peft deepspeed accelerate datasets sympy scikit-learn jsonlines "nvtx>=0.2.11" \
  || { echo PIP_FAIL; exit 1; }
# nvtx>=0.2.11 REQUIRED: deepspeed 0.19.5 calls nvtx.get_domain (absent in older nvtx) -> full-FT
# GRPO crashes with AttributeError at first step otherwise.

echo "== [4/5] flash_attn shadow =="
S=$HOME/.local/lib/python3.12/site-packages/flash_attn
mkdir -p $S/ops/triton
printf '%s\n' '__version__="2.4.2-shadow"' > $S/__init__.py
: > $S/ops/__init__.py; : > $S/ops/triton/__init__.py
cp /usr/local/lib/python3.12/dist-packages/flash_attn/ops/triton/rotary.py $S/ops/triton/rotary.py 2>/dev/null || true

echo "== [5/5] verify imports =="
/usr/bin/python3 -c "import vllm,trl,datasets,sklearn; print('IMPORTS_OK', vllm.__version__, trl.__version__)" || { echo IMPORT_FAIL; exit 1; }
touch $HOME/BOOTSTRAP_FAST_DONE
echo "==== BOOTSTRAP_FAST DONE $(date -u +%H:%M:%SZ) ===="
