#!/usr/bin/env bash
# Bootstrap a fresh pytorch-base-24.12 Greenland node for the coverage RL study.
# Idempotent-ish; writes ~/BOOTSTRAP_DONE on success. Run detached: it takes ~15-20 min.
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME
LOG=$HOME/bootstrap.log
exec > >(tee -a $LOG) 2>&1
echo "==== BOOTSTRAP START $(date -u +%H:%M:%SZ) on $(hostname) ===="

echo "== [1/6] clone repo =="
if [ ! -d $HOME/inference-time-uncertainty ]; then
  git clone --depth 1 https://github.com/ahmd-mohsin/inference-time-uncertainty.git $HOME/inference-time-uncertainty || { echo CLONE_FAIL; exit 1; }
fi
cd $HOME/inference-time-uncertainty

echo "== [2/6] pip install pinned stack (into ~/.local) =="
/usr/bin/python3 -m pip install --user --no-warn-script-location \
  vllm==0.23.0 trl==1.7.0 peft deepspeed accelerate datasets sympy scikit-learn jsonlines "nvtx>=0.2.11" \
  2>&1 | tail -3 || { echo PIP_FAIL; exit 1; }

echo "== [3/6] FIX: shadow broken flash_attn (torch-ABI mismatch) =="
S=$HOME/.local/lib/python3.12/site-packages/flash_attn
mkdir -p $S/ops/triton
echo '__version__="2.4.2-shadow"' > $S/__init__.py; : > $S/ops/__init__.py; : > $S/ops/triton/__init__.py
cp /usr/local/lib/python3.12/dist-packages/flash_attn/ops/triton/rotary.py $S/ops/triton/rotary.py 2>/dev/null || echo "  (rotary copy skipped)"
/usr/bin/python3 -c "from flash_attn.ops.triton.rotary import apply_rotary; print('  flash_attn shadow OK')" || echo "  WARN flash_attn shadow verify failed"

echo "== [4/6] prefetch base model + Oat-Zero (while online) =="
export PATH=$HOME/.local/bin:$PATH
/usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Math-7B'); print('  base cached')" 2>&1 | tail -1
/usr/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('sail/Qwen2.5-Math-7B-Oat-Zero'); print('  oat cached')" 2>&1 | tail -1

echo "== [5/6] prefetch datasets builder cache via repo loader (olympiad + omni) =="
/usr/bin/python3 -c "from datasets import load_dataset; load_dataset('math-ai/olympiadbench'); load_dataset('KbsdJames/Omni-MATH'); print('  raw datasets cached')" 2>&1 | tail -1
/usr/bin/python3 -c "import sys; sys.path.insert(0,'.'); from src.data.dataset import get_inference_dataset; d=get_inference_dataset({'dataset':{'name':'olympiad_bench','split':'test','n_problems':5,'seed':42}}); print('  olympiad loader OK n=',len(d))" 2>&1 | tail -1

echo "== [6/6] verify torch+vllm import =="
VLLM_ATTENTION_BACKEND=FLASHINFER /usr/bin/python3 -c "import torch,vllm,trl,deepspeed,accelerate; print('  torch',torch.__version__,'vllm',vllm.__version__,'trl',trl.__version__)" || { echo IMPORT_FAIL; exit 1; }

echo "==== BOOTSTRAP DONE $(date -u +%H:%M:%SZ) ===="
touch $HOME/BOOTSTRAP_DONE
