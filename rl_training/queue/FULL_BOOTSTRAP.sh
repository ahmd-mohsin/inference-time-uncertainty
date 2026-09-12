#!/bin/bash
# VALIDATED fresh pytorch-base-24.12 -> working vllm0.23 stack (torch cu126 + cu13 runtime for vllm engine). ~40min. $HOME paths.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 MAX_JOBS=32 CUDA_HOME=/usr/local/cuda
exec > $HOME/full_bootstrap.log 2>&1; set -x
mkdir -p $HOME/gu/logs $HOME/gu/comp_data
cd $HOME; [ -d inference-time-uncertainty/.git ] || git clone --depth 1 https://github.com/ahmd-mohsin/inference-time-uncertainty.git
cd inference-time-uncertainty && git pull --rebase 2>/dev/null||true
pip install --break-system-packages --ignore-installed vllm==0.23.0 2>&1|tail -2
pip install --break-system-packages "trl==1.7.0" peft deepspeed accelerate datasets sympy 2>&1|tail -2
pip install --break-system-packages --force-reinstall transformers==4.57.6 2>&1|tail -2
pip install --break-system-packages --force-reinstall --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126 2>&1|tail -2
pip install --break-system-packages --force-reinstall --no-deps torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126 2>&1|tail -2
pip install --break-system-packages --only-binary :all: nvidia-cuda-runtime-cu13 2>&1|tail -2
SP=$(python3 -c "import site;print(site.getusersitepackages())"); rm -rf $SP/flash_attn $SP/flash_attn_2_cuda.py 2>/dev/null
pip install --break-system-packages --force-reinstall --no-deps --no-build-isolation --no-cache-dir flash-attn 2>&1|tail -3
L13dir=$(dirname $(find ~/.local -name "libcudart.so.13*" 2>/dev/null|head -1))
echo "export LD_LIBRARY_PATH=$L13dir:\$LD_LIBRARY_PATH" >> $HOME/.bashrc
LD_LIBRARY_PATH="$L13dir:$LD_LIBRARY_PATH" python3 -c "import vllm,torch,torchvision,flash_attn,flash_attn_2_cuda,transformers,trl;print('WORKER_READY',vllm.__version__,torch.__version__)" && touch $HOME/WORKER_READY
echo "FULL_BOOTSTRAP_END $(date)"
