#!/bin/bash
# arg1=queue. torch is now cu126. Force flash-attn into ~/.local (wheel, else source build against torch2.11+cu126+nvcc12.6). Then launch queue.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 MAX_JOBS=32 CUDA_HOME=/usr/local/cuda
Q=$1; exec > $HOME/flashbuild.log 2>&1
SP=$(python3 -c "import site;print(site.getusersitepackages())"); rm -rf $SP/flash_attn $SP/flash_attn_2_cuda.py 2>/dev/null
echo "=== 1) try force-reinstall prebuilt wheel $(date) ==="
pip install --break-system-packages --force-reinstall --no-deps flash-attn 2>&1 | tail -3
if python3 -c "import flash_attn,flash_attn_2_cuda" 2>/dev/null; then echo "WHEEL_OK"; else
  echo "=== 2) source build (torch2.11+cu126, nvcc12.6 -> matching ABI) $(date) ==="
  pip install --break-system-packages --force-reinstall --no-deps --no-build-isolation --no-cache-dir flash-attn 2>&1 | tail -4
fi
python3 -c "import flash_attn,flash_attn_2_cuda;print('FLASH_OK',flash_attn.__version__)" 2>&1|tail -1
if python3 -c "import torch,flash_attn,flash_attn_2_cuda,vllm,transformers,trl" 2>/dev/null; then
  echo "STACK_OK $(python3 -c 'import torch;print(torch.__version__)')"
  cd $HOME/inference-time-uncertainty; rm -f $HOME/gu/RESULTS_$Q.md $HOME/gu/comp_data/*.jsonl
  VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup bash rl_training/queue/$Q.sh > $HOME/gu/${Q}_out.log 2>&1 & disown
  echo "QUEUE_LAUNCHED_$Q $(date)"
else echo "STILL_BROKEN"; python3 -c "import flash_attn_2_cuda" 2>&1|tail -1; fi
echo "FLASHBUILD_END $(date)"
