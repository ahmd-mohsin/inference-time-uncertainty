#!/bin/bash
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1
exec > $HOME/inst.log 2>&1
echo "START $(date)"
pip install --break-system-packages --ignore-installed vllm==0.23.0
echo "VLLM_RC=$?"
pip install --break-system-packages "trl==1.7.0" peft deepspeed accelerate datasets sympy
echo "TRL_RC=$?"
python3 -c "import vllm,trl;print('VERIFY OK', vllm.__version__, trl.__version__)"
echo "INST_DONE $(date)"
