#!/usr/bin/env bash
#
# greenland-setup-env.sh — Full environment setup on the Greenland p4d.24xlarge
set -euo pipefail

echo "============================================"
echo " Greenland Environment Setup"
echo " Instance: p4d.24xlarge (8x A100)"
echo "============================================"
echo ""

# ---- Miniconda ----
if [ ! -d "$HOME/miniconda3" ]; then
    echo "[1/6] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    echo "  ✓ Miniconda installed"
else
    echo "[1/6] Miniconda already installed ✓"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ---- Conda environment ----
echo "[2/6] Creating conda environment 'topo' (Python 3.12)..."
if conda env list | grep -q "topo"; then
    echo "  Environment 'topo' exists, updating..."
    conda activate topo
else
    conda create -n topo python=3.12 -y
    conda activate topo
fi
echo "  ✓ Environment 'topo' active ($(python --version))"
echo ""

# ---- PyTorch with CUDA ----
echo "[3/6] Installing PyTorch (CUDA 12.1)..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

# ---- Core ML dependencies ----
echo "[4/6] Installing transformers, datasets, vLLM..."
pip install --quiet \
    transformers>=4.45 \
    datasets \
    huggingface_hub \
    accelerate \
    vllm \
    numpy \
    scipy \
    pyyaml \
    jsonlines \
    tqdm \
    requests \
    matplotlib
echo "  ✓ Core ML deps installed"
echo ""

# ---- Topological persistence dependencies ----
echo "[5/6] Installing topological persistence deps (ripser, gudhi, persim)..."
pip install --quiet \
    ripser \
    persim \
    gudhi \
    scikit-learn
echo "  ✓ Topology deps installed"
echo ""

# ---- RL post-training dependencies (rl_training/, docs/RL.md) ----
echo "[5b/6] Installing RL post-training deps (trl, peft, deepspeed, sentence-transformers)..."
pip install --quiet \
    trl \
    peft \
    deepspeed \
    sentence-transformers \
    math-verify \
    latex2sympy2-extended
echo "  ✓ RL deps installed"
echo ""

# ---- Verification ----
echo "[6/6] Verifying full setup..."
echo ""

python -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.version.cuda}')
print(f'  GPUs:          {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.0f} GB)')
"
echo ""

python -c "
import transformers, datasets, vllm
print(f'  transformers:  {transformers.__version__}')
print(f'  datasets:      {datasets.__version__}')
print(f'  vllm:          {vllm.__version__}')
"
echo ""

python -c "
import ripser, gudhi, persim
print(f'  ripser:        {ripser.__version__}')
print(f'  gudhi:         {gudhi.__version__}')
print(f'  persim:        OK')
"
echo ""

cd ~/inference-time-uncertainty
python -c "
import sys; sys.path.insert(0, '.')
from topological_persistence.config import load_config
from topological_persistence.distances import cosine_distance_matrix
from topological_persistence.persistence import compute_persistence, compute_topological_signature
from topological_persistence.ceiling_detector import detect_ceiling
import numpy as np

D = cosine_distance_matrix(np.random.randn(8, 64))
sig = compute_topological_signature(D, max_dim=2, n_radii=50)
signal = detect_ceiling(sig)
print(f'  Pipeline test:  verdict={signal.verdict}, H1_features={signal.h1_n_features}')
print('  ✓ All topological_persistence imports OK')
"
echo ""

echo "============================================"
echo " ✅ Environment setup complete!"
echo ""
echo " To use:"
echo "   conda activate topo"
echo "   cd ~/inference-time-uncertainty"
echo "   python -m topological_persistence.run --model Qwen/Qwen3-32B --dataset aime_2024 --n-problems 5"
echo "============================================"
