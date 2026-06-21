#!/usr/bin/env bash
#
# greenland-instance-setup.sh — Run ON the Greenland instance (via deploy or manually)
set -euo pipefail

MODE="${1:-full}"

echo ">> Greenland instance setup (mode: $MODE)"

if [ "$MODE" = "full" ]; then
    echo ">> Setting up conda environment..."
    source ~/miniconda3/etc/profile.d/conda.sh

    if ! conda env list | grep -q "digte"; then
        conda create -n digte python=3.12 -y
    fi
    conda activate digte

    echo ">> Installing base dependencies..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.40 datasets huggingface_hub
    pip install numpy pyyaml jsonlines tqdm requests

    echo ">> Installing topological persistence deps..."
    pip install ripser persim gudhi matplotlib scipy

    echo ">> Installing vLLM..."
    pip install vllm

    echo ">> ✓ Full setup complete"

elif [ "$MODE" = "quick" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate digte

    echo ">> Quick install (topological deps only)..."
    pip install ripser persim gudhi matplotlib scipy
    echo ">> ✓ Quick setup complete"
fi

echo ""
echo ">> Verifying GPU access..."
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"
echo ""
echo ">> Verifying imports..."
python -c "
import sys; sys.path.insert(0, '.')
from topological_persistence.config import load_config
from topological_persistence.persistence import compute_persistence
print('All imports OK')
"
echo ">> ✓ Instance ready"
