#!/bin/bash
# arg1=queue. Downgrade transformers to 4.57.6 (consistent w/ vllm0.23; pulls matching tokenizers+hf_hub), verify, launch queue.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1
Q=$1; exec > $HOME/fix_and_run.log 2>&1
for t in $(seq 1 20); do
  echo "=== attempt $t $(date) ==="
  pip install --break-system-packages --force-reinstall "transformers==4.57.6" 2>&1 | tail -2
  if python3 -c "import transformers,tokenizers,huggingface_hub,vllm,trl,peft,deepspeed,datasets" 2>/dev/null; then
    python3 -c "import transformers,huggingface_hub,vllm,trl;print('DEPS_OK tf',transformers.__version__,'hub',huggingface_hub.__version__,'vllm',vllm.__version__)"
    cd $HOME/inference-time-uncertainty
    rm -f $HOME/gu/RESULTS_$Q.md $HOME/gu/comp_data/*.jsonl 2>/dev/null
    setsid nohup bash rl_training/queue/$Q.sh > $HOME/gu/${Q}_out.log 2>&1 & disown
    echo "QUEUE_LAUNCHED $Q $(date)"; break
  fi
  python3 -c "import transformers" 2>&1 | grep -iE "cannot import|ImportError|ResolutionImpossible|required" | tail -1
  sleep 8
done
echo "FIX_AND_RUN_END"
