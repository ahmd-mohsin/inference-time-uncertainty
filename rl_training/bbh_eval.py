# BBH eval: run a model on a BBH task (OOD family), report mean exact-match pass@1. Merges adapter if needed.
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.seq_recover import chat
from rl_training.bbh_util import load_bbh, bbh_match, BBH_PROMPT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--tag", default="bbh"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bbh(a.task)
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=0.7, top_p=0.95, max_tokens=768, stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, BBH_PROMPT.replace("{q}", it["q"])) for it in items], sp)
    p = sum(sum(int(bbh_match(s.text, it["gold"])) for s in o.outputs)/max(len(o.outputs),1) for it,o in zip(items,outs))/max(len(items),1)
    out = {"tag": a.tag, "task": a.task, "n": len(items), "mean_p": p}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True); json.dump(out, open(Path(a.output_dir)/f"bbh_{a.tag}.json","w"))
    print(f"[bbh_eval {a.tag} {a.task}] mean_p={p:.4f} n={len(items)}")
if __name__ == "__main__": main()
