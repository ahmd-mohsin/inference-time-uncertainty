"""Decoupled verified-preference (DPO) training from an RFT checkpoint on (prompt,chosen,rejected) pairs. LoRA."""
import argparse, os, sys, json
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--data",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--beta",type=float,default=0.1); ap.add_argument("--max-steps",type=int,default=300)
    ap.add_argument("--seed",type=int,default=1); ap.add_argument("--lr",type=float,default=5e-6); ap.add_argument("--bsz",type=int,default=4)
    ap.add_argument("--full",action="store_true",help="full-parameter DPO (for ZeRO-3 sharded multi-GPU launch; avoids LoRA+ZeRO-3 crash)")
    a=ap.parse_args()
    if os.environ.get("DPO_FULL")=="1": a.full=True   # robust to accelerate arg-passing
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    ds=load_dataset("json",data_files=a.data,split="train")
    # full-param (ZeRO-3 sharded) disables the ref-precompute (ref is sharded, kept live) and LoRA
    cfg=DPOConfig(output_dir=a.out,per_device_train_batch_size=a.bsz,gradient_accumulation_steps=2,
                  learning_rate=a.lr,max_steps=a.max_steps,logging_steps=20,save_steps=a.max_steps,
                  beta=a.beta,seed=a.seed,bf16=True,gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant":False},
                  max_length=int(os.environ.get("DPO_MAXLEN","768")),
                  warmup_ratio=0.03,lr_scheduler_type="cosine",report_to=[],
                  precompute_ref_log_probs=(not a.full))
    _tm=("all-linear" if any(k in a.model.lower() for k in ["phi","gemma"])
         else ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    peft=None if a.full else LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,task_type="CAUSAL_LM",target_modules=_tm)
    tr=DPOTrainer(model=a.model,args=cfg,train_dataset=ds,processing_class=tok,peft_config=peft)
    tr.train(); tr.save_model(a.out); tok.save_pretrained(a.out); print(f"saved DPO -> {a.out}")
if __name__=="__main__": main()
