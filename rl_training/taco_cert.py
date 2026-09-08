# TACO scale test of the Failure Certificate on HARD competitive-programming (stdin/stdout). On default-
# FAILED TACO problems, compare IID resampling vs CERTIFICATE (a concrete failing stdin case: input →
# expected-output vs the failed program's actual output, code ERASED). Tests whether (E_rich,∅) beats iid
# on genuinely hard tasks + difficulty stratification (does Δcert peak near the capability frontier?).
import argparse, json, os, sys, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.taco_recover import load_taco, run_stdin, build_prompt
from rl_training.code_passk import extract_code
from rl_training.seq_recover import chat

def stdin_cert(code, ins, outs, timeout=8):
    """First failing (input→expected vs got) as a certificate string; None if it passes all."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(code); path=f.name
    except Exception:
        return "The previous attempt did not run. Write a fresh correct solution."
    try:
        for inp, exp in zip(ins, outs):
            si = inp if isinstance(inp, str) else "\n".join(map(str, inp))
            eo = (exp if isinstance(exp, str) else "\n".join(map(str, exp))).strip()
            try:
                r = subprocess.run(["python3", path], input=si, capture_output=True, text=True, timeout=timeout)
            except Exception as e:
                return f"On the given input the previous attempt crashed/timed out ({type(e).__name__}). Expected output:\n{eo[:200]}\nWrite a fresh correct solution."
            got = (r.stdout or "").strip()
            if r.returncode != 0 or got != eo:
                return (f"Concrete counterexample — for input:\n{si[:300]}\nthe correct output is:\n{eo[:200]}\n"
                        f"but the previous attempt produced:\n{(got or '<none>')[:200]}\nDiagnose the specific error and write a fresh, correct solution.")
    finally:
        try: os.unlink(path)
        except Exception: pass
    return None

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_taco(a.difficulty, a.max_problems)
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM",0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n): return llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=1536, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"]))
    d0 = gen([build_prompt(it, mp) for it in items], 1)
    failed = []
    for it, o in zip(items, d0):
        code = extract_code(o.outputs[0].text)
        if not run_stdin(code, it["inputs"], it["outputs"]):
            failed.append({"it": it, "cert": stdin_cert(code, it["inputs"], it["outputs"])})
    K = a.k; iid_rec = cert_rec = 0
    if failed:
        iid = gen([build_prompt(f["it"], mp) for f in failed], K)
        for f, o in zip(failed, iid):
            if any(run_stdin(extract_code(s.text), f["it"]["inputs"], f["it"]["outputs"]) for s in o.outputs): iid_rec += 1
        cprompts = [chat(mp, build_prompt(f["it"], mp).split("<|im_start|>user\n")[-1].split("<|im_end|>")[0] + "\n\n" + (f["cert"] or "Previous attempt failed; write a fresh solution.")) for f in failed]
        cert = gen(cprompts, K)
        for f, o in zip(failed, cert):
            if any(run_stdin(extract_code(s.text), f["it"]["inputs"], f["it"]["outputs"]) for s in o.outputs): cert_rec += 1
    out = {"tag": a.tag, "model": mp, "difficulty": a.difficulty, "n_failed": len(failed),
           "iid_recovery": iid_rec/max(len(failed),1), "cert_recovery": cert_rec/max(len(failed),1),
           "cert_minus_iid": (cert_rec-iid_rec)/max(len(failed),1)}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"tacocert_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"tacocert_{a.tag}.json")
    json.dump(out, open(fp,"w")); print(f"[{a.tag} s{a.shard_index}] failed={len(failed)} cert-iid={out['cert_minus_iid']:+.3f}")

def merge(a):
    tot=ir=cr=0
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"tacocert_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): continue
        d=json.load(open(fp)); n=d["n_failed"]; tot+=n; ir+=d["iid_recovery"]*n; cr+=d["cert_recovery"]*n
    out={"tag":a.tag,"n_failed":tot,"iid_recovery":ir/max(tot,1),"cert_recovery":cr/max(tot,1),"cert_minus_iid":(cr-ir)/max(tot,1)}
    json.dump(out,open(Path(a.output_dir)/f"tacocert_{a.tag}.json","w"))
    print(f"[{a.tag}] n={tot} iid={out['iid_recovery']:.3f} cert={out['cert_recovery']:.3f} cert-iid={out['cert_minus_iid']:+.3f}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--difficulty",default="MEDIUM"); ap.add_argument("--max-problems",type=int,default=400)
    ap.add_argument("--k",type=int,default=6); ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag",default="tacocert")
    ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    a=ap.parse_args(); merge(a) if a.merge else run(a)

if __name__=="__main__": main()
