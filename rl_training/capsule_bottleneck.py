# Learned Minimal Failure State bottleneck (§11). Instead of the hand-built Failure Certificate, have the model
# COMPRESS (failed code + error) into a short <=128-token textual capsule z that helps a fresh solver but does NOT
# contain the failed program. Best-of-n over capsules (select the lowest-leakage sufficiently-short one). Compare
# recovery vs hand-cert vs iid, and measure the capsule's proposal-leakage (should be ~0 like the hand cert).
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests
from rl_training.seq_recover import base_task, chat
from rl_training.certify import make_certificate

CAPSULE_INSTR = ("A previous attempt at this problem failed. Below is that attempt and its error. Write a SHORT "
                 "FAILURE REPORT (at most ~100 words) for a fresh programmer who will NOT see the previous code: state "
                 "the specific failing input, what output is expected vs what was produced, and the class of bug to "
                 "avoid. Do NOT include, quote, or restate any of the previous code — only the diagnostic evidence.\n\n"
                 "Previous attempt:\n```python\n{code}\n```\nError/counterexample:\n{cert}\n\nFAILURE REPORT:")

def leakage_cosine(z, code):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1).fit_transform([z or "", code or ""])
        return float(cosine_similarity(X[0], X[1])[0,0])
    except Exception:
        return 0.0

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM",0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n, mt=1024, temp=1.0): return llm.generate(prompts, SamplingParams(n=n, temperature=temp, top_p=0.95, max_tokens=mt, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"]))
    d0 = gen([chat(mp, base_task(it)) for it in items], 1)
    failed = []
    for it, o in zip(items, d0):
        code = extract_code(o.outputs[0].text); ok, err = run_tests(code, it, return_err=True)
        if not ok:
            cert = make_certificate(code, it)
            if cert: failed.append({"it": it, "code": code, "err": err, "cert": cert})
    if len(failed) < 3:
        json.dump({"tag":a.tag,"n":0}, open(Path(a.output_dir)/f"cap_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json","w")); print("too few"); return
    # generate n capsule candidates per failed problem; pick lowest-leakage one within length budget (<=~140 words)
    N = a.n_capsules
    cap_out = gen([chat(mp, CAPSULE_INSTR.format(code=f["code"][:1200], cert=f["cert"])) for f in failed], N, mt=200, temp=1.0)
    # v2 selection: score each candidate capsule by an actual REPAIR PROBE (1 sample) minus a leakage penalty,
    # i.e. select the capsule that best drives a fresh repair — not merely the lowest-leakage one (v1 bug).
    for f, o in zip(failed, cap_out):
        f["caps"] = [s.text.strip()[:800] for s in o.outputs if s.text.strip()] or [f["cert"]]
    probe_prompts, probe_map = [], []
    for fi, f in enumerate(failed):
        for ci, c in enumerate(f["caps"]):
            probe_prompts.append(chat(mp, base_task(f["it"]) + "\n\n" + c)); probe_map.append((fi, ci))
    probe = gen(probe_prompts, 1)
    scores = {}
    for (fi, ci), o in zip(probe_map, probe):
        ok = run_tests(extract_code(o.outputs[0].text), failed[fi]["it"])
        c = failed[fi]["caps"][ci]; leak = leakage_cosine(c, failed[fi]["code"])
        scores.setdefault(fi, []).append((int(ok) - 0.1*leak, ci))
    for fi, f in enumerate(failed):
        best_ci = max(scores[fi])[1]; f["capsule"] = f["caps"][best_ci]; f["cap_leak"] = leakage_cosine(f["capsule"], f["code"])
    K = a.k
    arms = {"iid": [], "hand_cert": [], "capsule": []}
    def recov(prompts):
        outs = gen(prompts, K); return outs
    # iid
    for f, o in zip(failed, recov([chat(mp, base_task(f["it"])) for f in failed])):
        arms["iid"].append(any(run_tests(extract_code(s.text), f["it"]) for s in o.outputs))
    for f, o in zip(failed, recov([chat(mp, base_task(f["it"]) + "\n\n" + f["cert"]) for f in failed])):
        arms["hand_cert"].append(any(run_tests(extract_code(s.text), f["it"]) for s in o.outputs))
    for f, o in zip(failed, recov([chat(mp, base_task(f["it"]) + "\n\n" + f["capsule"]) for f in failed])):
        arms["capsule"].append(any(run_tests(extract_code(s.text), f["it"]) for s in o.outputs))
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n_failed": len(failed),
           "recovery": {k: sum(v)/max(len(v),1) for k,v in arms.items()},
           "cap_leak_mean": sum(f["cap_leak"] for f in failed)/len(failed),
           "handcert_leak_mean": sum(leakage_cosine(f["cert"], f["code"]) for f in failed)/len(failed),
           "cap_len_words_mean": sum(len(f["capsule"].split()) for f in failed)/len(failed)}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"cap_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"cap_{a.tag}.json")
    json.dump(out, open(fp,"w")); print(f"[{a.tag} s{a.shard_index}] failed={len(failed)} recov={out['recovery']} caplleak={out['cap_leak_mean']:.3f}")

def merge(a):
    agg = {"iid":[0,0],"hand_cert":[0,0],"capsule":[0,0]}; cl=[]; hl=[]; wl=[]; n=0
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"cap_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): continue
        d = json.load(open(fp))
        if not d.get("n_failed"): continue
        nf=d["n_failed"]; n+=nf
        for k in agg: agg[k][0]+=d["recovery"][k]*nf; agg[k][1]+=nf
        cl.append(d["cap_leak_mean"]*nf); hl.append(d["handcert_leak_mean"]*nf); wl.append(d["cap_len_words_mean"]*nf)
    rec={k:agg[k][0]/max(agg[k][1],1) for k in agg}
    out={"tag":a.tag,"n_failed":n,"recovery":rec,"capsule_leak":sum(cl)/max(n,1),"handcert_leak":sum(hl)/max(n,1),"capsule_len_words":sum(wl)/max(n,1)}
    json.dump(out, open(Path(a.output_dir)/f"cap_{a.tag}.json","w"))
    print(f"[{a.tag}] n={n} iid={rec['iid']:.3f} hand_cert={rec['hand_cert']:.3f} capsule={rec['capsule']:.3f} | cap_leak={out['capsule_leak']:.3f} handcert_leak={out['handcert_leak']:.3f} cap_len={out['capsule_len_words']:.0f}w")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--max-problems", type=int, default=-1); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-capsules", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="cap")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
