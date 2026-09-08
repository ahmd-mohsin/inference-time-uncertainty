# Proposal-Leakage measurement (Minimal Failure State main figure). For each problem generate TWO independent
# failed parents P_A, P_B. For a failure REPRESENTATION rep(.), build z_A=rep(P_A), z_B=rep(P_B). Leakage asks:
# can you recover WHICH parent produced z? Retrieval classifier D(z | P_A,P_B) = argmax cosine(z, parent_code).
# Leakage L = 2*(retrieval_acc - 0.5)  in [0,1].  full_code -> ~1 (z IS the code); error_only / certificate -> ~0
# (evidence carries little proposal identity). Pair with repair success (cert_ablate) for the Pareto main figure.
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests
from rl_training.seq_recover import base_task, chat, fb_msg
from rl_training.certify import make_certificate

REPS = ["full_code", "error_only", "certificate"]

def represent(rep, code, err, item):
    if rep == "full_code":   return code
    if rep == "error_only":  return fb_msg(err, True)            # error text, no code
    if rep == "certificate": return make_certificate(code, item) or fb_msg(err, True)
    return ""

def _cos_tfidf(z, pa, pb):
    """cosine(z,pa) vs cosine(z,pb) with a char n-gram TF-IDF fit on the three docs; return (sim_a, sim_b)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    docs = [z or "", pa or "", pb or ""]
    try:
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1).fit_transform(docs)
    except Exception:
        return 0.0, 0.0
    sa = float(cosine_similarity(X[0], X[1])[0,0]); sb = float(cosine_similarity(X[0], X[2])[0,0])
    return sa, sb

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM",0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n): return llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"]))
    # two independent attempts per problem
    outs = gen([chat(mp, base_task(it)) for it in items], 2)
    pairs = []
    for it, o in zip(items, outs):
        if len(o.outputs) < 2: continue
        cA, cB = extract_code(o.outputs[0].text), extract_code(o.outputs[1].text)
        okA, eA = run_tests(cA, it, return_err=True); okB, eB = run_tests(cB, it, return_err=True)
        if okA or okB: continue                      # need TWO distinct FAILED parents
        if cA.strip() == cB.strip(): continue         # need distinct parents
        pairs.append({"cA": cA, "cB": cB, "eA": eA, "eB": eB, "it": it})
    per_rep = {r: [0,0] for r in REPS}   # [correct, total] over both z_A->A and z_B->B
    for p in pairs:
        for r in REPS:
            zA = represent(r, p["cA"], p["eA"], p["it"]); zB = represent(r, p["cB"], p["eB"], p["it"])
            if not zA or not zB: continue
            saA, sbA = _cos_tfidf(zA, p["cA"], p["cB"])   # z_A: correct if closer to cA
            saB, sbB = _cos_tfidf(zB, p["cA"], p["cB"])   # z_B: correct if closer to cB
            per_rep[r][0] += int(saA > sbA) + int(sbB > saB); per_rep[r][1] += 2
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n_pairs": len(pairs),
           "per_rep": {r: {"correct": per_rep[r][0], "total": per_rep[r][1]} for r in REPS}}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"leak_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"leak_{a.tag}.json")
    json.dump(out, open(fp,"w")); print(f"[{a.tag} s{a.shard_index}] pairs={len(pairs)} -> {fp}")

def merge(a):
    agg = {r:[0,0] for r in REPS}; npairs=0
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"leak_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): continue
        d = json.load(open(fp)); npairs += d["n_pairs"]
        for r in REPS:
            if r in d["per_rep"]: agg[r][0]+=d["per_rep"][r]["correct"]; agg[r][1]+=d["per_rep"][r]["total"]
    res = {}
    for r in REPS:
        acc = agg[r][0]/max(agg[r][1],1); res[r] = {"retrieval_acc": acc, "leakage": 2*(acc-0.5), "n": agg[r][1]}
    out = {"tag": a.tag, "n_pairs": npairs, "per_rep": res}
    json.dump(out, open(Path(a.output_dir)/f"leak_{a.tag}.json","w"))
    print(f"[{a.tag}] pairs={npairs}")
    for r in REPS: print(f"  {r:<12} retrieval_acc={res[r]['retrieval_acc']:.3f}  LEAKAGE={res[r]['leakage']:+.3f}  (n={res[r]['n']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--max-problems", type=int, default=-1)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="leak")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
