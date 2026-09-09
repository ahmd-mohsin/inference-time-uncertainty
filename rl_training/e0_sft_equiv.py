# E0(a) (§56): is the hybrid trainer's SFT/NLL path the SAME operator as standalone SFT on an identical batch?
# Compares, on the SAME model + SAME (prompt,completion) batch:
#   (1) STANDALONE-SFT loss = MEAN over completion tokens of -logprob (what sft_train/SFTTrainer optimizes)
#   (2) HYBRID forward_kl path = -SUM over completion tokens of logprob (coverage_trainer.forward_kl_penalty),
#       optionally /len. Reports loss values, grad L2, and cosine(grad_sft, grad_hybrid).
# If cosine≈1 and losses match up to a constant reduction factor -> same operator (λ=1 IS SFT).
# If cosine<1 or reduction differs -> §44's flat λ-sweep was NOT a fair SFT proxy (reduction/weighting artifact).
# Usage: python -m rl_training.e0_sft_equiv --bank <bank.jsonl> --n 8
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B"); ap.add_argument("--bank", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(a.bank) if l.strip()][:a.n]

    def batch_logps(model):
        """per-example (sum_logp, n_comp_tokens) over completion tokens, teacher-forced."""
        res = []
        for r in rows:
            p, c = r["prompt"], r["completion"]
            pids = tok(p, add_special_tokens=False)["input_ids"]
            cids = tok(c, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([pids + cids], device=model.device)
            out = model(ids).logits.log_softmax(-1)
            # logprob of each completion token given prefix
            lp = 0.0; nt = 0
            for j in range(len(cids)):
                pos = len(pids) + j - 1
                lp = lp + out[0, pos, cids[j]]
                nt += 1
            res.append((lp, nt))
        return res

    def loss_and_grad(reduction):
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, trust_remote_code=True)
        model.train()
        lps = batch_logps(model)
        if reduction == "mean_token":   # standalone SFT: mean over ALL completion tokens
            tot_lp = sum(lp for lp, nt in lps); tot_nt = sum(nt for lp, nt in lps)
            loss = -tot_lp / tot_nt
        else:                            # hybrid forward_kl: mean over SEQUENCES of (-sum logp)
            loss = -sum(lp for lp, nt in lps) / len(lps)
        model.zero_grad(); loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        return float(loss), g

    l_sft, g_sft = loss_and_grad("mean_token")
    l_hyb, g_hyb = loss_and_grad("seq_sum")
    import torch as T
    cos = float(T.nn.functional.cosine_similarity(g_sft, g_hyb, dim=0))
    print(f"[E0a] standalone-SFT(mean-token) loss={l_sft:.4f} |g|={g_sft.norm():.3f}")
    print(f"[E0a] hybrid forward_kl(seq-sum)  loss={l_hyb:.4f} |g|={g_hyb.norm():.3f}")
    print(f"[E0a] cosine(grad_sft, grad_hybrid) = {cos:.4f}")
    print(f"[E0a] |g| ratio hybrid/sft = {g_hyb.norm()/g_sft.norm():.2f}  (reduction changes effective LR)")
    print("VERDICT: cos≈1 & ratio≈const -> same direction, differ only by LR scale (fixable). "
          "cos<0.99 -> different operator; §44 λ-sweep was not a fair SFT proxy.")

if __name__ == "__main__":
    main()
