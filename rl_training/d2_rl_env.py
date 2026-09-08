# DIRECTION 2 — controlled-env RL test (run BEFORE any LLM compute, per reviewer).
# Toy policy-gradient env where the random-m-test success objective J(theta)=E_a~pi[p_a^m] is EXACT.
# Compare three reward estimators at MATCHED per-rollout verification budget N, under RLOO (unstandardized,
# leave-one-out baseline — so group-std normalization does not obscure the objective comparison):
#   single   : 1[a fixed m-subset of the N tests all pass]         (sparse; E=p^m)
#   disjoint : mean over floor(N/m) DISJOINT m-suites of all-pass  (dense-ish; E=p^m; same budget)
#   Rhat_m   : C(c,m)/C(N,m)  (Rao-Blackwell over ALL m-subsets)   (dense; E=p^m; same budget)
# passfrac  : c/N  (WRONG objective E=p; included to show it optimizes the wrong thing)
# All of single/disjoint/Rhat are UNBIASED for p^m, so they optimize the SAME objective; the question is
# whether Rhat's lower reward variance gives faster/more stable convergence at matched N. Pure numpy.
import numpy as np
from math import comb

def reward(outcomes, m, kind):
    # outcomes: (N,) bool for one rollout's N i.i.d. tests
    N = len(outcomes); c = int(outcomes.sum())
    if kind == "single":   return float(outcomes[:m].all())
    if kind == "passfrac": return c / N
    if kind == "disjoint":
        g = N // m
        if g == 0: return float(outcomes[:m].all())
        return float(np.mean([outcomes[i*m:(i+1)*m].all() for i in range(g)]))
    if kind == "Rhat":     return (comb(c, m) / comb(N, m)) if c >= m else 0.0
    raise ValueError(kind)

def run(kind, p, m, N, G=16, T=400, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    K = len(p); theta = np.zeros(K)
    Jtraj = []
    for t in range(T):
        z = theta - theta.max(); pi = np.exp(z); pi /= pi.sum()
        acts = rng.choice(K, size=G, p=pi)
        rs = np.empty(G)
        for i, a in enumerate(acts):
            outc = rng.random(N) < p[a]
            rs[i] = reward(outc, m, kind)
        # RLOO leave-one-out baseline
        adv = rs - (rs.sum() - rs) / (G - 1)
        grad = np.zeros(K)
        for i, a in enumerate(acts):
            onehot = np.zeros(K); onehot[a] = 1.0
            grad += adv[i] * (onehot - pi)      # d log pi(a)/d theta
        theta += lr * grad / G
        z = theta - theta.max(); pi = np.exp(z); pi /= pi.sum()
        Jtraj.append(float((pi * (p ** m)).sum()))   # TRUE objective E_a[p_a^m]
    return np.array(Jtraj)

def main():
    p = np.array([0.35, 0.5, 0.65, 0.8, 0.92])   # 5 "programs"; optimum = highest p (max p^m)
    Jstar = (p ** 0).max()  # placeholder
    for (N, m) in ((8, 2), (8, 4), (12, 4)):
        opt = (p ** m).max()
        print(f"\n=== N={N} m={m}  (max p^m = {opt:.4f}) — mean final J over 20 seeds, higher=better ===")
        rows = {}
        for kind in ("single", "disjoint", "Rhat", "passfrac"):
            finals = []; auc = []
            for s in range(20):
                J = run(kind, p, m, N, seed=s)
                finals.append(J[-1]); auc.append(J.mean())
            rows[kind] = (np.mean(finals), np.std(finals), np.mean(auc))
        for kind in ("single", "disjoint", "Rhat", "passfrac"):
            fm, fs, ac = rows[kind]
            tgt = "(WRONG obj: maximizes p not p^m)" if kind == "passfrac" else ""
            print(f"  {kind:9s} final J={fm:.4f} ± {fs:.4f}   AUC(J)={ac:.4f}  {tgt}")
        best = max(("single","disjoint","Rhat"), key=lambda k: rows[k][2])
        print(f"  -> fastest convergence (AUC) among objective-preserving: {best}")

if __name__ == "__main__":
    main()
