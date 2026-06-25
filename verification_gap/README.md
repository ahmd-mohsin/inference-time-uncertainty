# Verification–Generation Gap

## The question

For a given problem, two capabilities measured on the **same 0–1 scale**:

- **G (generation):** can the model *produce* a correct answer? → `pass@k`
- **V (verification):** given candidate answers, can the model *pick* the correct one? → AUC of a YES/NO self-verification judgment separating correct from wrong candidates
- **Gap = V − G**

## The hypothesis (falsifiable, one line)

> **The verification gap predicts test-time scalability:** problems with high `V − G` are
> exactly the ones where best-of-N *with a verifier* lifts accuracy far above majority
> vote; problems with `V ≈ G` show no lift.

Mechanism (TTRL, core_intuitions #3): if the model can *judge* a correct answer it rarely
*generates*, test-time compute is just a search problem → solvable. If V ≈ G, more compute
is futile → genuine ceiling.

## Why this replaces the topological direction

The topological/diversity work measured properties of the *sample set* (diversity,
persistence, effective rank) and hoped they correlate with scalability. Three runs: they
did not (effective rank 0.80→0.66→chance at scale; entropy/NCD ≈ chance or tautological).
The **one signal that survived** was a hidden-state probe reading chain-correctness at
AUC 0.94 — i.e. the model *can verify*. This package studies that directly.

## What we measure, per problem

1. **G** = `pass@k` over N sampled chains (k ≤ N).
2. **V1** = generative self-verification AUC: prompt `"Is {answer} correct for {problem}?
   YES/NO"` for every distinct candidate answer; AUC of P(YES) vs ground-truth correctness.
   (AUC, not accuracy → immune to the model's YES-bias.)
3. **V2** = majority-vote-as-verifier: does the plurality answer match gold? (baseline)
4. **Gap = V1 − G**, plus the **selection lift**: accuracy of
   *verifier-best-of-N* (pick the candidate with highest P(YES)) minus *majority vote*.

## Decisive plot / number

Scatter **G (x) vs V1 (y)** → three regimes:
- high-V, low-G → recoverable (best-of-N / TTRL works)
- V ≈ G        → judging as hard as generating → ceiling
- low-V, low-G → can't even verify → hard ceiling

Then correlate `Gap` against `(verifier-best-of-N acc − majority-vote acc)`. Strong
positive ⇒ the gap is a real, cheap scalability signal.

## Critical fixes vs the topo runs
- `max_new_tokens = 32768` (topo run truncated 41% of chains at 16384 → contaminated labels)
- No hidden states needed → light, fast.
- Small models: `Qwen/Qwen3-4B` (default). `--model` overridable.

## Files
```
verification_gap/
├── config.py          # GapConfig (model, n_chains, n_verify, max_tokens, dataset)
├── verifier.py        # build_verify_prompt, parse_yes_no, P(YES) extraction
├── run_gap.py         # generate N chains -> verify candidates -> per-problem G/V/gap JSON
└── analyze_gap.py     # aggregate, regimes, gap-vs-lift correlation, plots
```

## Usage
```bash
# 8 GPUs, data-parallel over problems (one shard per GPU)
bash scripts/run_gap.sh
# single GPU
python -m verification_gap.run_gap --model Qwen/Qwen3-4B --dataset aime_all \
    --n-problems 90 --n-chains 16 --output-dir data/verification_gap_qwen4b
python -m verification_gap.analyze_gap --data-dir data/verification_gap_qwen4b
```
