# DIGTE — Decision-Instability Gated Token Expansion

A training-free inference-time compute allocation method for mathematical reasoning in LLMs. DIGTE detects semantic disagreement among short continuations at high-entropy token positions inside reasoning steps, then expands the token budget in-place at those positions — improving accuracy without retraining or fine-tuning.

---

## The Core Idea in Plain English

When a language model is generating a math solution step by step, most tokens are easy — filling in arithmetic, repeating a variable name, closing a parenthesis. But a small fraction of positions are genuinely hard: the model is about to commit to an operator, a substitution, or a logical conclusion, and the wrong choice sends the entire solution off a cliff.

DIGTE identifies these hard positions in two stages:

1. **Is entropy high AND are we at a mathematically important position?** (the cheap check)
2. **If yes — do multiple short sample continuations actually disagree with each other at the hidden-state level?** (the expensive check, only run when needed)

If both conditions are true, the model is genuinely uncertain about a semantically important decision. DIGTE responds by injecting a deliberation marker and generating ΔL=50 extra tokens at that exact position before continuing greedy decoding. The model effectively gets more space to work through the uncertainty before committing.

This differs from AdaDec (the closest prior work) in a critical way. AdaDec reranks which token to pick at uncertain positions — it makes a better local choice. DIGTE expands how much the model gets to think before it makes any choice — it gives the model more space to reason. For math, where the failure mode is *confidently applying the wrong rule* rather than *picking the wrong surface token*, expansion is the right intervention.

---

## How the Two Scripts Relate

```
calibrate.py  ──────────────────────────────────────────►  learned_thresholds.json
     │                                                              │
     │  (run once per model, offline)                               │
     │                                                              ▼
     └── produces τ_e and τ_d ──────────────────────►  run_inference.py
                                                           (uses thresholds at test time)
```

`calibrate.py` is a **one-time offline step**. It does not evaluate on test problems. It studies the model's own uncertainty patterns on GSM8K training data to learn two numbers: the entropy threshold (τ_e) and the disagreement threshold (τ_d). These thresholds are saved to a JSON file. Once you have that file, you never need to run `calibrate.py` again for that model.

`run_inference.py` is the **test-time script**. It loads those two thresholds, then generates solutions to MATH500 or AIME problems using whatever decoding mode you choose.

---

## What `calibrate.py` Does, Step by Step

`calibrate.py` runs four sub-phases internally:

### Phase 1a — Token data collection (`token_collector.py`)
The model generates solutions to 500 GSM8K training problems using greedy decoding. At every token position, it records: Shannon entropy of the logit distribution, logit margin (gap between top-1 and top-2 probability), top-1 probability, whether the token is in a semantic load zone (near an operator or transition phrase), and whether the step was correct (inferred from whether the partial answer matches the gold answer at that point).

This produces a JSONL file with ~50,000–100,000 token-level records. Crucially, this is done in **O(T) forward passes per problem**, not O(T²) — the logits are collected during the generation pass itself using the KV cache, not re-computed per position.

### Phase 1b — Entropy threshold optimization (`threshold_optimizer.py`)
Using only the records from semantic load zones, a grid search over 40 candidate τ_e values finds the threshold that maximizes AUROC for predicting incorrect steps, subject to a trigger rate constraint (0.5% to 15% of zone tokens). This is why DIGTE does not use AdaDec's logistic regression approach — grid search with a trigger rate constraint is simpler, requires no labeled training data, and is more directly interpretable.

The output: a single float τ_e.

### Phase 1c — Disagreement data collection (`disagreement_collector.py`)
The model runs again on the same 500 GSM8K problems, but now at every position where entropy exceeds τ_e, it samples K=3 short continuations of length L=12 tokens. The final-layer hidden state at the last token of each continuation is extracted, normalized, and pairwise cosine similarities are computed. The disagreement score d_t = 1 − mean_similarity. This is the novel signal that AdaDec does not use.

Only entropy-triggered positions are tested here, so the compute overhead is bounded by the trigger rate (~4% of positions × 3 continuations × 12 tokens = ~1.4 extra forward passes per problem on average).

### Phase 1d — Disagreement threshold optimization (`threshold_optimizer.py`)
Same grid search procedure as Phase 1b, but now over τ_d on the disagreement scores collected in Phase 1c. This finds the threshold above which d_t reliably predicts an incorrect step among the already-entropy-filtered positions.

The output: a single float τ_d.

### Phase 1e — Analysis and plots (`analysis.py`)
Generates three figures saved to `data/calibration_outputs/figures/`:
- Entropy distribution at zone positions (correct vs. incorrect steps), with τ_e marked
- Disagreement score distribution at entropy-triggered positions, with τ_d marked
- ROC comparison: entropy vs. logit margin vs. top-1 confidence as error predictors (this is the empirical support for Claim 1 in the paper)

Final output: `data/calibration_outputs/learned_thresholds.json` with τ_e, τ_d, and all metadata.

---

## What `run_inference.py` Does, Step by Step

`run_inference.py` loads the model, loads the thresholds calibrated by `calibrate.py`, and then runs generation on MATH500 (or AIME or GSM8K test) problems.

The core generation loop in `digte_generator.py` works as follows for each problem:

```
For each token position t:

  1. Run one forward pass → get logits, greedily select next token

  2. Check: is entropy H_t > τ_e  AND  is position t in a semantic load zone?
     If NO  → commit greedy token, move to t+1

  3. If YES (entropy triggered):
     Sample K=3 short continuations of length L=12
     Extract final hidden state of each continuation
     Compute d_t = 1 − mean pairwise cosine similarity

  4. Check: is d_t > τ_d?
     If NO  → commit greedy token, move to t+1

  5. If YES (expansion triggered):
     Inject deliberation marker into sequence
     Generate ΔL=50 extra tokens greedily (in-place expansion)
     After expansion completes, resume normal greedy decoding from t+1
```

This generates a sequence that is at most max_new_tokens long (expansions count toward the budget). The output includes the generated text, accuracy, token counts, trigger rates, and — if `--log_detail` is used — a full per-token trace.

For each problem the script writes one record to a JSONL predictions file and accumulates statistics for a JSON metrics file. Every 50 problems it logs a running accuracy.

---

## The Five Decoding Modes

`run_inference.py` supports six modes via `--mode`:

| Mode | What it does | Why it exists |
|---|---|---|
| `digte` | Full DIGTE: entropy gate + disagreement gate + in-place expansion | The proposed method |
| `greedy` | Standard greedy decoding, no intervention | Baseline, compute reference point |
| `beam` | Beam search width=3 | Standard strong baseline |
| `adadec` | Entropy gate + lookahead reranking (no expansion) | Closest prior work, directly comparable |
| `entropy_only` | Entropy gate + in-place expansion (no disagreement gate) | Ablation: isolates disagreement gate's contribution |
| `prompt_only` | Entropy gate + disagreement gate + marker injection (no extra tokens) | Ablation: isolates prompt engineering vs. compute expansion |

The ablation design is a 2×2 table:

```
                    No expansion          With expansion
No disagreement    greedy / entropy_only       —
With disagreement      prompt_only           digte ← full method
```

Running `run_ablation.py` executes all four ablation conditions and prints the comparison table automatically.

---

## Repository Structure

```
digte/
│
├── calibrate.py              # Phase 1 entry point — run once per model
├── run_inference.py          # Phase 2 entry point — run at test time
├── run_ablation.py           # Runs all 4 ablation conditions, prints table
├── validate_claim1.py        # Tests whether d_t beats entropy/margin/confidence
│
├── configs/
│   ├── calibration_config.yaml   # All Phase 1 hyperparameters
│   └── inference_config.yaml     # All Phase 2 hyperparameters
│
├── environment.yml           # Conda environment
│
├── src/
│   ├── data/
│   │   ├── dataset.py         # GSM8K, MATH500, AIME loaders + answer extraction
│   │   └── model_loader.py    # HuggingFace loading (float16, flash-attn)
│   │
│   ├── uncertainty/
│   │   ├── semantic_zone.py   # Detects operator + transition phrase positions
│   │   ├── entropy_filter.py  # Shannon entropy, logit margin, top-k probs
│   │   └── disagreement.py    # K-sample hidden-state cosine disagreement
│   │
│   ├── calibration/
│   │   ├── token_collector.py          # Phase 1a: entropy + step labels
│   │   ├── disagreement_collector.py   # Phase 1c: d_t at triggered positions
│   │   ├── threshold_optimizer.py      # Phase 1b/d: grid search for τ_e, τ_d
│   │   └── analysis.py                 # Phase 1e: plots and signal comparison
│   │
│   ├── inference/
│   │   └── digte_generator.py   # Core DIGTE generation loop
│   │
│   ├── baselines/
│   │   ├── greedy.py                    # Greedy decoding
│   │   ├── beam_search.py               # Beam search (width=3)
│   │   ├── adadec_math.py               # AdaDec lookahead reranking for math
│   │   ├── entropy_only_expansion.py    # Ablation: entropy gate only
│   │   └── prompt_only.py               # Ablation: marker injection only
│   │
│   └── evaluation/
│       ├── math_eval.py        # Boxed extraction, answer matching, scoring
│       ├── metrics.py          # ProblemResult, MetricsAggregator, compare_methods
│       ├── compute_matched.py  # Pareto curves, budget-matched comparison, table
│       └── trigger_analysis.py # Where DIGTE fired and whether it helped
│
└── tests/
    └── test_phase1.py    # 53 unit tests for all Phase 1 modules
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate digte
```

---

## Running the Full Pipeline

### Step 1 — Calibrate (once per model)

```bash
python calibrate.py \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --model_short_name qwen2.5-math-7b \
    --n_problems 500
```

This takes roughly 1–3 hours on a single A100 depending on model size. It writes:
- `data/calibration_outputs/token_data_qwen2.5-math-7b.jsonl` — per-token entropy records
- `data/calibration_outputs/token_data_qwen2.5-math-7b_disagreement.jsonl` — d_t records
- `data/calibration_outputs/learned_thresholds.json` — τ_e and τ_d
- `data/calibration_outputs/figures/` — three diagnostic plots

If calibration has already been run and you just want to re-optimize thresholds:

```bash
python calibrate.py --skip_collection --metric f1
```

### Step 2 — Validate Claim 1 (optional, from Phase 1 data)

```bash
python validate_claim1.py
```

Prints AUROC for each signal (entropy, logit margin, top-1 confidence, d_t) and whether d_t wins. This is the empirical support for the core claim that motivates using disagreement as the gating signal.

### Step 3 — Run inference

```bash
# DIGTE on MATH500
python run_inference.py --mode digte --dataset math500

# Greedy baseline for comparison
python run_inference.py --mode greedy --dataset math500

# AdaDec baseline (closest prior work)
python run_inference.py --mode adadec --dataset math500

# Run all four ablation conditions automatically
python run_ablation.py
```

Results are written to:
- `results/predictions_qwen2.5-math-7b_digte.jsonl` — one record per problem
- `results/metrics_qwen2.5-math-7b_digte.json` — aggregate statistics

### Step 4 — Evaluate on AIME 2024

```bash
python run_inference.py --mode digte --dataset aime_2024
python run_inference.py --mode greedy --dataset aime_2024
```

---

## Key Config Parameters

### calibration_config.yaml

| Parameter | Default | Meaning |
|---|---|---|
| `calibration.n_problems` | 500 | GSM8K problems to calibrate on |
| `calibration.k_continuations` | 3 | Number of short samples for d_t |
| `calibration.continuation_length` | 12 | Tokens per continuation |
| `calibration.temperature` | 0.8 | Sampling temperature for continuations |
| `calibration.metric` | auroc | Threshold selection metric: `auroc`, `f1`, `precision` |
| `calibration.min_trigger_rate` | 0.005 | Minimum fraction of zone tokens that must trigger |
| `calibration.max_trigger_rate` | 0.15 | Maximum fraction of zone tokens that can trigger |

### inference_config.yaml

| Parameter | Default | Meaning |
|---|---|---|
| `decoding.expansion_delta_l` | 50 | Extra tokens injected per expansion event |
| `decoding.k_continuations` | 3 | Continuations for disagreement (must match calibration) |
| `decoding.continuation_length` | 12 | Continuation length (must match calibration) |
| `decoding.min_tokens_before_trigger` | 20 | Warm-up: no triggering in first N tokens |
| `output.log_detail` | false | If true, writes per-token trace for every problem |

---

## Output Files

After running `run_inference.py --mode digte`:

**`results/predictions_qwen2.5-math-7b_digte.jsonl`** — one JSON object per problem:
```json
{
  "problem_id": 0,
  "gold_answer": "42",
  "extracted_answer": "42",
  "correct": true,
  "total_tokens": 312,
  "n_entropy_triggers": 4,
  "n_expansion_triggers": 2,
  "total_expansion_tokens": 118,
  "trigger_rate_entropy": 0.013,
  "trigger_rate_expansion": 0.006,
  "wall_time_sec": 3.8
}
```

**`results/metrics_qwen2.5-math-7b_digte.json`** — aggregate over all problems:
```json
{
  "accuracy": 0.632,
  "mean_tokens": 298.4,
  "mean_expansion_tokens": 41.2,
  "expansion_overhead_pct": 16.0,
  "mean_entropy_trigger_rate": 0.011,
  "mean_expansion_trigger_rate": 0.005
}
```

---

## Differences from AdaDec

| Dimension | AdaDec | DIGTE |
|---|---|---|
| Intervention | Reranks top-B tokens | Expands token budget in-place |
| Uncertainty signal | Shannon entropy alone | Entropy pre-filter + hidden-state disagreement |
| Trigger granularity | Any high-entropy token | High-entropy tokens in semantic load zones only |
| Domain | Code generation | Mathematical reasoning |
| Threshold learning | Logistic regression | Grid search with trigger rate constraint |
| Evaluation | HumanEval, MBPP (Pass@1) | MATH500, AIME 2024 (accuracy) |

The key insight is that AdaDec's reranking intervention is appropriate for code generation, where errors are often *wrong token ranking mistakes* (the right token exists but isn't top-1). For math, the dominant failure mode is *confidently wrong reasoning* — the model picks a token with high confidence but it commits to the wrong rule. Giving the model more space to reason (expansion) corrects this; picking a different top-k token (reranking) does not.

---

## Tests

```bash
python -m pytest tests/test_phase1.py -v
# 53 tests, all passing, no GPU required
```