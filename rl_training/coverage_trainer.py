# Model-coupled trainers for the two coverage-preservation techniques, both subclassing TRL's
# GRPOTrainer and reusing the CPU-tested pure logic in support_ratchet.py / projection.py:
#
#   RatchetGRPOTrainer   — Technique-1 SOFT form: add mu * mean(one-sided floor penalty) on a bank
#                          minibatch to the GRPO loss (optional dual ascent on mu).
#   ProjectionGRPOTrainer— Technique-1 HARD form: after each optimizer step, run bounded correction
#                          sub-steps on violating banked traces until feasible (projected gradient).
#
# Both operate on a BANK of base-correct traces (coverage_bank.load_bank): each entry has
# {prompt, completion, ref_logprob}. We compute the CURRENT policy's teacher-forced seq logprob on
# these fixed traces (off-policy) — the signal a reward cannot provide once the policy stops
# sampling the mode.
#
# NOTE: the GRPO gradient machinery is TRL's; we only add a bank term / correction loop. Kept in one
# file so train_grpo just picks the trainer class by flag. The bank tensorization helper
# (_bank_batch_logp) is factored out and CPU-testable with a tiny model (tests/test_bank_batch.py).

from __future__ import annotations
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.support_ratchet import (sequence_logprob, ratchet_penalty, dual_update,
                                         fraction_modes_alive)
from rl_training.projection import (ProjectionConfig, ProjectionState, violations,
                                    max_violation, should_project, batches)


def _tokenize_bank(bank, tokenizer, max_len=1280):
    """Pre-tokenize the bank ONCE: returns list of (input_ids, comp_mask, ref_logprob) python lists.
    comp_mask marks ONLY completion tokens (prompt tokens are context, not scored).

    max_len caps total length to bound the per-trace (T,V) logits memory in the constraint forward
    pass (V~151k for Qwen). We keep the prompt (needed as context) and TRUNCATE the completion from
    the FRONT, preserving the tail that contains the boxed answer — the constraint then floors the
    policy's prob on that answer-bearing tail. NOTE: ref_logprob was computed on the FULL trace, so
    for truncated traces the floor is slightly conservative (policy logp of the tail < full); this is
    safe (one-sided) but we log how many traces were truncated so the effect is auditable."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    items = []
    n_trunc = 0
    for r in bank:
        pids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        cids = tokenizer(r["completion"], add_special_tokens=False)["input_ids"]
        budget = max_len - len(pids)
        if budget < 1:                          # pathological: prompt alone exceeds max_len
            pids = pids[-(max_len // 2):]; budget = max_len - len(pids)
        if len(cids) > budget:                  # keep the answer-bearing TAIL of the completion
            cids = cids[-budget:]; n_trunc += 1
        ids = pids + cids
        m = [0] * len(pids) + [1] * len(cids)
        items.append((ids, m, float(r["ref_logprob"])))
    if n_trunc:
        print(f">> _tokenize_bank: truncated {n_trunc}/{len(bank)} traces to max_len={max_len} "
              f"(kept answer tail; floor is conservative on these)")
    return items, pad


def _bank_batch_logp(model, items, idxs, pad, device):
    """Teacher-forced summed policy logprob for a batch of banked traces (by index into `items`).
    Returns (policy_logp: Tensor[len(idxs)], ref_logp: Tensor[len(idxs)]). Differentiable in `model`.
    """
    chunk = [items[i] for i in idxs]
    maxlen = max(len(ids) for ids, _, _ in chunk)
    ids_b, mask_b, ref_b = [], [], []
    for ids, m, ref in chunk:
        ids_b.append(ids + [pad] * (maxlen - len(ids)))
        mask_b.append(m + [0] * (maxlen - len(m)))
        ref_b.append(ref)
    ids_t = torch.tensor(ids_b, device=device)
    mask_t = torch.tensor(mask_b, dtype=torch.float32, device=device)
    logits = model(ids_t).logits                                  # (B,T,V)
    # align: logits[:, t-1] predicts token t
    plog = sequence_logprob(logits[:, :-1, :], ids_t[:, 1:], mask_t[:, 1:])
    ref_t = torch.tensor(ref_b, dtype=plog.dtype, device=device)
    return plog, ref_t


# ---------------------------------------------------------------------------------------------
# Technique-1 SOFT: penalty added to the loss (Lagrangian). One-sided => no pass@1 tax.
# ---------------------------------------------------------------------------------------------
try:
    from trl import GRPOTrainer

    class RatchetGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, bank=None, tokenizer=None, alpha=0.5, mu=0.5,
                     bank_batch=2, dual=False, dual_kappa=0.0, dual_eta=0.1, mode="floor", **kw):
            super().__init__(*args, **kw)
            tk = tokenizer or self.processing_class
            self._items, self._pad = _tokenize_bank(bank, tk)
            self._alpha, self._mu = alpha, mu
            self._bank_batch = bank_batch
            self._dual, self._dk, self._de = dual, dual_kappa, dual_eta
            self._mode = mode   # "floor" = one-sided (ours) | "anchor" = symmetric PBA/DPH-RL baseline
            self._cursor = 0
            print(f">> RatchetGRPOTrainer: bank={len(self._items)} alpha={alpha} mu={mu} dual={dual} mode={mode}")

        def _next_bank_idxs(self):
            n = len(self._items)
            idxs = [(self._cursor + j) % n for j in range(min(self._bank_batch, n))]
            self._cursor = (self._cursor + len(idxs)) % n
            return idxs

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            base = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
            loss = base[0] if isinstance(base, tuple) else base
            dev = next(model.parameters()).device
            plog, reflog = _bank_batch_logp(model, self._items, self._next_bank_idxs(), self._pad, dev)
            if self._mode == "anchor":
                from rl_training.support_ratchet import anchor_penalty
                pen = anchor_penalty(plog, reflog, reduction="mean")
            else:
                pen = ratchet_penalty(plog, reflog, alpha=self._alpha, reduction="mean")
            if self._dual:
                self._mu = dual_update(self._mu, float(pen.detach()), self._dk, self._de)
            loss = loss + self._mu * pen
            try:
                self.log({"ratchet_penalty": float(pen.detach()), "ratchet_mu": float(self._mu),
                          "modes_alive": fraction_modes_alive(plog.detach(), reflog, self._alpha)})
            except Exception:
                pass
            return (loss, base[1]) if (return_outputs and isinstance(base, tuple)) else loss

    # -----------------------------------------------------------------------------------------
    # Technique-1 HARD: projected gradient. After each optimizer step, restore feasibility.
    # -----------------------------------------------------------------------------------------
    class ProjectionGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, bank=None, tokenizer=None, proj_cfg: ProjectionConfig = None, **kw):
            super().__init__(*args, **kw)
            tk = tokenizer or self.processing_class
            self._items, self._pad = _tokenize_bank(bank, tk)
            self._cfg = proj_cfg or ProjectionConfig()
            self._pstate = ProjectionState()
            self._wcursor = 0
            # dedicated optimizer for correction sub-steps (small lr, only touches policy params)
            self._corr_opt = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                             lr=self._cfg.lr)
            print(f">> ProjectionGRPOTrainer: bank={len(self._items)} alpha={self._cfg.alpha} "
                  f"max_steps={self._cfg.max_steps} lr={self._cfg.lr}")

        def _scan_logp(self, model, dev, which):
            """Policy + ref logp over the banked traces in `which` (indices). No grad.
            Scanning a SUBSAMPLE (not all 1172) each step keeps projection cost bounded: a full-bank
            scan every step made each step ~10min (1172 forward passes x max_steps). We instead check
            a rotating window of `check_sample` traces, correct any violators found, and rotate — so
            over several steps the whole bank is covered without paying the full scan every step."""
            pl, rl = [], []
            with torch.no_grad():
                for s in range(0, len(which), self._cfg.batch_size):
                    idxs = which[s:s + self._cfg.batch_size]
                    p, r = _bank_batch_logp(model, self._items, idxs, self._pad, dev)
                    pl += p.tolist(); rl += r.tolist()
            return pl, rl

        def _next_window(self):
            """Rotating window of check_sample bank indices to scan this projection step."""
            n = len(self._items); w = min(self._cfg.check_sample, n)
            idxs = [(self._wcursor + j) % n for j in range(w)]
            self._wcursor = (self._wcursor + w) % n
            return idxs

        def training_step(self, model, inputs, num_items_in_batch=None):
            # 1) normal GRPO step (TRL does forward/backward/opt.step internally via Trainer loop).
            loss = super().training_step(model, inputs, num_items_in_batch)
            self._pstate.step += 1
            if not should_project(self._pstate.step, self._cfg):
                return loss
            # 2) PROJECTION on a ROTATING WINDOW of the bank (bounded cost). Scan `check_sample`
            #    traces this step; correct any violators found within them, up to max_steps sub-steps.
            #    Over many steps the rotating window covers the whole bank without a full scan/step.
            dev = next(model.parameters()).device
            window = self._next_window()
            corr = 0
            for _ in range(self._cfg.max_steps):
                pl, rl = self._scan_logp(model, dev, window)
                if max_violation(pl, rl, self._cfg.alpha) <= self._cfg.tol:
                    break
                vlocal = violations(pl, rl, self._cfg.alpha)          # indices INTO window
                if not vlocal:
                    break
                vio = [window[i] for i in vlocal]                     # -> bank indices
                for bt in batches(vio, self._cfg.batch_size):
                    self._corr_opt.zero_grad()
                    p, r = _bank_batch_logp(model, self._items, bt, self._pad, dev)
                    gap = ratchet_penalty(p, r, alpha=self._cfg.alpha, reduction="mean")
                    gap.backward()
                    self._corr_opt.step()
                corr += 1
            pl, rl = self._scan_logp(model, dev, window)
            self._pstate.record(max_violation(pl, rl, self._cfg.alpha), len(violations(pl, rl, self._cfg.alpha)), corr)
            try:
                self.log({"proj_max_violation": self._pstate.last_max_violation,
                          "proj_n_violations": float(self._pstate.last_n_violations),
                          "modes_alive": fraction_modes_alive(torch.tensor(pl), torch.tensor(rl), self._cfg.alpha)})
            except Exception:
                pass
            return loss

except ImportError:
    # trl not present (e.g. local CPU box) — the pure logic + _bank_batch_logp are still importable.
    GRPOTrainer = None
