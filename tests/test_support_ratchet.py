# Unit tests for the support-ratchet constraint math (no model / GPU needed).
import math
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.support_ratchet import (
    sequence_logprob, ratchet_penalty, dual_update, fraction_modes_alive,
)


def test_one_sided_above_floor_zero_penalty():
    # policy ABOVE reference -> above floor -> zero penalty (the ratchet never pulls down).
    ref = torch.tensor([-10.0, -5.0])
    pol = torch.tensor([-8.0, -4.0])            # higher logp than ref
    assert float(ratchet_penalty(pol, ref, alpha=0.5)) == 0.0


def test_penalty_active_below_floor():
    # alpha=0.5 -> floor = ref + log(0.5) ~ ref - 0.693. Drop well below floor -> positive penalty.
    ref = torch.tensor([-5.0])
    pol = torch.tensor([-8.0])
    floor = -5.0 + math.log(0.5)
    expected = floor - (-8.0)                   # relu(floor - pol)
    assert abs(float(ratchet_penalty(pol, ref, alpha=0.5)) - expected) < 1e-5


def test_alpha_one_forbids_any_drop():
    # alpha=1 -> floor = ref -> any drop below ref penalized.
    ref = torch.tensor([-5.0])
    pol = torch.tensor([-5.5])
    assert abs(float(ratchet_penalty(pol, ref, alpha=1.0)) - 0.5) < 1e-5
    # exactly at ref -> zero
    assert float(ratchet_penalty(torch.tensor([-5.0]), ref, alpha=1.0)) == 0.0


def test_alpha_slack_tolerance():
    # smaller alpha => looser floor => same drop penalized less.
    ref = torch.tensor([-5.0]); pol = torch.tensor([-6.0])
    p_tight = float(ratchet_penalty(pol, ref, alpha=1.0))   # floor=-5
    p_loose = float(ratchet_penalty(pol, ref, alpha=0.5))   # floor=-5.693
    assert p_tight > p_loose >= 0.0


def test_reduction_modes():
    ref = torch.tensor([-5.0, -5.0]); pol = torch.tensor([-8.0, -5.0])
    none = ratchet_penalty(pol, ref, alpha=1.0, reduction="none")
    assert none.shape == (2,)
    assert abs(float(ratchet_penalty(pol, ref, alpha=1.0, reduction="sum")) - float(none.sum())) < 1e-6
    assert abs(float(ratchet_penalty(pol, ref, alpha=1.0, reduction="mean")) - float(none.mean())) < 1e-6


def test_sequence_logprob_masks_and_sums():
    # 2 seqs, T=3, V=4. Build logits so token logprob is known; mask out position 0.
    B, T, V = 2, 3, 4
    logits = torch.zeros(B, T, V)
    labels = torch.tensor([[0, 1, 2], [3, 3, 3]])
    mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])  # skip prompt token at t=0
    # uniform logits -> each token logprob = log(1/V) = -log(V); summed over 2 masked positions.
    lp = sequence_logprob(logits, labels, mask)
    assert torch.allclose(lp, torch.full((B,), -2 * math.log(V)), atol=1e-5)


def test_length_cancellation_relative_floor():
    # Because the floor is ref-relative, a longer trace (more negative logp) is treated fairly:
    # only the DROP from ref matters, not absolute magnitude.
    ref = torch.tensor([-50.0])          # long trace, very negative ref logp
    pol_same = torch.tensor([-50.0])     # policy matches ref exactly
    assert float(ratchet_penalty(pol_same, ref, alpha=1.0)) == 0.0
    pol_drop = torch.tensor([-51.0])     # 1 nat drop
    assert abs(float(ratchet_penalty(pol_drop, ref, alpha=1.0)) - 1.0) < 1e-5


def test_dual_update_increases_on_violation_clips():
    # violation (mean_penalty>kappa) raises mu; satisfaction lowers; clip to [0,mu_max].
    assert dual_update(0.5, mean_penalty=1.0, kappa=0.0, eta_mu=0.1) > 0.5
    assert dual_update(0.5, mean_penalty=0.0, kappa=0.1, eta_mu=0.1) < 0.5
    assert dual_update(0.0, mean_penalty=0.0, kappa=1.0, eta_mu=0.1) == 0.0     # clip low
    assert dual_update(5.0, mean_penalty=100.0, kappa=0.0, eta_mu=0.1, mu_max=5.0) == 5.0  # clip high


def test_fraction_modes_alive():
    ref = torch.tensor([-5.0, -5.0, -5.0, -5.0])
    pol = torch.tensor([-4.0, -5.0, -5.5, -9.0])   # alpha=1 floor=-5: alive if >=-5
    # -4>=-5 alive; -5>=-5 alive; -5.5 dead; -9 dead -> 2/4
    assert abs(fraction_modes_alive(pol, ref, alpha=1.0) - 0.5) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL RATCHET TESTS PASSED")
