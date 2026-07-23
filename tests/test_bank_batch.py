# CPU test of the model-coupled bank tensorization (_bank_batch_logp) using a STUB model that
# returns fixed logits — verifies padding, prompt/completion masking, logits/label alignment, and
# that ref_logprob is carried through. No transformers/GPU needed.
import math, sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.coverage_trainer import _tokenize_bank, _bank_batch_logp


class _StubTok:
    # whitespace tokenizer: each char-int token; deterministic. pad=0.
    pad_token_id = 0
    eos_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        # map "p1 p2 | c1 c2" style: tokens are ints after simple scheme; here use ord-based ids>=1
        ids = [ (ord(ch) % 50) + 1 for ch in text if not ch.isspace() ]
        return {"input_ids": ids}


class _StubOut:
    def __init__(self, logits): self.logits = logits


class _StubModel:
    """Returns UNIFORM logits over V so every token logprob = -log(V): lets us predict the sum."""
    def __init__(self, V=53): self.V = V
    def parameters(self):
        yield torch.zeros(1)  # for device lookup in other paths (unused here)
    def __call__(self, ids):
        B, T = ids.shape
        return _StubOut(torch.zeros(B, T, self.V))


def test_tokenize_bank_masks_completion_only():
    bank = [{"prompt": "ab", "completion": "cde", "ref_logprob": -3.0}]
    items, pad = _tokenize_bank(bank, _StubTok())
    ids, mask, ref = items[0]
    assert pad == 0
    assert len(ids) == len(mask) == 5          # 2 prompt + 3 completion chars
    assert mask == [0, 0, 1, 1, 1]             # only completion tokens scored
    assert ref == -3.0


def test_bank_batch_logp_uniform_matches_formula():
    V = 53
    bank = [{"prompt": "ab", "completion": "cde", "ref_logprob": -3.0},
            {"prompt": "a",  "completion": "bcde", "ref_logprob": -4.0}]
    items, pad = _tokenize_bank(bank, _StubTok())
    model = _StubModel(V=V)
    plog, ref = _bank_batch_logp(model, items, [0, 1], pad, device="cpu")
    # completion tokens scored: item0 has 3, item1 has 4. Uniform logits -> each = -log(V).
    # BUT alignment drops the first token (labels = ids[1:]), and mask is also shifted (mask[1:]).
    # item0 mask[1:] over positions = [0,1,1,1] -> 3 scored; item1 mask[1:] = [0,1,1,1] (a|bcde:
    #   ids mask=[0,1,1,1,1], mask[1:]=[1,1,1,1] -> 4 scored). Verify counts via known sum.
    assert abs(float(plog[0]) - (3 * -math.log(V))) < 1e-4
    assert abs(float(plog[1]) - (4 * -math.log(V))) < 1e-4
    assert ref.tolist() == [-3.0, -4.0]


def test_bank_batch_logp_padding_does_not_leak():
    # mixed lengths -> padding added to shorter seq must NOT contribute (mask zero on pad).
    V = 53
    bank = [{"prompt": "a", "completion": "b", "ref_logprob": -1.0},          # short
            {"prompt": "a", "completion": "bcdef", "ref_logprob": -5.0}]      # long
    items, pad = _tokenize_bank(bank, _StubTok())
    plog, _ = _bank_batch_logp(_StubModel(V=V), items, [0, 1], pad, device="cpu")
    # short seq: 1 completion token scored -> -log(V); long: 5 -> 5*-log(V)
    assert abs(float(plog[0]) - (1 * -math.log(V))) < 1e-4
    assert abs(float(plog[1]) - (5 * -math.log(V))) < 1e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL BANK-BATCH TESTS PASSED")
