# Verified Support Floor (VSF) trainer — Astra §B1 / award method.
# Subclass TRL GRPOTrainer to add a PERSISTENT, PROMPT-BALANCED replay channel over the verified bank:
#   L = L_grpo(fresh on-policy groups)  +  vsf_lambda * L_replay
#   L_replay = -mean_{x in covered prompts} log pi_theta(y | x),  y ~ verified bank(x), prompt-balanced (stratified traversal).
# This directly tests/repairs the dead-group failure: a success discovered once keeps supplying a learning signal even when the
# current on-policy group for x is all-failure. Prompt is the unit of coverage (not the trajectory) -> stratified cycling over
# distinct prompts gives every covered prompt >= 1 successful-sequence update per cycle (the "support floor").
import json
import random
import torch
import torch.nn.functional as F
from trl import GRPOTrainer


class VSFTrainer(GRPOTrainer):
    def __init__(self, *args, vsf_bank_path: str = "", vsf_lambda: float = 1.0,
                 vsf_bsz: int = 8, vsf_maxlen: int = 900, vsf_pg_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vsf_lambda = float(vsf_lambda)
        self.vsf_bsz = int(vsf_bsz)
        self.vsf_maxlen = int(vsf_maxlen)
        self.vsf_pg_weight = float(vsf_pg_weight)  # scale on the GRPO/PG loss; 0 => replay-only-online (memo W0 VSF-minus-PG)
        self._bank = {}          # prompt -> list[completion]
        self._prompts = []       # distinct covered prompts (coverage unit)
        self._ptr = 0
        if vsf_bank_path:
            with open(vsf_bank_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    p, c = r.get("prompt"), r.get("completion")
                    if p is None or c is None:
                        continue
                    self._bank.setdefault(p, [])
                    if len(self._bank[p]) < 4:      # cap diverse solutions per prompt
                        self._bank[p].append(c)
            self._prompts = list(self._bank.keys())
            random.Random(0).shuffle(self._prompts)
        n_sol = sum(len(v) for v in self._bank.values())
        print(f"[VSF] bank: {len(self._prompts)} covered prompts, {n_sol} verified solutions, "
              f"lambda={self.vsf_lambda}, replay_bsz={self.vsf_bsz}")

    def _tok(self):
        return getattr(self, "processing_class", None) or getattr(self, "_tokenizer", None)

    def _next_replay_batch(self):
        # stratified, prompt-balanced traversal: one prompt per slot, cycling shuffled prompt order (the support floor)
        items = []
        for _ in range(self.vsf_bsz):
            if self._ptr >= len(self._prompts):
                random.shuffle(self._prompts)
                self._ptr = 0
            p = self._prompts[self._ptr]
            self._ptr += 1
            items.append((p, random.choice(self._bank[p])))
        return items

    def _replay_loss(self, model):
        tok = self._tok()
        items = self._next_replay_batch()
        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        input_ids, labels = [], []
        for p, c in items:
            try:
                pre = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                              add_generation_prompt=True)
            except Exception:
                pre = tok(p + "\n", add_special_tokens=False)["input_ids"]
            cids = tok(c, add_special_tokens=False)["input_ids"] + [eos]
            ids = (pre + cids)[: self.vsf_maxlen]
            lab = ([-100] * len(pre) + cids)[: self.vsf_maxlen]
            input_ids.append(ids)
            labels.append(lab)
        maxlen = max(len(x) for x in input_ids)
        att = []
        for i in range(len(input_ids)):
            d = maxlen - len(input_ids[i])
            att.append([1] * len(input_ids[i]) + [0] * d)
            input_ids[i] = input_ids[i] + [pad] * d
            labels[i] = labels[i] + [-100] * d
        dev = next(model.parameters()).device
        ii = torch.tensor(input_ids, device=dev)
        am = torch.tensor(att, device=dev)
        ll = torch.tensor(labels, device=dev)
        logits = model(input_ids=ii, attention_mask=am).logits[:, :-1, :]
        tgt = ll[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss, extra = (out if isinstance(out, tuple) else (out, None))
        loss = self.vsf_pg_weight * loss   # W0: pg_weight=0 => replay-only-online (drop the GRPO term, keep verified replay)
        if self.vsf_lambda > 0 and self._prompts:
            rl = self._replay_loss(model)
            loss = loss + self.vsf_lambda * rl
            if self.state.global_step % 20 == 0:
                try:
                    self.log({"vsf_replay_loss": float(rl.detach())})
                except Exception:
                    pass
        return (loss, extra) if return_outputs else loss
