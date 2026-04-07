import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GreedyResult:
    generated_ids: list[int]
    generated_text: str
    prompt_len: int
    total_tokens: int
    n_entropy_triggers: int
    n_expansion_triggers: int
    total_expansion_tokens: int
    trigger_rate_entropy: float
    trigger_rate_expansion: float
    wall_time_sec: float


class GreedyGenerator:
    def __init__(self, model, tokenizer, cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> GreedyResult:
        t_start = time.time()
        prompt_len = prompt_ids.shape[1]

        out = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        generated_ids = out[0, prompt_len:].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total = len(generated_ids)

        return GreedyResult(
            generated_ids=generated_ids,
            generated_text=generated_text,
            prompt_len=prompt_len,
            total_tokens=total,
            n_entropy_triggers=0,
            n_expansion_triggers=0,
            total_expansion_tokens=0,
            trigger_rate_entropy=0.0,
            trigger_rate_expansion=0.0,
            wall_time_sec=time.time() - t_start,
        )