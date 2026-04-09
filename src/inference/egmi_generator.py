import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.uncertainty.entropy_filter import compute_entropy
from src.uncertainty.mixture_injector import EntropyGatedMixtureInjector, InjectionRecord

logger = logging.getLogger(__name__)


@dataclass
class EGMIResult:
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
    n_injections: int
    injection_positions: list[int]
    mean_injection_entropy: float
    trace: list = field(default_factory=list)


class EGMIGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        injector: EntropyGatedMixtureInjector,
        cfg: dict,
        log_detail: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.injector = injector
        self.log_detail = log_detail

        dec_cfg = cfg.get("decoding", {})
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.device = cfg["model"]["device"]

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> EGMIResult:
        t_start = time.time()
        prompt_len = prompt_ids.shape[1]

        embedding = self.model.get_input_embeddings()
        current_embeds = embedding(prompt_ids).clone()

        generated_ids: list[int] = []
        injections: list[InjectionRecord] = []

        out = self.model(inputs_embeds=current_embeds, return_dict=True)
        logits = out.logits[0, -1, :]
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        for step in range(self.max_new_tokens):
            entropy = compute_entropy(logits.unsqueeze(0)).item()

            if self.injector.should_inject(entropy, step):
                rec = InjectionRecord(
                    position=step,
                    entropy=entropy,
                    top_k=self.injector.top_k,
                    mixture_weight_spread=self.injector.mixture_weight_spread(logits),
                )
                injections.append(rec)

                mixture_embed = self.injector.build_mixture_embedding(logits)
                logits, current_embeds = self.injector.forward_with_mixture(
                    current_embeds, mixture_embed
                )
                logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)

                next_token_id = int(logits.argmax(dim=-1).item())

            else:
                next_token_id = int(logits.argmax(dim=-1).item())

            generated_ids.append(next_token_id)

            if next_token_id == self.eos_token_id:
                break

            token_embed = embedding(
                torch.tensor([[next_token_id]], device=self.device)
            )
            current_embeds = torch.cat([current_embeds, token_embed], dim=1)

            out = self.model(inputs_embeds=current_embeds, return_dict=True)
            logits = out.logits[0, -1, :]
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total = len(generated_ids)
        n_inj = len(injections)

        mean_inj_entropy = (
            float(sum(r.entropy for r in injections) / n_inj) if n_inj > 0 else 0.0
        )

        return EGMIResult(
            generated_ids=generated_ids,
            generated_text=generated_text,
            prompt_len=prompt_len,
            total_tokens=total,
            n_entropy_triggers=n_inj,
            n_expansion_triggers=n_inj,
            total_expansion_tokens=0,
            trigger_rate_entropy=n_inj / max(1, total),
            trigger_rate_expansion=n_inj / max(1, total),
            wall_time_sec=time.time() - t_start,
            n_injections=n_inj,
            injection_positions=[r.position for r in injections],
            mean_injection_entropy=mean_inj_entropy,
            trace=injections if self.log_detail else [],
        )