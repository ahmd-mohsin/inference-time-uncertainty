import logging
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class BeamResult:
    generated_ids: list[int]
    generated_text: str
    prompt_len: int
    total_tokens: int
    beam_width: int
    n_entropy_triggers: int
    n_expansion_triggers: int
    total_expansion_tokens: int
    trigger_rate_entropy: float
    trigger_rate_expansion: float
    wall_time_sec: float


class BeamSearchGenerator:
    def __init__(self, model, tokenizer, cfg: dict, beam_width: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.eos_token_id = tokenizer.eos_token_id
        self.device = cfg["model"]["device"]
        self.beam_width = beam_width

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> BeamResult:
        t_start = time.time()
        prompt_len = prompt_ids.shape[1]

        beams = [{
            "ids": prompt_ids.clone(),
            "past": None,
            "score": 0.0,
            "done": False,
            "generated": [],
        }]
        completed = []

        for _ in range(self.max_new_tokens):
            if not beams:
                break

            next_beams = []
            for beam in beams:
                if beam["done"]:
                    completed.append(beam)
                    continue

                out = self.model(
                    input_ids=beam["ids"] if beam["past"] is None else beam["ids"][:, -1:],
                    past_key_values=beam["past"],
                    use_cache=True,
                )
                log_probs = F.log_softmax(out.logits[:, -1, :], dim=-1)
                top_lp, top_ids = log_probs.topk(self.beam_width, dim=-1)

                for i in range(self.beam_width):
                    tid = top_ids[0, i].item()
                    new_score = beam["score"] + top_lp[0, i].item()
                    new_ids = torch.cat(
                        [beam["ids"], torch.tensor([[tid]], device=self.device)], dim=1
                    )
                    next_beams.append({
                        "ids": new_ids,
                        "past": out.past_key_values,
                        "score": new_score,
                        "done": tid == self.eos_token_id,
                        "generated": beam["generated"] + [tid],
                    })

            next_beams.sort(key=lambda x: x["score"], reverse=True)
            beams = []
            for cand in next_beams:
                if cand["done"]:
                    completed.append(cand)
                else:
                    beams.append(cand)
                if len(beams) >= self.beam_width:
                    break

            if len(completed) >= self.beam_width:
                break

        all_finished = completed + beams
        if not all_finished:
            return BeamResult(
                generated_ids=[], generated_text="", prompt_len=prompt_len,
                total_tokens=0, beam_width=self.beam_width,
                n_entropy_triggers=0, n_expansion_triggers=0, total_expansion_tokens=0,
                trigger_rate_entropy=0.0, trigger_rate_expansion=0.0,
                wall_time_sec=time.time() - t_start,
            )

        best = max(all_finished, key=lambda x: x["score"])
        generated_ids = best["generated"]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total = len(generated_ids)

        return BeamResult(
            generated_ids=generated_ids,
            generated_text=generated_text,
            prompt_len=prompt_len,
            total_tokens=total,
            beam_width=self.beam_width,
            n_entropy_triggers=0,
            n_expansion_triggers=0,
            total_expansion_tokens=0,
            trigger_rate_entropy=0.0,
            trigger_rate_expansion=0.0,
            wall_time_sec=time.time() - t_start,
        )