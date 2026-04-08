import logging
import time
from dataclasses import dataclass, field

import torch

from src.uncertainty.pretokencommitment import PreTokenCommitmentDetector, CommitmentProbe

logger = logging.getLogger(__name__)


@dataclass
class PTCSResult:
    generated_ids: list[int]
    generated_text: str
    prompt_len: int
    total_tokens: int
    commitment_divergence: float
    is_unstable: bool
    steered: bool
    selected_probe_idx: int
    n_entropy_triggers: int
    n_expansion_triggers: int
    total_expansion_tokens: int
    trigger_rate_entropy: float
    trigger_rate_expansion: float
    wall_time_sec: float
    probe_wall_time_sec: float
    trace: list = field(default_factory=list)


class PTCSGenerator:
    """
    Pre-Token Commitment Steering (PTCS).

    Before generating any token, runs K lightweight forward passes over the
    prompt with small embedding-layer noise. Measures cosine divergence of
    hidden states at the last prompt position across K passes.

    If divergence < threshold: model has a robust latent commitment.
    Generate greedily from unperturbed prompt. Cost: K forward passes only.

    If divergence >= threshold: commitment is fragile/unstable. Generate from
    the embedding that is closest to the centroid of the K probe hidden states —
    the most consensus-supported starting point — rather than from the
    highest-probability (potentially unstable) greedy start.

    This is NOT Best-of-N. Only one full generation is produced. The steering
    happens at the embedding level before token 0, not at the output level.
    """

    def __init__(
        self,
        model,
        tokenizer,
        detector: PreTokenCommitmentDetector,
        cfg: dict,
        log_detail: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.detector = detector
        self.log_detail = log_detail

        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.device = cfg["model"]["device"]

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> PTCSResult:
        t_start = time.time()
        prompt_len = prompt_ids.shape[1]

        t_probe_start = time.time()
        probe: CommitmentProbe = self.detector.probe(self.model, prompt_ids)
        probe_time = time.time() - t_probe_start

        steered = probe.is_unstable

        if steered:
            logger.debug(
                f"PTCS: unstable commitment (D={probe.divergence:.4f}), "
                f"steering to probe {probe.selected_idx}"
            )
            inputs_embeds = self.detector.get_steered_input(
                self.model, prompt_ids, probe.selected_idx
            )
            out = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        else:
            logger.debug(
                f"PTCS: stable commitment (D={probe.divergence:.4f}), greedy"
            )
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

        return PTCSResult(
            generated_ids=generated_ids,
            generated_text=generated_text,
            prompt_len=prompt_len,
            total_tokens=total,
            commitment_divergence=probe.divergence,
            is_unstable=steered,
            steered=steered,
            selected_probe_idx=probe.selected_idx,
            n_entropy_triggers=0,
            n_expansion_triggers=int(steered),
            total_expansion_tokens=0,
            trigger_rate_entropy=0.0,
            trigger_rate_expansion=float(steered),
            wall_time_sec=time.time() - t_start,
            probe_wall_time_sec=probe_time,
            trace=[],
        )