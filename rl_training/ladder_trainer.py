# Tier-1 ladder trainer (Award Register): subclass TRL GRPOTrainer to inject the estimator KNOBS that the
# success-gradient identity (§111) says must explain the RFT>>GRPO OOD gap. Transforms the group-normalized
# advantages produced by the parent (grpo_trainer.py:2310-2355) before they enter the loss:
#   none         : standard GRPO (advantage = R - group_mean, /std iff scale_rewards != none)
#   zeroneg (R3) : advantages.clamp(min=0)      -> zero negative advantages (only reinforce successes; keeps group-relative magnitude)
#   successcount (R4): (advantages > 0).float() -> unit weight per verified success => CROSS-PROMPT success-count weighting (RFT's implicit
#                      weighting), positives-only, no per-group normalization
# Combine with --scale-rewards {group,none} (R1) and --beta 0 / high epsilon (R2) to walk the full ladder from GRPO to RFT.
import torch
from trl import GRPOTrainer


class LadderGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, adv_transform: str = "none", **kwargs):
        self._adv_transform = adv_transform
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, *args, **kwargs):
        out = super()._generate_and_score_completions(*args, **kwargs)
        t = getattr(self, "_adv_transform", "none")
        if t == "none":
            return out
        adv = out["advantages"]
        if t == "zeroneg":
            out["advantages"] = adv.clamp(min=0)
        elif t == "successcount":
            out["advantages"] = (adv > 0).to(adv.dtype)
        else:
            raise ValueError(f"unknown adv_transform: {t}")
        return out
