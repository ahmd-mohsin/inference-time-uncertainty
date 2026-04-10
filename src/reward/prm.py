import logging
from typing import Optional, Protocol

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class PRMInterface(Protocol):

    def score_step(
        self,
        problem_text: str,
        solution_so_far: str,
        current_step: str,
    ) -> float:
        ...

    def score_steps_batch(
        self,
        problem_text: str,
        solutions_so_far: list[str],
        current_steps: list[str],
    ) -> list[float]:
        ...

class QwenMathPRM:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 4096,
        quantize_4bit: bool = False,
    ):
        from transformers import AutoTokenizer, AutoModel

        self.device = device
        self.max_length = max_length

        logger.info(f"Loading PRM: {model_name} (dtype={dtype}, 4bit={quantize_4bit})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"PRM loaded — {params / 1e9:.2f}B parameters")

    @torch.no_grad()
    def score_step(
        self,
        problem_text: str,
        solution_so_far: str,
        current_step: str,
    ) -> float:
        full_solution = solution_so_far + current_step
        conversation = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": problem_text},
            {"role": "assistant", "content": full_solution},
        ]

        conversation_str = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        input_ids = self.tokenizer(
            conversation_str,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )["input_ids"].to(self.model.device)

        outputs = self.model(input_ids=input_ids)
        if hasattr(outputs, "logits"):
            reward = torch.sigmoid(outputs.logits[0, -1]).item()
        elif hasattr(outputs, "last_hidden_state"):
            reward = self._extract_reward(outputs.last_hidden_state[0, -1])
        else:
            reward = 0.5

        return float(reward)

    @torch.no_grad()
    def score_steps_batch(
        self,
        problem_text: str,
        solutions_so_far: list[str],
        current_steps: list[str],
    ) -> list[float]:
        rewards = []
        for sol, step in zip(solutions_so_far, current_steps):
            rewards.append(self.score_step(problem_text, sol, step))
        return rewards

    def _extract_reward(self, hidden: torch.Tensor) -> float:
        return float(torch.sigmoid(hidden.mean()).item())

class DummyPRM:

    def score_step(self, problem_text, solution_so_far, current_step) -> float:
        return 0.5

    def score_steps_batch(self, problem_text, solutions_so_far, current_steps) -> list[float]:
        return [0.5] * len(solutions_so_far)

def load_prm(cfg: dict) -> PRMInterface:
    prm_cfg = cfg.get("prm", {})
    prm_type = prm_cfg.get("type", "qwen")

    if prm_type == "dummy":
        logger.info("Using DummyPRM (uniform rewards)")
        return DummyPRM()

    if prm_type == "qwen":
        return QwenMathPRM(
            model_name=prm_cfg.get("model_name", "Qwen/Qwen2.5-Math-PRM-7B"),
            device=prm_cfg.get("device", cfg.get("model", {}).get("device", "cuda")),
            dtype=prm_cfg.get("dtype", "bfloat16"),
            max_length=prm_cfg.get("max_length", 4096),
            quantize_4bit=prm_cfg.get("quantize_4bit", False),
        )

    raise ValueError(f"Unknown PRM type: {prm_type}")