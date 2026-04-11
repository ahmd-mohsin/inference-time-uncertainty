import logging
from typing import Optional, Protocol

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PRMInterface(Protocol):

    def score_step(
        self, problem_text: str, solution_so_far: str, current_step: str,
    ) -> float: ...

    def score_steps_batch(
        self, problem_text: str, solutions_so_far: list[str], current_steps: list[str],
    ) -> list[float]: ...


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        all_scores_res.append(positive_probs.cpu().tolist())
    return all_scores_res


class QwenMathPRM:

    STEP_SEP_TOKEN = "<extra_0>"

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
        self.model_name = model_name

        logger.info(f"Loading PRM: {model_name} (dtype={dtype}, 4bit={quantize_4bit})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
        load_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        self.step_sep_id = self.tokenizer.convert_tokens_to_ids(self.STEP_SEP_TOKEN)
        if self.step_sep_id is None or self.step_sep_id == self.tokenizer.unk_token_id:
            self.step_sep_id = self.tokenizer.encode(self.STEP_SEP_TOKEN, add_special_tokens=False)[0]

        self._run_self_test()

        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"PRM loaded — {params / 1e9:.2f}B parameters, sep_id={self.step_sep_id}")

    def _run_self_test(self):
        problem = "What is 2+2?"
        solution_steps = ["We need to add 2 and 2.", "2 + 2 = 4.", "The answer is \\boxed{4}."]
        input_ids = self._build_token_ids(problem, solution_steps)
        n_sep = (input_ids == self.step_sep_id).sum().item()
        logger.info(f"PRM self-test: {len(solution_steps)} steps -> {input_ids.shape[1]} tokens, {n_sep} separators")

        if n_sep != len(solution_steps):
            logger.error(f"PRM self-test FAILED: expected {len(solution_steps)} seps, got {n_sep}")
            return

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs[0]
            token_masks = (input_ids == self.step_sep_id)
            rewards = make_step_rewards(logits, token_masks)

        if rewards and rewards[0]:
            logger.info(f"PRM self-test PASSED: step rewards = {[round(r, 4) for r in rewards[0]]}")
        else:
            logger.error(f"PRM self-test FAILED: make_step_rewards returned empty")

    def _build_token_ids(self, problem_text: str, steps: list[str]) -> torch.Tensor:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": problem_text},
            {"role": "assistant", "content": "PLACEHOLDER"},
        ]

        conversation_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        before_placeholder, after_placeholder = conversation_str.split("PLACEHOLDER", 1)

        before_ids = self.tokenizer.encode(before_placeholder, add_special_tokens=False)
        after_ids = self.tokenizer.encode(after_placeholder, add_special_tokens=False)

        content_ids = []
        for i, step in enumerate(steps):
            step_ids = self.tokenizer.encode(step, add_special_tokens=False)
            content_ids.extend(step_ids)
            content_ids.append(self.step_sep_id)

        all_ids = before_ids + content_ids + after_ids
        return torch.tensor([all_ids], dtype=torch.long, device=self.model.device)

    def _solution_to_steps(self, full_solution: str) -> list[str]:
        steps = full_solution.split("\n\n")
        steps = [s.strip() for s in steps if s.strip()]
        if not steps:
            steps = [full_solution.strip()] if full_solution.strip() else ["(no content)"]
        return steps

    @torch.no_grad()
    def score_step(self, problem_text: str, solution_so_far: str, current_step: str) -> float:
        full_solution = solution_so_far + current_step
        steps = self._solution_to_steps(full_solution)
        input_ids = self._build_token_ids(problem_text, steps)

        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]

        token_masks = (input_ids == self.step_sep_id)
        n_sep = token_masks.sum().item()
        if n_sep == 0:
            return 0.5

        outputs = self.model(input_ids=input_ids)
        logits = outputs[0]
        step_rewards = make_step_rewards(logits, token_masks)

        if step_rewards and step_rewards[0]:
            return float(step_rewards[0][-1])
        return 0.5

    @torch.no_grad()
    def score_solution(self, problem_text: str, full_solution: str) -> tuple[float, list[float]]:
        steps = self._solution_to_steps(full_solution)
        input_ids = self._build_token_ids(problem_text, steps)

        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]

        token_masks = (input_ids == self.step_sep_id)
        if token_masks.sum().item() == 0:
            return 0.5, [0.5]

        outputs = self.model(input_ids=input_ids)
        logits = outputs[0]
        step_rewards = make_step_rewards(logits, token_masks)

        if step_rewards and step_rewards[0]:
            scores = step_rewards[0]
            return min(scores), scores
        return 0.5, [0.5]

    @torch.no_grad()
    def score_steps_batch(self, problem_text: str, solutions_so_far: list[str], current_steps: list[str]) -> list[float]:
        return [self.score_step(problem_text, sol, step) for sol, step in zip(solutions_so_far, current_steps)]


class DummyPRM:
    def score_step(self, problem_text, solution_so_far, current_step) -> float:
        return 0.5
    def score_solution(self, problem_text, full_solution) -> tuple[float, list[float]]:
        return 0.5, [0.5]
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