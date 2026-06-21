# Chain sampler using vLLM for batched generation with hidden state extraction.
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from topological_persistence.config import SamplingConfig

logger = logging.getLogger(__name__)


@dataclass
class Chain:
    text: str
    answer: str
    token_ids: list[int] = field(default_factory=list)
    hidden_states: Optional[np.ndarray] = None
    step_boundaries: list[int] = field(default_factory=list)
    n_tokens: int = 0
    truncated: bool = False


class VLLMSampler:
    def __init__(self, cfg: SamplingConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def _init_model(self):
        if self.model is not None:
            return
        from vllm import LLM, SamplingParams
        self.LLM = LLM
        self.SamplingParams = SamplingParams
        self.model = LLM(
            model=self.cfg.model_name,
            dtype=self.cfg.dtype,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=16384,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.model.get_tokenizer()

    def sample_chains(self, prompt: str, n: int) -> list[Chain]:
        self._init_model()
        params = self.SamplingParams(
            n=n,
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        outputs = self.model.generate([prompt], params)[0]
        chains = []
        for out in outputs.outputs:
            text = out.text
            answer = self._extract_answer(text)
            token_ids = list(out.token_ids) if out.token_ids else []
            truncated = out.finish_reason == "length"
            chains.append(Chain(
                text=text,
                answer=answer,
                token_ids=token_ids,
                n_tokens=len(token_ids),
                truncated=truncated,
            ))
        return chains

    def _extract_answer(self, text: str) -> str:
        import re
        matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        return matches[-1].strip() if matches else ""


class HFSampler:
    def __init__(self, cfg: SamplingConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.hidden_subsample = 32

    def _init_model(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading {self.cfg.model_name} with device_map='auto' and output_hidden_states=True")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=getattr(torch, self.cfg.dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"Model loaded. Device map: {len(self.model.hf_device_map)} layers distributed")

    @torch.no_grad()
    def sample_chains(self, prompt: str, n: int) -> list[Chain]:
        self._init_model()
        batch_input = self.tokenizer(
            [prompt] * n, return_tensors="pt", padding=True
        ).to(self.model.device)
        prompt_len = batch_input["input_ids"].shape[1]

        logger.info(f"  Generating {n} chains in batch (prompt_len={prompt_len})...")
        out = self.model.generate(
            **batch_input,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        chains = []
        all_hidden = self._collect_hidden_states_batch(out, prompt_len, n)

        for i in range(n):
            gen_ids = out.sequences[i, prompt_len:].tolist()
            if self.tokenizer.eos_token_id in gen_ids:
                eos_pos = gen_ids.index(self.tokenizer.eos_token_id)
                gen_ids = gen_ids[:eos_pos]
            pad_id = self.tokenizer.pad_token_id
            gen_ids = [t for t in gen_ids if t != pad_id]

            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            answer = self._extract_answer(text)
            truncated = len(gen_ids) >= self.cfg.max_new_tokens - 1
            hidden = all_hidden[i] if i < len(all_hidden) else np.array([])

            chains.append(Chain(
                text=text,
                answer=answer,
                token_ids=gen_ids,
                hidden_states=hidden,
                n_tokens=len(gen_ids),
                truncated=truncated,
            ))
            logger.info(f"    chain {i+1}/{n}: tokens={len(gen_ids)}, answer='{answer[:50]}', hidden_shape={hidden.shape if hidden.size > 0 else 'empty'}")

        torch.cuda.empty_cache()
        return chains

    def _collect_hidden_states_batch(self, out, prompt_len: int, batch_size: int) -> list[np.ndarray]:
        if not hasattr(out, "hidden_states") or not out.hidden_states:
            return [np.array([]) for _ in range(batch_size)]

        per_sample = [[] for _ in range(batch_size)]
        step_count = 0
        for step_idx, step_states in enumerate(out.hidden_states):
            if step_states is None or len(step_states) == 0:
                continue
            step_count += 1
            if step_count % self.hidden_subsample != 0:
                continue
            last_layer = step_states[-1]
            for i in range(min(batch_size, last_layer.shape[0])):
                h = last_layer[i, -1, :].cpu().float().numpy()
                per_sample[i].append(h)

        result = []
        for states in per_sample:
            if states:
                result.append(np.stack(states))
            else:
                result.append(np.array([]))
        return result

    def _extract_answer(self, text: str) -> str:
        import re
        matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        return matches[-1].strip() if matches else ""


class VLLMHiddenStateSampler:
    """Two-pass sampler: vLLM for fast generation, single HF forward pass for hidden states."""

    def __init__(self, cfg: SamplingConfig):
        self.cfg = cfg
        self.vllm_sampler = VLLMSampler(cfg)
        self._hf_model = None
        self._hf_tokenizer = None

    def _init_hf(self):
        if self._hf_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=getattr(torch, self.cfg.dtype),
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self._hf_model.eval()

    def sample_chains(self, prompt: str, n: int) -> list[Chain]:
        chains = self.vllm_sampler.sample_chains(prompt, n)
        self._init_hf()
        for chain in chains:
            chain.hidden_states = self._extract_hiddens(prompt, chain.text)
        return chains

    @torch.no_grad()
    def _extract_hiddens(self, prompt: str, generation: str) -> np.ndarray:
        full_text = prompt + generation
        input_ids = self._hf_tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=16384
        )["input_ids"].to(self._hf_model.device)
        prompt_ids = self._hf_tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        out = self._hf_model(input_ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, prompt_len:, :].cpu().float().numpy()
        torch.cuda.empty_cache()
        return h


def get_sampler(cfg: SamplingConfig):
    if cfg.use_vllm:
        return VLLMSampler(cfg)
    return HFSampler(cfg)
