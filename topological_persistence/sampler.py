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
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.model.device)
        prompt_len = input_ids.shape[1]
        chains = []

        for i in range(n):
            logger.info(f"  Generating chain {i+1}/{n} (prompt_len={prompt_len})...")
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
            gen_ids = out.sequences[0, prompt_len:].tolist()
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            answer = self._extract_answer(text)
            truncated = len(gen_ids) >= self.cfg.max_new_tokens - 1

            hidden_states = self._collect_hidden_states(out)

            chains.append(Chain(
                text=text,
                answer=answer,
                token_ids=gen_ids,
                hidden_states=hidden_states,
                n_tokens=len(gen_ids),
                truncated=truncated,
            ))
            logger.info(f"    chain {i+1}/{n}: tokens={len(gen_ids)}, answer='{answer[:50]}', hidden_shape={hidden_states.shape if hidden_states.size > 0 else 'empty'}")
            del out
            torch.cuda.empty_cache()

        return chains

    def _collect_hidden_states(self, out) -> np.ndarray:
        if not hasattr(out, "hidden_states") or not out.hidden_states:
            return np.array([])
        states = []
        for step_idx, step_states in enumerate(out.hidden_states):
            if step_states is None or len(step_states) == 0:
                continue
            if step_idx % self.hidden_subsample != 0:
                continue
            last_layer = step_states[-1]
            h = last_layer[0, -1, :].cpu().float().numpy()
            states.append(h)
        if states:
            return np.stack(states)
        return np.array([])

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


class VLLMThenHFSampler:
    """Fast generation via vLLM (TP=8), then single HF forward pass for hidden states."""

    def __init__(self, cfg: SamplingConfig):
        self.cfg = cfg
        self._vllm = None
        self._hf_model = None
        self._hf_tokenizer = None
        self.hidden_subsample = 32

    def _unload_hf(self):
        if self._hf_model is not None:
            del self._hf_model
            self._hf_model = None
            import gc; gc.collect()
            torch.cuda.empty_cache()
            logger.info("  HF model unloaded")

    def _init_vllm(self):
        if self._vllm is not None:
            return
        self._unload_hf()
        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams
        logger.info(f"Loading vLLM with TP={self.cfg.tensor_parallel_size}")
        self._vllm = LLM(
            model=self.cfg.model_name,
            dtype=self.cfg.dtype,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=16384 + 512,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.85,
        )
        self._hf_tokenizer = self._vllm.get_tokenizer()

    def _init_hf(self):
        if self._hf_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading HF model for hidden state extraction...")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=getattr(torch, self.cfg.dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        self._hf_model.eval()
        logger.info(f"HF model loaded. Device map: {len(self._hf_model.hf_device_map)} layers")

    def sample_chains(self, prompt: str, n: int) -> list[Chain]:
        self._init_vllm()
        params = self._SamplingParams(
            n=n,
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        logger.info(f"  vLLM generating {n} chains...")
        outputs = self._vllm.generate([prompt], params)[0]

        chains = []
        for i, out in enumerate(outputs.outputs):
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
            logger.info(f"    chain {i+1}/{n}: tokens={len(token_ids)}, answer='{answer[:50]}'")

        del self._vllm
        self._vllm = None
        torch.cuda.empty_cache()
        import gc; gc.collect()
        torch.cuda.empty_cache()

        logger.info("  Extracting hidden states via HF forward pass...")
        self._init_hf()
        for i, chain in enumerate(chains):
            chain.hidden_states = self._extract_hiddens(prompt, chain.text)
            logger.info(f"    chain {i+1}/{n}: hidden_shape={chain.hidden_states.shape if chain.hidden_states.size > 0 else 'empty'}")
            torch.cuda.empty_cache()

        return chains

    @torch.no_grad()
    def _extract_hiddens(self, prompt: str, generation: str) -> np.ndarray:
        full_text = prompt + generation
        input_ids = self._hf_tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=16384 + 512
        )["input_ids"].to(self._hf_model.device)
        prompt_ids = self._hf_tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        out = self._hf_model(input_ids, output_hidden_states=True)
        h_full = out.hidden_states[-1][0, prompt_len:, :].cpu().float().numpy()
        torch.cuda.empty_cache()

        if h_full.shape[0] > 0:
            indices = list(range(0, h_full.shape[0], self.hidden_subsample))
            if not indices:
                indices = [0]
            return h_full[indices]
        return np.array([])

    def _extract_answer(self, text: str) -> str:
        import re
        matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        return matches[-1].strip() if matches else ""


def get_sampler(cfg: SamplingConfig):
    if cfg.use_vllm:
        return VLLMThenHFSampler(cfg)
    return HFSampler(cfg)
