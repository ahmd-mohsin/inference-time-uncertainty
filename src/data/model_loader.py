import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, cfg: dict):
        self.model_name = cfg["model"]["name"]
        self.short_name = cfg["model"]["short_name"]
        self.dtype_str = cfg["model"].get("dtype", "float16")
        self.device = cfg["model"].get("device", "cuda")
        self.trust_remote_code = cfg["model"].get("trust_remote_code", True)
        self.attn_impl = cfg["model"].get("attn_implementation", "eager")
        self.max_new_tokens = cfg["model"].get("max_new_tokens", 2048)

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def _resolve_dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.dtype_str, torch.float16)

    def load(self) -> tuple:
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model: {self.model_name} dtype={self.dtype_str}")

        dtype = self._resolve_dtype()

        if self.device.startswith("cuda"):
            # Normalize "cuda" -> "cuda:0" so all layers land on one GPU
            device_map = self.device if ":" in self.device else "cuda:0"
        else:
            device_map = "auto"
        kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
            "device_map": device_map,
        }

        if self.attn_impl == "flash_attention_2":
            try:
                import flash_attn
                kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash_attention_2")
            except ImportError:
                logger.warning("flash_attn not installed, falling back to eager")
                kwargs["attn_implementation"] = "eager"
        else:
            kwargs["attn_implementation"] = self.attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

        logger.info(
            f"Model loaded — parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B"
        )

        return self.model, self.tokenizer

    def get_model(self) -> AutoModelForCausalLM:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self.model

    def get_tokenizer(self) -> AutoTokenizer:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call .load() first.")
        return self.tokenizer

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        return tokens["input_ids"].to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def token_to_str(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)