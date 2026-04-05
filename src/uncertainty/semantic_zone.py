import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MATH_OPERATORS = {
    "=", "+", "-", "*", "/", "^", "**",
    "≤", "≥", "≠", "∈", "∉", "⊂", "⊃", "∪", "∩",
    "→", "⇒", "⟹", "≡", "≈", "∝",
    r"\leq", r"\geq", r"\neq", r"\in", r"\notin",
    r"\subset", r"\supset", r"\cup", r"\cap",
    r"\implies", r"\equiv", r"\approx",
    r"\times", r"\cdot", r"\div",
    r"\frac", r"\sqrt", r"\sum", r"\prod", r"\int",
}

_DEFAULT_TRANSITION_PHRASES = {
    "therefore", "thus", "hence", "so",
    "we get", "we have", "we obtain", "we find",
    "it follows", "this gives", "this means",
    "which gives", "which means", "this implies",
    "substituting", "simplifying", "expanding",
    "factoring", "solving", "computing",
    "note that", "observe that", "recall that",
    "since", "because",
    "plugging in", "plugging", "substitution",
    "combining", "rearranging", "applying",
}


class SemanticLoadZoneClassifier:
    def __init__(
        self,
        math_operators: Optional[set] = None,
        transition_phrases: Optional[set] = None,
        context_window: int = 5,
    ):
        self.math_operators = math_operators or _DEFAULT_MATH_OPERATORS
        self.transition_phrases = transition_phrases or _DEFAULT_TRANSITION_PHRASES
        self.context_window = context_window

        self._transition_pattern = re.compile(
            r"\b("
            + "|".join(
                re.escape(p)
                for p in sorted(self.transition_phrases, key=len, reverse=True)
            )
            + r")\b",
            re.IGNORECASE,
        )
        self._operator_pattern = re.compile(
            r"("
            + "|".join(
                re.escape(op)
                for op in sorted(self.math_operators, key=len, reverse=True)
            )
            + r")"
        )

    def is_in_zone(self, decoded_prefix: str, current_token_str: str) -> bool:
        window_chars = self.context_window * 8
        context = (
            decoded_prefix[-window_chars:]
            if len(decoded_prefix) > window_chars
            else decoded_prefix
        )

        if self._operator_pattern.search(current_token_str):
            return True

        if self._operator_pattern.search(context[-20:]):
            return True

        if self._transition_pattern.search(context[-60:]):
            return True

        if self._is_line_initial_zone(context, current_token_str):
            return True

        return False

    def _is_line_initial_zone(self, context: str, current_token_str: str) -> bool:
        lines = context.split("\n")
        if not lines:
            return False
        last_line = lines[-1].strip()
        if len(last_line) <= 5:
            if self._operator_pattern.search(last_line + current_token_str):
                return True
        return False

    def batch_classify(
        self,
        decoded_prefixes: list[str],
        token_strs: list[str],
    ) -> list[bool]:
        return [
            self.is_in_zone(prefix, tok)
            for prefix, tok in zip(decoded_prefixes, token_strs)
        ]

    @classmethod
    def from_config(cls, cfg: dict) -> "SemanticLoadZoneClassifier":
        zone_cfg = cfg.get("semantic_load_zone", {})
        raw_ops = zone_cfg.get("math_operators", [])
        raw_phrases = zone_cfg.get("transition_phrases", [])
        ops = set(raw_ops) if raw_ops else None
        phrases = set(raw_phrases) if raw_phrases else None
        window = zone_cfg.get("context_window", 5)
        return cls(
            math_operators=ops,
            transition_phrases=phrases,
            context_window=window,
        )