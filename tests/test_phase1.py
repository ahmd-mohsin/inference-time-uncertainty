import json
import sys
import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataset:
    def test_extract_gsm8k_gold_hash_format(self):
        from src.data.dataset import _extract_gsm8k_gold
        assert _extract_gsm8k_gold("Step 1. ...\n#### 42") == "42"

    def test_extract_gsm8k_gold_with_comma(self):
        from src.data.dataset import _extract_gsm8k_gold
        assert _extract_gsm8k_gold("#### 1,234") == "1234"

    def test_extract_gsm8k_gold_fallback(self):
        from src.data.dataset import _extract_gsm8k_gold
        result = _extract_gsm8k_gold("The answer is 7")
        assert result == "The answer is 7"

    def test_answers_match_exact(self):
        from src.data.dataset import answers_match
        assert answers_match("42", "42")
        assert answers_match("42", "42.0")
        assert answers_match("1234", "1,234")

    def test_answers_match_numeric_tolerance(self):
        from src.data.dataset import answers_match
        assert answers_match("3.14159", "3.14159")
        assert not answers_match("3.1", "3.2")

    def test_answers_match_none(self):
        from src.data.dataset import answers_match
        assert not answers_match(None, "42")

    def test_answers_match_fraction(self):
        from src.data.dataset import answers_match
        assert answers_match("1/2", "1/2")

    def test_normalize_answer_integer(self):
        from src.data.dataset import normalize_answer
        assert normalize_answer("42.0") == "42"
        assert normalize_answer("  42  ") == "42"
        assert normalize_answer("1,234") == "1234"

    def test_extract_boxed_simple(self):
        from src.data.dataset import extract_boxed_answer
        assert extract_boxed_answer(r"the answer is \boxed{42}") == "42"

    def test_extract_boxed_nested(self):
        from src.data.dataset import extract_boxed_answer
        assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_extract_boxed_none(self):
        from src.data.dataset import extract_boxed_answer
        assert extract_boxed_answer("no boxed here") is None

    def test_extract_numeric_prefers_boxed(self):
        from src.data.dataset import extract_numeric_answer
        result = extract_numeric_answer(r"so the answer is \boxed{7} but also 99")
        assert result == "7"

    def test_extract_numeric_fallback(self):
        from src.data.dataset import extract_numeric_answer
        result = extract_numeric_answer("the answer is 42")
        assert result == "42"

    def test_format_prompt_qwen(self):
        from src.data.dataset import format_prompt
        problem = {"question": "What is 2+2?"}
        prompt = format_prompt(problem, "Qwen/Qwen2.5-Math-7B-Instruct")
        assert "<|im_start|>system" in prompt
        assert "What is 2+2?" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_format_prompt_generic(self):
        from src.data.dataset import format_prompt
        problem = {"question": "What is 2+2?"}
        prompt = format_prompt(problem, "llama-base")
        assert "What is 2+2?" in prompt
        assert "System:" in prompt


class TestSemanticZone:
    def setup_method(self):
        from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
        self.clf = SemanticLoadZoneClassifier()

    def test_operator_token_triggers(self):
        assert self.clf.is_in_zone("Let x be a number", "=") is True

    def test_latex_operator_triggers(self):
        assert self.clf.is_in_zone("We have", r"\frac") is True

    def test_transition_phrase_triggers(self):
        assert self.clf.is_in_zone("Step 1: we compute x+y therefore", "the") is True

    def test_neutral_token_no_trigger(self):
        result = self.clf.is_in_zone("Once upon a time", "a")
        assert result is False

    def test_recent_operator_in_context_triggers(self):
        assert self.clf.is_in_zone("x = 5 and y", "5") is True

    def test_from_config_empty_lists_uses_defaults(self):
        from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
        cfg = {"semantic_load_zone": {"math_operators": [], "transition_phrases": []}}
        clf = SemanticLoadZoneClassifier.from_config(cfg)
        assert clf.math_operators is not None
        assert len(clf.math_operators) > 0

    def test_from_config_custom_operators(self):
        from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
        cfg = {
            "semantic_load_zone": {
                "math_operators": ["CUSTOM_OP"],
                "transition_phrases": [],
                "context_window": 3,
            }
        }
        clf = SemanticLoadZoneClassifier.from_config(cfg)
        assert "CUSTOM_OP" in clf.math_operators

    def test_batch_classify(self):
        results = self.clf.batch_classify(
            ["Let x", "Once upon"],
            ["=", "a"],
        )
        assert results[0] is True
        assert results[1] is False


class TestEntropyFilter:
    def test_compute_entropy_uniform(self):
        from src.uncertainty.entropy_filter import compute_entropy
        logits = torch.zeros(1000)
        entropy = compute_entropy(logits).item()
        assert abs(entropy - np.log(1000)) < 0.01

    def test_compute_entropy_certain(self):
        from src.uncertainty.entropy_filter import compute_entropy
        logits = torch.zeros(1000)
        logits[0] = 100.0
        entropy = compute_entropy(logits).item()
        assert entropy < 0.01

    def test_compute_logit_margin(self):
        from src.uncertainty.entropy_filter import compute_logit_margin
        logits = torch.zeros(10)
        logits[0] = 5.0
        logits[1] = 2.0
        probs = torch.softmax(logits, dim=-1)
        margin = compute_logit_margin(logits).item()
        assert 0 <= margin <= 1

    def test_compute_top_probs(self):
        from src.uncertainty.entropy_filter import compute_top_probs
        logits = torch.zeros(10)
        logits[0] = 5.0
        top_probs = compute_top_probs(logits, k=2)
        assert top_probs.shape[-1] == 2
        assert top_probs[0] > top_probs[1]

    def test_should_trigger_below_min_tokens(self):
        from src.uncertainty.entropy_filter import EntropyPreFilter
        f = EntropyPreFilter(threshold=0.0, min_tokens_before_trigger=20)
        logits = torch.zeros(100)
        triggered, entropy = f.should_trigger(logits, position=5, in_semantic_zone=True)
        assert triggered is False

    def test_should_trigger_not_in_zone(self):
        from src.uncertainty.entropy_filter import EntropyPreFilter
        f = EntropyPreFilter(threshold=0.0, min_tokens_before_trigger=0)
        logits = torch.zeros(100)
        triggered, entropy = f.should_trigger(logits, position=25, in_semantic_zone=False)
        assert triggered is False

    def test_should_trigger_high_entropy(self):
        from src.uncertainty.entropy_filter import EntropyPreFilter
        f = EntropyPreFilter(threshold=0.5, min_tokens_before_trigger=0)
        logits = torch.zeros(1000)
        triggered, entropy = f.should_trigger(logits, position=25, in_semantic_zone=True)
        assert triggered is True
        assert entropy > 0.5

    def test_should_trigger_low_entropy(self):
        from src.uncertainty.entropy_filter import EntropyPreFilter
        f = EntropyPreFilter(threshold=5.0, min_tokens_before_trigger=0)
        logits = torch.zeros(1000)
        logits[0] = 100.0
        triggered, entropy = f.should_trigger(logits, position=25, in_semantic_zone=True)
        assert triggered is False

    def test_full_record(self):
        from src.uncertainty.entropy_filter import EntropyPreFilter
        f = EntropyPreFilter(threshold=1.0, min_tokens_before_trigger=0)
        logits = torch.zeros(100)
        record = f.full_record(
            logits=logits,
            position=5,
            token_id=0,
            token_str="=",
            in_semantic_zone=True,
        )
        assert record.position == 5
        assert record.entropy > 0
        assert 0 <= record.top1_prob <= 1
        assert 0 <= record.logit_margin <= 1


class TestDisagreementDetector:
    def _make_mock_model(self, hidden_dim: int = 32, vocab_size: int = 100):
        param = torch.zeros(1)

        class FakeOut:
            def __init__(self, past):
                self.logits = torch.randn(1, 1, vocab_size)
                hidden = torch.randn(1, 1, hidden_dim)
                self.hidden_states = [hidden, hidden]
                self.past_key_values = past if past is not None else (
                    (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),
                )

        def fake_forward(*args, **kwargs):
            past = kwargs.get("past_key_values")
            return FakeOut(past)

        model = MagicMock(side_effect=fake_forward)
        model.parameters.return_value = iter([param])
        return model

    def test_compute_returns_record(self):
        from src.uncertainty.disagreement import SemanticDisagreementDetector
        model = self._make_mock_model()
        detector = SemanticDisagreementDetector(k_continuations=2, continuation_length=3)
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        past = ((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),)
        record = detector.compute(model=model, input_ids=input_ids, past_key_values=past, position=5)
        assert 0.0 <= record.disagreement_score <= 1.0
        assert len(record.continuation_token_ids) == 2
        assert len(record.pairwise_similarities) == 1

    def test_disagreement_score_range(self):
        from src.uncertainty.disagreement import SemanticDisagreementDetector
        model = self._make_mock_model()
        detector = SemanticDisagreementDetector(k_continuations=3, continuation_length=4)
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        past = ((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),)
        record = detector.compute(model=model, input_ids=input_ids, past_key_values=past, position=0)
        assert 0.0 <= record.disagreement_score <= 1.0

    def test_should_expand_above_threshold(self):
        from src.uncertainty.disagreement import SemanticDisagreementDetector, ContinuationDisagreement
        detector = SemanticDisagreementDetector(disagreement_threshold=0.3)
        record = ContinuationDisagreement(
            position=0, disagreement_score=0.8,
            pairwise_similarities=[], continuation_token_ids=[],
            continuation_log_probs=[], triggered=True,
        )
        assert detector.should_expand(record) is True

    def test_should_not_expand_below_threshold(self):
        from src.uncertainty.disagreement import SemanticDisagreementDetector, ContinuationDisagreement
        detector = SemanticDisagreementDetector(disagreement_threshold=0.3)
        record = ContinuationDisagreement(
            position=0, disagreement_score=0.1,
            pairwise_similarities=[], continuation_token_ids=[],
            continuation_log_probs=[], triggered=False,
        )
        assert detector.should_expand(record) is False

    def test_clone_past_key_values(self):
        from src.uncertainty.disagreement import _clone_past_key_values
        past = (
            (torch.tensor([1.0]), torch.tensor([2.0])),
            (torch.tensor([3.0]), torch.tensor([4.0])),
        )
        cloned = _clone_past_key_values(past)
        assert cloned is not None
        assert len(cloned) == 2
        cloned[0][0][0] = 99.0
        assert past[0][0][0].item() == 1.0

    def test_clone_none_returns_none(self):
        from src.uncertainty.disagreement import _clone_past_key_values
        assert _clone_past_key_values(None) is None


class TestThresholdOptimizer:
    def _make_records(self, n: int = 200) -> list[dict]:
        rng = np.random.default_rng(42)
        records = []
        for i in range(n):
            is_correct = rng.random() > 0.4
            base_entropy = 0.5 if is_correct else 2.5
            entropy = float(rng.normal(base_entropy, 0.5))
            records.append({
                "problem_id": i // 50,
                "position": i % 50,
                "entropy": max(0.0, entropy),
                "logit_margin": float(rng.uniform(0.0, 0.5)),
                "top1_prob": float(rng.uniform(0.3, 0.9)),
                "top2_prob": float(rng.uniform(0.0, 0.3)),
                "in_semantic_zone": bool(rng.random() > 0.5),
                "token_id": int(rng.integers(0, 1000)),
                "token_str": "=",
                "is_correct_step": is_correct,
                "final_answer_correct": is_correct,
            })
        return records

    def _make_disagreement_records(self, n: int = 100) -> list[dict]:
        rng = np.random.default_rng(42)
        records = []
        for i in range(n):
            is_correct = rng.random() > 0.4
            base_d = 0.1 if is_correct else 0.7
            d_score = float(rng.beta(2, 5) if is_correct else rng.beta(5, 2))
            records.append({
                "problem_id": i // 20,
                "position": i % 20,
                "entropy": float(rng.uniform(1.5, 4.0)),
                "disagreement_score": d_score,
                "in_semantic_zone": True,
                "token_id": int(rng.integers(0, 1000)),
                "token_str": "=",
                "is_correct_step": is_correct,
                "final_answer_correct": is_correct,
                "entropy_triggered": True,
            })
        return records

    def _make_cfg(self) -> dict:
        return {
            "calibration": {
                "entropy_threshold_range": {"min": 0.1, "max": 4.0, "n_steps": 20},
                "disagreement_threshold_range": {"min": 0.05, "max": 0.95, "n_steps": 20},
                "metric": "auroc",
                "min_trigger_rate": 0.005,
                "max_trigger_rate": 0.8,
            }
        }

    def test_optimize_entropy_returns_valid_tau(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        records = self._make_records(300)
        result = opt.optimize_entropy_threshold(records)
        assert "tau_e" in result
        assert 0.1 <= result["tau_e"] <= 4.0
        assert "entropy_auroc_full" in result
        assert 0.0 <= result["entropy_auroc_full"] <= 1.0

    def test_optimize_entropy_empty_zone(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        records = [{"in_semantic_zone": False, "entropy": 1.0, "is_correct_step": True}]
        result = opt.optimize_entropy_threshold(records)
        assert result["tau_e"] == 1.0

    def test_optimize_disagreement_returns_valid_tau(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        records = self._make_disagreement_records(150)
        result = opt.optimize_disagreement_threshold(records)
        assert "tau_d" in result
        assert 0.05 <= result["tau_d"] <= 0.95
        assert 0.0 <= result["disagreement_auroc_full"] <= 1.0

    def test_optimize_disagreement_no_triggered(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        records = [{"entropy_triggered": False, "disagreement_score": 0.5, "is_correct_step": True}]
        result = opt.optimize_disagreement_threshold(records)
        assert result["tau_d"] == 0.3

    def test_compare_signals(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        records = self._make_records(300)
        result = opt.compare_signals(records)
        for key in ["entropy_auroc", "neg_margin_auroc", "neg_top1_auroc"]:
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    def test_save_and_load_thresholds(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "thresholds.json")
            opt.save_thresholds(
                model_short_name="test-model",
                tau_e=1.847,
                tau_d=0.312,
                metadata={"entropy_auroc_full": 0.72, "signal_comparison": {}},
                path=path,
            )
            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert "test-model" in data
            assert data["test-model"]["tau_e"] == 1.847
            assert data["test-model"]["tau_d"] == 0.312

            loaded_tau_e, loaded_tau_d = opt.load_thresholds("test-model", path)
            assert loaded_tau_e == 1.847
            assert loaded_tau_d == 0.312

    def test_save_thresholds_accumulates(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "thresholds.json")
            opt.save_thresholds("model-a", 1.0, 0.2, {}, path)
            opt.save_thresholds("model-b", 2.0, 0.4, {}, path)
            with open(path) as f:
                data = json.load(f)
            assert "model-a" in data
            assert "model-b" in data

    def test_load_thresholds_missing_file(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        with pytest.raises(FileNotFoundError):
            opt.load_thresholds("x", "/nonexistent/path.json")

    def test_load_thresholds_missing_model(self):
        from src.calibration.threshold_optimizer import ThresholdOptimizer
        cfg = self._make_cfg()
        opt = ThresholdOptimizer(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "thresholds.json")
            with open(path, "w") as f:
                json.dump({"other-model": {"tau_e": 1.0, "tau_d": 0.3}}, f)
            with pytest.raises(KeyError):
                opt.load_thresholds("missing-model", path)


class TestTokenCollectorLabels:
    def test_label_steps_final_wrong(self):
        from src.calibration.token_collector import TokenDataCollector
        collector = TokenDataCollector.__new__(TokenDataCollector)
        collector.tokenizer = MagicMock()
        collector.tokenizer.decode = lambda ids, **kw: "wrong text"

        with patch("src.calibration.token_collector.extract_numeric_answer", return_value=None):
            with patch("src.calibration.token_collector.answers_match", return_value=False):
                labels = collector._label_steps(
                    generated_ids=[1, 2, 3, 4, 5],
                    gold="42",
                    final_correct=False,
                )
        assert labels == [False] * 5

    def test_label_steps_final_correct_all_true(self):
        from src.calibration.token_collector import TokenDataCollector
        collector = TokenDataCollector.__new__(TokenDataCollector)
        collector.tokenizer = MagicMock()
        collector.tokenizer.decode = lambda ids, **kw: "text"

        call_count = [0]
        def mock_extract(text):
            call_count[0] += 1
            return "42" if call_count[0] >= 3 else None

        with patch("src.calibration.token_collector.extract_numeric_answer", side_effect=mock_extract):
            with patch("src.calibration.token_collector.answers_match", return_value=True):
                labels = collector._label_steps(
                    generated_ids=[1, 2, 3, 4, 5],
                    gold="42",
                    final_correct=True,
                )
        assert len(labels) == 5
        assert all(isinstance(v, bool) for v in labels)


class TestCalibrationAnalyzer:
    def _make_cfg(self, tmpdir: str) -> dict:
        return {
            "model": {"short_name": "test-model"},
            "output": {"figures_dir": str(Path(tmpdir) / "figures")},
        }

    def _make_token_records(self, n: int = 100) -> list[dict]:
        rng = np.random.default_rng(0)
        return [
            {
                "entropy": float(rng.uniform(0, 4)),
                "logit_margin": float(rng.uniform(0, 0.5)),
                "top1_prob": float(rng.uniform(0.1, 0.9)),
                "in_semantic_zone": bool(rng.random() > 0.4),
                "is_correct_step": bool(rng.random() > 0.4),
            }
            for _ in range(n)
        ]

    def _make_disagreement_records(self, n: int = 50) -> list[dict]:
        rng = np.random.default_rng(1)
        return [
            {
                "disagreement_score": float(rng.uniform(0, 1)),
                "is_correct_step": bool(rng.random() > 0.4),
                "entropy_triggered": True,
            }
            for _ in range(n)
        ]

    def test_plot_entropy_distribution_returns_stats(self):
        from src.calibration.analysis import CalibrationAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            analyzer = CalibrationAnalyzer(cfg)
            records = self._make_token_records()
            stats = analyzer.plot_entropy_distribution(records, tau_e=1.5, save=False)
            assert "correct_mean_entropy" in stats
            assert "trigger_rate_at_tau_e" in stats
            assert 0.0 <= stats["trigger_rate_at_tau_e"] <= 1.0

    def test_plot_disagreement_distribution_returns_stats(self):
        from src.calibration.analysis import CalibrationAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            analyzer = CalibrationAnalyzer(cfg)
            records = self._make_disagreement_records()
            stats = analyzer.plot_disagreement_distribution(records, tau_d=0.3, save=False)
            assert "expand_rate_at_tau_d" in stats
            assert 0.0 <= stats["expand_rate_at_tau_d"] <= 1.0

    def test_plot_empty_zone_records_returns_empty(self):
        from src.calibration.analysis import CalibrationAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            analyzer = CalibrationAnalyzer(cfg)
            records = [{"in_semantic_zone": False, "entropy": 1.0, "is_correct_step": True}]
            stats = analyzer.plot_entropy_distribution(records, tau_e=1.0, save=False)
            assert stats == {}

    def test_roc_comparison_returns_aurocs(self):
        from src.calibration.analysis import CalibrationAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            analyzer = CalibrationAnalyzer(cfg)
            records = self._make_token_records(200)
            for r in records:
                r["in_semantic_zone"] = True
            results = analyzer.plot_signal_comparison_roc(records, save=False)
            for key in ["Entropy ($H_t$)", "Neg. Logit Margin", "Neg. Top-1 Prob"]:
                if key in results:
                    assert 0.0 <= results[key] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])