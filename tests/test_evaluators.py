"""Tests for evaluators (exact_match, contains, llm_judge).

Tests cover:
    - Exact match: normalization, matching, no expected answer
    - Contains: substring found/not found, no expected answer
    - LLM Judge: JSON parsing from various messy formats
"""

from llm_eval_kit.evaluators.contains import ContainsEvaluator
from llm_eval_kit.evaluators.exact_match import ExactMatchEvaluator
from llm_eval_kit.evaluators.llm_judge import LLMJudgeEvaluator
from llm_eval_kit.models.base import BaseLLM
from llm_eval_kit.schemas import TestCase


# ── Exact Match ──────────────────────────────────────────────────────────────

class TestExactMatch:
    """Group exact match tests together using a class (just for organization)."""

    def setup_method(self):
        """Runs before each test — creates a fresh evaluator."""
        self.evaluator = ExactMatchEvaluator()

    def test_exact_match(self):
        tc = TestCase(prompt="test", expected="Paris")
        result = self.evaluator.evaluate(tc, "Paris")
        assert result.score == 1.0

    def test_case_insensitive(self):
        tc = TestCase(prompt="test", expected="Paris")
        result = self.evaluator.evaluate(tc, "paris")
        assert result.score == 1.0

    def test_strips_punctuation(self):
        """'Paris.' should match 'Paris' after normalization."""
        tc = TestCase(prompt="test", expected="Paris")
        result = self.evaluator.evaluate(tc, "Paris.")
        assert result.score == 1.0

    def test_strips_whitespace(self):
        tc = TestCase(prompt="test", expected="4")
        result = self.evaluator.evaluate(tc, "  4  ")
        assert result.score == 1.0

    def test_no_match(self):
        tc = TestCase(prompt="test", expected="Paris")
        result = self.evaluator.evaluate(tc, "London")
        assert result.score == 0.0

    def test_no_expected_answer(self):
        """Without expected answer, score should be 0.0."""
        tc = TestCase(prompt="Tell me a joke")
        result = self.evaluator.evaluate(tc, "Why did the chicken...")
        assert result.score == 0.0


# ── Contains ─────────────────────────────────────────────────────────────────

class TestContains:

    def setup_method(self):
        self.evaluator = ContainsEvaluator()

    def test_found_in_sentence(self):
        """'Paris' found inside a full sentence → 1.0."""
        tc = TestCase(prompt="test", expected="Paris")
        result = self.evaluator.evaluate(tc, "The capital of France is Paris.")
        assert result.score == 1.0

    def test_not_found(self):
        tc = TestCase(prompt="test", expected="Paris")
        result = self.evaluator.evaluate(tc, "The capital is London.")
        assert result.score == 0.0

    def test_case_insensitive(self):
        tc = TestCase(prompt="test", expected="paris")
        result = self.evaluator.evaluate(tc, "PARIS is beautiful")
        assert result.score == 1.0

    def test_no_expected_answer(self):
        tc = TestCase(prompt="test")
        result = self.evaluator.evaluate(tc, "anything")
        assert result.score == 0.0


# ── LLM Judge (parsing tests) ───────────────────────────────────────────────
# We don't test the full judge pipeline (that needs a real LLM).
# Instead, we test the JSON parsing logic, which is the tricky part.

class TestLLMJudgeParsing:
    """Test the _parse_judge_response static method directly."""

    def test_clean_json(self):
        """Judge returns perfect JSON."""
        score, reasoning = LLMJudgeEvaluator._parse_judge_response(
            '{"score": 0.8, "reasoning": "Good answer"}'
        )
        assert score == 0.8
        assert reasoning == "Good answer"

    def test_json_in_code_block(self):
        """Judge wraps JSON in markdown code blocks."""
        response = '```json\n{"score": 0.9, "reasoning": "Great"}\n```'
        score, reasoning = LLMJudgeEvaluator._parse_judge_response(response)
        assert score == 0.9

    def test_json_with_extra_text(self):
        """Judge adds text before/after the JSON."""
        response = 'Here is my evaluation:\n{"score": 0.7, "reasoning": "OK"}\nThank you.'
        score, reasoning = LLMJudgeEvaluator._parse_judge_response(response)
        assert score == 0.7

    def test_score_clamped_to_max(self):
        """Score > 1.0 should be clamped to 1.0."""
        score, _ = LLMJudgeEvaluator._parse_judge_response(
            '{"score": 1.5, "reasoning": "test"}'
        )
        assert score == 1.0

    def test_score_clamped_to_min(self):
        """Score < 0.0 should be clamped to 0.0."""
        score, _ = LLMJudgeEvaluator._parse_judge_response(
            '{"score": -0.5, "reasoning": "test"}'
        )
        assert score == 0.0

    def test_unparseable_response(self):
        """Completely invalid response → score 0.0."""
        score, reasoning = LLMJudgeEvaluator._parse_judge_response(
            "I cannot evaluate this."
        )
        assert score == 0.0
        assert "Could not parse" in reasoning
