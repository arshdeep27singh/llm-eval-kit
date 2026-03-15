"""Tests for data models (schemas.py).

Tests cover:
    - TestCase creation with defaults
    - EvalResult score validation (must be 0.0-1.0)
    - EvalReport computed properties (average, passed, total)
"""

import pytest

from llm_eval_kit.schemas import EvalReport, EvalResult, TestCase


# ── TestCase ─────────────────────────────────────────────────────────────────

def test_test_case_minimal():
    """TestCase only requires a prompt — everything else is optional."""
    tc = TestCase(prompt="What is 2+2?")
    assert tc.prompt == "What is 2+2?"
    assert tc.expected is None
    assert tc.context is None
    assert tc.tags == []


def test_test_case_full():
    """TestCase with all fields populated."""
    tc = TestCase(
        prompt="Capital of France?",
        expected="Paris",
        context="France is in Europe.",
        tags=["geography", "easy"],
    )
    assert tc.expected == "Paris"
    assert tc.context == "France is in Europe."
    assert tc.tags == ["geography", "easy"]


# ── EvalResult ───────────────────────────────────────────────────────────────

def test_eval_result_valid_score():
    """Score of 0.0 and 1.0 should be accepted (boundary values)."""
    tc = TestCase(prompt="test")
    result_zero = EvalResult(test_case=tc, response="x", score=0.0, evaluator_name="test")
    result_one = EvalResult(test_case=tc, response="x", score=1.0, evaluator_name="test")
    assert result_zero.score == 0.0
    assert result_one.score == 1.0


def test_eval_result_rejects_score_above_1():
    """Score > 1.0 should be rejected by Pydantic validation."""
    tc = TestCase(prompt="test")
    with pytest.raises(Exception):  # Pydantic ValidationError
        EvalResult(test_case=tc, response="x", score=1.5, evaluator_name="test")


def test_eval_result_rejects_negative_score():
    """Score < 0.0 should be rejected."""
    tc = TestCase(prompt="test")
    with pytest.raises(Exception):
        EvalResult(test_case=tc, response="x", score=-0.1, evaluator_name="test")


# ── EvalReport ───────────────────────────────────────────────────────────────

def _make_result(score: float) -> EvalResult:
    """Helper to create an EvalResult with a given score."""
    return EvalResult(
        test_case=TestCase(prompt="test"),
        response="test",
        score=score,
        evaluator_name="test",
    )


def test_report_average_score():
    """Average of [1.0, 0.5, 0.0] should be 0.5."""
    report = EvalReport(
        model_name="test",
        results=[_make_result(1.0), _make_result(0.5), _make_result(0.0)],
    )
    assert report.average_score == pytest.approx(0.5)


def test_report_passed_count():
    """Results with score >= 0.5 count as passed."""
    report = EvalReport(
        model_name="test",
        results=[_make_result(1.0), _make_result(0.5), _make_result(0.3)],
    )
    assert report.passed == 2  # 1.0 and 0.5 pass, 0.3 doesn't
    assert report.total == 3


def test_report_empty_results():
    """Empty report should have 0 average and 0 passed."""
    report = EvalReport(model_name="test", results=[])
    assert report.average_score == 0.0
    assert report.passed == 0
    assert report.total == 0
