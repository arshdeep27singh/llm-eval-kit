"""Tests for the runner (YAML loading, factories, pipeline).

Tests cover:
    - Factory functions (get_model, get_evaluator)
    - YAML config loading
    - Full dry-run pipeline
"""

import pytest

from llm_eval_kit.evaluators.contains import ContainsEvaluator
from llm_eval_kit.evaluators.exact_match import ExactMatchEvaluator
from llm_eval_kit.models.ollama import OllamaModel
from llm_eval_kit.runner import get_evaluator, get_model, load_suite, run_eval
from llm_eval_kit.schemas import EvalSuiteConfig, TestCase


# ── Factory: get_model ───────────────────────────────────────────────────────

def test_get_model_ollama():
    model = get_model("ollama", "llama3")
    assert isinstance(model, OllamaModel)
    assert model.name == "ollama/llama3"


def test_get_model_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_model("invalid_provider", "some-model")


# ── Factory: get_evaluator ───────────────────────────────────────────────────

def test_get_evaluator_exact_match():
    ev = get_evaluator("exact_match")
    assert isinstance(ev, ExactMatchEvaluator)


def test_get_evaluator_contains():
    ev = get_evaluator("contains")
    assert isinstance(ev, ContainsEvaluator)


def test_get_evaluator_unknown():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        get_evaluator("nonexistent")


# ── YAML loading ─────────────────────────────────────────────────────────────

def test_load_suite_sample(tmp_path):
    """Load a minimal YAML suite file.

    tmp_path is a pytest fixture — it creates a temporary directory
    that gets cleaned up automatically after the test.
    """
    yaml_content = """
model:
  provider: ollama
  name: llama3
evaluator: exact_match
test_cases:
  - prompt: "What is 2+2?"
    expected: "4"
  - prompt: "Capital of France?"
    expected: "Paris"
"""
    suite_file = tmp_path / "test_suite.yaml"
    suite_file.write_text(yaml_content)

    config = load_suite(suite_file)
    assert config.model_provider == "ollama"
    assert config.model_name == "llama3"
    assert config.evaluator == "exact_match"
    assert len(config.test_cases) == 2
    assert config.test_cases[0].prompt == "What is 2+2?"


def test_load_suite_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_suite("/nonexistent/path.yaml")


# ── Dry-run pipeline ────────────────────────────────────────────────────────

def test_dry_run_pipeline():
    """Full pipeline test using dry-run mode (no real LLM needed).

    The dry-run model returns expected answers, so all scores should be 1.0.
    """
    config = EvalSuiteConfig(
        model_provider="ollama",
        model_name="llama3",
        evaluator="exact_match",
        test_cases=[
            TestCase(prompt="What is 2+2?", expected="4"),
            TestCase(prompt="Capital of France?", expected="Paris"),
        ],
    )

    report = run_eval(config, dry_run=True)
    assert report.total == 2
    assert report.passed == 2
    assert report.average_score == 1.0
    assert report.model_name == "dry-run/mock"
