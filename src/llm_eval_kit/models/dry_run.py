"""
Dry-run model — a fake LLM for testing the pipeline.

Returns the expected answer if one exists (to simulate a perfect LLM),
or a placeholder string if no expected answer is provided.

This lets you verify:
    - YAML parsing works
    - Evaluator scoring works
    - Report formatting works
    ...all without needing a real LLM running.

Usage:
    llm-eval-kit run examples/sample_eval.yaml --dry-run
"""

from llm_eval_kit.models.base import BaseLLM


class DryRunModel(BaseLLM):
    """Fake model that returns expected answers for testing."""

    def __init__(self):
        self.name = "dry-run/mock"
        # This gets set by the runner for each test case
        self._expected: str | None = None

    def generate(self, prompt: str) -> str:
        """Return the expected answer if set, otherwise a placeholder."""
        if self._expected is not None:
            return self._expected
        return "[dry-run: no expected answer]"
