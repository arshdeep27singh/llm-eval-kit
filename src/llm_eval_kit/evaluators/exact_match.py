"""
Exact Match Evaluator — the simplest grading strategy.

Compares the LLM's response to the expected answer after normalizing both.
Returns 1.0 if they match, 0.0 if they don't.

Normalization means:
    - Lowercase everything        ("Paris" → "paris")
    - Strip whitespace            ("  paris  " → "paris")
    - Remove punctuation          ("paris." → "paris")

Why normalize? Because "Paris", "paris", "Paris.", and " Paris " are all
correct answers — we don't want to fail on trivial formatting differences.
"""

import re

from llm_eval_kit.evaluators.base import BaseEvaluator
from llm_eval_kit.schemas import EvalResult, TestCase


class ExactMatchEvaluator(BaseEvaluator):
    """Scores 1.0 if normalized response matches expected answer, else 0.0."""

    name: str = "exact_match"

    def evaluate(self, test_case: TestCase, response: str) -> EvalResult:
        if test_case.expected is None:
            return EvalResult(
                test_case=test_case,
                response=response,
                score=0.0,
                evaluator_name=self.name,
                reasoning="No expected answer provided — cannot do exact match.",
            )

        normalized_response = self._normalize(response)
        normalized_expected = self._normalize(test_case.expected)

        match = normalized_response == normalized_expected
        score = 1.0 if match else 0.0

        reasoning = (
            f"Match: '{normalized_response}' == '{normalized_expected}'"
            if match
            else f"No match: '{normalized_response}' != '{normalized_expected}'"
        )

        return EvalResult(
            test_case=test_case,
            response=response,
            score=score,
            evaluator_name=self.name,
            reasoning=reasoning,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip whitespace, remove punctuation."""
        text = text.lower().strip()
        # Remove all non-alphanumeric characters except spaces
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse multiple spaces into one
        text = re.sub(r"\s+", " ", text)
        return text
