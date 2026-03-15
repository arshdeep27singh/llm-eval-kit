"""
Contains Evaluator — checks if the expected answer appears anywhere in the response.

More forgiving than exact match. Useful when LLMs give verbose answers
but the key information is still present.

    Expected: "Paris"
    Response: "The capital of France is Paris, a beautiful city."
    Score:    1.0 (because "paris" is found in the normalized response)

Scoring:
    1.0 — expected answer found in response
    0.0 — expected answer NOT found in response

Still binary (1.0 or 0.0) but much more lenient than exact match.
"""

import re

from llm_eval_kit.evaluators.base import BaseEvaluator
from llm_eval_kit.schemas import EvalResult, TestCase


class ContainsEvaluator(BaseEvaluator):
    """Scores 1.0 if the normalized expected answer appears in the normalized response."""

    name: str = "contains"

    def evaluate(self, test_case: TestCase, response: str) -> EvalResult:
        if test_case.expected is None:
            return EvalResult(
                test_case=test_case,
                response=response,
                score=0.0,
                evaluator_name=self.name,
                reasoning="No expected answer provided — cannot check containment.",
            )

        normalized_response = self._normalize(response)
        normalized_expected = self._normalize(test_case.expected)

        found = normalized_expected in normalized_response
        score = 1.0 if found else 0.0

        reasoning = (
            f"Found '{normalized_expected}' in response"
            if found
            else f"'{normalized_expected}' not found in response"
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
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
