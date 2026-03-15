"""
Abstract base class for all evaluators.

Every evaluator (exact match, similarity, LLM judge, etc.) must inherit
from this class and implement the `evaluate` method. This ensures the
runner can use ANY evaluator interchangeably — it just calls evaluate().

This is the Strategy Pattern:
    - BaseEvaluator = the interface
    - ExactMatchEvaluator, SimilarityEvaluator, etc. = the strategies
    - Runner = the context that uses a strategy
"""

from abc import ABC, abstractmethod

from llm_eval_kit.schemas import EvalResult, TestCase


class BaseEvaluator(ABC):
    """Base class all evaluators must inherit from."""

    # Each evaluator must declare its name (used in reports)
    name: str = "base"

    @abstractmethod
    def evaluate(self, test_case: TestCase, response: str) -> EvalResult:
        """Grade an LLM's response to a test case.

        Args:
            test_case: The original test case (prompt, expected answer, etc.)
            response: What the LLM actually responded with.

        Returns:
            EvalResult with a score from 0.0 to 1.0
        """
        ...
