"""
Data models that flow through the entire pipeline.

    TestCase  →  Runner  →  EvalResult  →  EvalReport
    (input)      (LLM)      (one grade)    (all grades)
"""

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """One test case = one question to send to the LLM.

    Example in YAML:
        - prompt: "What is the capital of France?"
          expected: "Paris"
          context: "France is a country in Western Europe."
          tags: ["geography", "factual"]
    """

    # The prompt/question to send to the LLM (required)
    prompt: str

    # The correct/expected answer to compare against (optional — not all evals need it)
    expected: str | None = None

    # Additional context the LLM should use (useful for RAG-style evals)
    context: str | None = None

    # Tags for filtering/grouping results (e.g., ["math", "easy"])
    tags: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """The result of evaluating ONE test case.

    Contains the original test case, what the LLM said,
    and the score the evaluator gave it.
    """

    # The original test case (so we can show the question in reports)
    test_case: TestCase

    # What the LLM actually responded with
    response: str

    # Score from 0.0 (completely wrong) to 1.0 (perfect)
    score: float = Field(ge=0.0, le=1.0)

    # Which evaluator produced this score (e.g., "exact_match", "llm_judge")
    evaluator_name: str

    # Optional explanation of why this score was given
    # (especially useful for LLM-as-a-Judge which can explain its reasoning)
    reasoning: str | None = None


class EvalReport(BaseModel):
    """The final report containing ALL evaluation results.

    This is what gets passed to reporters for display.
    """

    # Which model was tested (e.g., "ollama/llama3", "openai/gpt-4")
    model_name: str

    # All individual results
    results: list[EvalResult]

    @property
    def average_score(self) -> float:
        """Calculate the average score across all results."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def total(self) -> int:
        """Total number of test cases evaluated."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of test cases with score >= 0.5."""
        return sum(1 for r in self.results if r.score >= 0.5)


class EvalSuiteConfig(BaseModel):
    """Configuration loaded from a YAML eval suite file.

    This is what the user writes in their YAML file:

        model:
          provider: ollama
          name: llama3
        evaluator: exact_match
        test_cases:
          - prompt: "What is 2+2?"
            expected: "4"
    """

    # Model configuration
    model_provider: str = "ollama"
    model_name: str = "llama3"

    # Which evaluator to use
    evaluator: str = "exact_match"

    # The test cases to run
    test_cases: list[TestCase]
