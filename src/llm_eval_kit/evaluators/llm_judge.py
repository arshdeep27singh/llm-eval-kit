"""
LLM-as-a-Judge Evaluator — uses a second LLM to grade responses.

This is the industry-standard approach for evaluating open-ended LLM outputs
where there's no single correct answer (e.g., "Write a poem", "Explain X").

How it works:
    1. Takes the original question + the LLM's response
    2. Builds a "judge prompt" asking a second LLM to grade it
    3. Sends the judge prompt to a judge model
    4. Parses the score (0.0-1.0) and reasoning from the judge's response

The judge model is configurable — it can be:
    - The same model being tested (free, but less reliable)
    - A stronger model like GPT-4 (more accurate, but costs money)
    - Any model accessible via our model providers

Usage in YAML:
    evaluator: llm_judge
    judge:
      provider: ollama          # or "openai", "anthropic"
      name: llama3.2:1b         # judge model name
"""

import json
import re

from llm_eval_kit.evaluators.base import BaseEvaluator
from llm_eval_kit.models.base import BaseLLM
from llm_eval_kit.schemas import EvalResult, TestCase

# The prompt template sent to the judge LLM.
# This is essentially "prompt engineering for grading."
#
# Key design choices:
#   - Ask for JSON output (easier to parse than free text)
#   - Score must be 0.0 to 1.0 (normalized scale)
#   - Ask for reasoning (explainability — WHY this score?)
#   - Include the expected answer if available (helps the judge)
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Your job is to grade how well an AI assistant answered a question.

## Question
{prompt}

## AI Response
{response}

{expected_section}

## Instructions
Rate the response on a scale from 0.0 to 1.0:
- 1.0 = Perfect, complete, accurate answer
- 0.7-0.9 = Good answer with minor issues
- 0.4-0.6 = Partially correct or incomplete
- 0.1-0.3 = Mostly wrong or very incomplete
- 0.0 = Completely wrong or irrelevant

You MUST respond with ONLY a JSON object in this exact format, nothing else:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Uses a second LLM to grade responses."""

    name: str = "llm_judge"

    def __init__(self, judge_model: BaseLLM):
        """
        Args:
            judge_model: The LLM to use as the judge. Can be any BaseLLM
                         (OllamaModel, OpenAIModel, AnthropicModel).
        """
        self.judge_model = judge_model

    def evaluate(self, test_case: TestCase, response: str) -> EvalResult:
        # Build the expected answer section (only include if provided)
        if test_case.expected:
            expected_section = f"## Expected Answer\n{test_case.expected}"
        else:
            expected_section = "## Expected Answer\nNo expected answer provided. Judge based on quality, accuracy, and completeness."

        # Build the full judge prompt from the template
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=test_case.prompt,
            response=response,
            expected_section=expected_section,
        )

        # Ask the judge LLM to grade the response
        judge_response = self.judge_model.generate(judge_prompt)

        # Parse the judge's JSON response
        score, reasoning = self._parse_judge_response(judge_response)

        return EvalResult(
            test_case=test_case,
            response=response,
            score=score,
            evaluator_name=self.name,
            reasoning=reasoning,
        )

    @staticmethod
    def _parse_judge_response(judge_response: str) -> tuple[float, str]:
        """Extract score and reasoning from the judge's JSON response.

        LLMs don't always return perfect JSON, so we try multiple strategies:
            1. Direct JSON parse (ideal case)
            2. Extract JSON from markdown code blocks (```json ... ```)
            3. Regex fallback to find score number anywhere in text

        Returns:
            (score, reasoning) tuple
        """
        # Strategy 1: Try direct JSON parse
        try:
            data = json.loads(judge_response.strip())
            score = float(data["score"])
            reasoning = data.get("reasoning", "No reasoning provided")
            return (max(0.0, min(1.0, score)), reasoning)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", judge_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                score = float(data["score"])
                reasoning = data.get("reasoning", "No reasoning provided")
                return (max(0.0, min(1.0, score)), reasoning)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Strategy 3: Find any JSON object in the response
        json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", judge_response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = float(data["score"])
                reasoning = data.get("reasoning", "No reasoning provided")
                return (max(0.0, min(1.0, score)), reasoning)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Strategy 4: Last resort — find any number that looks like a score
        score_match = re.search(r"(?:score|rating)[:\s]*([0-9]*\.?[0-9]+)", judge_response, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return (max(0.0, min(1.0, score)), f"Parsed from raw text: {judge_response[:100]}")

        # Give up — couldn't parse any score
        return (0.0, f"Could not parse judge response: {judge_response[:100]}")
