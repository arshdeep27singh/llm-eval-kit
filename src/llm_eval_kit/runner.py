"""
The runner — orchestrates the entire evaluation pipeline.

Responsibilities:
    1. Parse a YAML eval suite file into EvalSuiteConfig
    2. Create the right model provider (Ollama/OpenAI/Anthropic)
    3. Create the right evaluator (exact_match, etc.)
    4. Loop through every test case:
       - Send prompt to the LLM
       - Grade the response with the evaluator
       - Collect the result
    5. Return a complete EvalReport

This is the only file that knows about ALL parts of the system.
Models and evaluators don't know about each other — the runner connects them.
"""

from pathlib import Path

import yaml

from llm_eval_kit.evaluators.base import BaseEvaluator
from llm_eval_kit.evaluators.contains import ContainsEvaluator
from llm_eval_kit.evaluators.exact_match import ExactMatchEvaluator
from llm_eval_kit.evaluators.llm_judge import LLMJudgeEvaluator
from llm_eval_kit.models.anthropic import AnthropicModel
from llm_eval_kit.models.base import BaseLLM
from llm_eval_kit.models.ollama import OllamaModel
from llm_eval_kit.models.openai import OpenAIModel
from llm_eval_kit.schemas import EvalReport, EvalResult, EvalSuiteConfig


# ── Factory functions ────────────────────────────────────────────────────────
# These map string names (from YAML) to actual Python classes.
# When we add new providers/evaluators, we just add them here.

def get_model(provider: str, model_name: str) -> BaseLLM:
    """Create a model provider from its string name.

    Args:
        provider: "ollama", "openai", or "anthropic"
        model_name: Model-specific name like "llama3", "gpt-4o", etc.

    This is the Factory Pattern — convert a string into an object.
    """
    providers = {
        "ollama": OllamaModel,
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
    }
    if provider not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
    return providers[provider](model_name)


def get_evaluator(name: str, judge_model: BaseLLM | None = None) -> BaseEvaluator:
    """Create an evaluator from its string name.

    Args:
        name: "exact_match", "contains", or "llm_judge"
        judge_model: Required when name is "llm_judge" — the LLM that grades responses.
    """
    if name == "llm_judge":
        if judge_model is None:
            raise ValueError("llm_judge evaluator requires a judge model. Add a 'judge' section to your YAML.")
        return LLMJudgeEvaluator(judge_model=judge_model)

    evaluators = {
        "exact_match": ExactMatchEvaluator,
        "contains": ContainsEvaluator,
    }
    if name not in evaluators:
        available = ", ".join(evaluators.keys())
        raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
    return evaluators[name]()


# ── YAML loading ─────────────────────────────────────────────────────────────

def load_suite(path: str | Path) -> EvalSuiteConfig:
    """Load and validate a YAML eval suite file.

    The YAML file looks like:
        model:
          provider: ollama
          name: llama3
        evaluator: exact_match
        test_cases:
          - prompt: "What is 2+2?"
            expected: "4"

    We parse the YAML into a dict, then reshape it to match EvalSuiteConfig.
    Pydantic validates everything — wrong types, missing fields, etc.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Eval suite file not found: {file_path}")

    with open(file_path) as f:
        raw = yaml.safe_load(f)

    # Flatten the nested "model" dict into top-level fields
    model_config = raw.get("model", {})
    judge_config = raw.get("judge", {})
    return EvalSuiteConfig(
        model_provider=model_config.get("provider", "ollama"),
        model_name=model_config.get("name", "llama3"),
        evaluator=raw.get("evaluator", "exact_match"),
        judge_provider=judge_config.get("provider"),
        judge_model_name=judge_config.get("name"),
        test_cases=raw.get("test_cases", []),
    )


# ── Runner ───────────────────────────────────────────────────────────────────

def run_eval(config: EvalSuiteConfig, dry_run: bool = False) -> EvalReport:
    """Run the full evaluation pipeline.

    This is the main entry point. Given a config, it:
        1. Creates the model and evaluator
        2. Loops through each test case
        3. Sends the prompt to the LLM
        4. Grades the response
        5. Returns a complete report

    If dry_run=True, uses a fake model that returns expected answers.
    Useful for testing the pipeline without a real LLM.

    If a single test case fails (LLM error, timeout, etc.), it records
    the error and continues — one bad test doesn't stop the whole suite.
    """
    if dry_run:
        from llm_eval_kit.models.dry_run import DryRunModel
        model = DryRunModel()
    else:
        model = get_model(config.model_provider, config.model_name)

    # Create judge model if using LLM-as-a-Judge evaluator
    judge_model = None
    if config.evaluator == "llm_judge":
        if config.judge_provider and config.judge_model_name:
            judge_model = get_model(config.judge_provider, config.judge_model_name)
        else:
            # Default: use the same model as both the test subject and the judge
            judge_model = model

    evaluator = get_evaluator(config.evaluator, judge_model=judge_model)
    results: list[EvalResult] = []

    for i, test_case in enumerate(config.test_cases, start=1):
        print(f"  [{i}/{len(config.test_cases)}] {test_case.prompt[:60]}...")

        try:
            # In dry-run mode, feed the expected answer to the mock model
            if dry_run and hasattr(model, '_expected'):
                model._expected = test_case.expected

            # Step 1: Get LLM response
            response = model.generate(test_case.prompt)

            # Step 2: Grade the response
            result = evaluator.evaluate(test_case, response)

        except Exception as e:
            # If something goes wrong (network error, timeout, etc.),
            # record it as a 0.0 score with the error as reasoning
            result = EvalResult(
                test_case=test_case,
                response=f"ERROR: {e}",
                score=0.0,
                evaluator_name=evaluator.name,
                reasoning=f"Error during evaluation: {e}",
            )

        results.append(result)

    return EvalReport(model_name=model.name, results=results)
