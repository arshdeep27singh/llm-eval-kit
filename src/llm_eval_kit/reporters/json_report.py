"""
JSON Reporter — saves evaluation results to a JSON file.

Converts the EvalReport into a structured JSON document that includes:
    - Model name
    - Summary stats (average score, passed/total)
    - Every individual result with prompt, response, score, reasoning

Example output:
    {
        "model": "ollama/llama3.2:1b",
        "summary": {
            "total": 5,
            "passed": 4,
            "average_score": 0.8
        },
        "results": [
            {
                "prompt": "What is 2+2?",
                "expected": "4",
                "response": "4",
                "score": 1.0,
                "evaluator": "exact_match",
                "reasoning": "Match: '4' == '4'"
            },
            ...
        ]
    }

Usage:
    llm-eval-kit run examples/sample_eval.yaml --output results.json
"""

import json
from pathlib import Path

from llm_eval_kit.schemas import EvalReport


def save_json_report(report: EvalReport, output_path: str | Path) -> None:
    """Save an EvalReport as a JSON file.

    We manually build the dict instead of using Pydantic's .model_dump()
    because we want a flatter, cleaner structure for readability.
    """
    data = {
        "model": report.model_name,
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "average_score": round(report.average_score, 4),
        },
        "results": [
            {
                "prompt": result.test_case.prompt,
                "expected": result.test_case.expected,
                "response": result.response,
                "score": result.score,
                "evaluator": result.evaluator_name,
                "reasoning": result.reasoning,
                "tags": result.test_case.tags,
            }
            for result in report.results
        ],
    }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
