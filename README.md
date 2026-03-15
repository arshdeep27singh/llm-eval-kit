# llm-eval-kit

A lightweight Python CLI tool and library for evaluating LLM outputs using multiple strategies.

## Evaluation Strategies

- **Exact Match** — Direct string comparison
- **Semantic Similarity** — Cosine similarity via sentence embeddings
- **LLM-as-a-Judge** — Use a stronger LLM to grade outputs
- **Custom Rubrics** — Define your own scoring criteria

## Installation

```bash
pip install llm-eval-kit
```

## Quick Start

```bash
llm-eval-kit run examples/sample_eval.yaml
```

## Development

```bash
git clone https://github.com/arshdeep27singh/llm-eval-kit.git
cd llm-eval-kit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## License

MIT
