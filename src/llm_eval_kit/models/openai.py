"""
OpenAI model provider — GPT-4o, GPT-4, GPT-3.5-turbo, etc.

Requires an API key set as the OPENAI_API_KEY environment variable.
User pays per token to OpenAI — this tool does NOT charge anything.

Usage in eval suite:
    model:
      provider: openai
      name: gpt-4o

Set your key:
    export OPENAI_API_KEY="sk-..."
"""

import os

import httpx

from llm_eval_kit.models.base import BaseLLM


class OpenAIModel(BaseLLM):
    """Talks to the OpenAI Chat Completions API."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.name = f"openai/{model_name}"
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Get your key at https://platform.openai.com/api-keys"
            )

    def generate(self, prompt: str) -> str:
        """Send prompt to OpenAI's Chat Completions API.

        API docs: https://platform.openai.com/docs/api-reference/chat

        We use the /v1/chat/completions endpoint with the "user" role.
        The response is nested: choices[0].message.content
        """
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic output — important for evals!
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
