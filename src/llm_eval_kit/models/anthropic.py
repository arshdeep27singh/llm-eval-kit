"""
Anthropic model provider — Claude 3.5 Sonnet, Claude 3 Opus, etc.

Requires an API key set as the ANTHROPIC_API_KEY environment variable.
User pays per token to Anthropic — this tool does NOT charge anything.

Usage in eval suite:
    model:
      provider: anthropic
      name: claude-sonnet-4-20250514

Set your key:
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os

import httpx

from llm_eval_kit.models.base import BaseLLM


class AnthropicModel(BaseLLM):
    """Talks to the Anthropic Messages API."""

    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        self.model_name = model_name
        self.name = f"anthropic/{model_name}"
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Get your key at https://console.anthropic.com/settings/keys"
            )

    def generate(self, prompt: str) -> str:
        """Send prompt to Anthropic's Messages API.

        API docs: https://docs.anthropic.com/en/api/messages

        Anthropic's API differs from OpenAI:
            - Uses 'x-api-key' header (not Bearer token)
            - Requires 'anthropic-version' header
            - max_tokens is required (not optional)
            - Response is content[0].text (not choices[0].message.content)
        """
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "max_tokens": 1024,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
