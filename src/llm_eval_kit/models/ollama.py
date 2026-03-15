"""
Ollama model provider — runs LLMs locally for free.

Ollama (https://ollama.com) runs models like Llama 3, Mistral, Gemma
on your own machine. It exposes a REST API at localhost:11434.

Usage:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model: `ollama pull llama3`
    3. Use in eval suite:
        model:
          provider: ollama
          name: llama3

No API key needed. No cost. Runs entirely on your hardware.
"""

import httpx

from llm_eval_kit.models.base import BaseLLM


class OllamaModel(BaseLLM):
    """Talks to a locally running Ollama instance."""

    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.name = f"ollama/{model_name}"

    def generate(self, prompt: str) -> str:
        """Send prompt to Ollama's /api/generate endpoint.

        Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

        The request body:
            - model: which model to use (e.g., "llama3")
            - prompt: the text to send
            - stream: False = wait for full response (not token-by-token)
        """
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120.0,  # LLMs can be slow, especially on CPU
        )
        response.raise_for_status()
        return response.json()["response"]
