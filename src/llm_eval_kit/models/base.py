"""
Abstract base class for all model providers.

Every provider (Ollama, OpenAI, Anthropic) must inherit from this
and implement the `generate` method. The runner doesn't care which
LLM you're using — it just calls model.generate(prompt).

Same Strategy Pattern as evaluators:
    - BaseModel = the interface
    - OllamaModel, OpenAIModel, etc. = the strategies
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class all model providers must inherit from."""

    # Display name used in reports (e.g., "ollama/llama3", "openai/gpt-4o")
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return its text response.

        Args:
            prompt: The text to send to the model.

        Returns:
            The model's response as a string.

        Raises:
            ConnectionError: If the model API is unreachable.
        """
        ...
