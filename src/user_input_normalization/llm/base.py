"""LLM abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Large model client abstract interface.

    The normalization framework calls the large model through this interface;
    concrete implementations can be OpenAI / Anthropic / local models.
    """

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous chat call, returns model output text."""
