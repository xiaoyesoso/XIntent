"""Mock LLM client (for development / testing).

Simulates large model responses based on rule matching, no real API calls needed.
Supports registering custom response handlers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .base import LLMClient


# Type alias: response handler
ResponseHandler = Callable[[str, str], str]


class MockLLMClient(LLMClient):
    """Mock LLM client.

    Default behavior: returns simulated JSON based on keyword matching in user_prompt.
    Custom handlers can be registered via register_handler.
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[str, ResponseHandler]] = []
        self._default_handler: ResponseHandler | None = None
        self._call_log: list[dict[str, str]] = []

    def register_handler(self, keyword: str, handler: ResponseHandler) -> None:
        """Register a keyword-matching handler. Called when user_prompt contains keyword."""
        self._handlers.append((keyword, handler))

    def set_default_handler(self, handler: ResponseHandler) -> None:
        """Set the default handler."""
        self._default_handler = handler

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self._call_log.append({"system": system_prompt[:200], "user": user_prompt[:500]})
        for keyword, handler in self._handlers:
            if keyword in user_prompt:
                return handler(system_prompt, user_prompt)
        if self._default_handler:
            return self._default_handler(system_prompt, user_prompt)
        return self._default_response(user_prompt)

    @property
    def call_log(self) -> list[dict[str, str]]:
        """Call log (for testing)."""
        return self._call_log

    @staticmethod
    def _default_response(user_prompt: str) -> str:
        """Default simulated response: attempts to extract info from input to generate a basic normalization result."""
        return json.dumps(
            {
                "normalized_input": user_prompt[:200],
                "note": "MockLLM default response - register a handler for realistic output",
            },
            ensure_ascii=False,
        )
