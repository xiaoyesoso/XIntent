"""OpenAI-compatible LLM client.

Works with any API provider that implements the OpenAI SDK interface:
SiliconFlow, OpenAI, Azure OpenAI, DeepSeek, Moonshot, Together, etc.

Reads configuration from .env file:
    API_KEY=sk-...
    FLASH_LLM_MODEL=<flash-model>
    PRO_LLM_MODEL=<pro-model>
    FINE_TUNED_MODEL=<fine-tuned-model>   # D24: optional, falls back to PRO_LLM_MODEL
    BASE_URL=https://api.openai.com/v1
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMClient

logger = logging.getLogger(__name__)


def _find_env_file() -> Path | None:
    """Search for .env file in current dir and parent dirs."""
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        env_path = d / ".env"
        if env_path.exists():
            return env_path
    return None


class OpenAICompatibleClient(LLMClient):
    """LLM client for any OpenAI-compatible API.

    Loads API_KEY, FLASH_LLM_MODEL (or PRO_LLM_MODEL), BASE_URL from .env file.
    Use model_tier="flash" for lightweight tasks (default), "pro" for deep
    reasoning, or "fine_tuned" to use a D24 fine-tuned model (falls back to
    PRO_LLM_MODEL with a warning when FINE_TUNED_MODEL is not set).
    """

    def __init__(
        self,
        env_path: str | Path | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        model_tier: str = "flash",
        base_url: str | None = None,
    ) -> None:
        # Load .env file
        if env_path:
            load_dotenv(env_path)
        else:
            found = _find_env_file()
            if found:
                load_dotenv(found)

        self._api_key = api_key or os.getenv("API_KEY", "")
        self._base_url = base_url or os.getenv("BASE_URL") or "https://api.openai.com/v1"

        # Select model: explicit model > model_tier env var > flash default
        if model:
            self._model = model
        elif model_tier == "pro":
            self._model = os.getenv("PRO_LLM_MODEL") or ""
        elif model_tier == "fine_tuned":
            # D24: prefer the fine-tuned model; fall back to PRO_LLM_MODEL
            # with a warning so the caller knows the fine-tuned weights are
            # not actually being used.
            self._model = os.getenv("FINE_TUNED_MODEL") or ""
            if not self._model:
                self._model = os.getenv("PRO_LLM_MODEL") or ""
                if self._model:
                    logger.warning(
                        "FINE_TUNED_MODEL not set; falling back to "
                        "PRO_LLM_MODEL for model_tier='fine_tuned'."
                    )
        else:
            self._model = os.getenv("FLASH_LLM_MODEL") or ""

        if not self._api_key:
            raise ValueError(
                "API_KEY not found. Set it in .env file or pass as argument."
            )

        if not self._model:
            if model_tier == "pro":
                env_var = "PRO_LLM_MODEL"
            elif model_tier == "fine_tuned":
                env_var = "FINE_TUNED_MODEL (or PRO_LLM_MODEL fallback)"
            else:
                env_var = "FLASH_LLM_MODEL"
            raise ValueError(
                f"{env_var} not found. Set it in .env file or pass model as argument."
            )

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Call the LLM via OpenAI-compatible API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Support JSON response format if specified
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def embed(self, text: str, *, model: str | None = None) -> list[float]:
        """Generate embedding for text using the embedding model.

        Uses EMBEDDING_MODEL from .env if model is not specified.
        """
        embed_model = model or os.getenv("EMBEDDING_MODEL", "")
        if not embed_model:
            raise ValueError(
                "EMBEDDING_MODEL not found. Set it in .env file or pass model as argument."
            )
        response = self._client.embeddings.create(
            model=embed_model,
            input=text,
        )
        return response.data[0].embedding
