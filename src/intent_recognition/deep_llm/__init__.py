"""Deep reasoning LLM layer (Layer 3) - handles complex scenarios (D6)."""

from .classifier import DeepLLMClassifier
from .prompts import build_system_prompt, build_user_prompt

__all__ = ["DeepLLMClassifier", "build_system_prompt", "build_user_prompt"]
