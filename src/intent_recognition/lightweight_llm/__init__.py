"""Lightweight LLM layer (Layer 2) - D4, D5, D12, D17, D18, D28."""

from .candidate_retriever import (
    CandidateRetriever,
    HybridCandidateRetriever,
    LLMCoarseRetriever,
    VectorCandidateRetriever,
    create_retriever,
)
from .classifier import LightweightLLMClassifier
from .confidence_router import (
    ArbitrationDecision,
    ArbitrationInput,
    ConfidenceRouter,
)
from .dynamic_fewshot import DynamicFewShotInjector
from .multi_signal import MultiSignalFuser
from .prompts import (
    CAN_DO,
    CANNOT_DO,
    SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
    build_user_prompt,
    build_user_prompt_with_sections,
)

__all__ = [
    "LightweightLLMClassifier",
    "ConfidenceRouter",
    "ArbitrationInput",
    "ArbitrationDecision",
    "MultiSignalFuser",
    "CAN_DO",
    "CANNOT_DO",
    "SYSTEM_PROMPT_TEMPLATE",
    "build_system_prompt",
    "build_user_prompt",
    # D17: candidate retrieval
    "CandidateRetriever",
    "VectorCandidateRetriever",
    "LLMCoarseRetriever",
    "HybridCandidateRetriever",
    "create_retriever",
    # D18: dynamic few-shot injection
    "DynamicFewShotInjector",
    "build_user_prompt_with_sections",
]
