"""Configuration center (corresponds to tasks 1.6).

Manages configurable items such as confidence thresholds, top-k, and threshold promotion rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ClarifyConfig:
    """Clarification mechanism configuration (corresponds to D8)."""

    theta_clarify: float = 0.6  # Clarification trigger confidence threshold
    max_consecutive_clarifications: int = 3  # Maximum consecutive clarification count


@dataclass
class FewShotConfig:
    """few-shot retrieval configuration (corresponds to D5)."""

    top_k: int = 3  # Number of recalled examples
    enabled: bool = True


@dataclass
class VocabPromotionConfig:
    """Vocabulary threshold promotion rules (corresponds to D11)."""

    min_total_count: int = 100  # Total count threshold
    min_discussant_count: int = 3  # Discussant count threshold (exceeding promotes to public vocabulary)
    min_consecutive_count: int = 10  # Consecutive discussion count threshold
    require_human_review: bool = True  # Whether human review is required


@dataclass
class ContextConfig:
    """Context integration configuration (corresponds to D14)."""

    short_term_window: int = 10  # Short-term memory turns
    recall_top_k: int = 5  # Dialogue history recall count
    summary_max_tokens: int = 500  # Summary max tokens


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    agent_type: str = "general"  # general / domain
    merge_pre_with_intent: bool = False  # Domain Agent can merge (D2)
    enable_deep_normalization: bool = True
    enable_vocab_table: bool = True
    enable_attribute_resolution: bool = True


@dataclass
class Config:
    """Global configuration center."""

    clarify: ClarifyConfig = field(default_factory=ClarifyConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    vocab_promotion: VocabPromotionConfig = field(default_factory=VocabPromotionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build configuration from dict."""
        cfg = cls()
        if "clarify" in data:
            cfg.clarify = ClarifyConfig(**data["clarify"])
        if "fewshot" in data:
            cfg.fewshot = FewShotConfig(**data["fewshot"])
        if "vocab_promotion" in data:
            cfg.vocab_promotion = VocabPromotionConfig(**data["vocab_promotion"])
        if "context" in data:
            cfg.context = ContextConfig(**data["context"])
        if "pipeline" in data:
            cfg.pipeline = PipelineConfig(**data["pipeline"])
        return cfg

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "clarify": self.clarify.__dict__,
            "fewshot": self.fewshot.__dict__,
            "vocab_promotion": self.vocab_promotion.__dict__,
            "context": self.context.__dict__,
            "pipeline": self.pipeline.__dict__,
        }


# Default configuration singleton
_default_config: Config | None = None


def get_config() -> Config:
    """Get the default configuration singleton."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(cfg: Config) -> None:
    """Set global configuration."""
    global _default_config
    _default_config = cfg
