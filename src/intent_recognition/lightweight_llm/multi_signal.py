"""Multi-signal confidence fuser (D5).

Fuses multiple confidence signals into a single score using weighted average:
  - LLM confidence (default weight 0.5)
  - Rule match (default weight 0.2)
  - Vector similarity (default weight 0.2)
  - Historical accuracy (default weight 0.1)

Also returns a breakdown dict for transparency and debugging.
"""

from __future__ import annotations

from ..config import ConfidenceConfig


class MultiSignalFuser:
    """Fuses multiple confidence signals via weighted average."""

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self._config = config or ConfidenceConfig()

    def fuse(
        self,
        llm_confidence: float,
        rule_match: bool,
        vector_sim: float,
        historical_acc: float,
    ) -> tuple[float, dict[str, float]]:
        """Fuse signals into a single confidence score.

        Args:
            llm_confidence: LLM-output confidence in [0, 1].
            rule_match: Whether a code-layer rule matched (0/1 signal).
            vector_sim: Vector similarity score in [0, 1].
            historical_acc: Historical accuracy for this intent in [0, 1].

        Returns:
            Tuple of (fused_confidence, breakdown_dict).
        """
        rule_signal = 1.0 if rule_match else 0.0

        # Clamp inputs to [0, 1] defensively
        llm = max(0.0, min(1.0, llm_confidence))
        vec = max(0.0, min(1.0, vector_sim))
        hist = max(0.0, min(1.0, historical_acc))

        w = self._config
        total_weight = (
            w.weight_llm_confidence
            + w.weight_rule_match
            + w.weight_vector_similarity
            + w.weight_historical_accuracy
        )
        # Avoid division by zero if user zeroes out all weights
        if total_weight <= 0:
            return llm, {"llm_confidence": llm, "rule_match": rule_signal,
                         "vector_similarity": vec, "historical_accuracy": hist}

        fused = (
            w.weight_llm_confidence * llm
            + w.weight_rule_match * rule_signal
            + w.weight_vector_similarity * vec
            + w.weight_historical_accuracy * hist
        ) / total_weight

        breakdown = {
            "llm_confidence": llm,
            "rule_match": rule_signal,
            "vector_similarity": vec,
            "historical_accuracy": hist,
            "weight_llm": w.weight_llm_confidence,
            "weight_rule": w.weight_rule_match,
            "weight_vector": w.weight_vector_similarity,
            "weight_historical": w.weight_historical_accuracy,
            "fused": fused,
        }
        return fused, breakdown
