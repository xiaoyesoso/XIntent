"""Configuration for intent recognition (corresponding to D4, D5, D16)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfidenceConfig:
    """Confidence routing thresholds (D4)."""

    accept_threshold: float = 0.85  # >= accept -> directly accept
    clarify_threshold: float = 0.6  # >= clarify and < accept -> clarify
    # < clarify -> escalate to next layer

    # Multi-signal weights (D5)
    weight_llm_confidence: float = 0.5
    weight_rule_match: float = 0.2
    weight_vector_similarity: float = 0.2
    weight_historical_accuracy: float = 0.1


@dataclass
class ClarificationConfig:
    """Clarification limits (D10)."""

    max_consecutive_clarifications: int = 3
    # After max, degrade to best guess


@dataclass
class EvaluationConfig:
    """Evaluation metrics config (D15, D16)."""

    top_k: int = 3  # Top-K accuracy K value
    # Benchmark reference (D15)
    domain_specific_benchmark: float = 0.99
    general_agent_benchmark: float = 0.85


# ----------------------------------------------------------------------
# D17-D24 extension configs (all default OFF to preserve existing behavior)
# ----------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    """D17: Retrieval-based candidate narrowing config."""

    enable: bool = False
    method: str = "vector"  # "vector" | "llm_coarse" | "hybrid"
    top_n: int = 10
    dynamic_n: bool = True


@dataclass
class DynamicFewShotConfig:
    """D18: Dynamic few-shot injection config."""

    dynamic_enabled: bool = False
    dynamic_top_k: int = 3
    static_kind_tag: str = "static"


@dataclass
class OrthogonalityConfig:
    """D19: Intent orthogonality governance config."""

    enable_check: bool = False
    overlap_threshold: float = 0.7


@dataclass
class MultiIntentConfig:
    """D20: True multi-intent detection config."""

    enable: bool = True
    sequential_execution: bool = True
    filter_process_description: bool = True


@dataclass
class VectorFallbackConfig:
    """D21: Vector matching fallback config."""

    enable: bool = False
    similarity_threshold: float = 0.92
    top_k: int = 1


@dataclass
class ReuseStrategyConfig:
    """D22: Intent reuse with rollback config."""

    enable: bool = False
    rollback_on_failure_signal: bool = True
    rollback_on_tool_failure_count: int = 3


@dataclass
class ArbiterConfig:
    """D23: Multi-recognizer arbitration config."""

    enable: bool = False
    mode: str = "vote"  # "vote" | "weighted_score"
    recognizers: list[str] = field(default_factory=lambda: ["vector", "rule", "lightweight_llm"])
    weights: dict[str, float] = field(default_factory=lambda: {
        "vector": 0.8,
        "rule": 0.6,
        "lightweight_llm": 1.0,
    })


@dataclass
class FineTuningConfig:
    """D24: Fine-tuning integration point config (design-only, no training)."""

    enable: bool = False
    model: str | None = None  # Read from FINE_TUNED_MODEL env if None


# ----------------------------------------------------------------------
# D26-D31 extension configs (interview-insights change)
# All default to backward-compatible behavior: new fields are optional,
# new metrics are computed offline, new methodology is documentation-only.
# ----------------------------------------------------------------------


@dataclass
class EvidenceConfig:
    """D26: Evidence grading config.

    When ``enable_grading`` is True, slots are tagged with evidence grades
    and hard operations require verified evidence. Defaults to True because
    new fields are optional and do not change existing behavior.
    """

    enable_grading: bool = True
    require_verified_for_hard_ops: bool = True
    high_risk_intents: list[str] = field(default_factory=list)


@dataclass
class BoundaryConfig:
    """D27: sub_tasks vs independent_intents boundary config."""

    enable_sub_tasks: bool = True
    strict_mode: bool = False  # When True, ambiguous targets default to sub_tasks


@dataclass
class ArbitrationConfig:
    """D28: Five-factor routing arbitration config.

    When ``enable_five_factor`` is True, L2 confidence between clarify and
    accept thresholds triggers five-factor arbitration instead of the simple
    confidence-max decision. Defaults to True; existing fast paths (L1 hit,
    L2 >= accept_threshold) are unaffected.
    """

    enable_five_factor: bool = True
    candidate_gap_threshold: float = 0.1
    risk_aware_clarify: bool = True
    high_risk_intents: list[str] = field(default_factory=list)


@dataclass
class ProtocolConfig:
    """D31: Structured output protocol config."""

    enable_structured_output: bool = True
    deprecate_sub_intents: bool = False  # Keep sub_intents alias (will remove in future)


@dataclass
class ExtendedEvaluationConfig:
    """D29: Extended evaluation metrics config (offline metrics only)."""

    enable_confusion_matrix: bool = True
    enable_hard_sample_accuracy: bool = True
    enable_clarification_convergence: bool = True
    enable_slot_recall_completeness: bool = True
    enable_constraint_identification: bool = True
    enable_state_update_accuracy: bool = True
    online_feedback_loop: bool = True


@dataclass
class IntentRecognitionConfig:
    """Master configuration for intent recognition."""

    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    clarification: ClarificationConfig = field(default_factory=ClarificationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Layer enable flags
    enable_code_layer: bool = True
    enable_lightweight_llm: bool = True
    enable_deep_llm: bool = True

    # Few-shot injection
    fewshot_top_k: int = 3
    fewshot_enabled: bool = True

    # Hierarchical intent (D13)
    enable_hierarchical: bool = False

    # Implicit evaluation (D11)
    enable_implicit_eval: bool = True
    failure_signals: list[str] = field(default_factory=lambda: [
        "你理解错了",
        "不是这个意思",
        "不是，我不是",
        "不对，我要的是",
    ])

    # D17-D24 extension configs (all default OFF)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dynamic_fewshot: DynamicFewShotConfig = field(default_factory=DynamicFewShotConfig)
    orthogonality: OrthogonalityConfig = field(default_factory=OrthogonalityConfig)
    multi_intent: MultiIntentConfig = field(default_factory=MultiIntentConfig)
    vector_fallback: VectorFallbackConfig = field(default_factory=VectorFallbackConfig)
    reuse_strategy: ReuseStrategyConfig = field(default_factory=ReuseStrategyConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)

    # D26-D31 extension configs (interview-insights change)
    # All default to backward-compatible behavior.
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    arbitration: ArbitrationConfig = field(default_factory=ArbitrationConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    extended_evaluation: ExtendedEvaluationConfig = field(default_factory=ExtendedEvaluationConfig)

    # One-click preset: "balanced" | "high_accuracy" | "low_cost"
    accuracy_preset: str = "balanced"
