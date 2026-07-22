"""Data models for intent recognition (corresponding to D7, D8, D10)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class RecognitionSource(str, Enum):
    """Which layer produced the result."""

    CODE_LAYER = "code-layer"
    LIGHTWEIGHT_LLM = "lightweight-llm"
    DEEP_LLM = "deep-llm"
    # D17-D24 extension sources
    VECTOR_FALLBACK = "vector-fallback"
    REUSED = "reused"
    ARBITER_VOTE = "arbiter-vote"
    ARBITER_WEIGHTED = "arbiter-weighted"


class FewShotKind(str, Enum):
    """D18: Few-shot example kind."""

    STATIC = "static"   # Always injected (boundary cases, unclear/unsupported)
    DYNAMIC = "dynamic"  # Injected by retrieval on current input


class ConstraintType(str, Enum):
    """Hard constraint (must satisfy) or soft constraint (try to satisfy)."""

    HARD = "hard"
    SOFT = "soft"


class EvidenceGrade(str, Enum):
    """D26: Evidence grade.

    - ``VERIFIED``: grounded in current input, current context, or confirmed
      KeyFact. Safe to use as the direct basis for hard operations (inventory
      filter, order placement, payment).
    - ``PROVISIONAL``: inferred from historical profile, speculation, default
      values, or model generation. May be used for ranking or reducing
      clarification, but MUST be upgraded or explicitly disclosed before
      hard operations.
    """

    VERIFIED = "verified"
    PROVISIONAL = "provisional"


class Evidence(BaseModel):
    """D26: A piece of evidence with grade and source.

    Used to make intent recognition explainable and to enforce that hard
    operations only rely on verified facts.
    """

    content: str = Field(description="Evidence content, e.g. '当前输入：第二条'")
    grade: EvidenceGrade = Field(description="Evidence grade: verified or provisional")
    source: str = Field(
        description=(
            "Evidence source: 'current_input' | 'context' | 'key_fact' | "
            "'user_profile' | 'inferred' | 'default'"
        )
    )


class SlotValue(BaseModel):
    """D26: A slot value annotated with evidence grade.

    Slots extracted from the current input are tagged ``verified``; slots
    filled from historical profile / speculation / defaults are tagged
    ``provisional``. The ``evidence_ref`` points to the corresponding
    entry in ``verified_evidence`` or ``provisional_evidence``.
    """

    value: str = Field(description="Slot value")
    evidence_grade: EvidenceGrade = Field(
        default=EvidenceGrade.VERIFIED,
        description="Evidence grade for this slot value",
    )
    evidence_ref: str | None = Field(
        default=None,
        description="Reference to the Evidence.content that grounds this value",
    )


class SlotDefinition(BaseModel):
    """Definition of a slot for an intent."""

    name: str = Field(description="Slot name, e.g. 'category', 'budget_max'")
    required: bool = Field(default=False, description="Whether this slot is required")
    default: Any | None = Field(default=None, description="Default value if not provided")
    description: str = Field(default="", description="Slot description for LLM prompt")


class Constraint(BaseModel):
    """A hard or soft constraint extracted from user input."""

    type: ConstraintType = Field(description="Hard (must) or Soft (try)")
    expression: str = Field(description="Constraint expression, e.g. 'price<100'")
    raw_text: str = Field(default="", description="Original user text for this constraint")


class IntentDefinition(BaseModel):
    """Definition of a recognized intent."""

    name: str = Field(description="Intent name, e.g. 'product_recommendation'")
    description: str = Field(description="Human-readable description for LLM prompt")
    positive_examples: list[str] = Field(default_factory=list, description="Examples that match this intent")
    negative_examples: list[str] = Field(default_factory=list, description="Examples that do NOT match")
    slots: list[SlotDefinition] = Field(default_factory=list, description="Slot definitions")
    hard_constraints_enabled: bool = Field(default=True, description="Whether to extract hard constraints")
    soft_constraints_enabled: bool = Field(default=True, description="Whether to extract soft constraints")
    parent_intent: str | None = Field(default=None, description="Parent intent for hierarchical structure")


class FewShotExample(BaseModel):
    """D18: Few-shot example with static/dynamic kind tag."""

    text: str = Field(description="User input example")
    intent: str | None = Field(default=None, description="Expected intent (None for unclear/unsupported cases)")
    output: dict[str, Any] | None = Field(default=None, description="Full expected output dict, optional")
    kind: FewShotKind = Field(default=FewShotKind.DYNAMIC, description="static = always injected, dynamic = retrieval-based")
    description: str = Field(default="", description="Optional description for debugging")


class MultiIntentRelation(BaseModel):
    """D20: Dependency relation between sub-intents in a multi-intent input."""

    src: str = Field(description="Source intent (executes after dst)")
    dst: str = Field(description="Destination intent (executes before src)")
    constraints: list[str] = Field(default_factory=list, description="Dependency constraints, e.g. ['XX condition satisfied']")


class VectorMatchEntry(BaseModel):
    """D21: Pre-built 'user input -> intent' vector mapping entry."""

    text: str = Field(description="Original user input text")
    intent: str = Field(description="Bound intent name")
    vector: list[float] = Field(default_factory=list, description="Vector embedding of the text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata: source, added_at, etc.")


class OverlapReport(BaseModel):
    """D19: Overlap detection report between two intents."""

    intent_a: str = Field(description="First intent name")
    intent_b: str = Field(description="Second intent name")
    similarity: float = Field(description="Similarity score [0, 1]")
    suggestion: str = Field(default="", description="Governance suggestion: split or merge")


class SlotState(BaseModel):
    """Accumulated slot state across turns for a session."""

    session_id: str = Field(description="Session identifier")
    intent: str | None = Field(default=None, description="Current intent")
    slots: dict[str, Any] = Field(default_factory=dict, description="Extracted slot values")
    missing_slots: list[str] = Field(default_factory=list, description="Slots not yet provided")
    hard_constraints: list[Constraint] = Field(default_factory=list)
    soft_constraints: list[Constraint] = Field(default_factory=list)
    turn: int = Field(default=0, description="Current turn number")


class IntentRecognitionResult(BaseModel):
    """Complete output of the three-layer intent recognition pipeline."""

    intent: str | None = Field(default=None, description="Recognized intent name, or None if rejected")
    confidence: float = Field(default=0.0, description="Confidence score [0, 1]")
    source: RecognitionSource = Field(default=RecognitionSource.CODE_LAYER, description="Which layer produced this")
    slots: dict[str, Any] = Field(default_factory=dict, description="Extracted slot values")
    missing_slots: list[str] = Field(default_factory=list, description="Required slots not yet provided")
    need_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_question: str | None = Field(default=None, description="Question to ask user")
    hard_constraints: list[Constraint] = Field(default_factory=list)
    soft_constraints: list[Constraint] = Field(default_factory=list)
    rejection_reason: str | None = Field(default=None, description="Reason for rejection if unsupported")
    layer_reached: int = Field(default=1, description="Highest layer reached (1, 2, or 3)")
    signals: dict[str, float] = Field(default_factory=dict, description="Multi-signal confidence breakdown")
    sub_intents: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-intents for multi-intent input (deprecated alias of independent_intents)",
    )
    intent_switched: bool = Field(default=False, description="Whether intent switched from previous turn")
    previous_intent: str | None = Field(default=None, description="Previous turn's intent if switched")
    # D20: Multi-intent extension fields (default empty, backward compatible)
    relations: list[MultiIntentRelation] = Field(default_factory=list, description="Dependency relations between sub-intents")
    pending_intents: list[str] = Field(default_factory=list, description="Pending intents in topological order (first already executed)")

    # D26/D27/D31: Structured output protocol extension fields (all optional, backward compatible)
    normalized_query: str = Field(
        default="",
        description="D31: Normalized query string from normalization module (empty if normalization skipped)",
    )
    sub_tasks: list[str] = Field(
        default_factory=list,
        description="D27: Sub-tasks within the main flow (e.g. compare_with_reference, verify_delivery)",
    )
    independent_intents: list[str] = Field(
        default_factory=list,
        description="D27: Independent intents that trigger different business flows (primary field; sub_intents is deprecated alias)",
    )
    verified_evidence: list[Evidence] = Field(
        default_factory=list,
        description="D26: Evidence grounded in current input / context / confirmed KeyFact",
    )
    provisional_evidence: list[Evidence] = Field(
        default_factory=list,
        description="D26: Evidence inferred from history / speculation / defaults",
    )
    assumptions_disclosed: bool = Field(
        default=False,
        description="D26: True when provisional evidence was disclosed as assumptions instead of triggering clarification",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="D26: List of disclosed assumptions (e.g. '假设尺码为 M（来自历史画像）')",
    )
    arbitration_breakdown: dict[str, Any] = Field(
        default_factory=dict,
        description="D28: Five-factor arbitration breakdown (factor -> pass/fail/skipped) and reason; empty when arbitration did not run",
    )

    @model_validator(mode="after")
    def _sync_sub_intents_alias(self) -> "IntentRecognitionResult":
        """D27/D31: Keep ``sub_intents`` (deprecated) in sync with ``independent_intents``.

        ``independent_intents`` is the source of truth. ``sub_intents`` is kept
        as a deprecated alias for backward compatibility with existing clients.
        The two fields are always kept equal.
        """
        if self.independent_intents:
            if self.sub_intents != self.independent_intents:
                self.sub_intents = list(self.independent_intents)
        elif self.sub_intents:
            # Backward compat: someone set sub_intents directly (e.g. legacy L2/L3 code)
            self.independent_intents = list(self.sub_intents)
        return self

    @property
    def is_unsupported(self) -> bool:
        """True if the input was rejected as unsupported."""
        return self.intent is None and self.rejection_reason is not None

    @property
    def is_unclear(self) -> bool:
        """True if clarification is needed."""
        return self.need_clarification

    def get_slot_evidence(self, name: str) -> SlotValue | None:
        """D26: Get a slot value with evidence grade.

        If the slot value is a plain value (str/int/etc.), it is wrapped in a
        ``SlotValue`` with ``grade=verified`` (default for backward compat).
        If the slot value is already a ``SlotValue`` or a dict with
        ``evidence_grade``, it is returned as-is.
        """
        value = self.slots.get(name)
        if value is None:
            return None
        if isinstance(value, SlotValue):
            return value
        if isinstance(value, dict) and "evidence_grade" in value:
            return SlotValue(
                value=str(value.get("value", "")),
                evidence_grade=EvidenceGrade(value.get("evidence_grade", "verified")),
                evidence_ref=value.get("evidence_ref"),
            )
        # Plain value: default to verified (from current input)
        return SlotValue(value=str(value), evidence_grade=EvidenceGrade.VERIFIED)
