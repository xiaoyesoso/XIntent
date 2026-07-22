"""Confidence router (D4) and five-factor arbitration (D28).

Routes recognition results based on confidence thresholds:
  - >= accept_threshold (0.85)  -> "accept"
  - >= clarify_threshold (0.6)  -> "clarify"
  - < clarify_threshold         -> "escalate" (to Layer 3 / deep LLM)

D28 extends this with a five-factor arbitration that triggers when L2
confidence falls in the ambiguous zone ``[clarify_threshold,
accept_threshold)``. The five factors, evaluated in priority order, are:

  1. Rule validation         - hard-rule contradiction -> Clarify/Reject
  2. Slot completeness       - missing required slots -> Clarify
  3. Hard-constraint risk    - high-risk intent + provisional evidence -> Clarify
  4. Candidate gap           - Top1-Top2 confidence gap < threshold -> Escalate
  5. Confidence + history    - fused confidence + historical accuracy

Factor 5 reuses the D5 multi-signal fusion score; it does NOT replace it.
Confidence is treated as a signal, not a probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import ArbitrationConfig, ConfidenceConfig


@dataclass
class ArbitrationInput:
    """Input to the five-factor arbitration (D28).

    All factors are optional; ``None``/empty means the factor is skipped
    (treated as pass with no information).
    """

    confidence: float
    intent: str | None = None
    missing_slots: list[str] = field(default_factory=list)
    rule_matched: bool | None = None
    candidate_gap: float | None = None
    historical_acc: float | None = None
    provisional_required_slots: list[str] = field(default_factory=list)
    high_risk_intents: list[str] = field(default_factory=list)
    # Optional secondary intent (Top-2 prediction) for factor 4
    secondary_intent: str | None = None


@dataclass
class ArbitrationDecision:
    """Decision returned by :meth:`ConfidenceRouter.arbitrate`."""

    decision: str  # "accept" | "clarify" | "escalate" | "reject"
    reason: str
    factor_results: dict[str, str] = field(default_factory=dict)
    adjusted_confidence: float = 0.0


class ConfidenceRouter:
    """Routes based on confidence thresholds.

    Returns one of three decisions:
      - "accept":    confidence is high enough to directly accept
      - "clarify":   confidence is ambiguous; ask a clarification question
      - "escalate":  confidence is too low; escalate to Layer 3 (deep LLM)
    """

    def __init__(
        self,
        config: ConfidenceConfig | None = None,
        arbitration_config: ArbitrationConfig | None = None,
    ) -> None:
        self._config = config or ConfidenceConfig()
        self._arb_config = arbitration_config or ArbitrationConfig()

    @property
    def accept_threshold(self) -> float:
        return self._config.accept_threshold

    @property
    def clarify_threshold(self) -> float:
        return self._config.clarify_threshold

    def route(self, confidence: float) -> str:
        """Return the routing decision for a given confidence score."""
        if confidence >= self._config.accept_threshold:
            return "accept"
        if confidence >= self._config.clarify_threshold:
            return "clarify"
        return "escalate"

    # ------------------------------------------------------------------
    # D28: Five-factor arbitration
    # ------------------------------------------------------------------
    def arbitrate(self, inp: ArbitrationInput) -> ArbitrationDecision:
        """Run five-factor arbitration on an ambiguous-zone result.

        Pre-condition: caller should only invoke this when ``confidence``
        is in ``[clarify_threshold, accept_threshold)``. Outside this band
        the simple :meth:`route` decision stands (high-confidence accept or
        low-confidence escalate).

        The five factors are evaluated in priority order. The first failing
        factor that mandates Clarify/Escalate wins. Factor results are
        recorded for transparency.
        """
        cfg = self._arb_config
        factor_results: dict[str, str] = {}
        adjusted = inp.confidence

        # Factor 1: Rule validation
        # rule_matched=True -> positive (pass); False -> weak support (fail);
        # None -> skip (no rule signal available).
        if inp.rule_matched is None:
            factor_results["rule_validation"] = "skipped"
        elif inp.rule_matched:
            factor_results["rule_validation"] = "pass"
            adjusted = min(1.0, adjusted + 0.05)
        else:
            factor_results["rule_validation"] = "fail"
            adjusted = max(0.0, adjusted - 0.05)

        # Factor 2: Slot completeness
        if inp.missing_slots:
            factor_results["slot_completeness"] = "fail"
            return ArbitrationDecision(
                decision="clarify",
                reason=f"missing required slots: {inp.missing_slots}",
                factor_results=factor_results,
                adjusted_confidence=adjusted,
            )
        factor_results["slot_completeness"] = "pass"

        # Factor 3: Hard-constraint risk
        # High-risk intent with provisional required slots must clarify
        # (mirrors D26 hard-op check at the routing layer).
        if (
            cfg.risk_aware_clarify
            and inp.intent
            and inp.intent in cfg.high_risk_intents
            and inp.provisional_required_slots
        ):
            factor_results["hard_constraint_risk"] = "fail"
            return ArbitrationDecision(
                decision="clarify",
                reason=(
                    f"high-risk intent '{inp.intent}' has provisional "
                    f"required slots: {inp.provisional_required_slots}"
                ),
                factor_results=factor_results,
                adjusted_confidence=adjusted,
            )
        factor_results["hard_constraint_risk"] = "pass"

        # Factor 4: Candidate gap
        # If Top1 and Top2 are too close, escalate to L3 for a tie-break.
        if inp.candidate_gap is not None:
            if inp.candidate_gap < cfg.candidate_gap_threshold:
                factor_results["candidate_gap"] = "fail"
                return ArbitrationDecision(
                    decision="escalate",
                    reason=(
                        f"candidate gap {inp.candidate_gap:.3f} < "
                        f"threshold {cfg.candidate_gap_threshold}"
                    ),
                    factor_results=factor_results,
                    adjusted_confidence=adjusted,
                )
            factor_results["candidate_gap"] = "pass"
        else:
            factor_results["candidate_gap"] = "skipped"

        # Factor 5: Confidence + historical accuracy
        # If historical accuracy is poor AND adjusted confidence is still
        # below the accept threshold, clarify rather than accept.
        if inp.historical_acc is not None and inp.historical_acc < 0.5:
            factor_results["confidence_history"] = "fail"
            if adjusted < self._config.accept_threshold:
                return ArbitrationDecision(
                    decision="clarify",
                    reason=(
                        f"historical accuracy {inp.historical_acc:.2f} < 0.5 "
                        f"and adjusted confidence {adjusted:.2f} < accept"
                    ),
                    factor_results=factor_results,
                    adjusted_confidence=adjusted,
                )
        else:
            factor_results["confidence_history"] = "pass"

        # All factors passed (or skipped): accept
        return ArbitrationDecision(
            decision="accept",
            reason="all five factors passed",
            factor_results=factor_results,
            adjusted_confidence=adjusted,
        )

    def arbitrate_l2_l3(
        self,
        l2: Any,
        l3: Any,
    ) -> ArbitrationDecision:
        """Three-layer result-disagreement arbitration (D28 task 4.7).

        Called by the pipeline when L2 escalated to L3 and L3 produced a
        different intent than L2. L1 has priority (early-returns on hit),
        so by the time we reach L3 L1 has already missed; the disagreement
        is between L2 and L3.

        Rules:
        - If L3 confidence >= accept_threshold AND L3 has no missing slots
          -> accept L3 (high-confidence L3 overrides L2).
        - If L3 confidence < clarify_threshold AND L2 confidence < clarify_threshold
          -> force Clarify (three-layer ambiguity).
        - Otherwise run five-factor arbitration on the L3 result.
        """
        l2_conf = float(getattr(l2, "confidence", 0.0) or 0.0)
        l3_conf = float(getattr(l3, "confidence", 0.0) or 0.0)
        l3_intent = getattr(l3, "intent", None)
        l3_missing = list(getattr(l3, "missing_slots", []) or [])
        l3_slots = getattr(l3, "slots", {}) or {}

        # Detect provisional required slots on L3 (for factor 3)
        provisional_required: list[str] = []
        for name, value in l3_slots.items():
            if isinstance(value, dict) and value.get("evidence_grade") == "provisional":
                provisional_required.append(name)

        # If L3 is high-confidence with complete slots, accept it
        if l3_conf >= self._config.accept_threshold and not l3_missing:
            return ArbitrationDecision(
                decision="accept",
                reason="L3 high-confidence with complete slots overrides L2",
                factor_results={"l2_confidence": f"{l2_conf:.3f}", "l3_confidence": f"{l3_conf:.3f}"},
                adjusted_confidence=l3_conf,
            )

        # If both layers are below clarify threshold, force Clarify
        if l3_conf < self._config.clarify_threshold and l2_conf < self._config.clarify_threshold:
            return ArbitrationDecision(
                decision="clarify",
                reason=(
                    f"three-layer ambiguity: L2={l2_conf:.3f}, L3={l3_conf:.3f} "
                    f"both below clarify threshold"
                ),
                factor_results={"l2_confidence": f"{l2_conf:.3f}", "l3_confidence": f"{l3_conf:.3f}"},
                adjusted_confidence=l3_conf,
            )

        # Otherwise run five-factor arbitration on L3's result
        return self.arbitrate(ArbitrationInput(
            confidence=l3_conf,
            intent=l3_intent,
            missing_slots=l3_missing,
            rule_matched=None,
            candidate_gap=None,
            historical_acc=None,
            provisional_required_slots=provisional_required,
            high_risk_intents=self._arb_config.high_risk_intents,
        ))
