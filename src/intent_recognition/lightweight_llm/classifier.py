"""Lightweight LLM classifier (Layer 2) - D4, D5, D12, D27.

Calls a Flash/Mini model via the LLMClient interface, parses the structured
JSON output, and produces an IntentRecognitionResult. Uses the
ConfidenceRouter to decide accept / clarify / escalate, and the
MultiSignalFuser to combine LLM confidence with auxiliary signals.

D27 adds a lightweight rule-based boundary check that splits intra-flow
``sub_tasks`` from genuinely independent intents. This is a best-effort
heuristic at L2; L3 (deep LLM) does the rigorous determination.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import IntentRecognitionConfig
from ..intent_registry import IntentRegistry
from ..models import (
    Constraint,
    ConstraintType,
    IntentRecognitionResult,
    RecognitionSource,
)
from user_input_normalization.llm.base import LLMClient
from .confidence_router import ArbitrationInput, ConfidenceRouter
from .multi_signal import MultiSignalFuser
from .prompts import build_system_prompt, build_user_prompt


# D27: Phrases that indicate an intra-flow step rather than a separate
# business flow. Used by the simple boundary detector below.
_SUB_TASK_PATTERNS: tuple[str, ...] = (
    "对比", "比价", "比较价格", "查配送", "配送时间", "查参数",
    "看参数", "看评价", "查评价", "看详情", "看图片",
)

# D27: Markers suggesting the user is chaining independent intents.
_INDEPENDENT_MARKERS: tuple[str, ...] = (
    "顺便", "另外", "同时还要", "然后还要", "此外",
)


class LightweightLLMClassifier:
    """Layer 2 classifier: lightweight LLM-based intent recognition.

    Uses a Flash/Mini model for fast, cheap recognition. Escalates to
    Layer 3 (deep LLM) when confidence is below the clarify threshold.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: IntentRegistry,
        config: IntentRecognitionConfig | None = None,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._config = config or IntentRecognitionConfig()
        self._router = ConfidenceRouter(
            self._config.confidence,
            self._config.arbitration,
        )
        self._fuser = MultiSignalFuser(self._config.confidence)
        # Cache the system prompt (rebuild only if registry changes)
        self._system_prompt: str = build_system_prompt(self._registry)

    def classify(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        few_shot_examples: list[dict[str, Any]] | None = None,
        rule_matched: bool | None = None,
        vector_sim: float | None = None,
        historical_acc: float | None = None,
    ) -> IntentRecognitionResult:
        """Recognize intent via the lightweight LLM.

        Args:
            text: Normalized user input.
            context: Optional context dict (unused by default but reserved
                for future context-aware prompting).
            few_shot_examples: Optional few-shot examples to inject.
            rule_matched: Whether the code layer produced a rule match
                (boosts fused confidence via MultiSignalFuser). When None
                (default), the LLM confidence is trusted directly.
            vector_sim: Vector similarity score in [0, 1]. When None,
                not applied.
            historical_acc: Historical accuracy for this intent in [0, 1].
                When None, not applied.

        Returns:
            IntentRecognitionResult with source=LIGHTWEIGHT_LLM,
            layer_reached=2.
        """
        user_prompt = build_user_prompt(text, few_shot_examples)
        raw_response = self._llm.chat(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=2000,
        )

        data = self._extract_json(raw_response)
        if not data:
            # Parsing failed: return an escalate result so Layer 3 can retry
            return IntentRecognitionResult(
                intent=None,
                confidence=0.0,
                source=RecognitionSource.LIGHTWEIGHT_LLM,
                layer_reached=2,
                rejection_reason="LLM response parsing failed",
                signals={"llm_confidence": 0.0},
            )

        llm_confidence = float(data.get("confidence") or 0.0)
        intent_name = data.get("intent")

        # Fuse multi-signal confidence (D5). When no auxiliary signals are
        # provided (all None), trust the LLM confidence directly instead of
        # letting zero-valued aux signals dilute it.
        aux_provided = (
            rule_matched is not None
            or vector_sim is not None
            or historical_acc is not None
        )
        if aux_provided:
            fused_confidence, breakdown = self._fuser.fuse(
                llm_confidence=llm_confidence,
                rule_match=rule_matched or False,
                vector_sim=vector_sim or 0.0,
                historical_acc=historical_acc or 0.0,
            )
        else:
            fused_confidence = llm_confidence
            breakdown = {
                "llm_confidence": llm_confidence,
                "fused": llm_confidence,
            }

        # Parse slots
        slots_raw = data.get("slots") or {}
        slots: dict[str, Any] = {
            k: v for k, v in slots_raw.items() if v is not None
        }

        # Parse missing slots
        missing_slots = list(data.get("missing_slots") or [])

        # Parse constraints
        hard_constraints = self._parse_constraints(
            data.get("hard_constraints") or [], ConstraintType.HARD
        )
        soft_constraints = self._parse_constraints(
            data.get("soft_constraints") or [], ConstraintType.SOFT
        )

        # Validate intent is in registry (if not None)
        if intent_name is not None and not self._registry.has_intent(intent_name):
            # Unknown intent from LLM: treat as unsupported
            return IntentRecognitionResult(
                intent=None,
                confidence=fused_confidence,
                source=RecognitionSource.LIGHTWEIGHT_LLM,
                layer_reached=2,
                rejection_reason=f"LLM returned unknown intent: {intent_name}",
                signals=breakdown,
            )

        # Cross-check missing required slots
        if intent_name is not None:
            required = self._registry.get_required_slots(intent_name)
            provided_keys = set(slots.keys())
            missing_required = [
                s for s in required if s not in provided_keys
            ]
            # Merge with LLM-reported missing slots (dedupe, preserve order)
            for s in missing_required:
                if s not in missing_slots:
                    missing_slots.append(s)

        need_clarification = bool(data.get("need_clarification"))
        clarification_question = data.get("clarification_question")
        # Pitfall guard: dict.get with default may still return None
        clarification_question = clarification_question or None
        rejection_reason = data.get("rejection_reason") or None

        # Apply ConfidenceRouter decision (D4) with optional D28 five-factor
        # arbitration for the ambiguous zone [clarify, accept).
        decision = self._router.route(fused_confidence)
        arbitration_breakdown: dict[str, Any] = {}
        if (
            decision == "clarify"
            and self._config.arbitration.enable_five_factor
        ):
            # D28: run five-factor arbitration instead of the simple
            # clarify-decision. May upgrade to accept, downgrade to escalate,
            # or hold at clarify with a richer reason.
            arb_input = ArbitrationInput(
                confidence=fused_confidence,
                intent=intent_name,
                missing_slots=missing_slots,
                rule_matched=rule_matched,
                candidate_gap=self._extract_candidate_gap(data),
                historical_acc=historical_acc,
                provisional_required_slots=self._extract_provisional_required(intent_name, slots),
                high_risk_intents=self._config.arbitration.high_risk_intents,
            )
            arb_decision = self._router.arbitrate(arb_input)
            decision = arb_decision.decision
            # If arbitration downgrades to clarify, prefer its reason
            if decision == "clarify" and not clarification_question:
                clarification_question = "请补充更多信息以便确认您的意图。"
            # Record factor breakdown for transparency (kept separate from
            # ``signals`` because signals is typed dict[str, float]).
            arbitration_breakdown = {
                "factors": arb_decision.factor_results,
                "reason": arb_decision.reason,
                "adjusted_confidence": arb_decision.adjusted_confidence,
            }
        if decision == "escalate":
            # Signal escalation: keep result but mark for Layer 3 handling
            need_clarification = False
        elif decision == "clarify":
            need_clarification = True
            if clarification_question is None:
                clarification_question = "请补充更多信息以便确认您的意图。"
        # accept: leave flags as LLM reported

        # D27: Simple boundary detection (best-effort heuristic at L2).
        # Splits intra-flow steps (sub_tasks) from independent intents using
        # keyword rules. L3 does the rigorous determination.
        independent_intents: list[str] = []
        sub_tasks: list[str] = []
        if self._config.boundary.enable_sub_tasks:
            independent_intents, sub_tasks = self.detect_boundary_simple(
                text=text,
                intent=intent_name,
                legacy_sub_intents=self._parse_string_list(data.get("sub_intents")),
            )

        return IntentRecognitionResult(
            intent=intent_name,
            confidence=fused_confidence,
            source=RecognitionSource.LIGHTWEIGHT_LLM,
            layer_reached=2,
            slots=slots,
            missing_slots=missing_slots,
            need_clarification=need_clarification,
            clarification_question=clarification_question,
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            rejection_reason=rejection_reason,
            signals=breakdown,
            independent_intents=independent_intents,
            sub_tasks=sub_tasks,
            arbitration_breakdown=arbitration_breakdown,
        )

    # ------------------------------------------------------------------
    # D27: Simple boundary detection (keyword / rule based)
    # ------------------------------------------------------------------
    def detect_boundary_simple(
        self,
        text: str,
        intent: str | None,
        legacy_sub_intents: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Split intra-flow sub_tasks from independent intents at L2 (D27).

        Heuristics:
        1. If the text contains an independent-chaining marker (``顺便`` /
           ``另外`` / ``此外`` ...) AND there are LLM-returned sub_intents,
           treat those sub_intents whose names are registered intents as
           ``independent_intents``; the rest become ``sub_tasks``.
        2. Otherwise, scan the text for sub-task patterns (``对比`` / ``比价``
           / ``查配送`` ...) and collect them as ``sub_tasks``.
        3. In strict mode, ambiguous targets default to ``sub_tasks``; in
           non-strict mode they default to ``independent_intents`` only when
           a chaining marker is present.

        This is intentionally cheap (no LLM call) and is a best-effort
        pre-split. L3 owns the authoritative determination.

        Returns:
            ``(independent_intents, sub_tasks)`` tuple of string lists.
        """
        independent: list[str] = []
        sub_tasks: list[str] = []

        text_lower = text or ""
        has_chaining_marker = any(m in text_lower for m in _INDEPENDENT_MARKERS)

        # 1. If LLM returned sub_intents (legacy or via custom prompt), split
        #    them by registry membership: registered = independent, others =
        #    sub_tasks (only when chaining marker present).
        if legacy_sub_intents:
            for name in legacy_sub_intents:
                if self._registry.has_intent(name):
                    # In strict mode without a chaining marker, treat even
                    # registered intents as sub_tasks (conservative).
                    if has_chaining_marker and not self._config.boundary.strict_mode:
                        if name not in independent:
                            independent.append(name)
                    else:
                        if name not in sub_tasks:
                            sub_tasks.append(name)
                else:
                    if name not in sub_tasks:
                        sub_tasks.append(name)

        # 2. Scan text for sub-task patterns (intra-flow steps)
        for pat in _SUB_TASK_PATTERNS:
            if pat in text_lower:
                if pat not in sub_tasks:
                    sub_tasks.append(pat)

        # 3. If chaining marker present and we have an intent, ensure intent
        #    is in independent_intents (the user is chaining it with something)
        if has_chaining_marker and intent and intent not in independent:
            # Only push to independent if there is at least one other intent
            # already there (otherwise it's a single-intent flow).
            if independent:
                if intent not in independent:
                    independent.insert(0, intent)
            elif legacy_sub_intents:
                # No registered sub_intents but text indicates chaining;
                # surface the main intent as independent for L3 to adjudicate.
                independent.append(intent)

        return independent, sub_tasks

    @staticmethod
    def _parse_string_list(raw: Any) -> list[str]:
        """Parse a list of strings defensively (mirrors deep LLM helper)."""
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw if x is not None]

    # ------------------------------------------------------------------
    # D28: Helpers for five-factor arbitration inputs
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_candidate_gap(data: dict[str, Any]) -> float | None:
        """Extract Top1-Top2 confidence gap from LLM output (factor 4).

        The L2 prompt does not request top-k confidences by default, so this
        returns ``None`` when the field is absent (factor 4 is then skipped).
        """
        candidates = data.get("candidates") or data.get("topk") or []
        if not isinstance(candidates, list) or len(candidates) < 2:
            return None
        try:
            top1 = float(candidates[0].get("confidence", 0)) if isinstance(candidates[0], dict) else float(candidates[0])
            top2 = float(candidates[1].get("confidence", 0)) if isinstance(candidates[1], dict) else float(candidates[1])
            return abs(top1 - top2)
        except (TypeError, ValueError):
            return None

    def _extract_provisional_required(
        self,
        intent_name: str | None,
        slots: dict[str, Any],
    ) -> list[str]:
        """Identify required slots whose values are provisional (factor 3).

        Mirrors the D26 hard-op check: a slot is provisional when its value
        is a ``SlotValue``/dict tagged ``evidence_grade=provisional``.
        """
        if not intent_name or not self._registry.has_intent(intent_name):
            return []
        required = self._registry.get_required_slots(intent_name)
        provisional: list[str] = []
        for name in required:
            value = slots.get(name)
            if value is None:
                continue
            if isinstance(value, dict) and value.get("evidence_grade") == "provisional":
                provisional.append(name)
            # SlotValue instances are not produced by L2 currently; if they
            # were, we would also inspect ``value.evidence_grade`` here.
        return provisional

    @staticmethod
    def _parse_constraints(
        items: list[dict[str, Any]], ctype: ConstraintType
    ) -> list[Constraint]:
        """Parse a list of constraint dicts into Constraint objects."""
        result: list[Constraint] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            expression = item.get("expression") or item.get("value") or ""
            if not expression:
                continue
            result.append(
                Constraint(
                    type=ctype,
                    expression=expression,
                    raw_text=item.get("raw_text") or "",
                )
            )
        return result

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from text.

        Same pattern as user_input_normalization: tries direct parse, then
        ```json fenced block, then first {...} block.
        """
        if not text:
            return None
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try ```json ... ``` fenced block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try first { ... } block (greedy)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
