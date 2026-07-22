"""Deep reasoning LLM classifier (Layer 3, D6, D20).

Handles 5 complex scenarios: complex expressions, context dependency,
intent switching, multi-intent, implicit info completion.
D20: True multi-intent detection with relations, topological order,
and process-description filtering.
"""

from __future__ import annotations

import json
import re
from typing import Any

from user_input_normalization.llm.base import LLMClient

from ..config import IntentRecognitionConfig
from ..intent_registry import IntentRegistry
from ..models import (
    Constraint,
    ConstraintType,
    IntentRecognitionResult,
    MultiIntentRelation,
    RecognitionSource,
)
from .prompts import build_system_prompt, build_user_prompt


class DeepLLMClassifier:
    """Layer 3: Deep reasoning LLM for complex intent recognition."""

    def __init__(
        self,
        llm_client: LLMClient,
        registry: IntentRegistry,
        config: IntentRecognitionConfig | None = None,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._config = config or IntentRecognitionConfig()

    def classify(
        self,
        text: str,
        context: dict | None = None,
        dialogue_history: list[dict] | None = None,
        previous_intent: str | None = None,
    ) -> IntentRecognitionResult:
        """Classify intent using deep reasoning LLM.

        Handles complex expressions, context dependency, intent switching,
        multi-intent decomposition, and implicit info completion.
        D20: Also handles true multi-intent detection with sequential
        execution and process-description filtering.
        """
        system_prompt = build_system_prompt(self._registry)
        user_prompt = build_user_prompt(text, dialogue_history, previous_intent)

        response = self._llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        data = self._extract_json(response)
        if not data:
            return IntentRecognitionResult(
                intent=None,
                confidence=0.0,
                source=RecognitionSource.DEEP_LLM,
                layer_reached=3,
                rejection_reason="Deep LLM returned unparseable response",
            )

        intent = data.get("intent") or ""
        confidence = float(data.get("confidence", 0.5))
        slots = data.get("slots", {}) or {}
        missing = data.get("missing_slots", []) or []
        # D27: Parse both independent_intents (primary) and sub_tasks (intra-flow).
        # sub_intents is kept as a backward-compatible alias for independent_intents.
        independent_intents = self._parse_string_list(data.get("independent_intents", []))
        sub_tasks = self._parse_string_list(data.get("sub_tasks", []))
        # Backward compat: if LLM still returns sub_intents (legacy prompt),
        # treat it as independent_intents
        legacy_sub_intents = self._parse_string_list(data.get("sub_intents", []))
        if legacy_sub_intents and not independent_intents:
            independent_intents = legacy_sub_intents
        intent_switched = bool(data.get("intent_switched", False))

        # Parse constraints
        hard_constraints = self._parse_constraints(
            data.get("hard_constraints", []), ConstraintType.HARD
        )
        soft_constraints = self._parse_constraints(
            data.get("soft_constraints", []), ConstraintType.SOFT
        )

        # D20: Parse multi-intent extension fields.
        # IMPORTANT (D27): Only independent_intents enter D20 governance.
        # sub_tasks are intra-flow steps and must NOT be filtered, reordered,
        # or written into relations / pending_intents.
        relations = self._parse_relations(data.get("relations", []))
        pending_intents = self._parse_string_list(data.get("pending_intents", []))
        is_process_description = bool(data.get("is_process_description", False))

        # D20: Apply multi-intent governance (only to independent_intents)
        mi_cfg = self._config.multi_intent
        if mi_cfg.enable:
            # Process-description filter: collapse to single intent
            if mi_cfg.filter_process_description and is_process_description:
                independent_intents = []
                relations = []
                pending_intents = []
            # Sequential execution: pick topological-first as main, rest pending
            if mi_cfg.sequential_execution and len(independent_intents) > 1:
                ordered = _topological_sort(independent_intents, relations)
                # Always set main intent to the topological-first independent intent
                intent = ordered[0]
                pending_intents = ordered[1:]
        # When mi_cfg.enable is False: D20 governance is bypassed; the LLM's
        # raw independent_intents / relations / pending_intents are passed
        # through unchanged (no sequential reordering, no process filter).

        # Validate intent is registered
        if intent and not self._registry.has_intent(intent):
            intent = None
            confidence = 0.0

        return IntentRecognitionResult(
            intent=intent if intent else None,
            confidence=confidence,
            source=RecognitionSource.DEEP_LLM,
            slots=slots,
            missing_slots=missing,
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            independent_intents=independent_intents,
            sub_tasks=sub_tasks,
            intent_switched=intent_switched,
            previous_intent=previous_intent,
            layer_reached=3,
            signals={"llm_confidence": confidence},
            relations=relations,
            pending_intents=pending_intents,
        )

    @staticmethod
    def _parse_constraints(
        raw: list[Any], ctype: ConstraintType
    ) -> list[Constraint]:
        """Parse constraint list from LLM output."""
        result: list[Constraint] = []
        for item in raw:
            if isinstance(item, str):
                result.append(Constraint(type=ctype, expression=item, raw_text=item))
            elif isinstance(item, dict):
                expr = item.get("expression") or item.get("value") or str(item)
                raw_text = item.get("raw_text", expr)
                result.append(Constraint(type=ctype, expression=expr, raw_text=raw_text))
        return result

    @staticmethod
    def _parse_relations(raw: Any) -> list[MultiIntentRelation]:
        """Parse multi-intent dependency relations from LLM output (D20)."""
        if not isinstance(raw, list):
            return []
        result: list[MultiIntentRelation] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            src = item.get("src") or item.get("source") or ""
            dst = item.get("dst") or item.get("destination") or ""
            if not src or not dst:
                continue
            constraints_raw = item.get("constraints", []) or []
            constraints = [str(c) for c in constraints_raw] if isinstance(constraints_raw, list) else []
            result.append(MultiIntentRelation(
                src=str(src),
                dst=str(dst),
                constraints=constraints,
            ))
        return result

    @staticmethod
    def _parse_string_list(raw: Any) -> list[str]:
        """Parse a list of strings from LLM output (defensive)."""
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw if x is not None]

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON from text (handles raw, fenced, and embedded)."""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


def _topological_sort(
    intents: list[str], relations: list[MultiIntentRelation]
) -> list[str]:
    """Sort intents in topological order based on relations (D20).

    Semantics: in a relation ``src -> dst``, ``dst`` executes BEFORE ``src``.
    Returns the order such that earlier items should be executed first.
    Falls back to the original ``intents`` order if a cycle is detected or
    there are no usable relations.
    """
    if not intents or not relations:
        return list(intents)
    # Restrict to relations referencing intents in the list
    intent_set = set(intents)
    filtered = [
        r for r in relations
        if r.src in intent_set and r.dst in intent_set
    ]
    if not filtered:
        return list(intents)
    # Build in-degree: edge dst -> src (dst before src).
    # in_degree counts how many predecessors each node has.
    adj: dict[str, list[str]] = {n: [] for n in intents}
    in_degree: dict[str, int] = {n: 0 for n in intents}
    for r in filtered:
        # dst -> src
        adj[r.dst].append(r.src)
        in_degree[r.src] += 1
    # Kahn's algorithm, preserving original order for ties
    queue = [n for n in intents if in_degree[n] == 0]
    result: list[str] = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for nb in adj[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)
    # If cycle detected (not all nodes processed), fall back to original order
    if len(result) != len(intents):
        return list(intents)
    return result
