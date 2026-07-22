"""Cross-turn slot merger - accumulates slots across turns (D9)."""

from __future__ import annotations

from typing import Any

from ..intent_registry import IntentRegistry
from ..models import Constraint, ConstraintType, SlotState
from ..storage.base import SlotStateStore


class CrossTurnSlotMerger:
    """Merges slots across turns with latest-wins strategy and conflict detection."""

    def __init__(self, store: SlotStateStore) -> None:
        self._store = store
        self._conflicts: list[dict[str, Any]] = []

    def merge(
        self,
        session_id: str,
        new_slots: dict[str, Any],
        turn: int,
        intent: str | None = None,
        hard_constraints: list[Constraint] | None = None,
        soft_constraints: list[Constraint] | None = None,
    ) -> SlotState:
        """Merge new slots into existing state. Latest-wins for conflicts."""
        state = self._store.get(session_id)
        if state is None:
            state = SlotState(session_id=session_id, turn=turn)

        # Detect conflicts: same key with different non-None values
        for key, new_val in new_slots.items():
            if new_val is None:
                continue
            old_val = state.slots.get(key)
            if old_val is not None and old_val != new_val:
                self._conflicts.append({
                    "session_id": session_id,
                    "key": key,
                    "old_value": old_val,
                    "new_value": new_val,
                    "turn": turn,
                })

        # Merge slots (latest-wins)
        merged = self._store.merge_slots(session_id, new_slots, turn)
        merged = self._store.get(session_id) or SlotState(session_id=session_id, turn=turn)

        # Update intent if provided
        if intent:
            merged.intent = intent

        # Accumulate constraints (don't overwrite, append unique)
        if hard_constraints:
            existing_exprs = {c.expression for c in merged.hard_constraints}
            for c in hard_constraints:
                if c.expression not in existing_exprs:
                    merged.hard_constraints.append(c)
                    existing_exprs.add(c.expression)

        if soft_constraints:
            existing_exprs = {c.expression for c in merged.soft_constraints}
            for c in soft_constraints:
                if c.expression not in existing_exprs:
                    merged.soft_constraints.append(c)
                    existing_exprs.add(c.expression)

        merged.turn = turn
        self._store.save(merged)
        return merged

    def get_missing_required(
        self, state: SlotState, intent_name: str, registry: IntentRegistry
    ) -> list[str]:
        """Get required slots not yet filled."""
        required = registry.get_required_slots(intent_name)
        return [r for r in required if r not in state.slots or state.slots[r] is None]

    def update_missing_slots(
        self, state: SlotState, intent_name: str, registry: IntentRegistry
    ) -> SlotState:
        """Update missing_slots field in state based on current slots."""
        state.missing_slots = self.get_missing_required(state, intent_name, registry)
        self._store.save(state)
        return state

    @staticmethod
    def has_constraint_conflict(
        state: SlotState, new_constraints: list[Constraint]
    ) -> bool:
        """Check if new constraints conflict with existing ones."""
        for new_c in new_constraints:
            for existing in state.hard_constraints + state.soft_constraints:
                # Same expression but different type (hard vs soft) = conflict
                if (existing.expression == new_c.expression
                        and existing.type != new_c.type):
                    return True
                # Contradictory expressions (e.g., price<100 vs price>200)
                if (existing.type == ConstraintType.HARD
                        and new_c.type == ConstraintType.HARD
                        and _is_contradictory(existing.expression, new_c.expression)):
                    return True
        return False

    def get_conflicts(self) -> list[dict[str, Any]]:
        """Get all recorded slot conflicts."""
        return list(self._conflicts)

    def clear_conflicts(self) -> None:
        """Clear conflict log."""
        self._conflicts.clear()


def _is_contradictory(expr1: str, expr2: str) -> bool:
    """Heuristic check for contradictory constraint expressions."""
    # Simple check: same field with opposite operators
    # e.g., "price<100" vs "price>200"
    import re
    m1 = re.match(r"(\w+)([<>]=?)(\w+)", expr1)
    m2 = re.match(r"(\w+)([<>>=?])(\w+)", expr2)
    if m1 and m2 and m1.group(1) == m2.group(1):
        op1, op2 = m1.group(2), m2.group(2)
        if ("<" in op1 and ">" in op2) or (">" in op1 and "<" in op2):
            try:
                v1, v2 = float(m1.group(3)), float(m2.group(3))
                if v1 < v2 and "<" in op1:
                    return True
                if v1 > v2 and ">" in op1:
                    return True
            except ValueError:
                pass
    return False
