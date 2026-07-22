"""In-memory storage implementations for intent recognition."""

from __future__ import annotations

import math
from typing import Any

from ..models import (
    FewShotExample,
    FewShotKind,
    IntentRecognitionResult,
    SlotState,
    VectorMatchEntry,
)
from .base import EvaluationStore, FewShotStore, IntentHistoryStore, SlotStateStore, VectorMatchStore


class MemorySlotStateStore(SlotStateStore):
    """In-memory slot state store."""

    def __init__(self) -> None:
        self._states: dict[str, SlotState] = {}

    def get(self, session_id: str) -> SlotState | None:
        return self._states.get(session_id)

    def save(self, state: SlotState) -> None:
        self._states[state.session_id] = state

    def delete(self, session_id: str) -> None:
        self._states.pop(session_id, None)

    def merge_slots(
        self, session_id: str, new_slots: dict[str, Any], turn: int
    ) -> SlotState:
        state = self._states.get(session_id)
        if state is None:
            state = SlotState(session_id=session_id, turn=turn)
        # Latest-wins: new slots overwrite old (D9)
        for k, v in new_slots.items():
            if v is not None:
                state.slots[k] = v
        state.turn = turn
        self._states[session_id] = state
        return state


class MemoryIntentHistoryStore(IntentHistoryStore):
    """In-memory intent history store."""

    def __init__(self) -> None:
        self._history: dict[str, list[tuple[IntentRecognitionResult, int]]] = {}

    def add(self, session_id: str, result: IntentRecognitionResult, turn: int) -> None:
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append((result, turn))

    def get_history(self, session_id: str) -> list[IntentRecognitionResult]:
        return [r for r, _ in self._history.get(session_id, [])]

    def get_latest(self, session_id: str) -> IntentRecognitionResult | None:
        history = self._history.get(session_id, [])
        if not history:
            return None
        return history[-1][0]

    def get_previous_intent(self, session_id: str) -> str | None:
        history = self._history.get(session_id, [])
        if len(history) < 1:
            return None
        # Return the intent from the last entry
        return history[-1][0].intent


class MemoryEvaluationStore(EvaluationStore):
    """In-memory evaluation store."""

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []
        self._implicit: dict[str, list[dict[str, Any]]] = {}

    def record_result(
        self,
        test_case_id: str,
        predicted: IntentRecognitionResult,
        expected_intent: str | None,
        expected_slots: dict[str, Any] | None,
        is_correct: bool,
    ) -> None:
        self._results.append({
            "test_case_id": test_case_id,
            "predicted_intent": predicted.intent,
            "expected_intent": expected_intent,
            "predicted_slots": predicted.slots,
            "expected_slots": expected_slots,
            "is_correct": is_correct,
            "confidence": predicted.confidence,
            "source": predicted.source.value,
            "layer_reached": predicted.layer_reached,
        })

    def get_results(self) -> list[dict[str, Any]]:
        return list(self._results)

    def record_implicit_signal(
        self, session_id: str, turn: int, signal_type: str, is_failure: bool
    ) -> None:
        if session_id not in self._implicit:
            self._implicit[session_id] = []
        self._implicit[session_id].append({
            "session_id": session_id,
            "turn": turn,
            "signal_type": signal_type,
            "is_failure": is_failure,
        })

    def get_implicit_signals(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._implicit.get(session_id, []))


class MemoryFewShotStore(FewShotStore):
    """D18: In-memory few-shot store with kind filtering and simple text matching.

    Uses token-overlap similarity (Jaccard) for retrieval. Production should
    replace with a vector-backed implementation.
    """

    def __init__(self) -> None:
        self._examples: list[FewShotExample] = []

    def add(self, example: FewShotExample) -> None:
        self._examples.append(example)

    def bulk_add(self, examples: list[FewShotExample]) -> None:
        self._examples.extend(examples)

    def search(
        self,
        query: str,
        top_k: int = 3,
        kind: FewShotKind | None = None,
    ) -> list[FewShotExample]:
        # Filter by kind if specified
        candidates = [e for e in self._examples if kind is None or e.kind == kind]
        if not candidates:
            return []
        # Score by token-overlap Jaccard similarity
        query_tokens = self._tokenize(query)
        scored = [
            (e, self._jaccard(query_tokens, self._tokenize(e.text)))
            for e in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k] if _ > 0]

    def list_all(self, kind: FewShotKind | None = None) -> list[FewShotExample]:
        return [e for e in self._examples if kind is None or e.kind == kind]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        # Simple char-level tokenization for Chinese + whitespace split for English
        tokens: set[str] = set()
        for part in text.split():
            tokens.add(part.lower())
            # Add individual CJK chars
            for ch in part:
                if "\u4e00" <= ch <= "\u9fff":
                    tokens.add(ch)
        return tokens

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0


class MemoryVectorMatchStore(VectorMatchStore):
    """D21: In-memory vector match store using cosine similarity.

    Uses pure-Python math (no numpy dependency) for portability.
    """

    def __init__(self) -> None:
        self._entries: list[VectorMatchEntry] = []

    def add(self, entry: VectorMatchEntry) -> None:
        self._entries.append(entry)

    def bulk_add(self, entries: list[VectorMatchEntry]) -> None:
        self._entries.extend(entries)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 1,
    ) -> list[VectorMatchEntry]:
        if not self._entries or not query_vector:
            return []
        scored = [(e, self._cosine(query_vector, e.vector)) for e in self._entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def list_all(self) -> list[VectorMatchEntry]:
        return list(self._entries)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
