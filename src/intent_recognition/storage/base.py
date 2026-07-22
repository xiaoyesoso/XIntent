"""Abstract storage interfaces for intent recognition (D9, D11, D15, D18, D21)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import FewShotExample, FewShotKind, IntentRecognitionResult, SlotState, VectorMatchEntry


class SlotStateStore(ABC):
    """Stores accumulated slot state across turns (D9)."""

    @abstractmethod
    def get(self, session_id: str) -> SlotState | None:
        """Get slot state for a session."""

    @abstractmethod
    def save(self, state: SlotState) -> None:
        """Save or update slot state."""

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete slot state for a session."""

    @abstractmethod
    def merge_slots(
        self, session_id: str, new_slots: dict[str, Any], turn: int
    ) -> SlotState:
        """Merge new slots into existing state, returning updated state."""


class IntentHistoryStore(ABC):
    """Stores intent recognition history per session (D6, D11)."""

    @abstractmethod
    def add(self, session_id: str, result: IntentRecognitionResult, turn: int) -> None:
        """Add a recognition result to history."""

    @abstractmethod
    def get_history(self, session_id: str) -> list[IntentRecognitionResult]:
        """Get all recognition results for a session."""

    @abstractmethod
    def get_latest(self, session_id: str) -> IntentRecognitionResult | None:
        """Get the most recent recognition result."""

    @abstractmethod
    def get_previous_intent(self, session_id: str) -> str | None:
        """Get the intent from the previous turn (for intent switch detection)."""


class EvaluationStore(ABC):
    """Stores evaluation data for metrics (D11, D15)."""

    @abstractmethod
    def record_result(
        self,
        test_case_id: str,
        predicted: IntentRecognitionResult,
        expected_intent: str | None,
        expected_slots: dict[str, Any] | None,
        is_correct: bool,
    ) -> None:
        """Record a single test result."""

    @abstractmethod
    def get_results(self) -> list[dict[str, Any]]:
        """Get all recorded results."""

    @abstractmethod
    def record_implicit_signal(
        self, session_id: str, turn: int, signal_type: str, is_failure: bool
    ) -> None:
        """Record an implicit evaluation signal (D11)."""

    @abstractmethod
    def get_implicit_signals(self, session_id: str) -> list[dict[str, Any]]:
        """Get implicit signals for a session."""


class FewShotStore(ABC):
    """D18: Stores few-shot examples with static/dynamic kind tag."""

    @abstractmethod
    def add(self, example: FewShotExample) -> None:
        """Add a single few-shot example."""

    @abstractmethod
    def bulk_add(self, examples: list[FewShotExample]) -> None:
        """Add multiple few-shot examples."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 3,
        kind: FewShotKind | None = None,
    ) -> list[FewShotExample]:
        """Search for few-shot examples by query text.

        Args:
            query: User input text to match against.
            top_k: Maximum number of results to return.
            kind: Optional filter by FewShotKind. None means all kinds.
        """

    @abstractmethod
    def list_all(self, kind: FewShotKind | None = None) -> list[FewShotExample]:
        """List all few-shot examples, optionally filtered by kind."""


class VectorMatchStore(ABC):
    """D21: Stores 'user input -> intent' vector mappings for fallback matching."""

    @abstractmethod
    def add(self, entry: VectorMatchEntry) -> None:
        """Add a single vector match entry."""

    @abstractmethod
    def bulk_add(self, entries: list[VectorMatchEntry]) -> None:
        """Add multiple vector match entries."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 1,
    ) -> list[VectorMatchEntry]:
        """Search for nearest neighbors of query_vector.

        Args:
            query_vector: Vector embedding of the user input.
            top_k: Maximum number of results to return.

        Returns:
            List of VectorMatchEntry sorted by similarity (descending).
        """

    @abstractmethod
    def list_all(self) -> list[VectorMatchEntry]:
        """List all stored vector match entries."""
