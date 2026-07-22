"""Storage abstract interfaces (corresponds to D15 storage selection).

Defines abstract interfaces for various types of storage; concrete implementations can be in-memory, Redis, MySQL, vector database, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import (
    DialogueTurn,
    FewShotExample,
    KeyFact,
    UserProfile,
    VocabEntry,
    VocabLevel,
)


class KeyFactStore(ABC):
    """Key fact storage interface (corresponds to D6 pronoun resolution table cross-turn reuse)."""

    @abstractmethod
    def save(self, fact: KeyFact) -> None:
        """Save a key fact."""

    @abstractmethod
    def get(self, fact_id: str) -> KeyFact | None:
        """Get by ID."""

    @abstractmethod
    def get_by_session(self, session_id: str, fact_type: str | None = None) -> list[KeyFact]:
        """Get key fact list by session."""

    @abstractmethod
    def find_pronoun_resolution(
        self, session_id: str, pronoun: str
    ) -> KeyFact | None:
        """Find a resolved pronoun (cross-turn reuse D6)."""

    @abstractmethod
    def revoke(self, fact_id: str) -> None:
        """Revoke a key fact."""


class VocabStore(ABC):
    """Vocabulary table storage interface (corresponds to D10 two-level vocabulary)."""

    @abstractmethod
    def save(self, entry: VocabEntry) -> None:
        """Save a vocabulary entry."""

    @abstractmethod
    def get(self, vocab_id: str) -> VocabEntry | None:
        """Get by ID."""

    @abstractmethod
    def search(
        self,
        term: str,
        level: VocabLevel | None = None,
        industry: str | None = None,
        user_id: str | None = None,
    ) -> list[VocabEntry]:
        """Search by term (supports public + personal)."""

    @abstractmethod
    def search_semantic(
        self, query: str, top_k: int = 5, industry: str | None = None
    ) -> list[VocabEntry]:
        """Semantic search (vector database)."""

    @abstractmethod
    def list_candidates(self, status: str = "candidate") -> list[VocabEntry]:
        """List candidate vocabulary (pending promotion)."""

    @abstractmethod
    def update_status(self, vocab_id: str, status: str) -> None:
        """Update status (promote / approve)."""

    @abstractmethod
    def increment_count(
        self, term: str, user_id: str | None = None
    ) -> VocabEntry | None:
        """Increment occurrence count (for offline analysis)."""


class DialogueHistoryStore(ABC):
    """Dialogue history storage interface (corresponds to D14 dialogue history recall)."""

    @abstractmethod
    def append(self, session_id: str, turn: DialogueTurn) -> None:
        """Append a dialogue turn."""

    @abstractmethod
    def get_recent(self, session_id: str, n: int = 10) -> list[DialogueTurn]:
        """Get the most recent N turns (short-term memory)."""

    @abstractmethod
    def search_semantic(
        self, session_id: str, query: str, top_k: int = 5
    ) -> list[DialogueTurn]:
        """Semantic search of dialogue history (RAG long-term memory)."""

    @abstractmethod
    def get_summary(self, session_id: str) -> str | None:
        """Get dialogue summary."""

    @abstractmethod
    def set_summary(self, session_id: str, summary: str) -> None:
        """Set dialogue summary."""


class FewShotStore(ABC):
    """few-shot example library interface (corresponds to D5 retrieval injection)."""

    @abstractmethod
    def save(self, example: FewShotExample) -> None:
        """Save an example."""

    @abstractmethod
    def search(
        self, query: str, top_k: int = 3, input_type: str | None = None
    ) -> list[FewShotExample]:
        """Search for similar examples (vector / keyword)."""

    @abstractmethod
    def list_all(self) -> list[FewShotExample]:
        """List all (for debugging)."""


class UserProfileStore(ABC):
    """User profile storage interface (corresponds to D14 user profile layer)."""

    @abstractmethod
    def get(self, user_id: str) -> UserProfile | None:
        """Get user profile."""

    @abstractmethod
    def save(self, profile: UserProfile) -> None:
        """Save user profile."""

    @abstractmethod
    def update_preference(self, user_id: str, key: str, value: Any) -> None:
        """Update preference."""

    @abstractmethod
    def add_topic_tendency(self, user_id: str, topic: str, tendency: str) -> None:
        """Add topic tendency (e.g. '三国演义' -> '电视剧')."""
