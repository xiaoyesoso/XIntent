"""In-memory storage implementation (for development / testing, no Redis / MySQL / vector database needed).

All methods are synchronous; semantic search degrades to keyword matching.
Production environments can replace with Redis / MySQL / ChromaDB implementations.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from ..models import (
    DialogueTurn,
    FewShotExample,
    KeyFact,
    UserProfile,
    VocabEntry,
    VocabLevel,
    VocabStatus,
)
from .base import (
    DialogueHistoryStore,
    FewShotStore,
    KeyFactStore,
    UserProfileStore,
    VocabStore,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _tokenize(text: str) -> set[str]:
    """Simple tokenizer: Chinese characters by character + English words."""
    tokens: set[str] = set()
    for word in re.findall(r"[a-zA-Z]+", text):
        tokens.add(word.lower())
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.add(ch)
    return tokens


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Key fact storage
# ---------------------------------------------------------------------------


class MemoryKeyFactStore(KeyFactStore):
    """In-memory key fact storage."""

    def __init__(self) -> None:
        self._facts: dict[str, KeyFact] = {}

    def save(self, fact: KeyFact) -> None:
        if not fact.fact_id:
            fact.fact_id = _gen_id("fact")
        self._facts[fact.fact_id] = fact

    def get(self, fact_id: str) -> KeyFact | None:
        return self._facts.get(fact_id)

    def get_by_session(self, session_id: str, fact_type: str | None = None) -> list[KeyFact]:
        results = [
            f
            for f in self._facts.values()
            if f.session_id == session_id and f.status == "active"
        ]
        if fact_type:
            results = [f for f in results if f.fact_type.value == fact_type]
        return sorted(results, key=lambda f: f.turn)

    def find_pronoun_resolution(self, session_id: str, pronoun: str) -> KeyFact | None:
        for f in self._facts.values():
            if (
                f.session_id == session_id
                and f.fact_type.value == "pronoun_resolution"
                and f.status == "active"
                and f.content.get("pronoun") == pronoun
            ):
                return f
        return None

    def revoke(self, fact_id: str) -> None:
        if fact_id in self._facts:
            self._facts[fact_id].status = "revoked"


# ---------------------------------------------------------------------------
# Vocabulary table storage
# ---------------------------------------------------------------------------


class MemoryVocabStore(VocabStore):
    """In-memory vocabulary table storage."""

    def __init__(self) -> None:
        self._entries: dict[str, VocabEntry] = {}

    def save(self, entry: VocabEntry) -> None:
        if not entry.vocab_id:
            entry.vocab_id = _gen_id("vocab")
        self._entries[entry.vocab_id] = entry

    def get(self, vocab_id: str) -> VocabEntry | None:
        return self._entries.get(vocab_id)

    def search(
        self,
        term: str,
        level: VocabLevel | None = None,
        industry: str | None = None,
        user_id: str | None = None,
    ) -> list[VocabEntry]:
        results = []
        for e in self._entries.values():
            if e.term.lower() != term.lower():
                continue
            if e.status != VocabStatus.PROMOTED:
                continue
            if level and e.level != level:
                continue
            if industry and e.industry != industry and e.industry != "通用":
                continue
            if e.level == VocabLevel.PERSONAL and user_id and e.user_id != user_id:
                continue
            results.append(e)
        # Industry priority sorting: exact industry match > general
        if industry:
            results.sort(key=lambda e: 0 if e.industry == industry else 1)
        return results

    def search_semantic(
        self, query: str, top_k: int = 5, industry: str | None = None
    ) -> list[VocabEntry]:
        query_tokens = _tokenize(query)
        scored: list[tuple[float, VocabEntry]] = []
        for e in self._entries.values():
            if e.status != VocabStatus.PROMOTED:
                continue
            if industry and e.industry != industry and e.industry != "通用":
                continue
            entry_tokens = _tokenize(f"{e.term} {e.standard_meaning}")
            score = _jaccard_similarity(query_tokens, entry_tokens)
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def list_candidates(self, status: str = "candidate") -> list[VocabEntry]:
        return [e for e in self._entries.values() if e.status.value == status]

    def update_status(self, vocab_id: str, status: str) -> None:
        if vocab_id in self._entries:
            self._entries[vocab_id].status = VocabStatus(status)

    def increment_count(
        self, term: str, user_id: str | None = None
    ) -> VocabEntry | None:
        # Find existing candidate entry
        for e in self._entries.values():
            if e.term.lower() == term.lower():
                if user_id and e.level == VocabLevel.PERSONAL and e.user_id != user_id:
                    continue
                e.occurrence_count += 1
                e.updated_at = __import__("datetime").datetime.now()
                return e
        # Create new candidate
        entry = VocabEntry(
            vocab_id=_gen_id("vocab"),
            term=term,
            level=VocabLevel.PERSONAL if user_id else VocabLevel.PUBLIC,
            standard_meaning="",
            status=VocabStatus.CANDIDATE,
            user_id=user_id,
            occurrence_count=1,
            discussant_count=1,
        )
        self._entries[entry.vocab_id] = entry
        return entry


# ---------------------------------------------------------------------------
# Dialogue history storage
# ---------------------------------------------------------------------------


class MemoryDialogueHistoryStore(DialogueHistoryStore):
    """In-memory dialogue history storage."""

    def __init__(self) -> None:
        self._history: dict[str, list[DialogueTurn]] = {}
        self._summaries: dict[str, str] = {}

    def append(self, session_id: str, turn: DialogueTurn) -> None:
        self._history.setdefault(session_id, []).append(turn)

    def get_recent(self, session_id: str, n: int = 10) -> list[DialogueTurn]:
        turns = self._history.get(session_id, [])
        return turns[-n:] if n > 0 else turns

    def search_semantic(
        self, session_id: str, query: str, top_k: int = 5
    ) -> list[DialogueTurn]:
        query_tokens = _tokenize(query)
        turns = self._history.get(session_id, [])
        scored: list[tuple[float, DialogueTurn]] = []
        for t in turns:
            turn_tokens = _tokenize(t.content)
            score = _jaccard_similarity(query_tokens, turn_tokens)
            if score > 0:
                scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    def get_summary(self, session_id: str) -> str | None:
        return self._summaries.get(session_id)

    def set_summary(self, session_id: str, summary: str) -> None:
        self._summaries[session_id] = summary


# ---------------------------------------------------------------------------
# few-shot example storage
# ---------------------------------------------------------------------------


class MemoryFewShotStore(FewShotStore):
    """In-memory few-shot example storage."""

    def __init__(self) -> None:
        self._examples: dict[str, FewShotExample] = {}

    def save(self, example: FewShotExample) -> None:
        if not example.example_id:
            example.example_id = _gen_id("ex")
        self._examples[example.example_id] = example

    def search(
        self, query: str, top_k: int = 3, input_type: str | None = None
    ) -> list[FewShotExample]:
        query_tokens = _tokenize(query)
        scored: list[tuple[float, FewShotExample]] = []
        for ex in self._examples.values():
            if input_type:
                type_values = [t.value for t in ex.input_type]
                if input_type not in type_values and input_type not in [t.value for t in ex.input_type]:
                    pass  # Do not filter, as input types may not match exactly
            ex_tokens = _tokenize(f"{ex.input} {ex.context_summary}")
            score = _jaccard_similarity(query_tokens, ex_tokens)
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Filter out zero-score entries
        scored = [(s, e) for s, e in scored if s > 0]
        return [e for _, e in scored[:top_k]]

    def list_all(self) -> list[FewShotExample]:
        return list(self._examples.values())


# ---------------------------------------------------------------------------
# User profile storage
# ---------------------------------------------------------------------------


class MemoryUserProfileStore(UserProfileStore):
    """In-memory user profile storage."""

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    def save(self, profile: UserProfile) -> None:
        self._profiles[profile.user_id] = profile

    def update_preference(self, user_id: str, key: str, value: Any) -> None:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        self._profiles[user_id].preferences[key] = value

    def add_topic_tendency(self, user_id: str, topic: str, tendency: str) -> None:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        self._profiles[user_id].topic_tendencies[topic] = tendency
