"""Vector matching fallback recognizer (D21).

Acts as a code-layer fallback that compares the user input against a
pre-built ``text -> intent`` vector store. When the closest stored entry
has cosine similarity at or above ``similarity_threshold`` the recognizer
returns the bound intent with ``source=VECTOR_FALLBACK``; otherwise it
returns ``None`` so the waterfall can escalate to the next layer.

The default text vectorizer is a deterministic, dependency-free
character-hashing embedding into 128 dimensions (L2-normalized). It is
good enough for unit tests and simple demos; production deployments
should inject a real embedding function (e.g. ``OpenAIEmbeddingClient``)
via the ``vectorizer`` constructor argument.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from ..config import VectorFallbackConfig
from ..models import IntentRecognitionResult, RecognitionSource, VectorMatchEntry
from ..storage.base import VectorMatchStore


# Default embedding dimension used by the hashing vectorizer.
_DEFAULT_DIM = 128


def _default_vectorizer(text: str) -> list[float]:
    """Hashing-based 128-dim text vectorizer (no external deps).

    Each character is hashed to a dimension (``hash(char) % dim``); the
    corresponding bucket is incremented by 1. The resulting vector is
    L2-normalized so cosine similarity reduces to a dot product. Returns
    a zero vector when ``text`` is empty so callers can short-circuit
    safely.
    """
    if not text:
        return [0.0] * _DEFAULT_DIM

    vec = [0.0] * _DEFAULT_DIM
    for ch in text:
        idx = hash(ch) % _DEFAULT_DIM
        vec[idx] += 1.0

    # L2 normalize. If all entries are zero (theoretically impossible
    # here because we incremented at least one bucket), leave as-is.
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


class VectorMatcher:
    """D21: Code-layer vector matching fallback recognizer.

    Parameters
    ----------
    store:
        Backing :class:`VectorMatchStore` that holds the pre-built
        ``text -> intent`` entries.
    config:
        :class:`VectorFallbackConfig` (enable flag, similarity
        threshold, top-k).
    vectorizer:
        Optional callable mapping text to a vector. When ``None`` the
        :func:`_default_vectorizer` hashing embedding is used.
    """

    def __init__(
        self,
        store: VectorMatchStore,
        config: VectorFallbackConfig,
        vectorizer: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._vectorizer = vectorizer or _default_vectorizer

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def match(self, text: str) -> IntentRecognitionResult | None:
        """Attempt to match ``text`` against stored vector entries.

        Returns an :class:`IntentRecognitionResult` with
        ``source=VECTOR_FALLBACK`` when similarity is at or above the
        configured threshold; otherwise returns ``None`` to signal the
        caller to try the next recognizer / layer.
        """
        if not self._config.enable:
            return None

        if not text:
            return None

        query_vector = self._vectorizer(text)
        if not query_vector:
            return None

        entries = self._store.search(query_vector, top_k=self._config.top_k)
        if not entries:
            return None

        # The store returns entries sorted by similarity (descending);
        # take the first one as the best match.
        best = entries[0]
        similarity = self._cosine(query_vector, best.vector)

        if similarity < self._config.similarity_threshold:
            return None

        return IntentRecognitionResult(
            intent=best.intent,
            confidence=similarity,
            source=RecognitionSource.VECTOR_FALLBACK,
            layer_reached=1,
            signals={"vector_match": similarity},
        )

    # ------------------------------------------------------------------
    # Seeding / ingestion
    # ------------------------------------------------------------------
    def seed_entries(self, entries: list[VectorMatchEntry]) -> None:
        """Bulk-import pre-built entries (offline ingestion)."""
        self._store.bulk_add(entries)

    def add_entry(
        self,
        text: str,
        intent: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Vectorize ``text`` and append a new entry to the store."""
        vector = self._vectorizer(text)
        entry = VectorMatchEntry(
            text=text,
            intent=intent,
            vector=vector,
            metadata=metadata or {},
        )
        self._store.add(entry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two equal-length vectors.

        Mirrors :meth:`MemoryVectorMatchStore._cosine` so the matcher
        does not depend on a specific store implementation. Returns
        ``0.0`` for empty or length-mismatched vectors.
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
