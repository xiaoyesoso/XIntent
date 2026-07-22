"""Tests for the D21 vector matching fallback recognizer."""

import math

import pytest

from intent_recognition.code_layer.vector_matcher import (
    VectorMatcher,
    _default_vectorizer,
)
from intent_recognition.config import VectorFallbackConfig
from intent_recognition.models import (
    IntentRecognitionResult,
    RecognitionSource,
    VectorMatchEntry,
)
from intent_recognition.storage.memory import MemoryVectorMatchStore


def _make_config(**overrides) -> VectorFallbackConfig:
    """Build a VectorFallbackConfig with enable=True plus overrides."""
    defaults = {"enable": True, "similarity_threshold": 0.92, "top_k": 1}
    defaults.update(overrides)
    return VectorFallbackConfig(**defaults)


def _make_entry(text: str, intent: str) -> VectorMatchEntry:
    """Build a VectorMatchEntry pre-populated with the default vector."""
    return VectorMatchEntry(
        text=text,
        intent=intent,
        vector=_default_vectorizer(text),
        metadata={"source": "test"},
    )


class TestVectorMatcherMatch:
    """Tests for VectorMatcher.match()."""

    def test_high_similarity_match_returns_result(self):
        """Adding an entry with the same text yields a confident match."""
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, _make_config())
        matcher.add_entry("帮我推荐手机", "product_recommendation")

        result = matcher.match("帮我推荐手机")
        assert result is not None
        assert result.intent == "product_recommendation"
        # identical vectors -> cosine = 1.0 (use approx for float safety)
        assert result.confidence == pytest.approx(1.0)
        assert result.source == RecognitionSource.VECTOR_FALLBACK
        assert result.layer_reached == 1
        assert result.signals.get("vector_match") == pytest.approx(1.0)

    def test_low_similarity_no_match(self):
        """Very different text should not cross the similarity threshold."""
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, _make_config(similarity_threshold=0.99))
        matcher.add_entry("查询订单 12345", "order_query")

        # Completely different text - low overlap in hash buckets.
        result = matcher.match("XYZQRSTUVWXYZabcdef")
        assert result is None

    def test_disabled_config_returns_none(self):
        """When config.enable=False the matcher is a no-op."""
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, VectorFallbackConfig(enable=False))
        matcher.add_entry("继续", "continue")

        assert matcher.match("继续") is None

    def test_threshold_boundary_inclusive_match(self):
        """A similarity exactly at the threshold should still match.

        We craft a controlled scenario: two entries with the same intent
        and identical vectors. The query vector equals the stored vector,
        so cosine similarity is exactly 1.0. Setting the threshold to 1.0
        must still produce a match (>= comparison, not >).
        """
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, _make_config(similarity_threshold=1.0))
        matcher.add_entry("继续", "continue")

        result = matcher.match("继续")
        assert result is not None
        assert result.intent == "continue"
        assert result.confidence == pytest.approx(1.0)

    def test_empty_text_returns_none(self):
        """Empty input should not even hit the store."""
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, _make_config())
        matcher.add_entry("继续", "continue")

        assert matcher.match("") is None

    def test_no_entries_in_store_returns_none(self):
        """An empty store should produce no match."""
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, _make_config())
        assert matcher.match("任意文本") is None


class TestVectorMatcherSeeding:
    """Tests for seed_entries() and add_entry()."""

    def test_seed_entries_bulk_import(self):
        """seed_entries() inserts many entries at once via bulk_add."""
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, _make_config())
        entries = [
            _make_entry("继续", "continue"),
            _make_entry("查询订单", "order_query"),
            _make_entry("退款", "refund"),
        ]
        matcher.seed_entries(entries)

        assert len(store.list_all()) == 3
        # The first entry should be retrievable via a matching query.
        result = matcher.match("继续")
        assert result is not None
        assert result.intent == "continue"

    def test_add_entry_single_add(self):
        """add_entry() vectorizes the text and stores a new entry."""
        store = MemoryVectorMatchStore()
        matcher = VectorMatcher(store, _make_config())
        matcher.add_entry("帮我推荐手机", "product_recommendation", metadata={"k": "v"})

        all_entries = store.list_all()
        assert len(all_entries) == 1
        entry = all_entries[0]
        assert entry.text == "帮我推荐手机"
        assert entry.intent == "product_recommendation"
        assert entry.metadata == {"k": "v"}
        # The stored vector must be a unit vector (L2 norm == 1).
        norm = math.sqrt(sum(v * v for v in entry.vector))
        assert abs(norm - 1.0) < 1e-9


class TestDefaultVectorizer:
    """Tests for the dependency-free hashing vectorizer."""

    def test_default_vectorizer_produces_unit_vector(self):
        """The default vectorizer must return an L2-normalized vector."""
        vec = _default_vectorizer("hello world 你好")
        assert len(vec) == 128
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-9

    def test_default_vectorizer_empty_text(self):
        """Empty text should yield a zero vector (not crash)."""
        vec = _default_vectorizer("")
        assert len(vec) == 128
        assert all(v == 0.0 for v in vec)

    def test_default_vectorizer_deterministic(self):
        """Same input must always produce the same vector."""
        v1 = _default_vectorizer("确定性测试")
        v2 = _default_vectorizer("确定性测试")
        assert v1 == v2


class TestMemoryVectorMatchStoreCosine:
    """Direct tests on MemoryVectorMatchStore cosine similarity."""

    def test_identical_vectors_yield_perfect_similarity(self):
        """The store ranks identical vectors at similarity 1.0.

        We verify this indirectly: query with the same vector as the
        stored entry and confirm the entry is returned (and the matcher
        reports confidence 1.0).
        """
        store = MemoryVectorMatchStore()
        vec = _default_vectorizer("测试文本")
        store.add(VectorMatchEntry(text="测试文本", intent="t", vector=vec))

        results = store.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0].intent == "t"

        # And the matcher confirms cosine == 1.0 via the confidence field.
        matcher = VectorMatcher(store, _make_config(similarity_threshold=1.0))
        match = matcher.match("测试文本")
        assert match is not None
        assert match.confidence == pytest.approx(1.0)
