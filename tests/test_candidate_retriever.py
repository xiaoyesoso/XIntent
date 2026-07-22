"""Tests for D17: retrieval-based candidate narrowing.

Covers VectorCandidateRetriever, LLMCoarseRetriever, HybridCandidateRetriever,
the dynamic-N strategy, the factory function, and the disabled-default
behavior. Uses MockLLMClient to avoid real LLM API calls.
"""

import json

from intent_recognition import IntentDefinition
from intent_recognition.config import RetrievalConfig
from intent_recognition.lightweight_llm import (
    CandidateRetriever,
    HybridCandidateRetriever,
    LLMCoarseRetriever,
    VectorCandidateRetriever,
    create_retriever,
)
from user_input_normalization.llm.mock import MockLLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_intents() -> list[IntentDefinition]:
    """Build a small candidate pool (5 intents) for retriever tests."""
    return [
        IntentDefinition(
            name="product_recommendation",
            description="用户希望推荐商品，如手机、耳机等",
            positive_examples=["推荐手机", "有什么耳机推荐", "求推荐笔记本"],
        ),
        IntentDefinition(
            name="order_query",
            description="用户希望查询订单状态或物流",
            positive_examples=["我的订单到哪了", "查询物流"],
        ),
        IntentDefinition(
            name="refund",
            description="用户希望申请退款或退货退款",
            positive_examples=["我要退款", "退货退款"],
        ),
        IntentDefinition(
            name="weather_query",
            description="用户希望查询天气情况",
            positive_examples=["今天天气怎么样", "明天会下雨吗"],
        ),
        IntentDefinition(
            name="greeting",
            description="用户打招呼或闲聊",
            positive_examples=["你好", "嗨"],
        ),
    ]


# ---------------------------------------------------------------------------
# VectorCandidateRetriever
# ---------------------------------------------------------------------------


class TestVectorCandidateRetriever:
    """D17: pure-text Jaccard retrieval."""

    def test_returns_top_n_intents(self):
        """When top_n < len(all_intents), only Top-N are returned."""
        cfg = RetrievalConfig(enable=True, method="vector", top_n=2, dynamic_n=False)
        retriever = VectorCandidateRetriever(cfg)
        intents = _build_intents()
        result = retriever.retrieve("推荐手机", intents)
        assert len(result) == 2
        # The top match should be product_recommendation (shares tokens 推荐/手/机)
        assert result[0].name == "product_recommendation"

    def test_returns_all_when_top_n_exceeds_pool(self):
        """When top_n >= len(all_intents), no narrowing is performed."""
        cfg = RetrievalConfig(enable=True, method="vector", top_n=100, dynamic_n=False)
        retriever = VectorCandidateRetriever(cfg)
        intents = _build_intents()
        result = retriever.retrieve("推荐手机", intents)
        assert len(result) == len(intents)

    def test_empty_intent_list_returns_empty(self):
        cfg = RetrievalConfig(enable=True, method="vector", top_n=3)
        retriever = VectorCandidateRetriever(cfg)
        assert retriever.retrieve("anything", []) == []

    def test_default_config_works(self):
        """When config is None, defaults are used (top_n=10, dynamic_n=True)."""
        retriever = VectorCandidateRetriever()
        intents = _build_intents()
        # 5 intents < 10 default top_n -> all returned
        result = retriever.retrieve("推荐手机", intents)
        assert len(result) == len(intents)

    def test_ranking_uses_description_and_examples(self):
        """Querying for weather tokens should rank weather_query first."""
        cfg = RetrievalConfig(enable=True, method="vector", top_n=1, dynamic_n=False)
        retriever = VectorCandidateRetriever(cfg)
        intents = _build_intents()
        result = retriever.retrieve("今天天气怎么样", intents)
        assert len(result) == 1
        assert result[0].name == "weather_query"


# ---------------------------------------------------------------------------
# LLMCoarseRetriever
# ---------------------------------------------------------------------------


class TestLLMCoarseRetriever:
    """D17: LLM coarse retrieval."""

    def test_returns_top_n_via_llm_response(self):
        """LLM returns a JSON list of intent names; retriever maps them back."""
        mock = MockLLMClient()
        # LLM returns 2 intent names - both should map back to IntentDefinition
        mock.set_default_handler(
            lambda sys_p, usr_p: json.dumps(
                ["product_recommendation", "refund"], ensure_ascii=False
            )
        )
        cfg = RetrievalConfig(enable=True, method="llm_coarse", top_n=2, dynamic_n=False)
        retriever = LLMCoarseRetriever(mock, cfg)
        intents = _build_intents()
        result = retriever.retrieve("推荐手机", intents)
        assert len(result) == 2
        assert result[0].name == "product_recommendation"
        assert result[1].name == "refund"

    def test_parses_json_inside_fenced_block(self):
        """LLM may wrap JSON in ```json code blocks (pitfall guard)."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: "```json\n[\"order_query\"]\n```"
        )
        cfg = RetrievalConfig(enable=True, method="llm_coarse", top_n=1, dynamic_n=False)
        retriever = LLMCoarseRetriever(mock, cfg)
        intents = _build_intents()
        result = retriever.retrieve("查询订单", intents)
        assert len(result) == 1
        assert result[0].name == "order_query"

    def test_falls_back_when_llm_returns_garbage(self):
        """When LLM returns non-JSON, retriever falls back to first N candidates."""
        mock = MockLLMClient()
        mock.set_default_handler(lambda sys_p, usr_p: "this is not json")
        cfg = RetrievalConfig(enable=True, method="llm_coarse", top_n=2, dynamic_n=False)
        retriever = LLMCoarseRetriever(mock, cfg)
        intents = _build_intents()
        result = retriever.retrieve("anything", intents)
        # Should fall back to first 2 candidates from the registry order
        assert len(result) == 2
        assert result[0].name == intents[0].name

    def test_unknown_intent_names_are_filtered(self):
        """LLM names that aren't in the pool are silently dropped."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: json.dumps(
                ["product_recommendation", "nonexistent_intent"], ensure_ascii=False
            )
        )
        cfg = RetrievalConfig(enable=True, method="llm_coarse", top_n=2, dynamic_n=False)
        retriever = LLMCoarseRetriever(mock, cfg)
        intents = _build_intents()
        result = retriever.retrieve("推荐手机", intents)
        # Only the valid intent name survives
        assert len(result) == 1
        assert result[0].name == "product_recommendation"

    def test_truncates_to_top_n_when_llm_returns_more(self):
        """When LLM returns more than top_n names, result is truncated."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: json.dumps(
                ["product_recommendation", "order_query", "refund", "weather_query"],
                ensure_ascii=False,
            )
        )
        cfg = RetrievalConfig(enable=True, method="llm_coarse", top_n=2, dynamic_n=False)
        retriever = LLMCoarseRetriever(mock, cfg)
        intents = _build_intents()
        result = retriever.retrieve("推荐手机", intents)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# HybridCandidateRetriever
# ---------------------------------------------------------------------------


class TestHybridCandidateRetriever:
    """D17: hybrid = vector recall Top-2N + LLM precision Top-N."""

    def test_combines_vector_recall_and_llm_precision(self):
        """Vector narrows to 2N candidates, LLM picks N from those."""
        mock = MockLLMClient()
        # LLM will pick from the vector pool
        mock.set_default_handler(
            lambda sys_p, usr_p: json.dumps(
                ["product_recommendation", "refund"], ensure_ascii=False
            )
        )
        cfg = RetrievalConfig(enable=True, method="hybrid", top_n=2, dynamic_n=False)
        vector_r = VectorCandidateRetriever(cfg)
        llm_r = LLMCoarseRetriever(mock, cfg)
        hybrid = HybridCandidateRetriever(vector_r, llm_r)
        intents = _build_intents()
        result = hybrid.retrieve("推荐手机", intents)
        # Final output is Top-N=2 (after LLM precision)
        assert len(result) == 2
        names = [r.name for r in result]
        assert "product_recommendation" in names
        assert "refund" in names

    def test_hybrid_with_no_intents_returns_empty(self):
        cfg = RetrievalConfig(enable=True, method="hybrid", top_n=2, dynamic_n=False)
        vector_r = VectorCandidateRetriever(cfg)
        mock = MockLLMClient()
        llm_r = LLMCoarseRetriever(mock, cfg)
        hybrid = HybridCandidateRetriever(vector_r, llm_r)
        assert hybrid.retrieve("anything", []) == []


# ---------------------------------------------------------------------------
# Dynamic N strategy
# ---------------------------------------------------------------------------


class TestDynamicNStrategy:
    """D17: dynamic widening of N based on context or retry count."""

    def test_default_returns_base_n(self):
        """No context and no retry -> base N unchanged."""
        n = CandidateRetriever._compute_dynamic_n(
            base_n=10, context=None, retry_count=0
        )
        assert n == 10

    def test_context_free_pct_above_half_widens_to_1_5x(self):
        """When context_window_free_pct > 0.5, N becomes int(base_n * 1.5)."""
        ctx = {"context_window_free_pct": 0.7}
        n = CandidateRetriever._compute_dynamic_n(
            base_n=10, context=ctx, retry_count=0
        )
        assert n == int(10 * 1.5)  # 15

    def test_context_free_pct_exactly_half_does_not_widen(self):
        """Boundary: free_pct == 0.5 should NOT widen (condition is strict > 0.5)."""
        ctx = {"context_window_free_pct": 0.5}
        n = CandidateRetriever._compute_dynamic_n(
            base_n=10, context=ctx, retry_count=0
        )
        assert n == 10

    def test_retry_count_positive_doubles_with_cap(self):
        """retry_count > 0 -> min(base_n * 2, 100)."""
        n = CandidateRetriever._compute_dynamic_n(
            base_n=10, context=None, retry_count=1
        )
        assert n == 20

    def test_retry_count_capped_at_100(self):
        """min(base_n * 2, 100): base=80 -> 160 capped to 100."""
        n = CandidateRetriever._compute_dynamic_n(
            base_n=80, context=None, retry_count=1
        )
        assert n == 100

    def test_context_free_pct_takes_precedence_over_retry(self):
        """Free context window is checked first; should win over retry widening."""
        ctx = {"context_window_free_pct": 0.9}
        n = CandidateRetriever._compute_dynamic_n(
            base_n=10, context=ctx, retry_count=3
        )
        # 1.5x branch wins (15) over 2x branch (20)
        assert n == 15

    def test_invalid_context_free_pct_ignored(self):
        """Non-numeric context_window_free_pct should not trigger widening."""
        ctx = {"context_window_free_pct": "high"}
        n = CandidateRetriever._compute_dynamic_n(
            base_n=10, context=ctx, retry_count=0
        )
        assert n == 10

    def test_vector_retriever_uses_dynamic_n_when_enabled(self):
        """End-to-end: dynamic_n=True should widen the returned set when
        context_window_free_pct > 0.5."""
        cfg = RetrievalConfig(enable=True, method="vector", top_n=2, dynamic_n=True)
        retriever = VectorCandidateRetriever(cfg)
        intents = _build_intents()
        ctx = {"context_window_free_pct": 0.9}
        # base=2, widened to int(2*1.5)=3
        result = retriever.retrieve("推荐手机", intents, context=ctx)
        assert len(result) == 3

    def test_retry_count_read_from_context(self):
        """When context contains retry_count > 0, N is doubled (capped at 100)."""
        cfg = RetrievalConfig(enable=True, method="vector", top_n=2, dynamic_n=True)
        retriever = VectorCandidateRetriever(cfg)
        intents = _build_intents()
        ctx = {"retry_count": 1}
        # base=2, retry widened to min(4, 100)=4
        result = retriever.retrieve("推荐手机", intents, context=ctx)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateRetriever:
    """D17: factory dispatch on config.method."""

    def test_returns_none_when_disabled(self):
        """enable=False -> None (no narrowing, default behavior)."""
        cfg = RetrievalConfig(enable=False, method="vector")
        assert create_retriever(cfg) is None

    def test_vector_method_returns_vector_retriever(self):
        cfg = RetrievalConfig(enable=True, method="vector", top_n=2)
        r = create_retriever(cfg)
        assert isinstance(r, VectorCandidateRetriever)

    def test_llm_coarse_method_returns_llm_retriever(self):
        cfg = RetrievalConfig(enable=True, method="llm_coarse", top_n=2)
        mock = MockLLMClient()
        r = create_retriever(cfg, llm_client=mock)
        assert isinstance(r, LLMCoarseRetriever)

    def test_llm_coarse_method_without_llm_client_returns_none(self):
        """LLM-required method without an llm_client is treated as disabled."""
        cfg = RetrievalConfig(enable=True, method="llm_coarse", top_n=2)
        assert create_retriever(cfg, llm_client=None) is None

    def test_hybrid_method_returns_hybrid_retriever(self):
        cfg = RetrievalConfig(enable=True, method="hybrid", top_n=2)
        mock = MockLLMClient()
        r = create_retriever(cfg, llm_client=mock)
        assert isinstance(r, HybridCandidateRetriever)

    def test_hybrid_method_without_llm_client_returns_none(self):
        cfg = RetrievalConfig(enable=True, method="hybrid", top_n=2)
        assert create_retriever(cfg, llm_client=None) is None

    def test_unknown_method_returns_none(self):
        cfg = RetrievalConfig(enable=True, method="bogus", top_n=2)
        assert create_retriever(cfg) is None


# ---------------------------------------------------------------------------
# Independent behavior (L1-hit scenario simulation)
# ---------------------------------------------------------------------------


class TestIndependentBehavior:
    """D17: retriever is independent of L1 - just a function call.

    When L1 (code layer) hits, the pipeline should NOT call the retriever at
    all.  We simulate this by simply never invoking ``retrieve`` - and verify
    that the retriever, when it IS called, behaves correctly on its own.
    """

    def test_retriever_does_not_modify_input_list(self):
        """Retriever must not mutate the caller's list (no side effects)."""
        cfg = RetrievalConfig(enable=True, method="vector", top_n=2, dynamic_n=False)
        retriever = VectorCandidateRetriever(cfg)
        intents = _build_intents()
        original_len = len(intents)
        _ = retriever.retrieve("推荐手机", intents)
        # Input list should be unchanged
        assert len(intents) == original_len

    def test_disabled_default_does_not_affect_existing_behavior(self):
        """Default IntentRecognitionConfig has retrieval disabled, so the
        factory returns None - simulating the unchanged pre-D17 behavior."""
        from intent_recognition import IntentRecognitionConfig

        cfg = IntentRecognitionConfig()
        assert cfg.retrieval.enable is False
        assert create_retriever(cfg.retrieval) is None


# ---------------------------------------------------------------------------
# Backward compat / sanity
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """D17 must not break existing import paths."""

    def test_lightweight_llm_package_exports_symbols(self):
        """All new symbols are re-exported from the package."""
        from intent_recognition.lightweight_llm import (
            CandidateRetriever as CR,
            HybridCandidateRetriever as HCR,
            LLMCoarseRetriever as LCR,
            VectorCandidateRetriever as VCR,
            create_retriever as cr,
        )
        assert CR is CandidateRetriever
        assert HCR is HybridCandidateRetriever
        assert LCR is LLMCoarseRetriever
        assert VCR is VectorCandidateRetriever
        assert cr is create_retriever
