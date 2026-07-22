"""Attribute + retrieval-based anaphora resolution regression tests (corresponds to task 9.8).

Covers the "Jiming Temple" case (D9):
- Before the holiday, the user was recommended Jiming Temple for cherry blossom viewing
- After the holiday, the user inputs "你上次推荐的看樱花的那个地方……"
- The system completes resolution via two-step inference + compensation mechanism

Test points:
- Attribute extraction (task 9.1)
- Vector retrieval recall (task 9.2)
- Two-step inference (task 9.3)
- Retrieval failure compensation mechanism (task 9.4)
- Confidence assessment (task 9.5)
- Resolution result writeback (task 9.6)
- Recall quality monitoring (task 9.7)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from user_input_normalization.attribute_resolution import AttributeResolver
from user_input_normalization.llm.mock import MockLLMClient
from user_input_normalization.models import (
    AttributeResolutionResult,
    DialogueTurn,
    FactType,
)
from user_input_normalization.storage.memory import (
    MemoryDialogueHistoryStore,
    MemoryKeyFactStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_client() -> MockLLMClient:
    """Provide a MockLLMClient with attribute extraction / inference handlers registered."""
    client = MockLLMClient()

    def handler(system_prompt: str, user_prompt: str) -> str:
        # Attribute extraction
        if "属性关键词提取" in system_prompt:
            return json.dumps(
                {"attributes": ["樱花", "推荐", "看樱花"]},
                ensure_ascii=False,
            )
        # Anaphora resolution inference
        if "指代消解助手" in system_prompt:
            # Check whether recalled details contain Jiming Temple
            if any("鸡鸣寺" in d for d in [user_prompt]):
                return json.dumps(
                    {
                        "resolved_to": "鸡鸣寺",
                        "confidence": 0.92,
                        "reasoning": "召回细节中明确提到推荐鸡鸣寺看樱花",
                    },
                    ensure_ascii=False,
                )
            # Recalled details are insufficient for inference
            return json.dumps(
                {
                    "resolved_to": "",
                    "confidence": 0.3,
                    "reasoning": "召回内容不足以确定指代对象",
                },
                ensure_ascii=False,
            )
        return json.dumps({"note": "mock"}, ensure_ascii=False)

    client.set_default_handler(handler)
    return client


@pytest.fixture
def dialogue_store() -> MemoryDialogueHistoryStore:
    """A store pre-filled with the Jiming Temple cherry blossom dialogue history."""
    store = MemoryDialogueHistoryStore()
    # Simulate the pre-holiday conversation: assistant recommends Jiming Temple for cherry blossoms
    store.append(
        "s1",
        DialogueTurn(
            turn=1,
            role="user",
            content="假期想去一个看樱花的地方，有推荐吗？",
        ),
    )
    store.append(
        "s1",
        DialogueTurn(
            turn=2,
            role="assistant",
            content="我推荐鸡鸣寺，那里看樱花非常出名",
            summary="推荐鸡鸣寺看樱花",
        ),
    )
    store.append(
        "s1",
        DialogueTurn(
            turn=3,
            role="user",
            content="鸡鸣寺的樱花什么时候开？",
        ),
    )
    store.append(
        "s1",
        DialogueTurn(
            turn=4,
            role="assistant",
            content="鸡鸣寺的樱花通常在三月开放",
        ),
    )
    store.set_summary("s1", "用户询问看樱花的地方，推荐了鸡鸣寺")
    return store


@pytest.fixture
def key_fact_store() -> MemoryKeyFactStore:
    return MemoryKeyFactStore()


@pytest.fixture
def resolver(
    llm_client: MockLLMClient,
    dialogue_store: MemoryDialogueHistoryStore,
    key_fact_store: MemoryKeyFactStore,
) -> AttributeResolver:
    return AttributeResolver(
        llm_client=llm_client,
        dialogue_store=dialogue_store,
        key_fact_store=key_fact_store,
    )


@pytest.fixture
def empty_resolver(
    llm_client: MockLLMClient,
    key_fact_store: MemoryKeyFactStore,
) -> AttributeResolver:
    """A resolver using empty dialogue history (for testing the compensation mechanism)."""
    return AttributeResolver(
        llm_client=llm_client,
        dialogue_store=MemoryDialogueHistoryStore(),
        key_fact_store=key_fact_store,
    )


# ---------------------------------------------------------------------------
# task 9.1: Attribute extraction
# ---------------------------------------------------------------------------


class TestExtractAttributes:
    """Attribute extraction (task 9.1)."""

    def test_extract_from_cherry_blossom_input(
        self, resolver: AttributeResolver
    ) -> None:
        """Extract attribute keywords from '看樱花的那个地方'."""
        attrs = resolver.extract_attributes("你上次推荐的看樱花的那个地方")
        # Should contain "樱花" (rule-based extraction)
        assert "樱花" in attrs
        # Should contain "看樱花" (verb-object combination)
        assert "看樱花" in attrs
        # Should contain "推荐" or "上次推荐"
        assert any("推荐" in a for a in attrs)

    def test_extract_from_recommendation_input(
        self, resolver: AttributeResolver
    ) -> None:
        """Extract attribute keywords from '上次推荐的那个'."""
        attrs = resolver.extract_attributes("上次推荐的那个")
        assert any("上次" in a for a in attrs)
        assert "推荐" in attrs or "上次推荐" in attrs

    def test_extract_from_multi_attribute_input(
        self, resolver: AttributeResolver
    ) -> None:
        """Multi-attribute joint anaphora: '穿红色衣服、戴眼镜的同事'."""
        attrs = resolver.extract_attributes("上次那个穿红色衣服、戴眼镜的同事")
        # Should extract "红色" and "红色衣服" (wearing red clothes)
        assert any("红色" in a for a in attrs)
        # Should extract "眼镜"-related
        assert any("眼镜" in a for a in attrs)

    def test_extract_empty_input(self, resolver: AttributeResolver) -> None:
        """Empty input should return an empty list."""
        assert resolver.extract_attributes("") == []

    def test_extract_deduplicates(
        self, resolver: AttributeResolver
    ) -> None:
        """Extraction results should be deduplicated."""
        attrs = resolver.extract_attributes("看樱花的那个看樱花的地方")
        # Should have no duplicates
        assert len(attrs) == len(set(attrs))


# ---------------------------------------------------------------------------
# task 9.2: Vector retrieval recall
# ---------------------------------------------------------------------------


class TestRecallDetails:
    """Vector retrieval recall (task 9.2)."""

    def test_recall_returns_relevant_details(
        self, resolver: AttributeResolver
    ) -> None:
        """Retrieving with attribute keywords should recall relevant dialogue."""
        details = resolver.recall_details("s1", ["樱花", "推荐", "看樱花"], top_k=5)
        # Should recall dialogue containing "鸡鸣寺"
        assert len(details) > 0
        assert any("鸡鸣寺" in d for d in details)

    def test_recall_with_empty_attributes(
        self, resolver: AttributeResolver
    ) -> None:
        """Empty attribute list should return empty results."""
        assert resolver.recall_details("s1", []) == []

    def test_recall_with_no_matching(
        self, resolver: AttributeResolver
    ) -> None:
        """When no attributes match, should return an empty list."""
        details = resolver.recall_details("s1", ["量子力学", "区块链"], top_k=5)
        assert details == []

    def test_recall_top_k_limit(
        self, resolver: AttributeResolver
    ) -> None:
        """The number of recalls should be limited by top_k."""
        details = resolver.recall_details("s1", ["樱花"], top_k=2)
        assert len(details) <= 2

    def test_recall_deduplicates(
        self, resolver: AttributeResolver
    ) -> None:
        """Recall results should be deduplicated."""
        details = resolver.recall_details("s1", ["樱花", "鸡鸣寺"], top_k=10)
        assert len(details) == len(set(details))


# ---------------------------------------------------------------------------
# task 9.3: Two-step inference
# ---------------------------------------------------------------------------


class TestInfer:
    """Two-step inference (task 9.3)."""

    def test_infer_with_relevant_details(
        self, resolver: AttributeResolver
    ) -> None:
        """Should infer Jiming Temple based on relevant recalled details."""
        recalled = [
            "我推荐鸡鸣寺，那里看樱花非常出名",
            "鸡鸣寺的樱花通常在三月开放",
        ]
        resolved, confidence = resolver.infer(
            "s1", "你上次推荐的看樱花的那个地方", recalled
        )
        assert resolved == "鸡鸣寺"
        assert confidence >= 0.7

    def test_infer_with_empty_details(
        self, resolver: AttributeResolver
    ) -> None:
        """Should return low confidence when there are no recalled details."""
        resolved, confidence = resolver.infer("s1", "那个地方", [])
        assert resolved == ""
        assert confidence == 0.0

    def test_infer_confidence_in_range(
        self, resolver: AttributeResolver
    ) -> None:
        """Confidence should be within [0, 1]."""
        recalled = ["我推荐鸡鸣寺，那里看樱花非常出名"]
        _, confidence = resolver.infer(
            "s1", "看樱花的那个地方", recalled
        )
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# task 9.4: Retrieval failure compensation mechanism
# ---------------------------------------------------------------------------


class TestCompensation:
    """Retrieval failure compensation mechanism (task 9.4)."""

    def test_compensation_returns_tuple(
        self, resolver: AttributeResolver
    ) -> None:
        """The compensation mechanism should return a 4-tuple."""
        result = resolver.compensation(
            "s1", "看樱花的那个地方", ["樱花", "推荐"]
        )
        assert len(result) == 4
        resolved, confidence, used, tool = result
        assert isinstance(resolved, str)
        assert isinstance(confidence, float)
        assert isinstance(used, bool)
        assert isinstance(tool, str)

    def test_compensation_marks_used(
        self, resolver: AttributeResolver
    ) -> None:
        """The compensation mechanism should mark compensation_used=True."""
        _, _, used, _ = resolver.compensation(
            "s1", "看樱花的那个地方", ["樱花"]
        )
        assert used is True

    def test_compensation_returns_tool_name(
        self, resolver: AttributeResolver
    ) -> None:
        """The compensation mechanism should return a tool name."""
        _, _, _, tool = resolver.compensation(
            "s1", "看樱花的那个地方", ["樱花"]
        )
        assert tool == "search_history_by_attributes"

    def test_compensation_with_empty_attributes(
        self, resolver: AttributeResolver
    ) -> None:
        """With an empty attribute list, the compensation mechanism should return low confidence."""
        resolved, confidence, used, _ = resolver.compensation(
            "s1", "那个地方", []
        )
        assert used is True
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# task 9.5: Confidence assessment
# ---------------------------------------------------------------------------


class TestAssessConfidence:
    """Confidence assessment (task 9.5)."""

    def test_high_confidence_with_strong_match(
        self, resolver: AttributeResolver
    ) -> None:
        """Strong match should return high confidence."""
        details = [
            "我推荐鸡鸣寺，那里看樱花非常出名",
            "鸡鸣寺的樱花通常在三月开放",
        ]
        confidence = resolver.assess_confidence("鸡鸣寺", details)
        assert confidence >= 0.7

    def test_low_confidence_with_no_match(
        self, resolver: AttributeResolver
    ) -> None:
        """No match should return low confidence."""
        details = ["今天天气不错"]
        confidence = resolver.assess_confidence("鸡鸣寺", details)
        assert confidence < 0.5

    def test_zero_confidence_with_empty(
        self, resolver: AttributeResolver
    ) -> None:
        """Empty results or empty details should return 0."""
        assert resolver.assess_confidence("", ["detail"]) == 0.0
        assert resolver.assess_confidence("鸡鸣寺", []) == 0.0

    def test_confidence_in_range(
        self, resolver: AttributeResolver
    ) -> None:
        """Confidence should be within [0, 1]."""
        details = ["鸡鸣寺看樱花"]
        confidence = resolver.assess_confidence("鸡鸣寺", details)
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# task 9.6: Resolution result writeback
# ---------------------------------------------------------------------------


class TestWriteback:
    """Resolution result writeback (task 9.6)."""

    def test_writeback_saves_to_key_fact_store(
        self,
        resolver: AttributeResolver,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Resolution results should be written back as a PRONOUN_RESOLUTION type KeyFact."""
        result = AttributeResolutionResult(
            pronoun="看樱花的那个地方",
            extracted_attributes=["樱花", "推荐", "看樱花"],
            recalled_details=["我推荐鸡鸣寺看樱花"],
            resolved_to="鸡鸣寺",
            confidence=0.92,
        )
        resolver.writeback("s1", 5, result)
        facts = key_fact_store.get_by_session("s1", fact_type="pronoun_resolution")
        assert len(facts) >= 1
        fact = facts[0]
        assert fact.content["pronoun"] == "看樱花的那个地方"
        assert fact.content["resolved_to"] == "鸡鸣寺"
        assert "樱花" in fact.content["extracted_attributes"]

    def test_writeback_skips_empty_result(
        self,
        resolver: AttributeResolver,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Unresolved results should not be written back."""
        result = AttributeResolutionResult(
            pronoun="那个地方",
            extracted_attributes=["樱花"],
            resolved_to="",
            confidence=0.0,
        )
        resolver.writeback("s1", 5, result)
        facts = key_fact_store.get_by_session("s1", fact_type="pronoun_resolution")
        assert len(facts) == 0

    def test_writeback_records_confidence(
        self,
        resolver: AttributeResolver,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Writeback should record confidence."""
        result = AttributeResolutionResult(
            pronoun="看樱花的那个地方",
            extracted_attributes=["樱花"],
            resolved_to="鸡鸣寺",
            confidence=0.88,
        )
        resolver.writeback("s1", 5, result)
        facts = key_fact_store.get_by_session("s1", fact_type="pronoun_resolution")
        assert len(facts) >= 1
        assert facts[0].confidence == 0.88
        assert facts[0].content["confidence"] == 0.88


# ---------------------------------------------------------------------------
# task 9.7: Recall quality monitoring
# ---------------------------------------------------------------------------


class TestRecallStats:
    """Recall quality monitoring (task 9.7)."""

    def test_stats_initial_state(
        self, resolver: AttributeResolver
    ) -> None:
        """Initial state stats should be zero."""
        stats = resolver.get_recall_stats()
        assert stats["total_resolutions"] == 0
        assert stats["successful_recalls"] == 0
        assert stats["failed_recalls"] == 0

    def test_stats_after_successful_resolve(
        self, resolver: AttributeResolver
    ) -> None:
        """Stats should update after a successful resolution."""
        resolver.resolve("s1", "你上次推荐的看樱花的那个地方")
        stats = resolver.get_recall_stats()
        assert stats["total_resolutions"] >= 1
        assert stats["successful_recalls"] >= 1

    def test_stats_includes_rates(
        self, resolver: AttributeResolver
    ) -> None:
        """Stats should include recall success rate and compensation trigger rate."""
        resolver.resolve("s1", "你上次推荐的看樱花的那个地方")
        stats = resolver.get_recall_stats()
        assert "recall_success_rate" in stats
        assert "compensation_trigger_rate" in stats
        assert 0.0 <= stats["recall_success_rate"] <= 1.0

    def test_stats_avg_confidence(
        self, resolver: AttributeResolver
    ) -> None:
        """Stats should include average confidence."""
        resolver.resolve("s1", "你上次推荐的看樱花的那个地方")
        stats = resolver.get_recall_stats()
        assert "avg_confidence" in stats
        assert stats["avg_confidence"] >= 0.0


# ---------------------------------------------------------------------------
# End-to-end: Jiming Temple case (D9)
# ---------------------------------------------------------------------------


class TestJimingTempleCase:
    """Jiming Temple case end-to-end tests (D9)."""

    def test_full_resolution_flow(
        self, resolver: AttributeResolver
    ) -> None:
        """Complete two-step inference flow: user input -> resolves to Jiming Temple."""
        result = resolver.resolve(
            "s1", "你上次推荐的看樱花的那个地方……"
        )
        # Should be an AttributeResolutionResult instance
        assert isinstance(result, AttributeResolutionResult)
        # Extracted attributes should contain "樱花"
        assert any("樱花" in a for a in result.extracted_attributes)
        # Recalled details should mention Jiming Temple
        assert any("鸡鸣寺" in d for d in result.recalled_details)
        # Resolution result should be "鸡鸣寺"
        assert result.resolved_to == "鸡鸣寺"
        # Confidence should be relatively high
        assert result.confidence >= 0.7

    def test_resolution_written_back(
        self,
        resolver: AttributeResolver,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Resolution results should be written back to the key fact store."""
        resolver.resolve("s1", "你上次推荐的看樱花的那个地方……")
        facts = key_fact_store.get_by_session("s1", fact_type="pronoun_resolution")
        assert len(facts) >= 1
        assert facts[0].content["resolved_to"] == "鸡鸣寺"

    def test_pronoun_extracted(
        self, resolver: AttributeResolver
    ) -> None:
        """Should extract the attribute anaphora phrase."""
        result = resolver.resolve("s1", "你上次推荐的看樱花的那个地方……")
        # The pronoun field should contain "樱花" or "那个地方"
        assert "樱花" in result.pronoun or "那个" in result.pronoun

    def test_reuse_after_writeback(
        self,
        resolver: AttributeResolver,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """After writeback, the resolution result should be findable in the key fact store."""
        resolver.resolve("s1", "你上次推荐的看樱花的那个地方……")
        # Should be findable via find_pronoun_resolution (if pronoun matches exactly)
        facts = key_fact_store.get_by_session("s1", fact_type="pronoun_resolution")
        assert len(facts) >= 1
        # Verify content structure
        content = facts[0].content
        assert content["resolved_to"] == "鸡鸣寺"
        assert content["source"] == "attribute_resolution"

    def test_empty_history_triggers_compensation(
        self,
        empty_resolver: AttributeResolver,
    ) -> None:
        """Empty dialogue history should trigger the compensation mechanism."""
        result = empty_resolver.resolve(
            "s_empty", "你上次推荐的看樱花的那个地方……"
        )
        # Compensation mechanism should be triggered (because dialogue history is empty)
        assert result.compensation_used is True
        assert result.tool_called == "search_history_by_attributes"
        # With empty history, resolution result should be empty or low confidence
        assert result.confidence < 0.7 or result.resolved_to == ""

    def test_low_confidence_when_no_match(
        self,
        empty_resolver: AttributeResolver,
    ) -> None:
        """Should return low confidence when there is no matching history."""
        result = empty_resolver.resolve(
            "s_empty", "那个看量子力学的地方"
        )
        # Confidence should be low
        assert result.confidence < 0.7
