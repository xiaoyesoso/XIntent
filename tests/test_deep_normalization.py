"""Deep normalization regression tests (corresponding to task 5.7).

Coverage:
- Adjective quantification executed in the deep stage (task 5.2)
- External fact resolution (task 5.3)
- Observation-dependent backtracking resolution (task 5.4)
- Context window management (task 5.5)
- Result writeback to key fact storage (task 5.6)
- ReAct loop single-step integration (task 5.1)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from user_input_normalization.config import Config
from user_input_normalization.deep_normalization import DeepNormalizer
from user_input_normalization.llm.mock import MockLLMClient
from user_input_normalization.models import (
    CompletenessStatus,
    FactType,
    KeyFact,
    NormalizationResult,
    PronounResolution,
    QuantifiableAdjective,
    SubjectPredicateObject,
)
from user_input_normalization.quantification import QuantificationEngine
from user_input_normalization.storage.memory import (
    MemoryDialogueHistoryStore,
    MemoryKeyFactStore,
    MemoryVocabStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_client() -> MockLLMClient:
    """Provide a MockLLMClient with external fact resolution / backtracking inference handlers registered."""
    client = MockLLMClient()

    def external_handler(system_prompt: str, user_prompt: str) -> str:
        # Handle external fact resolution
        if "外部事实" in system_prompt:
            return json.dumps(
                {
                    "resolved": True,
                    "resolved_entity": "商品X",
                    "evidence": "价格查询工具返回，商品X 当前价格最低",
                    "needs_clarification": False,
                },
                ensure_ascii=False,
            )
        # Handle backtracking resolution
        if "回溯" in system_prompt:
            return json.dumps(
                {
                    "resolutions": [
                        {
                            "pronoun": "那个",
                            "resolved_to": "商品D",
                            "confidence": 0.88,
                            "evidence_source": "Observation 回溯（商品C已下架）",
                            "corrected": True,
                            "correction_reason": "商品C已下架，切换至候选商品D",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        # Default response
        return json.dumps({"note": "mock"}, ensure_ascii=False)

    client.set_default_handler(external_handler)
    return client


@pytest.fixture
def key_fact_store() -> MemoryKeyFactStore:
    return MemoryKeyFactStore()


@pytest.fixture
def dialogue_store() -> MemoryDialogueHistoryStore:
    return MemoryDialogueHistoryStore()


@pytest.fixture
def engine(llm_client: MockLLMClient) -> QuantificationEngine:
    return QuantificationEngine(llm_client=llm_client, vocab_store=MemoryVocabStore())


@pytest.fixture
def normalizer(
    llm_client: MockLLMClient,
    engine: QuantificationEngine,
    key_fact_store: MemoryKeyFactStore,
    dialogue_store: MemoryDialogueHistoryStore,
) -> DeepNormalizer:
    return DeepNormalizer(
        llm_client=llm_client,
        quantification_engine=engine,
        key_fact_store=key_fact_store,
        dialogue_store=dialogue_store,
    )


def _make_pre_result(
    adjective: str = "性价比",
    quantified: bool = False,
) -> NormalizationResult:
    """Construct a pre-normalization result containing an unquantified adjective."""
    return NormalizationResult(
        normalized_input="帮我推荐一个更有性价比的牛仔裤",
        spo=SubjectPredicateObject(subject="我", predicate="推荐", obj="牛仔裤"),
        quantifiable_adjectives=[
            QuantifiableAdjective(
                adjective=adjective,
                quantified=quantified,
                quantified_value=None if not quantified else {"price_range": [180, 220]},
            )
        ],
        raw_input="帮我推荐一个更有性价比的牛仔裤",
    )


# ---------------------------------------------------------------------------
# task 5.2: Adjective quantification executed in the deep stage
# ---------------------------------------------------------------------------


class TestQuantifyAdjectives:
    """Adjective quantification executed in the deep stage."""

    def test_quantify_unquantified_adjective(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Unquantified adjectives should be quantified."""
        adjectives = [
            QuantifiableAdjective(adjective="性价比", quantified=False),
        ]
        context = {"current_price": 200}
        result = normalizer.quantify_adjectives(adjectives, context)
        assert len(result) == 1
        assert result[0].adjective == "性价比"
        assert result[0].quantified is True
        assert result[0].quantified_value is not None
        assert "price_range" in result[0].quantified_value

    def test_quantify_preserves_already_quantified(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Already-quantified adjectives should remain unchanged."""
        original_value = {"price_range": [100, 150]}
        adjectives = [
            QuantifiableAdjective(
                adjective="性价比",
                quantified=True,
                quantified_value=original_value,
            ),
            QuantifiableAdjective(adjective="更好", quantified=False),
        ]
        context = {"current_price": 200}
        result = normalizer.quantify_adjectives(adjectives, context)
        # First one keeps its original value
        assert result[0].quantified is True
        assert result[0].quantified_value == original_value
        # Second one is quantified
        assert result[1].quantified is True
        assert result[1].quantified_value is not None

    def test_quantify_multiple_adjectives(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Multiple unquantified adjectives should all be quantified."""
        adjectives = [
            QuantifiableAdjective(adjective="性价比", quantified=False),
            QuantifiableAdjective(adjective="更好", quantified=False),
            QuantifiableAdjective(adjective="更便宜", quantified=False),
        ]
        context = {"current_price": 200, "current_tier": 1}
        result = normalizer.quantify_adjectives(adjectives, context)
        assert all(adj.quantified for adj in result)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# task 5.3: External fact resolution
# ---------------------------------------------------------------------------


class TestResolveExternalFact:
    """External fact resolution (combining Observation)."""

    def test_resolve_with_observation_dict(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Resolve external facts based on dict-form Observation."""
        observation = {
            "tool": "price_query",
            "items": [
                {"name": "商品X", "price": 99},
                {"name": "商品Y", "price": 150},
            ],
        }
        result = normalizer.resolve_external_fact(
            "s1", "现在最便宜的是哪个？", observation
        )
        assert result["resolved"] is True
        assert result["resolved_entity"] == "商品X"
        assert "Observation" in result["evidence"] or "价格" in result["evidence"]
        assert result["needs_clarification"] is False

    def test_resolve_with_observation_string(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Resolve based on string-form Observation."""
        result = normalizer.resolve_external_fact(
            "s1", "现在最便宜的", "价格查询返回：商品X 99元，商品Y 150元"
        )
        assert result["resolved"] is True
        assert result["resolved_entity"] is not None

    def test_resolve_without_observation_does_not_fabricate(
        self, normalizer: DeepNormalizer
    ) -> None:
        """When Observation is missing, fabrication is forbidden; should mark for clarification."""
        result = normalizer.resolve_external_fact(
            "s1", "现在最便宜的", None
        )
        assert result["resolved"] is False
        assert result["resolved_entity"] is None
        assert result["needs_clarification"] is True
        assert "禁止伪造" in result["evidence"]

    def test_resolve_with_empty_observation_does_not_fabricate(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Empty Observation should also forbid fabrication."""
        result = normalizer.resolve_external_fact("s1", "现在最便宜的", "")
        assert result["resolved"] is False
        assert result["needs_clarification"] is True


# ---------------------------------------------------------------------------
# task 5.4: Observation-dependent backtracking resolution
# ---------------------------------------------------------------------------


class TestReResolveWithObservation:
    """Observation-dependent backtracking resolution."""

    def test_re_resolve_corrects_invalid_pronoun(
        self, normalizer: DeepNormalizer
    ) -> None:
        """When Observation shows the original resolution target is invalid, backtrack-correct."""
        pre_result = _make_pre_result()
        pre_result.pronoun_resolutions = [
            PronounResolution(
                pronoun="那个",
                resolved_to="商品C",
                confidence=0.85,
                evidence_source="对话历史第2轮",
            )
        ]
        observation = {"tool_result": "商品C已下架"}
        updated = normalizer.re_resolve_with_observation("s1", pre_result, observation)
        assert len(updated) >= 1
        # Should have a correction: resolved_to is no longer 商品C
        assert any(pr.resolved_to != "商品C" for pr in updated)
        # Correction record should appear in evidence_source
        corrected = [pr for pr in updated if "回溯修正" in pr.evidence_source]
        assert len(corrected) >= 1

    def test_re_resolve_keeps_when_no_pronouns(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Return an empty list when there is no anaphora resolution table."""
        pre_result = _make_pre_result()
        pre_result.pronoun_resolutions = []
        updated = normalizer.re_resolve_with_observation("s1", pre_result, "some obs")
        assert updated == []

    def test_re_resolve_keeps_when_observation_empty(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Keep the original resolution result when Observation is empty."""
        pre_result = _make_pre_result()
        pre_result.pronoun_resolutions = [
            PronounResolution(
                pronoun="那个",
                resolved_to="商品C",
                confidence=0.85,
                evidence_source="对话历史第2轮",
            )
        ]
        updated = normalizer.re_resolve_with_observation("s1", pre_result, "")
        assert len(updated) == 1
        assert updated[0].resolved_to == "商品C"


# ---------------------------------------------------------------------------
# task 5.5: Context window management
# ---------------------------------------------------------------------------


class TestManageContextWindow:
    """Context window management."""

    def test_window_contains_observation(
        self, normalizer: DeepNormalizer
    ) -> None:
        """Window should contain Observation."""
        pre_result = _make_pre_result()
        observation = {"current_price": 200}
        window = normalizer.manage_context_window("s1", observation, pre_result)
        assert "Observation" in window
        assert "200" in window

    def test_window_contains_key_facts(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Window should contain key facts."""
        key_fact_store.save(
            KeyFact(
                fact_id="f1",
                session_id="s1",
                turn=1,
                fact_type=FactType.KEY_FACT,
                content={"topic": "brand", "value": "TCC方案"},
            )
        )
        pre_result = _make_pre_result()
        window = normalizer.manage_context_window("s1", None, pre_result)
        assert "关键事实" in window
        assert "TCC" in window

    def test_window_contains_recent_dialogue(
        self,
        normalizer: DeepNormalizer,
        dialogue_store: MemoryDialogueHistoryStore,
    ) -> None:
        """Window should contain recent dialogue."""
        from user_input_normalization.models import DialogueTurn

        dialogue_store.append(
            "s1",
            DialogueTurn(turn=1, role="user", content="我要买牛仔裤"),
        )
        pre_result = _make_pre_result()
        window = normalizer.manage_context_window("s1", None, pre_result)
        assert "近期对话" in window
        assert "牛仔裤" in window

    def test_window_contains_summary(
        self,
        normalizer: DeepNormalizer,
        dialogue_store: MemoryDialogueHistoryStore,
    ) -> None:
        """Window should contain the dialogue summary (can be trimmed)."""
        dialogue_store.set_summary("s1", "用户在选购牛仔裤")
        pre_result = _make_pre_result()
        window = normalizer.manage_context_window("s1", None, pre_result)
        assert "对话摘要" in window or "选购" in window

    def test_window_priority_key_facts_first(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Key facts should rank before Observation."""
        key_fact_store.save(
            KeyFact(
                fact_id="f1",
                session_id="s1",
                turn=1,
                fact_type=FactType.KEY_FACT,
                content={"k": "v"},
            )
        )
        pre_result = _make_pre_result()
        window = normalizer.manage_context_window(
            "s1", {"current_price": 200}, pre_result
        )
        facts_pos = window.find("关键事实")
        obs_pos = window.find("Observation")
        assert facts_pos < obs_pos


# ---------------------------------------------------------------------------
# task 5.6: Result writeback
# ---------------------------------------------------------------------------


class TestWriteback:
    """Result writeback to key fact storage."""

    def test_writeback_quantification(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Quantification parameters should be written back as a KeyFact of type QUANTIFICATION."""
        result = _make_pre_result(adjective="性价比", quantified=True)
        result.quantifiable_adjectives[0].quantified_value = {"price_range": [180, 220]}
        normalizer.writeback("s1", 3, result)
        facts = key_fact_store.get_by_session("s1", fact_type="quantification")
        assert len(facts) >= 1
        assert facts[0].content["adjective"] == "性价比"
        assert facts[0].content["quantified_value"]["price_range"] == [180, 220]

    def test_writeback_external_fact(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """External fact resolution results should be written back."""
        result = _make_pre_result()
        external_resolved = {
            "resolved": True,
            "resolved_entity": "商品X",
            "evidence": "价格查询返回",
            "raw_observation": "商品X 99元",
        }
        normalizer.writeback("s1", 3, result, external_resolved=external_resolved)
        facts = key_fact_store.get_by_session("s1", fact_type="key_fact")
        assert len(facts) >= 1
        assert facts[0].content["type"] == "external_fact_resolution"
        assert facts[0].content["resolved_entity"] == "商品X"

    def test_writeback_correction_record(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Backtracking correction records should be written back as type PRONOUN_RESOLUTION."""
        result = _make_pre_result()
        re_resolved = [
            PronounResolution(
                pronoun="那个",
                resolved_to="商品D",
                confidence=0.88,
                evidence_source="Observation 回溯（商品C已下架）",
            )
        ]
        normalizer.writeback("s1", 3, result, re_resolved=re_resolved)
        facts = key_fact_store.get_by_session("s1", fact_type="pronoun_resolution")
        assert len(facts) >= 1
        assert facts[0].content["resolved_to"] == "商品D"
        assert facts[0].content["correction_recorded"] is True

    def test_writeback_skips_unquantified(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """Unquantified adjectives should not be written back."""
        result = _make_pre_result(quantified=False)
        normalizer.writeback("s1", 3, result)
        facts = key_fact_store.get_by_session("s1", fact_type="quantification")
        assert len(facts) == 0


# ---------------------------------------------------------------------------
# task 5.1: Main entry process + ReAct integration
# ---------------------------------------------------------------------------


class TestProcessAndReactStep:
    """Main entry process and ReAct single-step integration."""

    def test_process_quantifies_adjectives(
        self, normalizer: DeepNormalizer
    ) -> None:
        """process should quantify unquantified adjectives."""
        pre_result = _make_pre_result(adjective="性价比", quantified=False)
        result = normalizer.process(
            session_id="s1",
            turn=1,
            pre_result=pre_result,
            user_id="u1",
            observation={"current_price": 200},
        )
        assert all(adj.quantified for adj in result.quantifiable_adjectives)
        assert result.quantifiable_adjectives[0].quantified_value is not None

    def test_process_with_observation_resolves_external_fact(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """process should complete external fact resolution and writeback when Observation is present."""
        pre_result = _make_pre_result()
        normalizer.process(
            session_id="s1",
            turn=2,
            pre_result=pre_result,
            user_id="u1",
            observation={"current_price": 200, "items": [{"name": "商品X"}]},
        )
        # Should writeback quantification results
        facts = key_fact_store.get_by_session("s1")
        assert any(f.fact_type == FactType.QUANTIFICATION for f in facts)

    def test_process_updates_completeness(
        self, normalizer: DeepNormalizer
    ) -> None:
        """process should update the completeness check result."""
        pre_result = _make_pre_result(quantified=False)
        result = normalizer.process(
            session_id="s1",
            turn=1,
            pre_result=pre_result,
            user_id="u1",
            observation={"current_price": 200},
        )
        assert result.completeness is not None
        assert result.completeness.adjectives_quantified is True
        assert result.completeness.result == CompletenessStatus.COMPLETE

    def test_process_without_observation(
        self, normalizer: DeepNormalizer
    ) -> None:
        """process should still quantify adjectives when there is no Observation."""
        pre_result = _make_pre_result(adjective="性价比", quantified=False)
        # Inject current_price via historical quantification facts
        result = normalizer.process(
            session_id="s1",
            turn=1,
            pre_result=pre_result,
            user_id="u1",
            observation=None,
        )
        # Even without Observation, rule-matched adjectives should be quantified
        # (Note: without current_price, quantified values keep placeholders, but quantified=True)
        assert result.quantifiable_adjectives[0].quantified is True

    def test_process_does_not_mutate_input(
        self, normalizer: DeepNormalizer
    ) -> None:
        """process should not modify the input pre_result."""
        pre_result = _make_pre_result(quantified=False)
        original_quantified = pre_result.quantifiable_adjectives[0].quantified
        normalizer.process(
            session_id="s1",
            turn=1,
            pre_result=pre_result,
            user_id="u1",
            observation={"current_price": 200},
        )
        # Input remains unchanged
        assert pre_result.quantifiable_adjectives[0].quantified == original_quantified

    def test_react_step_integration(
        self,
        normalizer: DeepNormalizer,
        key_fact_store: MemoryKeyFactStore,
    ) -> None:
        """react_step should complete deep normalization within Thought/Action/Observation."""
        pre_result = _make_pre_result(quantified=False)
        result = normalizer.react_step(
            session_id="s1",
            turn=1,
            pre_result=pre_result,
            thought="需要查询价格并量化性价比",
            action="price_query(items=牛仔裤)",
            observation={"current_price": 200},
            user_id="u1",
        )
        # Should complete quantification
        assert all(adj.quantified for adj in result.quantifiable_adjectives)
        # Should writeback key facts
        facts = key_fact_store.get_by_session("s1")
        assert any(f.fact_type == FactType.QUANTIFICATION for f in facts)

    def test_process_stats_tracked(
        self, normalizer: DeepNormalizer
    ) -> None:
        """process should update statistics metrics."""
        pre_result = _make_pre_result(quantified=False)
        normalizer.process(
            session_id="s1",
            turn=1,
            pre_result=pre_result,
            user_id="u1",
            observation={"current_price": 200},
        )
        stats = normalizer.get_stats()
        assert stats["total_processed"] >= 1
        assert stats["adjectives_quantified"] >= 1
        assert stats["writebacks"] >= 1
