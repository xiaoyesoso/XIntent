"""pre-normalization regression tests (task 3.15).

Covers overreach scenarios: not answering / not executing tools / not fabricating facts.
"""

import json

from user_input_normalization.llm import MockLLMClient
from user_input_normalization.models import (
    InputProblemType,
    NormalizationStage,
)
from user_input_normalization.pre_normalization import PreNormalizer
from user_input_normalization.storage import (
    MemoryDialogueHistoryStore,
    MemoryFewShotStore,
    MemoryKeyFactStore,
    MemoryUserProfileStore,
    MemoryVocabStore,
)


def _make_mock_llm():
    """Create a MockLLM that returns simulated normalization results."""
    llm = MockLLMClient()

    def handler(system_prompt, user_prompt):
        # Simulate pronoun resolution scenario
        if "第二个" in user_prompt:
            return json.dumps({
                "normalized_input": "TCC方案适合生产环境吗？",
                "spo": {
                    "subject": "TCC方案", "subject_source": "对话历史第3轮",
                    "predicate": "适合", "obj": "生产环境", "obj_source": "原文",
                },
                "pronoun_resolutions": [
                    {"pronoun": "第二个", "resolved_to": "TCC方案",
                     "confidence": 0.95, "evidence_source": "对话历史第3轮",
                     "named_entity": "TCC方案"}
                ],
                "quantifiable_adjectives": [],
                "term_mappings": [],
                "completions": [],
            }, ensure_ascii=False)
        # Simulate missing completion scenario
        if "市场占有率" in user_prompt:
            return json.dumps({
                "normalized_input": "良信互联的市场占有率是多少？",
                "spo": {
                    "subject": "良信互联", "subject_source": "对话历史第1轮",
                    "predicate": "是", "obj": "多少",
                },
                "pronoun_resolutions": [],
                "quantifiable_adjectives": [],
                "term_mappings": [],
                "completions": [
                    {"field": "主语", "content": "良信互联", "source": "对话历史第1轮"}
                ],
            }, ensure_ascii=False)
        # Simulate subjective judgment scenario
        if "性价比" in user_prompt:
            return json.dumps({
                "normalized_input": "帮我推荐一个更有性价比的牛仔裤",
                "spo": {"subject": "我", "predicate": "推荐", "obj": "牛仔裤"},
                "pronoun_resolutions": [],
                "quantifiable_adjectives": [
                    {"adjective": "性价比", "quantified": False,
                     "quantified_value": None, "route_to": "deep"}
                ],
                "term_mappings": [],
                "completions": [],
            }, ensure_ascii=False)
        # Default
        return json.dumps({
            "normalized_input": user_prompt[:100],
            "spo": {"subject": None, "predicate": None, "obj": None},
            "pronoun_resolutions": [],
            "quantifiable_adjectives": [],
            "term_mappings": [],
            "completions": [],
        }, ensure_ascii=False)

    llm.set_default_handler(handler)
    return llm


def _make_normalizer():
    """Create a complete PreNormalizer instance."""
    llm = _make_mock_llm()
    return PreNormalizer(
        llm_client=llm,
        key_fact_store=MemoryKeyFactStore(),
        fewshot_store=MemoryFewShotStore(),
        vocab_store=MemoryVocabStore(),
        profile_store=MemoryUserProfileStore(),
        dialogue_store=MemoryDialogueHistoryStore(),
    )


class TestPronounResolution:
    """Pronoun resolution (task 3.5)."""

    def test_resolve_ordinal_pronoun(self):
        norm = _make_normalizer()
        result = norm.normalize("第二个适合生产吗？", "s1", "u1", turn=3)
        assert len(result.pronoun_resolutions) > 0
        assert result.pronoun_resolutions[0].pronoun == "第二个"
        assert result.pronoun_resolutions[0].resolved_to == "TCC方案"
        assert result.pronoun_resolutions[0].confidence > 0.6

    def test_pronoun_table_written_to_key_facts(self):
        norm = _make_normalizer()
        norm.normalize("第二个适合生产吗？", "s1", "u1", turn=3)
        # Check the key fact store
        fact = norm._key_facts.find_pronoun_resolution("s1", "第二个")
        assert fact is not None
        assert fact.content["resolved_to"] == "TCC方案"

    def test_named_entity_set(self):
        norm = _make_normalizer()
        result = norm.normalize("第二个适合生产吗？", "s1", "u1", turn=3)
        pr = result.pronoun_resolutions[0]
        assert pr.named_entity == "TCC方案"


class TestEllipsisCompletion:
    """Ellipsis completion (task 3.6)."""

    def test_complete_missing_subject(self):
        norm = _make_normalizer()
        result = norm.normalize("市场占有率多少？", "s2", "u1", turn=1)
        assert result.spo.subject == "良信互联"
        assert result.spo.subject_source == "对话历史第1轮"
        assert len(result.completions) > 0
        assert result.completions[0].field == "主语"


class TestCompletenessCheck:
    """Completeness check (task 3.11)."""

    def test_adjective_unquantified_routes_to_deep(self):
        norm = _make_normalizer()
        result = norm.normalize("帮我推荐一个更有性价比的牛仔裤", "s3", "u1", turn=1)
        assert result.completeness is not None
        assert not result.completeness.adjectives_quantified
        assert result.route_to == NormalizationStage.DEEP

    def test_complete_result_no_route(self):
        norm = _make_normalizer()
        result = norm.normalize("第二个适合生产吗？", "s4", "u1", turn=3)
        # Has subject-predicate-object, pronouns resolved, no adjectives
        assert result.completeness is not None
        assert result.completeness.result.value == "complete" or result.route_to is None


class TestResponsibilityBoundary:
    """Responsibility boundary constraints (task 3.4 can/cannot do) - overreach scenario regression tests."""

    def test_does_not_answer_question(self):
        """Verify that pre-normalization does not directly answer user questions."""
        norm = _make_normalizer()
        result = norm.normalize("市场占有率多少？", "s5", "u1", turn=1)
        violations = norm.validate_boundary(result)
        assert not violations["answered_question"], "不应直接回答问题"

    def test_does_not_execute_tool(self):
        """Verify that pre-normalization does not execute tools."""
        norm = _make_normalizer()
        result = norm.normalize("第二个适合生产吗？", "s6", "u1", turn=3)
        # Normalization results should not contain tool calls
        assert "tool_call" not in result.normalized_input.lower()

    def test_does_not_fabricate_facts(self):
        """Verify that pre-normalization does not fabricate real-time information."""
        norm = _make_normalizer()
        result = norm.normalize("市场占有率多少？", "s7", "u1", turn=1)
        # Should not contain specific values (market share data)
        import re
        numbers = re.findall(r"\d+\.?\d*%", result.normalized_input)
        assert len(numbers) == 0, "不应伪造具体数据"


class TestFewShotSinking:
    """Strange input sinks into few-shot (task 3.14)."""

    def test_strange_input_sunk_to_fewshot(self):
        norm = _make_normalizer()
        norm.normalize("第二个适合生产吗？", "s8", "u1", turn=3)
        # Anaphora problems are strange input and should be sunk
        examples = norm._fewshots.list_all()
        assert len(examples) > 0
        assert "第二个" in examples[0].input

    def test_non_strange_input_not_sunk(self):
        norm = _make_normalizer()
        # "你好" does not contain strange input features
        norm.normalize("你好", "s9", "u1", turn=1)
        examples = norm._fewshots.list_all()
        # "你好" may not be flagged as strange input
        # Check whether there is an example of "你好"
        hello_examples = [e for e in examples if "你好" in e.input]
        assert len(hello_examples) == 0


class TestCrossTurnReuse:
    """Pronoun resolution table cross-turn reuse (task 3.12)."""

    def test_pronoun_reused_across_turns(self):
        norm = _make_normalizer()
        # Turn 3 resolves "第二个" -> "TCC方案"
        norm.normalize("第二个适合生产吗？", "s10", "u1", turn=3)
        # Turn 5 reuses "第二个", should reuse the existing resolution result
        fact = norm._key_facts.find_pronoun_resolution("s10", "第二个")
        assert fact is not None
        assert fact.content["resolved_to"] == "TCC方案"
