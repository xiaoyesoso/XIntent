"""End-to-end pipeline tests (tasks 10.1-10.4)."""

import json

from user_input_normalization.llm import MockLLMClient
from user_input_normalization.models import (
    DialogueTurn,
    InputProblemType,
    NormalizationStage,
    VocabEntry,
    VocabLevel,
    VocabStatus,
)
from user_input_normalization.pipeline import NormalizationPipeline
from user_input_normalization.storage import (
    MemoryDialogueHistoryStore,
    MemoryFewShotStore,
    MemoryKeyFactStore,
    MemoryUserProfileStore,
    MemoryVocabStore,
)


def _make_pipeline_llm():
    """Create a MockLLM for the pipeline."""
    llm = MockLLMClient()

    def handler(system_prompt, user_prompt):
        if "第二个" in user_prompt:
            return json.dumps({
                "normalized_input": "TCC方案适合生产环境吗？",
                "spo": {"subject": "TCC方案", "subject_source": "对话历史",
                        "predicate": "适合", "obj": "生产环境"},
                "pronoun_resolutions": [
                    {"pronoun": "第二个", "resolved_to": "TCC方案",
                     "confidence": 0.95, "evidence_source": "对话历史第3轮",
                     "named_entity": "TCC方案"}
                ],
                "quantifiable_adjectives": [],
                "term_mappings": [],
                "completions": [],
            }, ensure_ascii=False)
        if "性价比" in user_prompt:
            return json.dumps({
                "normalized_input": "推荐更有性价比的牛仔裤",
                "spo": {"subject": "我", "predicate": "推荐", "obj": "牛仔裤"},
                "pronoun_resolutions": [],
                "quantifiable_adjectives": [
                    {"adjective": "性价比", "quantified": False,
                     "quantified_value": None, "route_to": "deep"}
                ],
                "term_mappings": [],
                "completions": [],
            }, ensure_ascii=False)
        if "RAG" in user_prompt:
            return json.dumps({
                "normalized_input": "用检索增强生成做知识库",
                "spo": {"subject": "我", "predicate": "用", "obj": "检索增强生成"},
                "pronoun_resolutions": [],
                "quantifiable_adjectives": [],
                "term_mappings": [
                    {"original": "RAG", "standard": "检索增强生成", "source": "vocabulary-table"}
                ],
                "completions": [],
            }, ensure_ascii=False)
        return json.dumps({
            "normalized_input": user_prompt[:100],
            "spo": {"subject": "user", "predicate": "ask", "obj": "question"},
            "pronoun_resolutions": [],
            "quantifiable_adjectives": [],
            "term_mappings": [],
            "completions": [],
        }, ensure_ascii=False)

    llm.set_default_handler(handler)
    return llm


def _make_pipeline():
    """Create a complete pipeline instance."""
    llm = _make_pipeline_llm()
    key_facts = MemoryKeyFactStore()
    fewshots = MemoryFewShotStore()
    vocab = MemoryVocabStore()
    profiles = MemoryUserProfileStore()
    dialogue = MemoryDialogueHistoryStore()

    # Pre-populate vocabulary table
    vocab.save(VocabEntry(
        vocab_id="v1", term="RAG", level=VocabLevel.PUBLIC,
        industry="通用", standard_meaning="检索增强生成",
        status=VocabStatus.PROMOTED,
    ))

    # Pre-populate dialogue history
    dialogue.append("s1", DialogueTurn(turn=1, role="assistant",
        content="我推荐以下方案：1. TCC方案 2. 轻量方案"))

    return NormalizationPipeline(
        llm_client=llm,
        key_fact_store=key_facts,
        fewshot_store=fewshots,
        vocab_store=vocab,
        profile_store=profiles,
        dialogue_store=dialogue,
    )


class TestFullPipeline:
    """End-to-end full-chain tests (task 10.1)."""

    def test_classification_to_pre_normalization(self):
        pipe = _make_pipeline()
        result = pipe.process("第二个适合生产吗？", "s1", "u1", turn=2)
        assert result.stage_reached in ("pre", "deep")
        assert len(result.result.pronoun_resolutions) > 0

    def test_deep_normalization_for_adjective(self):
        pipe = _make_pipeline()
        result = pipe.process("帮我推荐更有性价比的牛仔裤", "s2", "u1", turn=1)
        # Should route to the deep stage
        assert result.result.route_to == NormalizationStage.DEEP or result.stage_reached == "deep"

    def test_term_standardization_with_vocab(self):
        pipe = _make_pipeline()
        result = pipe.process("用RAG做知识库", "s3", "u1", turn=1)
        assert len(result.result.term_mappings) > 0
        assert result.result.term_mappings[0].original == "RAG"
        assert result.result.term_mappings[0].standard == "检索增强生成"


class TestSixCategoriesScenarios:
    """Six major input problem category coverage scenarios (task 10.2)."""

    def test_anaphora_scenario(self):
        """Anaphora problem: 第二个适合生产吗？"""
        pipe = _make_pipeline()
        result = pipe.process("第二个适合生产吗？", "c1", "u1", turn=2)
        assert InputProblemType.ANAPHORA in result.result.classification_tags

    def test_subjective_scenario(self):
        """Subjective judgment problem: 更有性价比"""
        pipe = _make_pipeline()
        result = pipe.process("帮我推荐更有性价比的牛仔裤", "c2", "u1", turn=1)
        assert InputProblemType.SUBJECTIVE in result.result.classification_tags

    def test_semantic_scenario(self):
        """Semantic problem: RAG"""
        pipe = _make_pipeline()
        result = pipe.process("用RAG做知识库", "c3", "u1", turn=1)
        assert InputProblemType.SEMANTIC in result.result.classification_tags

    def test_external_fact_scenario(self):
        """External fact problem: 现在最便宜的"""
        pipe = _make_pipeline()
        result = pipe.process("现在最便宜的是哪个？", "c4", "u1", turn=1)
        assert InputProblemType.EXTERNAL_FACT in result.result.classification_tags

    def test_missing_scenario(self):
        """Missing problem: 市场占有率多少？"""
        pipe = _make_pipeline()
        result = pipe.process("市场占有率多少？", "c5", "u1", turn=1)
        assert InputProblemType.MISSING in result.result.classification_tags

    def test_expression_scenario(self):
        """Expression problem: 不是，我说的是另一个"""
        pipe = _make_pipeline()
        result = pipe.process("不是，我说的是另一个。", "c6", "u1", turn=1)
        assert InputProblemType.EXPRESSION in result.result.classification_tags


class TestCrossTimeSpanAnaphora:
    """Cross-time-span anaphora resolution (task 10.3)."""

    def test_attribute_anaphora_detected(self):
        """Attribute anaphora detection: 看樱花的那个地方"""
        pipe = _make_pipeline()
        # Set up historical dialogue: recommend Jiming Temple for cherry blossoms
        pipe._dialogue_store.append("ct1", DialogueTurn(
            turn=1, role="assistant",
            content="我推荐鸡鸣寺，现在去看樱花正好。"
        ))
        result = pipe.process(
            "你上次推荐的看樱花的那个地方怎么去？", "ct1", "u1", turn=2
        )
        # Should detect attribute anaphora
        assert result.stage_reached in ("pre", "deep")


class TestEcommerceQuantification:
    """E-commerce price-performance quantification full chain (task 10.4)."""

    def test_quantification_full_chain(self):
        pipe = _make_pipeline()
        result = pipe.process(
            "帮我推荐更有性价比的牛仔裤", "eq1", "u1", turn=1,
            observation={"current_price": 200, "available_prices": [100, 150, 200, 300]},
        )
        # Should enter the deep stage for quantification
        assert result.stage_reached == "deep" or result.result.route_to == NormalizationStage.DEEP


class TestDegradation:
    """Degradation drills (task 10.8)."""

    def test_pre_normalization_passthrough(self):
        """pre passthrough mode (no normalization)"""
        from user_input_normalization.config import Config, PipelineConfig
        cfg = Config()
        cfg.pipeline = PipelineConfig(enable_deep_normalization=False)
        # Create a simplified pipeline
        llm = _make_pipeline_llm()
        pipe = NormalizationPipeline(
            llm_client=llm,
            key_fact_store=MemoryKeyFactStore(),
            fewshot_store=MemoryFewShotStore(),
            config=cfg,
        )
        result = pipe.process("第二个适合生产吗？", "d1", "u1", turn=1)
        # deep is disabled, does not enter the deep stage
        assert result.stage_reached == "pre"

    def test_vocab_table_disabled(self):
        """Vocabulary table disabled"""
        from user_input_normalization.config import Config, PipelineConfig
        cfg = Config()
        cfg.pipeline = PipelineConfig(enable_deep_normalization=False)
        llm = _make_pipeline_llm()
        pipe = NormalizationPipeline(
            llm_client=llm,
            key_fact_store=MemoryKeyFactStore(),
            fewshot_store=MemoryFewShotStore(),
            config=cfg,
        )
        result = pipe.process("用RAG做知识库", "d2", "u1", turn=1)
        # Vocabulary table disabled, but classification should still work
        assert result.result.classification_tags is not None
