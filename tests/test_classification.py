"""Input problem classifier regression tests (corresponding to task 2.4).

Coverage:
- Six categories of typical examples (at least 2 per category)
- Multi-category simultaneous recognition
- Routing logic (PRE / DEEP)
- Primary category priority
- LLM fallback
"""

from __future__ import annotations

import json

import pytest

from user_input_normalization.classification import (
    ClassificationResult,
    ClassificationRule,
    InputClassifier,
    match_rules,
)
from user_input_normalization.llm.mock import MockLLMClient
from user_input_normalization.models import InputProblemType, NormalizationStage


# ---------------------------------------------------------------------------
# Six categories single-category recognition
# ---------------------------------------------------------------------------


class TestAnaphoraClassification:
    """Anaphora (ANAPHORA)."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    @pytest.mark.parametrize(
        "text",
        [
            "第二个适合生产吗？",
            "刚才那个项目怎么包装？",
            "前一个方案不行。",
            "上一个版本还有问题。",
            "这个可以吗？",
            "那个不太行。",
        ],
    )
    def test_anaphora_detected(self, text: str) -> None:
        tags = self.classifier.classify(text)
        assert InputProblemType.ANAPHORA in tags, f"应识别为指代问题: {text}"

    def test_anaphora_sub_type(self) -> None:
        """Sub-types should be recorded as 序号指代/时间指代, etc."""
        result = self.classifier.classify_detailed("第二个适合生产吗？")
        assert result.primary_category == InputProblemType.ANAPHORA
        assert "指代问题" in result.sub_types
        assert result.sub_types["指代问题"] == "序号指代"

    def test_anaphora_evidence_span(self) -> None:
        """Matched evidence should be recorded."""
        result = self.classifier.classify_detailed("刚才那个项目怎么包装？")
        assert any("刚才" in span or "那个" in span for span in result.evidence_spans)


class TestMissingClassification:
    """Missing (MISSING)."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    @pytest.mark.parametrize(
        "text",
        [
            "市场占有率多少？",
            "帮我优化一下。",
            "方案怎么样？",
            "适合吗？",
            "看一下。",
        ],
    )
    def test_missing_detected(self, text: str) -> None:
        tags = self.classifier.classify(text)
        assert InputProblemType.MISSING in tags, f"应识别为缺失问题: {text}"

    def test_missing_sub_type_缺宾语(self) -> None:
        result = self.classifier.classify_detailed("帮我优化一下。")
        assert result.primary_category == InputProblemType.MISSING
        assert result.sub_types.get("缺失问题") == "缺宾语"


class TestExpressionClassification:
    """Expression (EXPRESSION)."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    @pytest.mark.parametrize(
        "text",
        [
            "不是，我说的是另一个。",
            "这个不太行，换个更像面试能讲的。",
            "算了，不对，换一个。",
            "然后，然后就是那个，怎么说呢。",
        ],
    )
    def test_expression_detected(self, text: str) -> None:
        tags = self.classifier.classify(text)
        assert InputProblemType.EXPRESSION in tags, f"应识别为表达问题: {text}"

    def test_expression_sub_type_临时改口(self) -> None:
        result = self.classifier.classify_detailed("不是，我说的是另一个。")
        assert result.primary_category == InputProblemType.EXPRESSION
        assert result.sub_types.get("表达问题") == "临时改口"


class TestSemanticClassification:
    """Semantic (SEMANTIC)."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    @pytest.mark.parametrize(
        "text",
        [
            "RAG 怎么实现？",
            "这个方案不够 P8。",
            "我们需要找一个抓手。",
            "做闭环。",
            "用 DDD 重构。",
        ],
    )
    def test_semantic_detected(self, text: str) -> None:
        tags = self.classifier.classify(text)
        assert InputProblemType.SEMANTIC in tags, f"应识别为词义问题: {text}"


class TestSubjectiveClassification:
    """Subjective (SUBJECTIVE)."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    @pytest.mark.parametrize(
        "text",
        [
            "哪个最有性价比？",
            "再高级一点。",
            "更好的方案是什么？",
            "最划算的是哪个？",
            "再便宜一点。",
        ],
    )
    def test_subjective_detected(self, text: str) -> None:
        tags = self.classifier.classify(text)
        assert InputProblemType.SUBJECTIVE in tags, f"应识别为主观判断问题: {text}"


class TestExternalFactClassification:
    """External fact (EXTERNAL_FACT)."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    @pytest.mark.parametrize(
        "text",
        [
            "最近哪个框架更火？",
            "现在最便宜的是哪个？",
            "当前最流行的方案是什么？",
            "最新版本是多少？",
            "实时价格多少？",
        ],
    )
    def test_external_fact_detected(self, text: str) -> None:
        tags = self.classifier.classify(text)
        assert InputProblemType.EXTERNAL_FACT in tags, (
            f"应识别为外部事实问题: {text}"
        )


# ---------------------------------------------------------------------------
# Multi-category simultaneous recognition
# ---------------------------------------------------------------------------


class TestMultiCategoryClassification:
    """Multi-category simultaneous recognition (corresponds to spec "multi-category simultaneous recognition")."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    def test_anaphora_plus_subjective(self) -> None:
        """Anaphora + Subjective ("刚才那个最有性价比的方案")."""
        text = "刚才那个最有性价比的方案适合生产吗？"
        tags = self.classifier.classify(text)
        assert InputProblemType.ANAPHORA in tags
        assert InputProblemType.SUBJECTIVE in tags

    def test_anaphora_plus_missing_plus_semantic(self) -> None:
        """Anaphora + Missing + Semantic ("帮我看看那个 RAG 的情况")."""
        text = "帮我看看那个 RAG 的情况。"
        tags = self.classifier.classify(text)
        assert InputProblemType.ANAPHORA in tags
        assert InputProblemType.MISSING in tags
        assert InputProblemType.SEMANTIC in tags

    def test_primary_category_anaphora_first(self) -> None:
        """Primary category priority: Anaphora takes priority over Subjective (needs resolution first)."""
        text = "刚才那个最有性价比的方案适合生产吗？"
        result = self.classifier.classify_detailed(text)
        assert result.primary_category == InputProblemType.ANAPHORA

    def test_primary_category_missing_priority(self) -> None:
        """Primary category priority: Missing takes priority over Semantic (needs completion first)."""
        text = "帮我看看那个 RAG 的情况。"
        result = self.classifier.classify_detailed(text)
        assert result.primary_category == InputProblemType.MISSING

    def test_multi_evidence_spans(self) -> None:
        """On multi-category match, evidence_spans should contain multiple evidence fragments."""
        text = "刚才那个最有性价比的方案适合生产吗？"
        result = self.classifier.classify_detailed(text)
        assert len(result.evidence_spans) >= 2


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


class TestRoute:
    """Classification result routing (corresponds to task 2.3)."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    @pytest.mark.parametrize(
        "tag,expected",
        [
            (InputProblemType.ANAPHORA, NormalizationStage.PRE),
            (InputProblemType.MISSING, NormalizationStage.PRE),
            (InputProblemType.EXPRESSION, NormalizationStage.PRE),
            (InputProblemType.SEMANTIC, NormalizationStage.PRE),
            (InputProblemType.SUBJECTIVE, NormalizationStage.DEEP),
            (InputProblemType.EXTERNAL_FACT, NormalizationStage.DEEP),
        ],
    )
    def test_single_tag_route(
        self, tag: InputProblemType, expected: NormalizationStage
    ) -> None:
        assert self.classifier.route([tag]) == expected

    def test_pre_tags_route_to_pre(self) -> None:
        """PRE-category tag combinations route to PRE."""
        tags = [
            InputProblemType.ANAPHORA,
            InputProblemType.MISSING,
            InputProblemType.EXPRESSION,
        ]
        assert self.classifier.route(tags) == NormalizationStage.PRE

    def test_mixed_tags_route_to_deep(self) -> None:
        """PRE + DEEP mixed tags route to DEEP (requires two-stage coordination)."""
        tags = [InputProblemType.ANAPHORA, InputProblemType.SUBJECTIVE]
        assert self.classifier.route(tags) == NormalizationStage.DEEP

    def test_empty_tags_route_to_pre(self) -> None:
        """No tags: default route to PRE."""
        assert self.classifier.route([]) == NormalizationStage.PRE


# ---------------------------------------------------------------------------
# Routing + classification end-to-end
# ---------------------------------------------------------------------------


class TestClassifyAndRoute:
    """End-to-end: auto-route after classification."""

    def setup_method(self) -> None:
        self.classifier = InputClassifier(enable_llm_fallback=False)

    def test_anaphora_input_routes_to_pre(self) -> None:
        tags = self.classifier.classify("第二个适合生产吗？")
        assert self.classifier.route(tags) == NormalizationStage.PRE

    def test_subjective_input_routes_to_deep(self) -> None:
        tags = self.classifier.classify("哪个最有性价比？")
        assert self.classifier.route(tags) == NormalizationStage.DEEP

    def test_external_fact_routes_to_deep(self) -> None:
        tags = self.classifier.classify("最近哪个框架更火？")
        assert self.classifier.route(tags) == NormalizationStage.DEEP

    def test_mixed_input_routes_to_deep(self) -> None:
        tags = self.classifier.classify("刚才那个最有性价比的方案适合生产吗？")
        assert self.classifier.route(tags) == NormalizationStage.DEEP


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------


class TestLLMFallback:
    """LLM fallback (when rules do not match)."""

    def test_llm_fallback_used_when_no_rule_match(self) -> None:
        """When rules do not match, call LLM for fallback."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys, usr: json.dumps(
                {
                    "categories": ["主观判断问题"],
                    "primary_category": "主观判断问题",
                    "confidence": 0.8,
                },
                ensure_ascii=False,
            )
        )
        classifier = InputClassifier(
            llm_client=mock, enable_llm_fallback=True
        )
        # An input with no rule match
        result = classifier.classify_detailed("来个差不多的就行")
        assert result.llm_used is True
        assert InputProblemType.SUBJECTIVE in result.categories
        assert result.confidence == 0.8

    def test_llm_not_used_when_rule_matches(self) -> None:
        """When rules match, do not call LLM."""
        mock = MockLLMClient()
        classifier = InputClassifier(
            llm_client=mock, enable_llm_fallback=True
        )
        result = classifier.classify_detailed("第二个适合生产吗？")
        assert result.rule_matched is True
        assert result.llm_used is False

    def test_llm_invalid_json_returns_empty(self) -> None:
        """When LLM returns invalid JSON, degrade to empty result."""
        mock = MockLLMClient()
        mock.set_default_handler(lambda sys, usr: "not a json")
        classifier = InputClassifier(
            llm_client=mock, enable_llm_fallback=True
        )
        result = classifier.classify_detailed("来个差不多的就行")
        assert result.categories == []
        assert result.llm_used is True


# ---------------------------------------------------------------------------
# Rule matching utilities
# ---------------------------------------------------------------------------


class TestMatchRules:
    """Rule matching utility functions."""

    def test_match_rules_returns_dict(self) -> None:
        matches = match_rules("第二个适合生产吗？")
        assert isinstance(matches, dict)
        assert InputProblemType.ANAPHORA in matches

    def test_match_rules_no_match(self) -> None:
        matches = match_rules("今天天气不错")
        # Should not match any explicit category (may match meaningless SEMANTIC all-caps rule, filtered out)
        # Here only assert no match for ANAPHORA / MISSING / SUBJECTIVE / EXTERNAL_FACT
        for pt in [
            InputProblemType.ANAPHORA,
            InputProblemType.MISSING,
            InputProblemType.SUBJECTIVE,
            InputProblemType.EXTERNAL_FACT,
        ]:
            assert pt not in matches


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases."""

    def test_empty_input(self) -> None:
        classifier = InputClassifier(enable_llm_fallback=False)
        tags = classifier.classify("")
        assert tags == []

    def test_whitespace_input(self) -> None:
        classifier = InputClassifier(enable_llm_fallback=False)
        tags = classifier.classify("   ")
        assert tags == []

    def test_normal_input_no_tags(self) -> None:
        """Normal input should not match any category."""
        classifier = InputClassifier(enable_llm_fallback=False)
        tags = classifier.classify("请帮我介绍一下 TCC 方案的原理。")
        # Should not match Subjective/External fact
        assert InputProblemType.SUBJECTIVE not in tags
        assert InputProblemType.EXTERNAL_FACT not in tags

    def test_confidence_high_when_rule_matched(self) -> None:
        """When rules match, confidence should be 1.0."""
        classifier = InputClassifier(enable_llm_fallback=False)
        result = classifier.classify_detailed("第二个适合生产吗？")
        assert result.confidence == 1.0
