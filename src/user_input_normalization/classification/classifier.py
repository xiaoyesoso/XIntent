"""Input problem classifier (corresponding to tasks 2.2, 2.3).

Two-stage classification based on rules + LLM:
1. First use rule matching to quickly hit explicit categories (high accuracy)
2. For inputs that rules cannot cover or have low confidence, optionally call LLM as fallback

Classification results drive subsequent routing:
- Anaphora/Missing/Expression/Semantic -> pre-normalization
- Subjective -> deep-normalization (quantification)
- External fact -> deep-normalization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, get_config
from ..llm.base import LLMClient
from ..llm.mock import MockLLMClient
from ..models import InputProblemType, NormalizationStage
from .rules import ClassificationRule, match_rules


# ---------------------------------------------------------------------------
# Classification result data structure (corresponds to spec "classification output structure")
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Classification result (corresponds to the structured classification output in the spec)."""

    categories: list[InputProblemType] = field(default_factory=list)
    primary_category: InputProblemType | None = None
    sub_types: dict[str, str] = field(default_factory=dict)
    evidence_spans: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rule_matched: bool = False  # Whether matched purely by rules
    llm_used: bool = False  # Whether LLM fallback was used


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class InputClassifier:
    """Input problem classifier.

    Prefers rule matching (high accuracy, low latency); falls back to LLM when rules cannot decide.

    Args:
        llm_client: LLM client (optional; may not be used when rules match with sufficient confidence)
        config: Global configuration
        enable_llm_fallback: Whether to enable LLM fallback (default True)
        confidence_threshold: Rule match is treated as 1.0; below this threshold triggers clarification
            (this class does not trigger clarification directly; it only returns confidence for upper-layer judgment)
    """

    # Primary category priority: used to determine primary_category when multiple categories match
    # Higher priority comes first (consistent with processing dependency order)
    # Based on spec scenarios: "帮我看看那个 RAG 的情况" primary category is Missing (request intent must be clarified first)
    #                        "刚才那个最有性价比的方案" primary category is Anaphora (only when Missing does not match does Anaphora take over)
    _PRIMARY_PRIORITY: list[InputProblemType] = [
        InputProblemType.MISSING,  # Missing needs request intent clarified first
        InputProblemType.ANAPHORA,  # Anaphora needs resolution first
        InputProblemType.EXPRESSION,  # Expression needs correction first
        InputProblemType.SEMANTIC,  # Semantic needs disambiguation first
        InputProblemType.SUBJECTIVE,  # Subjective needs quantification
        InputProblemType.EXTERNAL_FACT,  # External needs tools
    ]

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: Config | None = None,
        *,
        enable_llm_fallback: bool = True,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or get_config()
        self.enable_llm_fallback = enable_llm_fallback
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Main entry: classification
    # ------------------------------------------------------------------

    def classify(self, input_text: str) -> list[InputProblemType]:
        """Classify the input text and return the list of matched problem types (multi-category supported).

        Args:
            input_text: User's original input

        Returns:
            List of matched InputProblemType (may be empty, meaning no obvious problem)
        """
        result = self.classify_detailed(input_text)
        return result.categories

    def classify_detailed(self, input_text: str) -> ClassificationResult:
        """Detailed version of classification, returns the full structured result."""
        if not input_text or not input_text.strip():
            return ClassificationResult(confidence=1.0, rule_matched=True)

        # Step 1: Rule matching
        matches = match_rules(input_text)
        if matches:
            return self._build_result_from_rules(matches)

        # Step 2: LLM fallback
        if self.enable_llm_fallback and self.llm_client is not None:
            return self._classify_via_llm(input_text)

        # Neither rule match nor LLM: return empty result
        return ClassificationResult(confidence=0.5, rule_matched=False)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, tags: list[InputProblemType]) -> NormalizationStage:
        """Route to the corresponding stage based on classification tags.

        - Anaphora/Missing/Expression/Semantic -> PRE
        - Subjective/External fact -> DEEP
        - When multiple tags match simultaneously, DEEP takes priority (requires two-stage coordination)

        Args:
            tags: List of classification tags

        Returns:
            NormalizationStage.PRE or NormalizationStage.DEEP
        """
        if not tags:
            # No tags: default to pre stage for basic normalization
            return NormalizationStage.PRE

        deep_types = {InputProblemType.SUBJECTIVE, InputProblemType.EXTERNAL_FACT}
        # Any tag belonging to deep processing types -> route to DEEP
        # (spec requires: inputs matching both MUST be handled by two-stage coordination,
        #   i.e. PRE stage does resolution/completion, DEEP stage does quantification/external fact)
        if any(tag in deep_types for tag in tags):
            return NormalizationStage.DEEP
        return NormalizationStage.PRE

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_result_from_rules(
        self,
        matches: dict[InputProblemType, list[tuple[ClassificationRule, str]]],
    ) -> ClassificationResult:
        """Build ClassificationResult from rule match results."""
        categories = list(matches.keys())
        # Determine primary category by sorting according to _PRIMARY_PRIORITY
        primary = self._pick_primary(categories)

        sub_types: dict[str, str] = {}
        evidence_spans: list[str] = []
        for problem_type, hits in matches.items():
            # Sub-type takes the first matched rule's sub_type
            for rule, span in hits:
                if rule.sub_type and problem_type.value not in sub_types:
                    sub_types[problem_type.value] = rule.sub_type
                if span not in evidence_spans:
                    evidence_spans.append(span)

        return ClassificationResult(
            categories=categories,
            primary_category=primary,
            sub_types=sub_types,
            evidence_spans=evidence_spans,
            confidence=1.0,  # Rule match treated as high confidence
            rule_matched=True,
            llm_used=False,
        )

    def _pick_primary(
        self, categories: list[InputProblemType]
    ) -> InputProblemType | None:
        """Select the primary category according to processing dependency order."""
        if not categories:
            return None
        for pt in self._PRIMARY_PRIORITY:
            if pt in categories:
                return pt
        return categories[0]

    def _classify_via_llm(self, input_text: str) -> ClassificationResult:
        """Call LLM for fallback classification (for vague inputs not matched by rules)."""
        assert self.llm_client is not None
        system_prompt = (
            "你是用户输入问题分类器。请将用户输入归类到以下六大类问题中的零个或多"
            "个类别，并以 JSON 返回。类别取值：\n"
            "- 指代问题\n- 缺失问题\n- 表达问题\n- 词义问题\n"
            "- 主观判断问题\n- 外部事实问题\n\n"
            '输出 JSON 格式：{"categories": ["..."], "primary_category": "...", '
            '"confidence": 0.0}'
        )
        user_prompt = f"用户输入：{input_text}\n\n请输出分类 JSON。"
        try:
            response = self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            return self._parse_llm_response(response, input_text)
        except Exception:
            # LLM call failed: degrade to empty result
            return ClassificationResult(
                categories=[],
                primary_category=None,
                confidence=0.3,
                rule_matched=False,
                llm_used=True,
            )

    def _parse_llm_response(
        self, response: str, input_text: str
    ) -> ClassificationResult:
        """Parse the JSON returned by the LLM."""
        try:
            data: dict[str, Any] = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re

            m = re.search(r"\{[\s\S]*\}", response)
            if not m:
                return ClassificationResult(
                    confidence=0.3, rule_matched=False, llm_used=True
                )
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return ClassificationResult(
                    confidence=0.3, rule_matched=False, llm_used=True
                )

        # String category name -> enum
        valid_values = {t.value for t in InputProblemType}
        raw_categories = data.get("categories", [])
        categories: list[InputProblemType] = []
        for c in raw_categories:
            if isinstance(c, str) and c in valid_values:
                categories.append(InputProblemType(c))

        primary_raw = data.get("primary_category")
        primary: InputProblemType | None = None
        if isinstance(primary_raw, str) and primary_raw in valid_values:
            primary = InputProblemType(primary_raw)
        elif categories:
            primary = self._pick_primary(categories)

        confidence = float(data.get("confidence", 0.7))

        return ClassificationResult(
            categories=categories,
            primary_category=primary,
            sub_types={},
            evidence_spans=[],
            confidence=confidence,
            rule_matched=False,
            llm_used=True,
        )


# ---------------------------------------------------------------------------
# Convenience factory function
# ---------------------------------------------------------------------------


def create_default_classifier(
    llm_client: LLMClient | None = None,
) -> InputClassifier:
    """Create a default classifier (uses MockLLMClient as fallback when no LLM is provided)."""
    if llm_client is None:
        llm_client = MockLLMClient()
    return InputClassifier(llm_client=llm_client)
