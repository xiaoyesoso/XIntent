"""Quantification engine (corresponding to tasks 6.1-6.3, 6.5-6.6 / D13).

Transforms subjective judgment words into executable tool-call parameters.
Core capabilities:
1. Rule-based quantification (default Spec rules in rules.py)
2. Fallback to LLM quantification for unknown adjectives
3. Context-aware quantification (combining user profile, previous-round candidate attributes)
4. Explainability output
5. Dynamic registration of new rules (integrated with vocabulary, task 6.7)
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Config, get_config
from ..llm.base import LLMClient
from ..models import (
    QuantifiableAdjective,
    QuantificationRule,
    UserProfile,
)
from ..storage.base import VocabStore
from .rules import DEFAULT_RULES, get_alternative_rule, get_rule


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fill_template(template: Any, context: dict[str, Any]) -> Any:
    """Recursively fill placeholders in the template.

    Supports placeholders like {current_price}, {current_tier}.
    For placeholders not provided, keep them as-is or replace with an empty string.
    When the template string itself is a single placeholder (e.g. "{current_tier_plus_1}"),
    preserve the original type (int/float) without stringifying.
    """
    if isinstance(template, str):
        # If the whole string is a single placeholder, return the value with its original type
        single_match = re.match(r"^\{(\w+)\}$", template.strip())
        if single_match:
            key = single_match.group(1)
            if key in context:
                return context[key]
            return template
        # Simple string template substitution
        try:
            return template.format_map(_SafeDict(context))
        except (KeyError, IndexError, ValueError):
            # Degrade to per-key replacement
            result = template
            for key, val in context.items():
                result = result.replace("{" + key + "}", str(val))
            return result
    if isinstance(template, dict):
        return {k: _fill_template(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill_template(item, context) for item in template]
    return template


class _SafeDict(dict):
    """Safe dict: missing keys return the {key} string as-is."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


def _parse_range(text: str) -> list[Any] | str:
    """Try to parse a string like "[150, 200]" into a list; return the original string on failure."""
    if not isinstance(text, str):
        return text
    match = re.match(r"^\[(.+),(.+)\]$", text.strip())
    if not match:
        return text
    try:
        return [float(match.group(1).strip()), float(match.group(2).strip())]
    except ValueError:
        return text


def _build_context_for_template(context: dict[str, Any]) -> dict[str, Any]:
    """Derive placeholder variables required by the template from the original context."""
    current_price = _safe_float(context.get("current_price"), 0.0)
    current_tier = int(_safe_float(context.get("current_tier"), 0))

    # Price range ±10%
    price_lower_10 = round(current_price * 0.9, 2)
    price_upper_10 = round(current_price * 1.1, 2)
    # Price range down 25%
    price_lower_25 = round(current_price * 0.75, 2)
    # Price range up 25%
    price_upper_25 = round(current_price * 1.25, 2)

    return {
        "current_price": current_price,
        "current_tier": current_tier,
        "current_tier_plus_1": current_tier + 1,
        "current_price_lower": price_lower_10,
        "current_price_upper": price_upper_10,
        "current_price_lower_25": price_lower_25,
        "current_price_upper_25": price_upper_25,
        # Previous-round candidate attributes
        "current_quality_rank": context.get("current_quality_rank", ""),
        "current_brand_tier": context.get("current_brand_tier", ""),
        # User profile related
        "user_industry": context.get("user_industry", "通用"),
        "user_preference": context.get("user_preference", ""),
    }


class QuantificationEngine:
    """Quantification engine (corresponds to D13).

    Responsible for transforming subjective judgment words into tool-call parameters.
    Uses rules first, and falls back to LLM when no rule matches.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vocab_store: VocabStore | None = None,
        config: Config | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.vocab_store = vocab_store
        self.config = config or get_config()
        # Rule table, supports dynamic registration (task 6.7)
        self._rules: dict[str, QuantificationRule] = dict(DEFAULT_RULES)
        # Record the rule and basis used for each quantification, for explainability
        self._last_rule: QuantificationRule | None = None
        self._last_context: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core: quantification (tasks 6.1-6.3)
    # ------------------------------------------------------------------

    def quantify(
        self, adjective: str, context: dict[str, Any] | None = None
    ) -> QuantifiableAdjective:
        """Quantify a judgment word into tool parameters.

        Args:
            adjective: Subjective judgment word, e.g. "性价比", "再高级一点"
            context: Context, may include current_price, current_tier,
                     user_profile, current_quality_rank, etc.

        Returns:
            QuantifiableAdjective, quantified=True means already quantified,
            quantified_value is the filled tool parameters
        """
        context = context or {}
        template_ctx = _build_context_for_template(context)

        # 1) Check rules first
        rule = self._rules.get(adjective)
        if rule is not None:
            # If the user profile indicates price sensitivity, switch "性价比" to the alternative strategy
            if adjective in ("性价比", "划算"):
                user_profile = context.get("user_profile")
                if isinstance(user_profile, UserProfile) and self._is_price_sensitive(user_profile):
                    alt = get_alternative_rule(adjective)
                    if alt is not None:
                        rule = alt
            return self._quantify_with_rule(adjective, rule, template_ctx, context)

        # 2) Unknown adjective falls back to LLM
        return self._quantify_with_llm(adjective, template_ctx, context)

    def _quantify_with_rule(
        self,
        adjective: str,
        rule: QuantificationRule,
        template_ctx: dict[str, Any],
        original_context: dict[str, Any],
    ) -> QuantifiableAdjective:
        """Fill template parameters based on a rule."""
        filled = _fill_template(rule.tool_params_template, template_ctx)
        # Try to parse strings like "[150, 200]" into lists
        if isinstance(filled, dict):
            filled = {
                k: _parse_range(v) if isinstance(v, str) else v
                for k, v in filled.items()
            }

        self._last_rule = rule
        self._last_context = original_context
        return QuantifiableAdjective(
            adjective=adjective,
            quantified=True,
            quantified_value=filled,
            route_to=None,  # Already quantified, no need to route to deep
        )

    def _quantify_with_llm(
        self,
        adjective: str,
        template_ctx: dict[str, Any],
        original_context: dict[str, Any],
    ) -> QuantifiableAdjective:
        """Unknown adjective falls back to LLM quantification."""
        system_prompt = (
            "你是一个形容词量化助手。将用户输入的主观判断词转化为可执行的工具调用参数。"
            "输出必须是 JSON 格式，包含字段：\n"
            "- tool_params: dict, 工具参数\n"
            "- strategy: str, 量化策略描述\n"
            "- explanation: str, 量化依据\n"
            "注意：禁止粗暴处理（如\"性价比高\"=>\"价格最低\"），必须综合考虑多个维度。"
        )
        user_prompt = (
            f"形容词: {adjective}\n"
            f"上下文: {json.dumps(template_ctx, ensure_ascii=False, default=str)}\n"
            f"原始上下文: {json.dumps(original_context, ensure_ascii=False, default=str)}\n"
            "请输出 JSON。"
        )
        response = self.llm_client.chat(system_prompt, user_prompt)

        # Parse LLM response - extract JSON from response (may be wrapped in ```json blocks)
        data = self._extract_json(response)
        if data is not None:
            tool_params = data.get("tool_params", data)
            strategy = data.get("strategy", "llm_generated")
            explanation = data.get("explanation", "")
        else:
            tool_params = {"note": "LLM 返回解析失败", "raw": response}
            strategy = "llm_fallback"
            explanation = "LLM 返回无法解析，使用原始响应作为参考。"

        # Dynamically register as a rule (task 6.7) for later reuse
        new_rule = QuantificationRule(
            adjective=adjective,
            strategy=strategy,
            tool_params_template=tool_params,
            explanation=explanation,
        )
        self._rules[adjective] = new_rule
        self._last_rule = new_rule
        self._last_context = original_context

        return QuantifiableAdjective(
            adjective=adjective,
            quantified=True,
            quantified_value=tool_params,
            route_to=None,
        )

    # ------------------------------------------------------------------
    # Context-aware quantification (task 6.5)
    # ------------------------------------------------------------------

    def quantify_with_context(
        self,
        adjective: str,
        user_profile: UserProfile | None,
        current_context: dict[str, Any],
    ) -> QuantifiableAdjective:
        """Context-aware quantification.

        Combines user profile, previous-round candidate attributes and other context
        to infer the user's understanding of the adjective.

        Args:
            adjective: Subjective judgment word
            user_profile: User profile (one of the three layers of context in D14)
            current_context: Previous-round candidate attributes, key facts, etc.

        Returns:
            QuantifiableAdjective
        """
        # Inject user profile information into context
        enriched_ctx = dict(current_context)
        if user_profile is not None:
            enriched_ctx["user_profile"] = user_profile
            enriched_ctx["user_industry"] = user_profile.industry
            # Flatten user preferences into context
            if user_profile.preferences:
                enriched_ctx["user_preference"] = user_profile.preferences
            # Topic tendencies may affect quantification strategy
            if user_profile.topic_tendencies:
                enriched_ctx["topic_tendencies"] = user_profile.topic_tendencies

        # Call the base quantification method
        result = self.quantify(adjective, enriched_ctx)

        # Insufficient context detection: if the adjective requires a baseline
        # (e.g. "再高级一点" requires current_tier) but context does not provide it,
        # route to the deep stage
        if adjective in ("再高级一点", "再好一点") and not current_context.get("current_tier"):
            result.route_to = None  # Already attempted quantification, but flagged as missing baseline
            result.quantified_value = result.quantified_value or {}
            if isinstance(result.quantified_value, dict):
                result.quantified_value["warning"] = "缺少上一轮候选属性，建议触发澄清"

        return result

    # ------------------------------------------------------------------
    # Explainability (task 6.6)
    # ------------------------------------------------------------------

    def explain(
        self, adjective: str, result: QuantifiableAdjective
    ) -> str:
        """Generate a human-readable explanation of the quantification basis.

        Args:
            adjective: Adjective
            result: Quantification result

        Returns:
            Quantification basis explanation text
        """
        rule = self._last_rule or self._rules.get(adjective)
        if rule is None:
            return (
                f"形容词 \"{adjective}\" 未命中任何规则，"
                f"量化结果由 LLM 生成，参数：{result.quantified_value}。"
            )

        # Fill placeholders in explanation
        template_ctx = _build_context_for_template(self._last_context)
        explanation = _fill_template(rule.explanation, template_ctx)

        parts = [
            f"形容词：{adjective}",
            f"量化策略：{rule.strategy}",
            f"工具参数：{result.quantified_value}",
            f"量化依据：{explanation}",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Dynamic rule registration (task 6.7, integrated with vocabulary)
    # ------------------------------------------------------------------

    def register_rule(self, adjective: str, rule: QuantificationRule) -> None:
        """Register a new quantification rule (task 6.7).

        Allows dynamic loading of rules from the vocabulary.
        When quantification rule entries exist in the vocabulary, they take priority.

        Args:
            adjective: Adjective
            rule: QuantificationRule instance
        """
        if rule.adjective != adjective:
            # Keep adjective consistent with rule.adjective
            rule = rule.model_copy(update={"adjective": adjective})
        self._rules[adjective] = rule

    def list_rules(self) -> dict[str, QuantificationRule]:
        """List all currently registered rules (for debugging)."""
        return dict(self._rules)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract JSON object from text (handles ```json code blocks)."""
        import re

        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting from ```json ... ``` block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _is_price_sensitive(profile: UserProfile) -> bool:
        """Determine whether the user is price-sensitive."""
        # Explicitly marked in preferences
        if profile.preferences.get("price_sensitive") is True:
            return True
        if profile.preferences.get("price_sensitive") == "true":
            return True
        # Topic tendencies contain keywords like "便宜", "低价" multiple times
        tendencies_str = " ".join(profile.topic_tendencies.values()).lower()
        if any(kw in tendencies_str for kw in ("便宜", "低价", "划算", "省钱")):
            return True
        return False
