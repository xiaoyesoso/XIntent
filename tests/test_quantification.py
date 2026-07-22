"""Quantification engine tests (corresponding to task 6.8)."""

from __future__ import annotations

import json

import pytest

from user_input_normalization.config import Config
from user_input_normalization.llm.mock import MockLLMClient
from user_input_normalization.models import (
    QuantifiableAdjective,
    QuantificationRule,
    UserProfile,
)
from user_input_normalization.quantification import (
    DEFAULT_RULES,
    QuantificationEngine,
    get_alternative_rule,
    get_rule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_client() -> MockLLMClient:
    """Provide a MockLLMClient with an "unknown adjective" handler registered."""
    client = MockLLMClient()

    def handler(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "tool_params": {
                    "custom_param": "llm_generated",
                    "note": "由 LLM 动态生成",
                },
                "strategy": "llm_generated",
                "explanation": "未知形容词，由 LLM 推断生成参数。",
            },
            ensure_ascii=False,
        )

    client.set_default_handler(handler)
    return client


@pytest.fixture
def engine(llm_client: MockLLMClient) -> QuantificationEngine:
    return QuantificationEngine(llm_client=llm_client)


# ---------------------------------------------------------------------------
# rules.py tests
# ---------------------------------------------------------------------------


class TestRules:
    def test_default_rules_contains_key_adjectives(self) -> None:
        """Default rules should contain core adjectives."""
        for adj in ("性价比", "划算", "再高级一点", "再好一点", "更便宜", "更好"):
            assert adj in DEFAULT_RULES, f"缺少默认规则：{adj}"

    def test_get_rule_returns_matched(self) -> None:
        rule = get_rule("性价比")
        assert rule is not None
        assert rule.adjective == "性价比"
        assert "price_range" in rule.tool_params_template

    def test_get_rule_returns_none_for_unknown(self) -> None:
        assert get_rule("不存在的词") is None

    def test_get_alternative_rule_for_xingjiabi(self) -> None:
        """性价比 should have an alternative strategy (same quality lower price)."""
        alt = get_alternative_rule("性价比")
        assert alt is not None
        assert alt.strategy == "same_quality_lower_price"

    def test_rule_has_explanation(self) -> None:
        """Each rule should have an explainability note."""
        for adj, rule in DEFAULT_RULES.items():
            assert rule.explanation, f"规则 {adj} 缺少 explanation"


# ---------------------------------------------------------------------------
# quantify tests (tasks 6.1-6.3)
# ---------------------------------------------------------------------------


class TestQuantify:
    def test_quantify_xingjiabi_with_current_price(self, engine: QuantificationEngine) -> None:
        """性价比 quantification: generate price range based on previous-round candidate price."""
        result = engine.quantify("性价比", {"current_price": 200})
        assert result.adjective == "性价比"
        assert result.quantified is True
        assert result.quantified_value is not None
        # Default strategy: same price better quality, price range ±10%
        value = result.quantified_value
        assert "price_range" in value
        price_range = value["price_range"]
        # Should be [180.0, 220.0]
        assert isinstance(price_range, list)
        assert price_range[0] == 180.0
        assert price_range[1] == 220.0
        # Quality rank should be top 30%
        assert value["quality_rank"] == "top 30%"

    def test_quantify_xingjiabi_two_strategies(self, engine: QuantificationEngine) -> None:
        """Both strategies of 性价比 are accessible."""
        # Default strategy
        result_default = engine.quantify("性价比", {"current_price": 200})
        assert result_default.quantified_value is not None
        assert result_default.quantified_value["quality_rank"] == "top 30%"

        # Alternative strategy: triggered by price-sensitive user profile
        price_sensitive_profile = UserProfile(
            user_id="u1",
            preferences={"price_sensitive": True},
        )
        result_alt = engine.quantify(
            "性价比",
            {"current_price": 200, "user_profile": price_sensitive_profile},
        )
        assert result_alt.quantified_value is not None
        # When price-sensitive, switch to "same quality lower price" strategy, quality rank top 50%
        assert result_alt.quantified_value["quality_rank"] == "top 50%"

    def test_quantify_zaigaoji_upgrade(self, engine: QuantificationEngine) -> None:
        """再高级一点: upgrade one tier relative to the current tier."""
        result = engine.quantify("再高级一点", {"current_tier": 2, "current_price": 200})
        assert result.quantified is True
        assert result.quantified_value is not None
        # tier should be 3 (current_tier + 1)
        assert result.quantified_value["tier"] == 3

    def test_quantify_gengbianyi_price_reduction(self, engine: QuantificationEngine) -> None:
        """更便宜: price reduction strategy."""
        result = engine.quantify("更便宜", {"current_price": 200})
        assert result.quantified is True
        assert result.quantified_value is not None
        price_range = result.quantified_value["price_range"]
        # Should be [150.0, 200.0] (down 25% to 150, upper bound is current price)
        assert isinstance(price_range, list)
        assert price_range[0] == 180.0  # current_price * 0.9
        assert price_range[1] == 200.0  # current_price (no upper expansion for 更便宜)

    def test_quantify_genghao_quality_improvement(self, engine: QuantificationEngine) -> None:
        """更好: quality improvement strategy."""
        result = engine.quantify("更好", {"current_price": 200})
        assert result.quantified is True
        assert result.quantified_value is not None
        assert result.quantified_value["quality_rank"] == "top 20%"
        assert result.quantified_value["brand_tier"] == "high"

    def test_quantify_unknown_adjective_falls_back_to_llm(
        self, engine: QuantificationEngine, llm_client: MockLLMClient
    ) -> None:
        """Unknown adjective falls back to LLM quantification."""
        result = engine.quantify("再卷一点", {"current_price": 100})
        assert result.quantified is True
        assert result.quantified_value is not None
        # LLM-generated parameters
        assert result.quantified_value.get("custom_param") == "llm_generated"
        # Should be registered as a rule (task 6.7)
        assert "再卷一点" in engine.list_rules()
        # Should have a call record
        assert len(llm_client.call_log) >= 1

    def test_quantify_result_is_quantifiable_adjective(self, engine: QuantificationEngine) -> None:
        """Quantification result should be a QuantifiableAdjective instance."""
        result = engine.quantify("性价比", {"current_price": 200})
        assert isinstance(result, QuantifiableAdjective)


# ---------------------------------------------------------------------------
# Context-aware quantification tests (task 6.5)
# ---------------------------------------------------------------------------


class TestQuantifyWithContext:
    def test_with_user_profile_industry(self, engine: QuantificationEngine) -> None:
        """Quantify combined with the user profile's industry."""
        profile = UserProfile(
            user_id="u1",
            industry="IT",
            preferences={"price_sensitive": False},
        )
        result = engine.quantify_with_context(
            "性价比", profile, {"current_price": 200}
        )
        assert result.quantified is True
        assert result.quantified_value is not None

    def test_price_sensitive_profile_switches_strategy(
        self, engine: QuantificationEngine
    ) -> None:
        """Price-sensitive user profile should switch 性价比 strategy to "same quality lower price"."""
        profile = UserProfile(
            user_id="u1",
            preferences={"price_sensitive": True},
        )
        result = engine.quantify_with_context(
            "性价比", profile, {"current_price": 200}
        )
        assert result.quantified_value is not None
        # Should switch to top 50% (alternative strategy)
        assert result.quantified_value["quality_rank"] == "top 50%"

    def test_missing_baseline_for_relative_upgrade(
        self, engine: QuantificationEngine
    ) -> None:
        """再高级一点 should give a warning when previous-round candidate attributes are missing."""
        profile = UserProfile(user_id="u1")
        result = engine.quantify_with_context(
            "再高级一点", profile, current_context={}
        )
        assert result.quantified_value is not None
        assert "warning" in result.quantified_value

    def test_no_user_profile(self, engine: QuantificationEngine) -> None:
        """Should quantify normally when user_profile is None."""
        result = engine.quantify_with_context(
            "更好", None, {"current_price": 200}
        )
        assert result.quantified is True


# ---------------------------------------------------------------------------
# Explainability tests (task 6.6)
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_returns_human_readable(self, engine: QuantificationEngine) -> None:
        """explain should return a human-readable quantification basis."""
        result = engine.quantify("性价比", {"current_price": 200})
        explanation = engine.explain("性价比", result)
        assert isinstance(explanation, str)
        assert "性价比" in explanation
        assert "量化策略" in explanation
        assert "量化依据" in explanation

    def test_explain_includes_strategy_and_params(
        self, engine: QuantificationEngine
    ) -> None:
        """Explanation should include the quantification strategy and tool parameters."""
        result = engine.quantify("再高级一点", {"current_tier": 1})
        explanation = engine.explain("再高级一点", result)
        assert "relative_upgrade" in explanation
        assert "tier" in explanation

    def test_explain_for_llm_generated_rule(
        self, engine: QuantificationEngine
    ) -> None:
        """LLM-generated rules should also be explainable."""
        result = engine.quantify("再卷一点", {"current_price": 100})
        explanation = engine.explain("再卷一点", result)
        assert isinstance(explanation, str)
        assert "再卷一点" in explanation


# ---------------------------------------------------------------------------
# Dynamic rule registration tests (task 6.7)
# ---------------------------------------------------------------------------


class TestRegisterRule:
    def test_register_new_rule(self, engine: QuantificationEngine) -> None:
        """After registering a new rule, it should be hit by quantification."""
        custom_rule = QuantificationRule(
            adjective="再奢华一点",
            strategy="luxury_upgrade",
            tool_params_template={
                "brand_tier": "luxury",
                "min_config_level": 3,
            },
            explanation="升级至奢华档位。",
        )
        engine.register_rule("再奢华一点", custom_rule)
        result = engine.quantify("再奢华一点", {"current_price": 500})
        assert result.quantified is True
        assert result.quantified_value is not None
        assert result.quantified_value["brand_tier"] == "luxury"

    def test_register_overrides_existing_rule(
        self, engine: QuantificationEngine
    ) -> None:
        """Registering a rule with the same name should override the default rule."""
        original = engine.quantify("更好", {"current_price": 200})
        assert original.quantified_value is not None
        assert original.quantified_value["quality_rank"] == "top 20%"

        new_rule = QuantificationRule(
            adjective="更好",
            strategy="custom_quality",
            tool_params_template={"quality_rank": "top 5%"},
            explanation="自定义质量提升规则。",
        )
        engine.register_rule("更好", new_rule)
        result = engine.quantify("更好", {"current_price": 200})
        assert result.quantified_value is not None
        assert result.quantified_value["quality_rank"] == "top 5%"

    def test_register_rule_adjective_mismatch_corrected(
        self, engine: QuantificationEngine
    ) -> None:
        """When adjective and rule.adjective are inconsistent on registration, should auto-correct."""
        rule = QuantificationRule(
            adjective="原形容词",
            strategy="test",
            tool_params_template={"k": "v"},
            explanation="测试",
        )
        engine.register_rule("新形容词", rule)
        registered = engine.list_rules().get("新形容词")
        assert registered is not None
        assert registered.adjective == "新形容词"
