"""Adjective quantification rule definitions (corresponding to task 6.4 / D13 Spec judgment rules).

Transforms subjective judgment words (e.g. "性价比", "再高级一点", "更便宜", "更好") into
executable tool-call parameter templates.
Each rule contains:
- adjective: the adjective itself
- strategy: quantification strategy description (e.g. "same price better quality", "relative upgrade")
- tool_params_template: tool parameter template, supports placeholders like {current_price}
- explanation: explainability note (why these parameters were chosen)

Note: Rules only define the "template"; actual value filling is done by engine.py at runtime based on context.
"""

from __future__ import annotations

from ..models import QuantificationRule


# ---------------------------------------------------------------------------
# Default Spec judgment rules (task 6.4)
# ---------------------------------------------------------------------------

# "性价比" / "划算": two strategies
# Strategy 1: at the same price, better quality, configuration, service or experience (more reasonable interpretation)
# Strategy 2: at the same quality, lower price
# Default uses Strategy 1 (more reasonable); engine can switch to Strategy 2 based on user profile
_XINGJIABI_RULE_PRIMARY = QuantificationRule(
    adjective="性价比",
    strategy="same_price_better_quality",
    tool_params_template={
        "price_range": "[{current_price_lower}, {current_price_upper}]",
        "quality_rank": "top 30%",
        "quality_weight": "higher",
        "brand_tier": "mid",
        "sort_by": "quality_desc",
    },
    explanation=(
        "同等价格下，质量、配置、服务或体验更好。"
        "参考上一轮候选价格 {current_price}，给出 ±10% 的价格区间，"
        "并要求质量排名前 30%。"
    ),
)

_XINGJIABI_RULE_SECONDARY = QuantificationRule(
    adjective="性价比",
    strategy="same_quality_lower_price",
    tool_params_template={
        "price_range": "[{current_price_lower}, {current_price_upper}]",
        "quality_rank": "top 50%",
        "brand_tier": "same",
        "sort_by": "price_asc",
    },
    explanation=(
        "同等质量下，价格更低。"
        "参考上一轮候选价格 {current_price}，下浮约 25% 给出价格区间，"
        "保持品牌等级不变。"
    ),
)

_HUASUAN_RULE = QuantificationRule(
    adjective="划算",
    strategy="same_price_better_quality",
    tool_params_template={
        "price_range": "[{current_price_lower}, {current_price_upper}]",
        "quality_rank": "top 30%",
        "quality_weight": "higher",
        "brand_tier": "mid",
        "sort_by": "quality_desc",
    },
    explanation="\"划算\"等同于\"性价比\"，采用同等价格更好质量策略。",
)

# "再高级一点" / "再好一点": relative upgrade strategy (tier + 1)
_ZAIGAOJI_RULE = QuantificationRule(
    adjective="再高级一点",
    strategy="relative_upgrade",
    tool_params_template={
        "tier": "{current_tier_plus_1}",
        "min_config_level": 2,
        "brand_tier": "high",
        "price_range": "[{current_price}, {current_price_upper}]",
    },
    explanation=(
        "相对当前层级升级一级。"
        "参考当前 tier={current_tier}，升级至 tier={current_tier_plus_1}，"
        "价格区间上限可适当上浮。"
    ),
)

_ZAIHAO_RULE = QuantificationRule(
    adjective="再好一点",
    strategy="relative_upgrade",
    tool_params_template={
        "tier": "{current_tier_plus_1}",
        "quality_rank": "top 20%",
        "min_config_level": 2,
        "brand_tier": "high",
    },
    explanation="\"再好一点\"等同于\"再高级一点\"，相对当前层级升级一级。",
)

# "更便宜": price reduction strategy
_GENGBIANYI_RULE = QuantificationRule(
    adjective="更便宜",
    strategy="price_reduction",
    tool_params_template={
        "price_range": "[{current_price_lower}, {current_price}]",
        "quality_rank": "top 50%",
        "brand_tier": "same",
        "sort_by": "price_asc",
    },
    explanation=(
        "在当前价格基础上下浮约 25%，保持品牌等级与质量排名前 50%。"
    ),
)

# "更好": quality improvement strategy
_GENGHAO_RULE = QuantificationRule(
    adjective="更好",
    strategy="quality_improvement",
    tool_params_template={
        "quality_rank": "top 20%",
        "quality_weight": "higher",
        "brand_tier": "high",
        "min_config_level": 2,
        "price_range": "[{current_price}, {current_price_upper}]",
    },
    explanation=(
        "提升质量排名至前 20%，品牌等级提升至 high，价格区间上限可适当上浮。"
    ),
)


# ---------------------------------------------------------------------------
# Default rule dictionary (task 6.4)
# ---------------------------------------------------------------------------

DEFAULT_RULES: dict[str, QuantificationRule] = {
    "性价比": _XINGJIABI_RULE_PRIMARY,
    "划算": _HUASUAN_RULE,
    "再高级一点": _ZAIGAOJI_RULE,
    "再好一点": _ZAIHAO_RULE,
    "更便宜": _GENGBIANYI_RULE,
    "更好": _GENGHAO_RULE,
}

# Alternative strategy for "性价比" (same quality, lower price)
ALTERNATIVE_RULES: dict[str, QuantificationRule] = {
    "性价比": _XINGJIABI_RULE_SECONDARY,
    "划算": _XINGJIABI_RULE_SECONDARY,
}


def get_rule(adjective: str) -> QuantificationRule | None:
    """Get the default quantification rule for an adjective.

    Args:
        adjective: Adjective, e.g. "性价比", "再高级一点"

    Returns:
        Matching QuantificationRule, or None if no match
    """
    return DEFAULT_RULES.get(adjective)


def get_alternative_rule(adjective: str) -> QuantificationRule | None:
    """Get the alternative strategy rule (e.g. "same quality lower price" for "性价比").

    Args:
        adjective: Adjective

    Returns:
        Alternative QuantificationRule, or None if no alternative
    """
    return ALTERNATIVE_RULES.get(adjective)
