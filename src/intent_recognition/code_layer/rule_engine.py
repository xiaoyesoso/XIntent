"""Rule engine (D3).

Rule-based matching that combines context state + normalized input.
Rules are callable functions: (text, context) -> IntentRecognitionResult | None.
"""

from __future__ import annotations

from typing import Any, Callable

from ..models import IntentRecognitionResult, RecognitionSource


# Type alias: rule function
RuleFunc = Callable[[str, dict[str, Any]], IntentRecognitionResult | None]


def _make_result(intent: str, signals_key: str = "rule_match") -> IntentRecognitionResult:
    """Build a code-layer match result."""
    return IntentRecognitionResult(
        intent=intent,
        confidence=1.0,
        source=RecognitionSource.CODE_LAYER,
        layer_reached=1,
        signals={signals_key: 1.0},
    )


def _builtin_continue_rule(text: str, context: dict[str, Any]) -> IntentRecognitionResult | None:
    """Match "继续" or variations when context indicates an active flow."""
    stripped = text.strip()
    # Only trigger continue rule if context says we are mid-flow
    if stripped == "继续" and context.get("in_flow") is True:
        return _make_result("continue")
    return None


def _builtin_order_query_rule(
    text: str, context: dict[str, Any]
) -> IntentRecognitionResult | None:
    """Match "查询订单 ..." regex pattern."""
    import re

    if re.search(r"查询订单\s*\S*", text):
        return _make_result("order_query")
    return None


class RuleEngine:
    """Rule-based intent matcher.

    Rules are functions that take (text, context) and return an
    IntentRecognitionResult or None. Rules are evaluated in insertion order;
    the first non-None result wins.

    Built-in rules:
      - "继续" -> continue intent (when context["in_flow"] is True)
      - "查询订单.*" -> order_query intent
    """

    def __init__(self, rules: list[RuleFunc] | None = None) -> None:
        # Built-in rules first, then custom rules
        self._rules: list[RuleFunc] = [
            _builtin_continue_rule,
            _builtin_order_query_rule,
        ]
        if rules:
            self._rules.extend(rules)

    def add_rule(self, rule_func: RuleFunc) -> None:
        """Register a custom rule at the end of the rule chain."""
        self._rules.append(rule_func)

    def match(
        self, text: str, context: dict[str, Any] | None = None
    ) -> IntentRecognitionResult | None:
        """Evaluate rules in order; return first match or None."""
        if not text:
            return None
        ctx = context or {}
        for rule in self._rules:
            try:
                result = rule(text, ctx)
            except Exception:
                # Defensive: a misbehaving rule should not break the chain
                continue
            if result is not None:
                return result
        return None
