"""Tests for the code layer (Layer 1) - D3.

Covers PageGuidanceMatcher, KeywordMatcher, RuleEngine, and the
orchestrating CodeLayerClassifier. All matches must set source=CODE_LAYER,
confidence=1.0, layer_reached=1, and produce zero LLM calls.
"""

from intent_recognition.code_layer import (
    CodeLayerClassifier,
    KeywordMatcher,
    PageGuidanceMatcher,
    RuleEngine,
)
from intent_recognition.models import RecognitionSource


# ---------------------------------------------------------------------------
# Page guidance matcher
# ---------------------------------------------------------------------------


class TestPageGuidanceMatcher:
    """UI event -> intent mapping."""

    def test_click_next_event_matches_continue(self):
        m = PageGuidanceMatcher()
        result = m.match("click:next")
        assert result is not None
        assert result.intent == "continue"
        assert result.confidence == 1.0
        assert result.source == RecognitionSource.CODE_LAYER
        assert result.layer_reached == 1

    def test_page_navigation_matches_intent(self):
        m = PageGuidanceMatcher()
        result = m.match("page:order_list")
        assert result is not None
        assert result.intent == "order_query"

    def test_no_event_returns_none(self):
        """Pass-through when no UI event is provided."""
        m = PageGuidanceMatcher()
        assert m.match(None) is None
        assert m.match("") is None

    def test_unknown_event_returns_none(self):
        m = PageGuidanceMatcher()
        assert m.match("click:unknown_button") is None

    def test_context_is_accepted_but_not_required(self):
        m = PageGuidanceMatcher()
        result = m.match("click:submit", context={"session_id": "s1"})
        assert result is not None
        assert result.intent == "submit"

    def test_register_custom_event(self):
        m = PageGuidanceMatcher()
        m.register_event("click:like", "like_product")
        result = m.match("click:like")
        assert result is not None
        assert result.intent == "like_product"


# ---------------------------------------------------------------------------
# Keyword matcher
# ---------------------------------------------------------------------------


class TestKeywordMatcher:
    """Keyword and regex matching for simple inputs."""

    def test_keyword_continue_matches(self):
        m = KeywordMatcher()
        result = m.match("继续")
        assert result is not None
        assert result.intent == "continue"
        assert result.confidence == 1.0
        assert result.source == RecognitionSource.CODE_LAYER
        assert result.layer_reached == 1

    def test_keyword_next_step_matches_continue(self):
        m = KeywordMatcher()
        result = m.match("下一步")
        assert result is not None
        assert result.intent == "continue"

    def test_keyword_with_whitespace_stripped(self):
        m = KeywordMatcher()
        result = m.match("  继续  ")
        assert result is not None
        assert result.intent == "continue"

    def test_regex_order_query_matches(self):
        m = KeywordMatcher()
        result = m.match("查询订单 12345")
        assert result is not None
        assert result.intent == "order_query"

    def test_regex_refund_matches(self):
        m = KeywordMatcher()
        result = m.match("退款 12345")
        assert result is not None
        assert result.intent == "refund"

    def test_no_match_returns_none(self):
        """No keyword/regex match -> pass to L2."""
        m = KeywordMatcher()
        assert m.match("帮我推荐一个适合学生用的手机") is None

    def test_empty_text_returns_none(self):
        m = KeywordMatcher()
        assert m.match("") is None

    def test_keyword_takes_priority_over_regex(self):
        """Keyword exact match wins before regex search."""
        m = KeywordMatcher()
        # "继续" is a keyword; even if a regex could match, keyword wins
        result = m.match("继续")
        assert result is not None
        assert result.intent == "continue"
        assert result.signals.get("keyword_match") == 1.0

    def test_register_custom_keyword(self):
        m = KeywordMatcher()
        m.register_keyword("好的呀", "confirm")
        result = m.match("好的呀")
        assert result is not None
        assert result.intent == "confirm"

    def test_register_custom_regex(self):
        m = KeywordMatcher()
        m.register_regex(r"购买\s*\d+\s*件", "purchase")
        result = m.match("购买 3 件商品")
        assert result is not None
        assert result.intent == "purchase"


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------


class TestRuleEngine:
    """Context-aware rule matching on normalized input."""

    def test_continue_rule_requires_in_flow_context(self):
        """The '继续' rule only fires when context['in_flow'] is True."""
        engine = RuleEngine()
        # Without context flag -> no match (falls through to L2)
        assert engine.match("继续", context={}) is None
        # With in_flow=True -> match
        result = engine.match("继续", context={"in_flow": True})
        assert result is not None
        assert result.intent == "continue"
        assert result.confidence == 1.0
        assert result.source == RecognitionSource.CODE_LAYER

    def test_order_query_rule_regex_match(self):
        engine = RuleEngine()
        result = engine.match("查询订单 12345", context={})
        assert result is not None
        assert result.intent == "order_query"

    def test_no_rule_matches_returns_none(self):
        engine = RuleEngine()
        assert engine.match("随便说点什么", context={}) is None

    def test_add_custom_rule(self):
        engine = RuleEngine()

        def my_rule(text: str, context: dict) -> None:
            if text.strip() == "帮我推荐手机":
                from intent_recognition.models import IntentRecognitionResult
                return IntentRecognitionResult(
                    intent="product_recommendation",
                    confidence=1.0,
                    source=RecognitionSource.CODE_LAYER,
                    layer_reached=1,
                )
            return None

        engine.add_rule(my_rule)
        result = engine.match("帮我推荐手机", context={})
        assert result is not None
        assert result.intent == "product_recommendation"

    def test_first_matching_rule_wins(self):
        engine = RuleEngine()

        def first_rule(text, context):
            if text == "ping":
                from intent_recognition.models import IntentRecognitionResult
                return IntentRecognitionResult(
                    intent="first",
                    confidence=1.0,
                    source=RecognitionSource.CODE_LAYER,
                    layer_reached=1,
                )
            return None

        def second_rule(text, context):
            if text == "ping":
                from intent_recognition.models import IntentRecognitionResult
                return IntentRecognitionResult(
                    intent="second",
                    confidence=1.0,
                    source=RecognitionSource.CODE_LAYER,
                    layer_reached=1,
                )
            return None

        engine.add_rule(first_rule)
        engine.add_rule(second_rule)
        result = engine.match("ping", context={})
        assert result is not None
        assert result.intent == "first"

    def test_misbehaving_rule_does_not_break_chain(self):
        """A rule that raises should be skipped, not crash the engine."""
        engine = RuleEngine()

        def bad_rule(text, context):
            raise RuntimeError("boom")

        def good_rule(text, context):
            if text == "hello":
                from intent_recognition.models import IntentRecognitionResult
                return IntentRecognitionResult(
                    intent="greet",
                    confidence=1.0,
                    source=RecognitionSource.CODE_LAYER,
                    layer_reached=1,
                )
            return None

        engine.add_rule(bad_rule)
        engine.add_rule(good_rule)
        result = engine.match("hello", context={})
        assert result is not None
        assert result.intent == "greet"

    def test_rule_engine_works_on_normalized_text(self):
        """Rule engine operates on the normalized input (whitespace stripped).

        This is the 'normalized input dependency' requirement: the rule engine
        should treat its input as already-normalized text, not raw user input.
        """
        engine = RuleEngine()
        # Stripped normalized input
        result = engine.match("查询订单ABC999", context={})
        assert result is not None
        assert result.intent == "order_query"


# ---------------------------------------------------------------------------
# Orchestrating classifier
# ---------------------------------------------------------------------------


class TestCodeLayerClassifier:
    """CodeLayerClassifier orchestrates the three matchers."""

    def test_button_click_next_step_matched(self):
        """Button click '下一步' -> matched intent (via page guidance)."""
        clf = CodeLayerClassifier()
        result = clf.classify(text="", event="click:next")
        assert result is not None
        assert result.intent == "continue"
        assert result.confidence == 1.0
        assert result.source == RecognitionSource.CODE_LAYER
        assert result.layer_reached == 1

    def test_keyword_continue_matched(self):
        """Keyword '继续' -> continue intent."""
        clf = CodeLayerClassifier()
        result = clf.classify(text="继续")
        assert result is not None
        assert result.intent == "continue"

    def test_regex_order_query_matched(self):
        """Regex match '查询订单 12345' -> order_query."""
        clf = CodeLayerClassifier()
        result = clf.classify(text="查询订单 12345")
        assert result is not None
        assert result.intent == "order_query"

    def test_rule_engine_with_context_state(self):
        """Rule engine fires when context state is set."""
        clf = CodeLayerClassifier()
        result = clf.classify(text="继续", context={"in_flow": True})
        assert result is not None
        assert result.intent == "continue"

    def test_no_match_returns_none(self):
        """Unmatched input returns None (signal to escalate to L2)."""
        clf = CodeLayerClassifier()
        result = clf.classify(text="帮我推荐一个适合学生用的手机")
        assert result is None

    def test_page_guidance_takes_priority_over_keyword(self):
        """When both event and text are provided, event wins."""
        clf = CodeLayerClassifier()
        # Text "继续" would match keyword, but event should take priority
        result = clf.classify(text="继续", event="page:cart")
        assert result is not None
        assert result.intent == "view_cart"

    def test_keyword_takes_priority_over_rule_engine(self):
        """Keyword match fires before rule engine."""
        clf = CodeLayerClassifier()
        # "继续" matches keyword directly (no need for in_flow context)
        result = clf.classify(text="继续", context={"in_flow": False})
        assert result is not None
        assert result.intent == "continue"

    def test_zero_llm_calls(self):
        """Code layer must never call the LLM."""
        # We verify by ensuring no LLMClient is even referenced by the
        # classifier. The classifier has no llm_client attribute.
        clf = CodeLayerClassifier()
        assert not hasattr(clf, "_llm")
        assert not hasattr(clf, "llm_client")

    def test_all_matches_set_required_fields(self):
        """Every match sets source, confidence, layer_reached correctly."""
        clf = CodeLayerClassifier()
        for text, event in [("继续", None), ("", "click:next"), ("查询订单 1", None)]:
            result = clf.classify(text=text, event=event)
            assert result is not None
            assert result.source == RecognitionSource.CODE_LAYER
            assert result.confidence == 1.0
            assert result.layer_reached == 1

    def test_performance_under_10ms(self):
        """Code layer should complete in < 10ms (target: deterministic, fast)."""
        import time

        clf = CodeLayerClassifier()
        # Warm up
        clf.classify(text="继续")
        # Measure
        start = time.perf_counter()
        for _ in range(100):
            clf.classify(text="继续")
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 100
        # Generous threshold: well under 10ms per call on any modern machine
        assert avg_ms < 10.0, f"Average {avg_ms:.3f}ms exceeds 10ms target"
