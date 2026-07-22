"""Tests for the D23 multi-recognizer arbiter."""

import pytest

from intent_recognition.config import ArbiterConfig
from intent_recognition.models import (
    IntentRecognitionResult,
    RecognitionSource,
)
from intent_recognition.multi_recognizer_arbiter import (
    MultiRecognizerArbiter,
    Recognizer,
)


def _make_result(
    intent: str | None,
    confidence: float = 0.9,
    source: RecognitionSource = RecognitionSource.CODE_LAYER,
) -> IntentRecognitionResult:
    """Build a minimal IntentRecognitionResult for tests."""
    return IntentRecognitionResult(
        intent=intent,
        confidence=confidence,
        source=source,
        layer_reached=1,
    )


def _constant_recognizer(intent: str | None, confidence: float = 0.9) -> Recognizer:
    """Build a recognizer that always returns the same intent."""

    def _recognizer(text: str, context: dict | None = None) -> IntentRecognitionResult | None:
        return _make_result(intent, confidence)

    return _recognizer


def _none_recognizer() -> Recognizer:
    """Build a recognizer that always returns None (no opinion)."""

    def _recognizer(text: str, context: dict | None = None) -> IntentRecognitionResult | None:
        return None

    return _recognizer


def _make_vote_config(**overrides) -> ArbiterConfig:
    """Build an enabled vote-mode ArbiterConfig with 3 recognizers."""
    defaults = {
        "enable": True,
        "mode": "vote",
        "recognizers": ["vector", "rule", "lightweight_llm"],
        "weights": {"vector": 0.8, "rule": 0.6, "lightweight_llm": 1.0},
    }
    defaults.update(overrides)
    return ArbiterConfig(**defaults)


class TestArbiterVoteMode:
    """Tests for vote-mode arbitration."""

    def test_clear_winner_returns_arbiter_vote_result(self):
        """2 of 3 recognizers vote for A -> A wins via ARBITER_VOTE."""
        cfg = _make_vote_config()
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": _constant_recognizer("product_recommendation", 0.8),
            "rule": _constant_recognizer("product_recommendation", 0.9),
            "lightweight_llm": _constant_recognizer("order_query", 0.95),
        })

        result = arbiter.arbitrate("帮我推荐手机")
        assert result is not None
        assert result.intent == "product_recommendation"
        assert result.source == RecognitionSource.ARBITER_VOTE
        # confidence = avg of the supporting recognizers (0.8, 0.9)
        assert result.confidence == pytest.approx((0.8 + 0.9) / 2)
        assert result.layer_reached == 1
        assert result.signals.get("arbiter_votes") == 2.0

    def test_tie_returns_none(self):
        """1 vs 1 vote (with 2 recognizers) is a tie -> None."""
        cfg = _make_vote_config(
            recognizers=["a", "b"],
            weights={"a": 0.5, "b": 0.5},
        )
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "a": _constant_recognizer("intent_a"),
            "b": _constant_recognizer("intent_b"),
        })

        assert arbiter.arbitrate("hello") is None

    def test_three_way_tie_returns_none(self):
        """All 3 recognizers disagree -> 1-vs-1-vs-1 tie -> None."""
        cfg = _make_vote_config()
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": _constant_recognizer("a"),
            "rule": _constant_recognizer("b"),
            "lightweight_llm": _constant_recognizer("c"),
        })

        assert arbiter.arbitrate("hello") is None


class TestArbiterWeightedMode:
    """Tests for weighted_score-mode arbitration."""

    def test_weighted_score_picks_highest_weighted_intent(self):
        """Even with 1 vs 2 votes, the higher-weighted intent wins.

        Scenario:
          - vector (weight 0.8) votes intent_a with confidence 1.0
                                   -> score 0.8
          - rule (weight 0.6) votes intent_b with confidence 1.0
                                                 -> score 0.6
          - lightweight_llm (weight 1.0) votes intent_b with confidence 1.0
                                                           -> score 1.0
        intent_b total: 1.6, intent_a total: 0.8 -> intent_b wins.
        confidence = 1.6 / (0.8 + 0.6 + 1.0) = 1.6 / 2.4
        """
        cfg = _make_vote_config(mode="weighted_score")
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": _constant_recognizer("intent_a", 1.0),
            "rule": _constant_recognizer("intent_b", 1.0),
            "lightweight_llm": _constant_recognizer("intent_b", 1.0),
        })

        result = arbiter.arbitrate("hello")
        assert result is not None
        assert result.intent == "intent_b"
        assert result.source == RecognitionSource.ARBITER_WEIGHTED
        assert result.confidence == pytest.approx(1.6 / 2.4)
        assert result.signals.get("arbiter_score") == pytest.approx(1.6)
        assert result.signals.get("arbiter_total_weight") == pytest.approx(2.4)

    def test_weighted_score_confidence_clamped_to_one(self):
        """Confidence must never exceed 1.0 even with high weights."""
        cfg = _make_vote_config(
            mode="weighted_score",
            recognizers=["a", "b"],
            weights={"a": 5.0, "b": 5.0},
        )
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "a": _constant_recognizer("intent_a", 1.0),
            "b": _constant_recognizer("intent_a", 1.0),
        })

        result = arbiter.arbitrate("hello")
        assert result is not None
        assert result.intent == "intent_a"
        # score = 5 + 5 = 10, total_weight = 10, raw ratio = 1.0
        assert result.confidence <= 1.0
        assert result.confidence == pytest.approx(1.0)


class TestArbiterGuardClauses:
    """Tests for the early-exit / validation paths."""

    def test_disabled_config_returns_none(self):
        """When enable=False the arbiter is a no-op."""
        cfg = _make_vote_config(enable=False)
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": _constant_recognizer("a"),
            "rule": _constant_recognizer("a"),
            "lightweight_llm": _constant_recognizer("a"),
        })

        assert arbiter.arbitrate("hello") is None

    def test_fewer_than_two_recognizers_returns_none(self):
        """Only one configured recognizer -> no arbitration possible."""
        cfg = _make_vote_config(recognizers=["only_one"])
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "only_one": _constant_recognizer("a"),
        })

        assert arbiter.arbitrate("hello") is None

    def test_unknown_recognizer_name_raises_value_error(self):
        """A configured name with no registered callable must raise."""
        cfg = _make_vote_config(recognizers=["vector", "missing"])
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": _constant_recognizer("a"),
        })

        with pytest.raises(ValueError, match="Unknown recognizer: missing"):
            arbiter.arbitrate("hello")

    def test_validate_recognizers_raises_for_unknown_name(self):
        """validate_recognizers() surfaces configuration errors early."""
        cfg = _make_vote_config(recognizers=["vector", "ghost"])
        arbiter = MultiRecognizerArbiter(cfg, recognizers={"vector": _constant_recognizer("a")})

        with pytest.raises(ValueError, match="Unknown recognizer: ghost"):
            arbiter.validate_recognizers()

    def test_only_one_recognizer_returns_result_directly(self):
        """When exactly one recognizer has an opinion, skip arbitration.

        We configure 3 recognizers but only one returns non-None; the
        arbiter should return that recognizer's result directly (no
        ARBITER_VOTE / ARBITER_WEIGHTED source mutation).
        """
        cfg = _make_vote_config()
        lone_result = _make_result("lone_intent", confidence=0.7)
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": lambda text, ctx=None: lone_result,
            "rule": _none_recognizer(),
            "lightweight_llm": _none_recognizer(),
        })

        result = arbiter.arbitrate("hello")
        assert result is lone_result
        assert result.intent == "lone_intent"
        assert result.confidence == 0.7

    def test_all_recognizers_return_none_yields_none(self):
        """When every recognizer abstains, the arbiter has nothing to say."""
        cfg = _make_vote_config()
        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": _none_recognizer(),
            "rule": _none_recognizer(),
            "lightweight_llm": _none_recognizer(),
        })

        assert arbiter.arbitrate("hello") is None


class TestArbiterRegistration:
    """Tests for the register() API."""

    def test_register_adds_callable(self):
        """register() makes a new callable available by name."""
        cfg = _make_vote_config(recognizers=["a", "b"], weights={"a": 1.0, "b": 1.0})
        arbiter = MultiRecognizerArbiter(cfg)
        arbiter.register("a", _constant_recognizer("intent_a"))
        arbiter.register("b", _constant_recognizer("intent_a"))

        result = arbiter.arbitrate("hello")
        assert result is not None
        assert result.intent == "intent_a"

    def test_register_overwrites_existing_callable(self):
        """Re-registering a name replaces the previous callable."""
        cfg = _make_vote_config(recognizers=["a", "b"], weights={"a": 1.0, "b": 1.0})
        arbiter = MultiRecognizerArbiter(cfg)
        arbiter.register("a", _constant_recognizer("intent_a"))
        arbiter.register("b", _constant_recognizer("intent_b"))
        # First run -> tie
        assert arbiter.arbitrate("hello") is None
        # Re-register b to also vote intent_a
        arbiter.register("b", _constant_recognizer("intent_a"))
        result = arbiter.arbitrate("hello")
        assert result is not None
        assert result.intent == "intent_a"


class TestArbiterDefensiveBehavior:
    """Tests for graceful handling of misbehaving recognizers."""

    def test_misbehaving_recognizer_skipped(self):
        """A recognizer that raises is treated as 'no opinion'."""
        cfg = _make_vote_config()

        def boom(text, ctx=None):
            raise RuntimeError("kaboom")

        arbiter = MultiRecognizerArbiter(cfg, recognizers={
            "vector": boom,
            "rule": _constant_recognizer("intent_a"),
            "lightweight_llm": _constant_recognizer("intent_a"),
        })

        result = arbiter.arbitrate("hello")
        assert result is not None
        assert result.intent == "intent_a"
        # 2 of the 3 (surviving) recognizers voted intent_a.
        assert result.source == RecognitionSource.ARBITER_VOTE
