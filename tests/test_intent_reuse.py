"""Tests for D22: intent reuse strategy with rollback."""

from __future__ import annotations

import pytest

from intent_recognition import IntentRecognitionResult, RecognitionSource
from intent_recognition.config import ReuseStrategyConfig
from intent_recognition.intent_reuse_strategy import (
    INTENT_SWITCH_MARKERS,
    IntentReuseStrategy,
)
from intent_recognition.storage import MemoryIntentHistoryStore


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _seed_history(
    store: MemoryIntentHistoryStore,
    session_id: str,
    intent: str,
    turn: int = 1,
) -> None:
    """Add a recognition result with the given intent to the history store."""
    result = IntentRecognitionResult(
        intent=intent,
        confidence=0.95,
        source=RecognitionSource.CODE_LAYER,
        layer_reached=1,
    )
    store.add(session_id, result, turn)


def _make_strategy(
    *,
    enable: bool = True,
    rollback_on_failure_signal: bool = True,
    rollback_on_tool_failure_count: int = 3,
    failure_signals: list[str] | None = None,
    history: MemoryIntentHistoryStore | None = None,
) -> IntentReuseStrategy:
    return IntentReuseStrategy(
        history_store=history or MemoryIntentHistoryStore(),
        config=ReuseStrategyConfig(
            enable=enable,
            rollback_on_failure_signal=rollback_on_failure_signal,
            rollback_on_tool_failure_count=rollback_on_tool_failure_count,
        ),
        failure_signals=failure_signals if failure_signals is not None
        else ["你理解错了", "不是这个意思"],
    )


# ----------------------------------------------------------------------
# try_reuse
# ----------------------------------------------------------------------
class TestTryReuse:
    def test_reuse_returns_previous_intent_when_enabled(self):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", "product_recommendation", turn=1)
        strategy = _make_strategy(history=store)

        result = strategy.try_reuse("s1", "再推荐一款类似的", turn=2)

        assert result is not None
        assert result.intent == "product_recommendation"
        assert result.source == RecognitionSource.REUSED
        assert result.confidence == 1.0
        assert result.layer_reached == 0
        assert result.signals == {"reused": 1.0}

    def test_reuse_returns_none_when_disabled(self):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", "product_recommendation", turn=1)
        strategy = _make_strategy(enable=False, history=store)

        assert strategy.try_reuse("s1", "继续推荐", turn=2) is None

    def test_reuse_returns_none_for_first_turn_no_history(self):
        store = MemoryIntentHistoryStore()
        strategy = _make_strategy(history=store)

        # First turn: no previous intent recorded yet.
        assert strategy.try_reuse("s1", "推荐手机", turn=1) is None

    def test_reuse_returns_none_when_intent_switch_marker_detected(self):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", "product_recommendation", turn=1)
        strategy = _make_strategy(history=store)

        # Each marker should suppress reuse.
        for marker in INTENT_SWITCH_MARKERS:
            text = f"{marker}，我想查订单"
            assert strategy.try_reuse("s1", text, turn=2) is None, (
                f"marker {marker!r} should suppress reuse"
            )

    def test_reuse_returns_none_when_failure_signal_detected(self):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", "product_recommendation", turn=1)
        strategy = _make_strategy(
            history=store,
            failure_signals=["你理解错了", "不是这个意思"],
        )

        # Failure signal in the text -> do not reuse.
        assert strategy.try_reuse("s1", "你理解错了，我是想查订单", turn=2) is None
        assert strategy.try_reuse("s1", "不是这个意思", turn=2) is None

    def test_reuse_no_failure_signals_allows_continuation(self):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", "order_query", turn=1)
        # Empty failure_signals list -> only markers block reuse.
        strategy = _make_strategy(history=store, failure_signals=[])

        result = strategy.try_reuse("s1", "再查一下另一个订单", turn=2)
        assert result is not None
        assert result.intent == "order_query"
        assert result.source == RecognitionSource.REUSED

    def test_enabled_property_reflects_config(self):
        strategy = _make_strategy(enable=True)
        assert strategy.enabled is True

        strategy = _make_strategy(enable=False)
        assert strategy.enabled is False


# ----------------------------------------------------------------------
# should_rollback
# ----------------------------------------------------------------------
class TestShouldRollback:
    def test_rollback_returns_true_on_failure_signal(self):
        strategy = _make_strategy(
            rollback_on_failure_signal=True,
            failure_signals=["你理解错了"],
        )
        assert strategy.should_rollback("s1", "你理解错了") is True

    def test_rollback_returns_false_when_failure_signal_disabled(self):
        strategy = _make_strategy(
            rollback_on_failure_signal=False,
            failure_signals=["你理解错了"],
        )
        assert strategy.should_rollback("s1", "你理解错了") is False

    def test_rollback_returns_true_on_tool_failure_count_exceeded(self):
        strategy = _make_strategy(rollback_on_tool_failure_count=3)
        # Equal to threshold -> rollback.
        assert strategy.should_rollback("s1", "继续", tool_failure_count=3) is True
        # Above threshold -> rollback.
        assert strategy.should_rollback("s1", "继续", tool_failure_count=5) is True

    def test_rollback_returns_false_below_tool_failure_threshold(self):
        strategy = _make_strategy(rollback_on_tool_failure_count=3)
        assert strategy.should_rollback("s1", "继续", tool_failure_count=2) is False
        assert strategy.should_rollback("s1", "继续", tool_failure_count=0) is False

    def test_rollback_returns_false_on_clean_input(self):
        strategy = _make_strategy(
            rollback_on_failure_signal=True,
            rollback_on_tool_failure_count=3,
            failure_signals=["你理解错了"],
        )
        assert strategy.should_rollback("s1", "继续推荐", tool_failure_count=0) is False


# ----------------------------------------------------------------------
# rollback_and_recognize
# ----------------------------------------------------------------------
class TestRollbackAndRecognize:
    def test_rollback_and_recognize_calls_recognize_fn(self):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", "product_recommendation", turn=1)
        strategy = _make_strategy(history=store)

        captured: dict = {}

        def fake_recognize(text: str, session_id: str, turn: int) -> IntentRecognitionResult:
            captured["text"] = text
            captured["session_id"] = session_id
            captured["turn"] = turn
            return IntentRecognitionResult(
                intent="order_query",
                confidence=0.9,
                source=RecognitionSource.LIGHTWEIGHT_LLM,
                layer_reached=2,
            )

        result = strategy.rollback_and_recognize(
            "s1", "切换，查订单", turn=2, recognize_fn=fake_recognize
        )

        # The recognize_fn was called with the forwarded args.
        assert captured == {"text": "切换，查订单", "session_id": "s1", "turn": 2}
        # And its result is returned verbatim.
        assert result.intent == "order_query"
        assert result.source == RecognitionSource.LIGHTWEIGHT_LLM
        assert result.layer_reached == 2

    def test_rollback_and_recognize_returns_real_result_not_reused(self):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", "product_recommendation", turn=1)
        strategy = _make_strategy(history=store)

        def fake_recognize(text: str, session_id: str, turn: int) -> IntentRecognitionResult:
            return IntentRecognitionResult(
                intent="order_query",
                confidence=0.88,
                source=RecognitionSource.LIGHTWEIGHT_LLM,
                layer_reached=2,
            )

        result = strategy.rollback_and_recognize(
            "s1", "我要问别的", turn=2, recognize_fn=fake_recognize
        )
        # Source must NOT be REUSED.
        assert result.source != RecognitionSource.REUSED
        assert result.source == RecognitionSource.LIGHTWEIGHT_LLM


# ----------------------------------------------------------------------
# INTENT_SWITCH_MARKERS sanity
# ----------------------------------------------------------------------
class TestIntentSwitchMarkers:
    def test_markers_are_non_empty_unique_strings(self):
        assert len(INTENT_SWITCH_MARKERS) >= 4
        assert all(isinstance(m, str) and m for m in INTENT_SWITCH_MARKERS)
        assert len(set(INTENT_SWITCH_MARKERS)) == len(INTENT_SWITCH_MARKERS)

    def test_required_markers_present(self):
        # The task spec lists these four as required markers.
        for required in ("换个话题", "不是这个", "我要问别的", "切换"):
            assert required in INTENT_SWITCH_MARKERS
