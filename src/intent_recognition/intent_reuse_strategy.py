"""Intent reuse strategy with rollback (D22).

When the user keeps talking about the same task across turns, the previous
turn's intent can be reused directly without re-running the three-layer
waterfall. This avoids unnecessary LLM calls and keeps cross-turn slot
accumulation stable.

The strategy is opt-in (``ReuseStrategyConfig.enable=False`` by default) and
supports rollback: when an implicit failure signal (D11) or repeated tool
failures are detected, the reused intent is discarded and a fresh
recognition pass is triggered.

Design note: this module is intentionally side-effect free with respect to
the pipeline. It only reads from ``IntentHistoryStore`` and produces
``IntentRecognitionResult`` objects; the caller (pipeline) decides whether
to persist the result back into history.
"""

from __future__ import annotations

from typing import Callable

from .config import ReuseStrategyConfig
from .models import IntentRecognitionResult, RecognitionSource
from .storage.base import IntentHistoryStore


# Substring markers that strongly suggest the user is switching away from
# the previous intent. If any of these appear in the input we do NOT reuse
# the previous intent and instead let the waterfall re-recognize.
INTENT_SWITCH_MARKERS: list[str] = [
    "换个话题",
    "不是这个",
    "我要问别的",
    "切换",
    "换一个",
    "不是说这个",
    "我想问的是",
]


class IntentReuseStrategy:
    """D22: Reuse the previous turn's intent, with rollback support.

    Parameters
    ----------
    history_store:
        Store used to look up the previous turn's intent.
    config:
        ``ReuseStrategyConfig`` (enable flag + rollback thresholds).
    failure_signals:
        Optional list of implicit failure signal phrases (D11). When none
        is provided the strategy still works but only relies on the
        ``INTENT_SWITCH_MARKERS`` and tool-failure-count triggers.
    """

    def __init__(
        self,
        history_store: IntentHistoryStore,
        config: ReuseStrategyConfig,
        failure_signals: list[str] | None = None,
    ) -> None:
        self._history_store = history_store
        self._config = config
        # Defensive copy so external mutation does not affect detection.
        self._failure_signals: list[str] = list(failure_signals or [])

    # ------------------------------------------------------------------
    # Reuse
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether the reuse strategy is enabled."""
        return self._config.enable

    def try_reuse(
        self,
        session_id: str,
        text: str,
        turn: int,
    ) -> IntentRecognitionResult | None:
        """Attempt to reuse the previous turn's intent.

        Returns a reused ``IntentRecognitionResult`` (``source=REUSED``) when:
        - the strategy is enabled, AND
        - a previous intent exists in history, AND
        - the input does not contain an intent-switch marker or failure signal.

        Returns ``None`` otherwise, signaling the caller to fall back to the
        normal three-layer waterfall.
        """
        if not self._config.enable:
            return None

        prev = self._history_store.get_previous_intent(session_id)
        if prev is None:
            # First turn or no history yet - nothing to reuse.
            return None

        # Intent-switch markers indicate the user is moving on.
        if self._contains_intent_switch_marker(text):
            return None

        # Failure signals (D11) also invalidate reuse.
        if self._contains_failure_signal(text):
            return None

        return IntentRecognitionResult(
            intent=prev,
            confidence=1.0,
            source=RecognitionSource.REUSED,
            layer_reached=0,
            signals={"reused": 1.0},
        )

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------
    def should_rollback(
        self,
        session_id: str,
        text: str,
        tool_failure_count: int = 0,
    ) -> bool:
        """Decide whether a reused intent should be rolled back.

        Triggers:
        - ``rollback_on_failure_signal=True`` and the text contains a
          failure signal phrase.
        - ``tool_failure_count >= rollback_on_tool_failure_count``.

        The ``session_id`` is accepted for API symmetry and future
        per-session heuristics; it is not currently used.
        """
        if self._config.rollback_on_failure_signal and self._contains_failure_signal(text):
            return True

        if (
            self._config.rollback_on_tool_failure_count > 0
            and tool_failure_count >= self._config.rollback_on_tool_failure_count
        ):
            return True

        return False

    def rollback_and_recognize(
        self,
        session_id: str,
        text: str,
        turn: int,
        recognize_fn: Callable[..., IntentRecognitionResult],
    ) -> IntentRecognitionResult:
        """Discard the reused intent and trigger a real recognition pass.

        ``recognize_fn`` is expected to be the pipeline's ``recognize``
        method (or any callable with the same signature). It is invoked
        positionally as ``recognize_fn(text, session_id, turn)``.
        """
        return recognize_fn(text, session_id, turn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _contains_intent_switch_marker(self, text: str) -> bool:
        """Return True if ``text`` contains any intent-switch marker."""
        if not text:
            return False
        return any(marker in text for marker in INTENT_SWITCH_MARKERS)

    def _contains_failure_signal(self, text: str) -> bool:
        """Return True if ``text`` contains any registered failure signal."""
        if not text or not self._failure_signals:
            return False
        return any(signal and signal in text for signal in self._failure_signals)
