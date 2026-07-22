"""Rejection & clarification handler (D10, D11).

Handles unsupported-input rejection, unclear-input clarification, implicit
failure-signal detection, convergence tracking, and max-clarification limits.
"""

from __future__ import annotations

from ..config import IntentRecognitionConfig
from ..intent_registry import IntentRegistry
from ..models import IntentRecognitionResult, RecognitionSource
from ..storage.base import IntentHistoryStore


# Slot-name -> human-friendly Chinese label for clarification questions.
# Keys are intentionally lower_snake_case slot names.
_SLOT_LABELS: dict[str, str] = {
    "category": "类目",
    "budget": "预算",
    "budget_max": "最高预算",
    "budget_min": "最低预算",
    "price": "价格",
    "price_max": "最高价格",
    "price_min": "最低价格",
    "brand": "品牌",
    "color": "颜色",
    "size": "尺寸",
    "quantity": "数量",
    "region": "地区",
    "date": "日期",
    "time": "时间",
}


def _slot_label(slot: str) -> str:
    """Return a human-friendly label for a slot name."""
    return _SLOT_LABELS.get(slot, slot)


class RejectionClarificationHandler:
    """Handler for rejection (D10) and clarification (D10/D11) flows.

    Responsibilities:
    - Produce ``unsupported`` results when input falls outside the candidate
      intent set, listing what the system *does* support.
    - Produce ``unclear`` results with targeted clarification questions when
      required slots are missing or the intent boundary is ambiguous.
    - Detect implicit failure signals (e.g. "你理解错了") from the next turn
      to mark the previous recognition as failed.
    - Track convergence after clarification and enforce the
      ``max_consecutive_clarifications`` limit.
    """

    def __init__(
        self,
        registry: IntentRegistry,
        history_store: IntentHistoryStore | None = None,
        config: IntentRecognitionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._history = history_store
        self._config = config or IntentRecognitionConfig()

        # session_id -> list of (turn, resolved) convergence records
        self._convergence: dict[str, list[tuple[int, bool]]] = {}
        # session_id -> number of consecutive unresolved clarifications
        self._consecutive: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Rejection
    # ------------------------------------------------------------------
    def handle_unsupported(self, text: str) -> IntentRecognitionResult:
        """Return an ``unsupported`` result that lists supported intents.

        The result has ``intent=None`` and a ``rejection_reason`` that explains
        what the system supports, so downstream components can surface a
        friendly message instead of silently dropping the input.
        """
        names = self._registry.list_names()
        descriptions = self._registry.build_prompt_description()
        supported_block = ", ".join(names) if names else "(无已注册意图)"
        reason = (
            f"intent_not_in_candidate_list | "
            f"unsupported_content={text!r} | "
            f"supported_intents=[{supported_block}]\n"
            f"{descriptions}"
        )
        return IntentRecognitionResult(
            intent=None,
            confidence=0.0,
            source=RecognitionSource.CODE_LAYER,
            rejection_reason=reason,
            layer_reached=1,
        )

    # ------------------------------------------------------------------
    # Clarification
    # ------------------------------------------------------------------
    def handle_unclear(
        self,
        text: str,
        candidates: list[str],
        missing_slots: list[str],
    ) -> IntentRecognitionResult:
        """Return an ``unclear`` result asking for the missing information.

        ``candidates`` are the plausible intent names (e.g. top-k from the
        recognizer). ``missing_slots`` drives the clarification question.
        """
        question = self._build_clarification_question(missing_slots, candidates)
        reason_parts: list[str] = ["required_slot_missing"]
        if missing_slots:
            reason_parts.append("missing_slots=" + ",".join(missing_slots))
        if candidates:
            reason_parts.append("ambiguous_intents=" + ",".join(candidates))
        reason = " | ".join(reason_parts)
        return IntentRecognitionResult(
            intent=candidates[0] if candidates else None,
            confidence=0.0,
            source=RecognitionSource.CODE_LAYER,
            missing_slots=list(missing_slots),
            need_clarification=True,
            clarification_question=question,
            rejection_reason=None,
            layer_reached=1,
            sub_intents=list(candidates),
        )

    def _build_clarification_question(
        self,
        missing_slots: list[str],
        candidates: list[str] | None = None,
    ) -> str:
        """Build a concise clarification question.

        - For missing slots, use the template "您的{label}大概是多少？" for
          numeric slots and "您希望提供哪类{label}？" for categorical ones.
        - Multiple missing slots are merged into one question.
        - If only ambiguous candidates are provided (no slots), ask the user
          to choose between them.
        """
        if not missing_slots and candidates:
            joined = " / ".join(candidates)
            return f"您是想做以下哪个操作？可选：{joined}"

        parts: list[str] = []
        for slot in missing_slots:
            label = _slot_label(slot)
            if any(k in slot for k in ("budget", "price", "quantity")):
                parts.append(f"您的{label}大概是多少？")
            else:
                parts.append(f"您希望提供哪类{label}？")
        return "".join(parts) if parts else "请问您能补充更多信息吗？"

    # ------------------------------------------------------------------
    # Implicit failure detection (D11)
    # ------------------------------------------------------------------
    def check_implicit_failure(self, text: str) -> bool:
        """Return True if ``text`` contains an implicit failure signal.

        Signals come from ``config.failure_signals`` (e.g. "你理解错了",
        "不是这个意思"). Matching is substring-based so partial phrases such
        as "你理解错了，我是想查询订单" still trigger.
        """
        if not text:
            return False
        for signal in self._config.failure_signals:
            if signal and signal in text:
                return True
        return False

    # ------------------------------------------------------------------
    # Convergence tracking (D10)
    # ------------------------------------------------------------------
    def track_convergence(
        self, session_id: str, turn: int, resolved: bool
    ) -> None:
        """Record whether the user converged after a clarification.

        When ``resolved`` is True the consecutive-unresolved counter resets;
        otherwise it increments. The counter feeds
        :meth:`check_max_clarifications`.
        """
        self._convergence.setdefault(session_id, []).append((turn, resolved))
        if resolved:
            self._consecutive[session_id] = 0
        else:
            self._consecutive[session_id] = (
                self._consecutive.get(session_id, 0) + 1
            )

    def check_max_clarifications(self, session_id: str) -> bool:
        """Return True if max consecutive clarifications is exceeded.

        After the limit is reached the caller should degrade to a best guess
        (or hand off to a human) rather than clarifying again.
        """
        limit = self._config.clarification.max_consecutive_clarifications
        return self._consecutive.get(session_id, 0) >= limit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_convergence_history(self, session_id: str) -> list[tuple[int, bool]]:
        """Return the recorded convergence history for a session."""
        return list(self._convergence.get(session_id, []))

    def get_consecutive_count(self, session_id: str) -> int:
        """Return the current consecutive-unresolved clarification count."""
        return self._consecutive.get(session_id, 0)
