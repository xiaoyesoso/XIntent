"""Clarification mechanism module (corresponds to D8).

When pronoun resolution, word sense, subjective judgment, and other low-confidence content cannot be inferred,
actively ask the user to clarify rather than guessing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..config import Config, get_config
from ..models import (
    ClarificationReason,
    ClarificationRequest,
    NormalizationResult,
)


class ClarificationState(BaseModel):
    """Clarification state: tracks the clarification count and history of the current session."""

    session_id: str
    pending_clarification: ClarificationRequest | None = None
    consecutive_count: int = 0
    history: list[ClarificationRequest] = Field(default_factory=list)
    paused: bool = False  # Whether normalization is paused waiting for user response


class ClarificationHandler:
    """Clarification mechanism handler (corresponds to D8 clarification trigger conditions).

    Trigger conditions:
    1. Pronoun resolution cannot infer a specific person, object, or location (confidence below threshold theta_clarify, default 0.6)
    2. Word sense disambiguation confidence is low
    3. Subjective judgment words cannot be quantified
    4. Omission completion lacks evidence
    5. Unknown vocabulary and no corresponding entry in vocabulary-table

    Core principle: do not guess, do not fabricate; low-confidence content must be clarified.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or get_config()
        self._states: dict[str, ClarificationState] = {}

    def _get_state(self, session_id: str) -> ClarificationState:
        if session_id not in self._states:
            self._states[session_id] = ClarificationState(session_id=session_id)
        return self._states[session_id]

    @property
    def threshold(self) -> float:
        """Clarification trigger confidence threshold."""
        return self._config.clarify.theta_clarify

    @property
    def max_consecutive(self) -> int:
        """Maximum consecutive clarification count."""
        return self._config.clarify.max_consecutive_clarifications

    def should_clarify(self, confidence: float) -> bool:
        """Determine whether confidence is below threshold and clarification is needed (task 4.1)."""
        return confidence < self.threshold

    def create_request(
        self,
        reason: ClarificationReason,
        item: str,
        confidence: float,
        candidates: list[str] | None = None,
        question: str | None = None,
    ) -> ClarificationRequest:
        """Create a clarification request (task 4.2).

        Generates structured clarification output: reason / item to clarify / candidate list / clarification question.
        """
        if question is None:
            question = self._generate_question(reason, item, candidates)
        return ClarificationRequest(
            reason=reason,
            item=item,
            candidates=candidates or [],
            question=question,
            confidence=confidence,
        )

    def trigger(
        self,
        session_id: str,
        reason: ClarificationReason,
        item: str,
        confidence: float,
        candidates: list[str] | None = None,
        question: str | None = None,
    ) -> ClarificationRequest | None:
        """Trigger clarification (task 4.3 dialogue flow integration).

        If the maximum consecutive clarification count has been reached, returns None (degrades to pass-through).
        Otherwise pauses normalization and returns the clarification request.
        """
        state = self._get_state(session_id)

        # Check whether the maximum consecutive clarification count is exceeded (task 4.4)
        if state.consecutive_count >= self.max_consecutive:
            # Exceeded limit, degrade to pass-through to avoid clarification loops
            state.paused = False
            state.pending_clarification = None
            return None

        request = self.create_request(reason, item, confidence, candidates, question)
        state.pending_clarification = request
        state.consecutive_count += 1
        state.history.append(request)
        state.paused = True
        return request

    def receive_response(self, session_id: str, user_response: str) -> str:
        """Receive user clarification response and resume normalization (task 4.3).

        Returns the user response content for the normalization module to reprocess.
        """
        state = self._get_state(session_id)
        state.paused = False
        state.pending_clarification = None
        return user_response

    def is_paused(self, session_id: str) -> bool:
        """Check whether the session is paused waiting for clarification."""
        return self._get_state(session_id).paused

    def get_pending(self, session_id: str) -> ClarificationRequest | None:
        """Get the pending clarification request."""
        return self._get_state(session_id).pending_clarification

    def reset(self, session_id: str) -> None:
        """Reset the session clarification state (called after successfully completing a round of normalization)."""
        state = self._get_state(session_id)
        state.consecutive_count = 0
        state.paused = False
        state.pending_clarification = None

    def check_result(
        self, session_id: str, result: NormalizationResult
    ) -> list[ClarificationRequest]:
        """Check whether the normalization result needs to trigger clarification (task 4.1).

        Iterates over the pronoun resolution table, term mappings, etc. to find low-confidence content.
        """
        requests: list[ClarificationRequest] = []

        # Check pronoun resolution confidence
        for pr in result.pronoun_resolutions:
            if self.should_clarify(pr.confidence):
                req = self.trigger(
                    session_id,
                    ClarificationReason.PRONOUN_RESOLUTION_FAILED,
                    pr.pronoun,
                    pr.confidence,
                    candidates=[pr.resolved_to] if pr.resolved_to else [],
                )
                if req:
                    requests.append(req)

        # Check completeness check result
        if result.completeness:
            c = result.completeness
            if not c.spo_complete:
                req = self.trigger(
                    session_id,
                    ClarificationReason.COMPLETION_NO_EVIDENCE,
                    "主谓宾不完整",
                    0.0,
                )
                if req:
                    requests.append(req)
            if not c.pronouns_resolved:
                req = self.trigger(
                    session_id,
                    ClarificationReason.PRONOUN_RESOLUTION_FAILED,
                    "存在未消解代词",
                    0.0,
                )
                if req:
                    requests.append(req)

        return requests

    @staticmethod
    def _generate_question(
        reason: ClarificationReason,
        item: str,
        candidates: list[str] | None,
    ) -> str:
        """Generate clarification question (task 4.2 structured output)."""
        if candidates:
            candidates_str = "、".join(candidates[:5])
            return f"您说的'{item}'是指以下哪一个？{candidates_str}"
        reason_questions = {
            ClarificationReason.PRONOUN_RESOLUTION_FAILED: f"您说的'{item}'具体是指什么？请提供更多信息。",
            ClarificationReason.WORD_SENSE_LOW_CONFIDENCE: f"'{item}'在这个上下文中是什么含义？请确认。",
            ClarificationReason.SUBJECTIVE_NO_BASELINE: f"您提到的'{item}'，能具体说明您的标准或期望吗？",
            ClarificationReason.COMPLETION_NO_EVIDENCE: f"关于'{item}'，能否补充更多上下文信息？",
            ClarificationReason.UNKNOWN_VOCAB: f"'{item}'是什么意思？请解释一下。",
        }
        return reason_questions.get(
            reason, f"关于'{item}'，请提供更多信息以便准确理解。"
        )

    def get_stats(self, session_id: str) -> dict[str, Any]:
        """Get clarification statistics (for quality evaluation)."""
        state = self._get_state(session_id)
        return {
            "total_clarifications": len(state.history),
            "consecutive_count": state.consecutive_count,
            "is_paused": state.paused,
            "reasons": [r.reason.value for r in state.history],
        }
