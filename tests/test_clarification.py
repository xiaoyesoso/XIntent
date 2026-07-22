"""Clarification mechanism regression tests (task 4.5)."""

from user_input_normalization.clarification import ClarificationHandler
from user_input_normalization.models import (
    ClarificationReason,
    CompletenessCheck,
    CompletenessStatus,
    NormalizationResult,
    PronounResolution,
)


class TestClarificationTrigger:
    """Trigger condition judgment (task 4.1)."""

    def setup_method(self):
        self.handler = ClarificationHandler()

    def test_low_confidence_triggers_clarify(self):
        assert self.handler.should_clarify(0.3) is True

    def test_high_confidence_no_clarify(self):
        assert self.handler.should_clarify(0.9) is False

    def test_threshold_boundary(self):
        # Default threshold is 0.6; below triggers clarification
        assert self.handler.should_clarify(0.59) is True
        assert self.handler.should_clarify(0.6) is False

    def test_trigger_pronoun_resolution_failed(self):
        req = self.handler.trigger(
            "sess1", ClarificationReason.PRONOUN_RESOLUTION_FAILED,
            "那个帅气的同事", 0.3,
        )
        assert req is not None
        assert req.reason == ClarificationReason.PRONOUN_RESOLUTION_FAILED
        assert req.item == "那个帅气的同事"
        assert self.handler.is_paused("sess1") is True

    def test_trigger_unknown_vocab(self):
        req = self.handler.trigger(
            "sess1", ClarificationReason.UNKNOWN_VOCAB, "XQR", 0.1,
        )
        assert req is not None
        assert "XQR" in req.question


class TestClarificationOutput:
    """Clarification output format (task 4.2)."""

    def setup_method(self):
        self.handler = ClarificationHandler()

    def test_output_with_candidates(self):
        req = self.handler.create_request(
            ClarificationReason.PRONOUN_RESOLUTION_FAILED,
            "第二个", 0.4,
            candidates=["TCC方案", "轻量方案"],
        )
        assert req.candidates == ["TCC方案", "轻量方案"]
        assert "TCC方案" in req.question

    def test_output_without_candidates(self):
        req = self.handler.create_request(
            ClarificationReason.PRONOUN_RESOLUTION_FAILED,
            "那个", 0.2,
        )
        assert req.candidates == []
        assert "具体" in req.question or "什么" in req.question

    def test_question_for_each_reason(self):
        for reason in ClarificationReason:
            req = self.handler.create_request(reason, "test", 0.3)
            assert req.question  # non-empty


class TestClarificationFlow:
    """Clarification dialogue flow integration (task 4.3)."""

    def setup_method(self):
        self.handler = ClarificationHandler()

    def test_pause_and_resume(self):
        self.handler.trigger("s1", ClarificationReason.UNKNOWN_VOCAB, "XQR", 0.1)
        assert self.handler.is_paused("s1")
        response = self.handler.receive_response("s1", "XQR是某个项目的代号")
        assert response == "XQR是某个项目的代号"
        assert not self.handler.is_paused("s1")

    def test_get_pending(self):
        self.handler.trigger("s1", ClarificationReason.UNKNOWN_VOCAB, "XQR", 0.1)
        pending = self.handler.get_pending("s1")
        assert pending is not None
        assert pending.item == "XQR"


class TestClarificationLimit:
    """Clarification count limit (task 4.4)."""

    def setup_method(self):
        self.handler = ClarificationHandler()

    def test_max_consecutive_limit(self):
        # Default maximum is 3 times
        for i in range(3):
            req = self.handler.trigger("s1", ClarificationReason.UNKNOWN_VOCAB, f"word{i}", 0.1)
            assert req is not None
        # The 4th time should return None (degraded passthrough)
        req = self.handler.trigger("s1", ClarificationReason.UNKNOWN_VOCAB, "word3", 0.1)
        assert req is None

    def test_reset_after_success(self):
        for i in range(2):
            self.handler.trigger("s1", ClarificationReason.UNKNOWN_VOCAB, f"w{i}", 0.1)
        self.handler.reset("s1")
        # After reset, clarification can continue
        req = self.handler.trigger("s1", ClarificationReason.UNKNOWN_VOCAB, "new", 0.1)
        assert req is not None


class TestCheckResult:
    """Normalization result checking."""

    def setup_method(self):
        self.handler = ClarificationHandler()

    def test_low_confidence_pronoun_triggers(self):
        result = NormalizationResult(
            raw_input="那个", normalized_input="那个",
            pronoun_resolutions=[
                PronounResolution(
                    pronoun="那个", resolved_to="某物",
                    confidence=0.3, evidence_source="无",
                )
            ],
        )
        requests = self.handler.check_result("s1", result)
        assert len(requests) > 0

    def test_high_confidence_no_trigger(self):
        result = NormalizationResult(
            raw_input="第二个", normalized_input="TCC方案",
            pronoun_resolutions=[
                PronounResolution(
                    pronoun="第二个", resolved_to="TCC方案",
                    confidence=0.95, evidence_source="对话历史第3轮",
                )
            ],
        )
        requests = self.handler.check_result("s1", result)
        assert len(requests) == 0

    def test_incomplete_spo_triggers(self):
        result = NormalizationResult(
            raw_input="帮我看看", normalized_input="帮我看看",
            completeness=CompletenessCheck(
                spo_complete=False, pronouns_resolved=True,
                adjectives_quantified=True,
                result=CompletenessStatus.INCOMPLETE_MISSING_ARGUMENT,
            ),
        )
        requests = self.handler.check_result("s1", result)
        assert len(requests) > 0
