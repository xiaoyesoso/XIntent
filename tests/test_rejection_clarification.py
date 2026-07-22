"""Rejection & clarification handler tests (Group 6: D10, D11)."""

from intent_recognition import (
    IntentDefinition,
    IntentRecognitionConfig,
    SlotDefinition,
)
from intent_recognition.rejection_clarification import (
    RejectionClarificationHandler,
)
from intent_recognition.storage import MemoryIntentHistoryStore


def _build_registry():
    """Build a small registry with two intents for testing."""
    from intent_recognition import IntentRegistry

    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="product_recommendation",
        description="用户希望推荐商品",
        positive_examples=["推荐手机", "有什么耳机推荐"],
        slots=[
            SlotDefinition(name="category", required=True, description="商品类目"),
            SlotDefinition(name="price", required=True, description="价格区间"),
            SlotDefinition(name="budget", required=False, description="预算"),
        ],
    ))
    reg.register(IntentDefinition(
        name="order_query",
        description="用户希望查询订单",
        positive_examples=["我的订单到哪了"],
    ))
    return reg


class TestRejection:
    """D10: unsupported rejection."""

    def setup_method(self):
        self.handler = RejectionClarificationHandler(
            _build_registry(),
            MemoryIntentHistoryStore(),
            IntentRecognitionConfig(),
        )

    def test_reject_unsupported_intent(self):
        result = self.handler.handle_unsupported("帮我生成一个奥特曼动画")
        assert result.intent is None
        assert result.rejection_reason is not None
        assert result.is_unsupported is True

    def test_rejection_reason_lists_supported_intents(self):
        result = self.handler.handle_unsupported("帮我生成一个奥特曼动画")
        assert result.rejection_reason is not None
        assert "product_recommendation" in result.rejection_reason
        assert "order_query" in result.rejection_reason

    def test_rejection_includes_unsupported_content(self):
        result = self.handler.handle_unsupported("帮我生成一个奥特曼动画")
        assert "帮我生成一个奥特曼动画" in (result.rejection_reason or "")

    def test_rejection_does_not_classify_into_candidate(self):
        result = self.handler.handle_unsupported("帮我生成一个奥特曼动画")
        # Must NOT force-classify into a registered intent
        assert result.intent not in {"product_recommendation", "order_query"}

    def test_rejection_layer_and_source(self):
        result = self.handler.handle_unsupported("anything unsupported")
        assert result.layer_reached == 1
        assert result.source.value == "code-layer"


class TestClarification:
    """D10: unclear clarification."""

    def setup_method(self):
        self.handler = RejectionClarificationHandler(
            _build_registry(),
            MemoryIntentHistoryStore(),
            IntentRecognitionConfig(),
        )

    def test_clarify_missing_price_slot(self):
        result = self.handler.handle_unclear(
            "帮我推荐一条裤子",
            candidates=["product_recommendation"],
            missing_slots=["price"],
        )
        assert result.need_clarification is True
        assert result.is_unclear is True
        assert result.clarification_question is not None
        # Question must ask about the missing slot
        assert "价格" in result.clarification_question

    def test_clarify_missing_category_slot(self):
        result = self.handler.handle_unclear(
            "帮我推荐一个",
            candidates=["product_recommendation"],
            missing_slots=["category"],
        )
        assert result.need_clarification is True
        assert result.clarification_question is not None
        assert "类目" in result.clarification_question

    def test_clarify_multiple_missing_slots_merged(self):
        result = self.handler.handle_unclear(
            "帮我推荐",
            candidates=["product_recommendation"],
            missing_slots=["category", "price"],
        )
        assert result.need_clarification is True
        question = result.clarification_question or ""
        # Both slots should be addressed in a single merged question
        assert "类目" in question
        assert "价格" in question

    def test_clarify_records_missing_slots(self):
        result = self.handler.handle_unclear(
            "帮我推荐",
            candidates=["product_recommendation"],
            missing_slots=["category", "price"],
        )
        assert result.missing_slots == ["category", "price"]

    def test_clarify_candidates_stored_as_sub_intents(self):
        result = self.handler.handle_unclear(
            "帮我看看这个笔记本",
            candidates=["product_query", "product_comparison"],
            missing_slots=[],
        )
        assert result.need_clarification is True
        assert result.clarification_question is not None
        assert "product_query" in result.clarification_question
        assert result.sub_intents == ["product_query", "product_comparison"]

    def test_clarify_question_for_budget_slot(self):
        result = self.handler.handle_unclear(
            "推荐手机",
            candidates=["product_recommendation"],
            missing_slots=["budget"],
        )
        assert result.clarification_question is not None
        assert "预算" in result.clarification_question


class TestImplicitFailure:
    """D11: implicit failure signal detection."""

    def setup_method(self):
        self.handler = RejectionClarificationHandler(
            _build_registry(),
            MemoryIntentHistoryStore(),
            IntentRecognitionConfig(),
        )

    def test_detect_failure_signal_you_understood_wrong(self):
        assert self.handler.check_implicit_failure("你理解错了，我是想查询订单") is True

    def test_detect_failure_signal_not_what_i_meant(self):
        assert self.handler.check_implicit_failure("不是这个意思，我要的是手机") is True

    def test_detect_failure_signal_not_i_am_not(self):
        # Configured in default failure_signals
        assert self.handler.check_implicit_failure("不是，我不是这个意思") is True

    def test_detect_failure_signal_no_i_want(self):
        # Configured in default failure_signals
        assert self.handler.check_implicit_failure("不对，我要的是手机") is True

    def test_normal_input_no_failure_signal(self):
        assert self.handler.check_implicit_failure("帮我推荐手机") is False

    def test_empty_text_no_failure_signal(self):
        assert self.handler.check_implicit_failure("") is False

    def test_custom_failure_signals(self):
        cfg = IntentRecognitionConfig()
        cfg.failure_signals = ["我说错了", "重新理解一下"]
        handler = RejectionClarificationHandler(
            _build_registry(),
            MemoryIntentHistoryStore(),
            cfg,
        )
        assert handler.check_implicit_failure("我说错了，重新来") is True
        assert handler.check_implicit_failure("帮我推荐手机") is False


class TestConvergenceTracking:
    """D10: convergence tracking after clarification."""

    def setup_method(self):
        self.handler = RejectionClarificationHandler(
            _build_registry(),
            MemoryIntentHistoryStore(),
            IntentRecognitionConfig(),
        )

    def test_resolved_resets_consecutive_count(self):
        # Two unresolved, then resolved -> counter resets to 0
        self.handler.track_convergence("s1", turn=1, resolved=False)
        self.handler.track_convergence("s1", turn=2, resolved=False)
        self.handler.track_convergence("s1", turn=3, resolved=True)
        assert self.handler.get_consecutive_count("s1") == 0
        assert not self.handler.check_max_clarifications("s1")

    def test_unresolved_increments_consecutive_count(self):
        self.handler.track_convergence("s1", turn=1, resolved=False)
        assert self.handler.get_consecutive_count("s1") == 1
        self.handler.track_convergence("s1", turn=2, resolved=False)
        assert self.handler.get_consecutive_count("s1") == 2

    def test_convergence_history_recorded(self):
        self.handler.track_convergence("s1", turn=1, resolved=False)
        self.handler.track_convergence("s1", turn=2, resolved=True)
        history = self.handler.get_convergence_history("s1")
        assert len(history) == 2
        assert history[0] == (1, False)
        assert history[1] == (2, True)

    def test_max_clarifications_exceeded_after_default_3(self):
        # Default max_consecutive_clarifications == 3
        self.handler.track_convergence("s1", turn=1, resolved=False)
        assert not self.handler.check_max_clarifications("s1")
        self.handler.track_convergence("s1", turn=2, resolved=False)
        assert not self.handler.check_max_clarifications("s1")
        self.handler.track_convergence("s1", turn=3, resolved=False)
        # At limit -> exceeded
        assert self.handler.check_max_clarifications("s1") is True

    def test_max_clarifications_resets_after_resolution(self):
        # Two unresolved, then resolved, then two more unresolved -> not exceeded
        for turn, resolved in [(1, False), (2, False), (3, True), (4, False), (5, False)]:
            self.handler.track_convergence("s1", turn=turn, resolved=resolved)
        assert not self.handler.check_max_clarifications("s1")

    def test_max_clarifications_custom_limit(self):
        cfg = IntentRecognitionConfig()
        cfg.clarification.max_consecutive_clarifications = 2
        handler = RejectionClarificationHandler(
            _build_registry(),
            MemoryIntentHistoryStore(),
            cfg,
        )
        handler.track_convergence("s1", turn=1, resolved=False)
        handler.track_convergence("s1", turn=2, resolved=False)
        assert handler.check_max_clarifications("s1") is True

    def test_isolated_sessions(self):
        self.handler.track_convergence("s1", turn=1, resolved=False)
        self.handler.track_convergence("s2", turn=1, resolved=False)
        assert self.handler.get_consecutive_count("s1") == 1
        assert self.handler.get_consecutive_count("s2") == 1
        # Resolving s1 must not affect s2
        self.handler.track_convergence("s1", turn=2, resolved=True)
        assert self.handler.get_consecutive_count("s1") == 0
        assert self.handler.get_consecutive_count("s2") == 1
