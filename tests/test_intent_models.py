"""Tests for intent recognition data models and registry."""

from intent_recognition import (
    Constraint,
    ConstraintType,
    Evidence,
    EvidenceGrade,
    IntentDefinition,
    IntentRecognitionConfig,
    IntentRecognitionResult,
    IntentRegistry,
    RecognitionSource,
    SlotDefinition,
    SlotState,
    SlotValue,
)


class TestModels:
    """Test data models."""

    def test_intent_recognition_result_defaults(self):
        r = IntentRecognitionResult()
        assert r.intent is None
        assert r.confidence == 0.0
        assert r.source == RecognitionSource.CODE_LAYER
        assert r.layer_reached == 1
        assert r.is_unsupported is False
        assert r.is_unclear is False

    def test_is_unsupported(self):
        r = IntentRecognitionResult(intent=None, rejection_reason="超出能力范围")
        assert r.is_unsupported is True

    def test_is_unclear(self):
        r = IntentRecognitionResult(need_clarification=True, clarification_question="你要什么价位？")
        assert r.is_unclear is True

    def test_constraint_types(self):
        hard = Constraint(type=ConstraintType.HARD, expression="price<100")
        soft = Constraint(type=ConstraintType.SOFT, expression="type=牛仔裤")
        assert hard.type == ConstraintType.HARD
        assert soft.type == ConstraintType.SOFT

    def test_slot_definition(self):
        s = SlotDefinition(name="category", required=True)
        assert s.required is True
        assert s.default is None

    def test_intent_definition_with_slots(self):
        d = IntentDefinition(
            name="product_recommendation",
            description="推荐商品",
            slots=[
                SlotDefinition(name="category", required=True),
                SlotDefinition(name="budget", required=False, default=1000),
            ],
        )
        assert len(d.slots) == 2
        assert d.slots[0].name == "category"
        assert d.slots[1].default == 1000

    def test_slot_state(self):
        s = SlotState(session_id="s1")
        assert s.intent is None
        assert s.slots == {}
        assert s.missing_slots == []


class TestRegistry:
    """Test intent registry."""

    def _build_registry(self):
        reg = IntentRegistry()
        reg.register(IntentDefinition(
            name="product_recommendation",
            description="用户希望推荐商品",
            positive_examples=["推荐手机", "有什么耳机推荐"],
            negative_examples=["查询订单状态 -> order_query"],
            slots=[
                SlotDefinition(name="category", required=True, description="商品类目"),
                SlotDefinition(name="budget_max", required=False, description="最高预算"),
            ],
        ))
        reg.register(IntentDefinition(
            name="order_query",
            description="用户希望查询订单",
            positive_examples=["我的订单到哪了"],
        ))
        reg.register(IntentDefinition(
            name="refund",
            description="用户希望退款",
            parent_intent="order_query",
        ))
        return reg

    def test_register_and_get(self):
        reg = self._build_registry()
        d = reg.get("product_recommendation")
        assert d is not None
        assert d.name == "product_recommendation"

    def test_get_nonexistent(self):
        reg = self._build_registry()
        assert reg.get("nonexistent") is None

    def test_list_all(self):
        reg = self._build_registry()
        all_intents = reg.list_all()
        assert len(all_intents) == 3

    def test_list_names(self):
        reg = self._build_registry()
        names = reg.list_names()
        assert "product_recommendation" in names
        assert "order_query" in names
        assert "refund" in names

    def test_list_root_intents(self):
        reg = self._build_registry()
        roots = reg.list_root_intents()
        root_names = [d.name for d in roots]
        assert "product_recommendation" in root_names
        assert "order_query" in root_names
        assert "refund" not in root_names

    def test_list_children(self):
        reg = self._build_registry()
        children = reg.list_children("order_query")
        assert len(children) == 1
        assert children[0].name == "refund"

    def test_has_intent(self):
        reg = self._build_registry()
        assert reg.has_intent("product_recommendation") is True
        assert reg.has_intent("nonexistent") is False

    def test_get_required_slots(self):
        reg = self._build_registry()
        required = reg.get_required_slots("product_recommendation")
        assert required == ["category"]

    def test_get_all_slots(self):
        reg = self._build_registry()
        all_slots = reg.get_all_slots("product_recommendation")
        assert "category" in all_slots
        assert "budget_max" in all_slots

    def test_build_prompt_description(self):
        reg = self._build_registry()
        desc = reg.build_prompt_description()
        assert "product_recommendation" in desc
        assert "推荐商品" in desc

    def test_build_slot_description(self):
        reg = self._build_registry()
        desc = reg.build_slot_description("product_recommendation")
        assert "category" in desc
        assert "必选" in desc
        assert "budget_max" in desc
        assert "可选" in desc


class TestConfig:
    """Test configuration."""

    def test_defaults(self):
        cfg = IntentRecognitionConfig()
        assert cfg.confidence.accept_threshold == 0.85
        assert cfg.confidence.clarify_threshold == 0.6
        assert cfg.clarification.max_consecutive_clarifications == 3
        assert cfg.evaluation.top_k == 3

    def test_multi_signal_weights(self):
        cfg = IntentRecognitionConfig()
        total = (
            cfg.confidence.weight_llm_confidence
            + cfg.confidence.weight_rule_match
            + cfg.confidence.weight_vector_similarity
            + cfg.confidence.weight_historical_accuracy
        )
        assert abs(total - 1.0) < 0.01

    def test_failure_signals(self):
        cfg = IntentRecognitionConfig()
        assert "你理解错了" in cfg.failure_signals
        assert len(cfg.failure_signals) >= 3


class TestD26D31Models:
    """Tests for D26/D27/D31 model extensions (Evidence, SlotValue, alias)."""

    def test_evidence_grade_enum(self):
        assert EvidenceGrade.VERIFIED.value == "verified"
        assert EvidenceGrade.PROVISIONAL.value == "provisional"

    def test_evidence_model(self):
        ev = Evidence(content="当前输入：我要M码", grade=EvidenceGrade.VERIFIED, source="current_input")
        assert ev.content == "当前输入：我要M码"
        assert ev.grade == EvidenceGrade.VERIFIED
        assert ev.source == "current_input"

    def test_slot_value_defaults_verified(self):
        sv = SlotValue(value="M")
        assert sv.value == "M"
        assert sv.evidence_grade == EvidenceGrade.VERIFIED
        assert sv.evidence_ref is None

    def test_slot_value_provisional(self):
        sv = SlotValue(value="M", evidence_grade=EvidenceGrade.PROVISIONAL, evidence_ref="历史画像")
        assert sv.evidence_grade == EvidenceGrade.PROVISIONAL
        assert sv.evidence_ref == "历史画像"

    def test_result_d26_d31_defaults(self):
        r = IntentRecognitionResult()
        assert r.normalized_query == ""
        assert r.sub_tasks == []
        assert r.independent_intents == []
        assert r.verified_evidence == []
        assert r.provisional_evidence == []
        assert r.assumptions_disclosed is False
        assert r.assumptions == []
        assert r.arbitration_breakdown == {}

    def test_sub_intents_alias_sync_from_independent(self):
        """Setting independent_intents should sync sub_intents (deprecated alias)."""
        r = IntentRecognitionResult(independent_intents=["a", "b"])
        assert r.sub_intents == ["a", "b"]
        assert r.independent_intents == ["a", "b"]

    def test_sub_intents_alias_sync_from_sub_intents(self):
        """Setting sub_intents (legacy) should sync independent_intents."""
        r = IntentRecognitionResult(sub_intents=["x", "y"])
        assert r.independent_intents == ["x", "y"]

    def test_get_slot_evidence_wraps_plain_value(self):
        """get_slot_evidence wraps plain values as verified SlotValue."""
        r = IntentRecognitionResult(slots={"size": "M"})
        sv = r.get_slot_evidence("size")
        assert sv is not None
        assert sv.value == "M"
        assert sv.evidence_grade == EvidenceGrade.VERIFIED

    def test_get_slot_evidence_returns_none_for_missing(self):
        r = IntentRecognitionResult(slots={"size": "M"})
        assert r.get_slot_evidence("color") is None


class TestD26D31Config:
    """Tests for D26-D31 config groups."""

    def test_evidence_config_defaults(self):
        cfg = IntentRecognitionConfig()
        assert cfg.evidence.enable_grading is True
        assert cfg.evidence.require_verified_for_hard_ops is True
        assert cfg.evidence.high_risk_intents == []

    def test_boundary_config_defaults(self):
        cfg = IntentRecognitionConfig()
        assert cfg.boundary.enable_sub_tasks is True
        assert cfg.boundary.strict_mode is False

    def test_arbitration_config_defaults(self):
        cfg = IntentRecognitionConfig()
        assert cfg.arbitration.enable_five_factor is True
        assert cfg.arbitration.candidate_gap_threshold == 0.1
        assert cfg.arbitration.risk_aware_clarify is True

    def test_protocol_config_defaults(self):
        cfg = IntentRecognitionConfig()
        assert cfg.protocol.enable_structured_output is True
        assert cfg.protocol.deprecate_sub_intents is False

    def test_extended_evaluation_config_defaults(self):
        cfg = IntentRecognitionConfig()
        assert cfg.extended_evaluation.enable_confusion_matrix is True
        assert cfg.extended_evaluation.online_feedback_loop is True
