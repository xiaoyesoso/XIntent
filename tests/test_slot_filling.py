"""Tests for slot filling (D7, D8, D9, D26)."""

import json

from user_input_normalization.llm.mock import MockLLMClient

from intent_recognition import (
    Constraint,
    ConstraintType,
    Evidence,
    EvidenceGrade,
    IntentDefinition,
    IntentRecognitionConfig,
    IntentRecognitionPipeline,
    IntentRegistry,
    SlotDefinition,
    SlotState,
    SlotValue,
)
from intent_recognition.slot_filling import (
    ConstraintExtractor,
    CrossTurnSlotMerger,
    SlotExtractor,
)
from intent_recognition.storage import MemorySlotStateStore


def _build_registry() -> IntentRegistry:
    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="product_recommendation",
        description="推荐商品",
        slots=[
            SlotDefinition(name="category", required=True, description="商品类目"),
            SlotDefinition(name="budget_max", required=False, description="最高预算"),
            SlotDefinition(name="usage_scenario", required=False, description="使用场景"),
        ],
    ))
    return reg


class TestSlotExtractor:
    """Test slot extraction from user input."""

    def test_slot_extraction(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "slots": {
                "category": "笔记本电脑",
                "usage_scenario": "写代码",
            },
            "missing_slots": ["budget_max"],
        }))
        reg = _build_registry()
        extractor = SlotExtractor(llm, reg)
        slots, missing = extractor.extract(
            "帮我推荐一台笔记本，主要是写代码，最好轻一点",
            "product_recommendation",
        )
        assert slots.get("category") == "笔记本电脑"
        assert slots.get("usage_scenario") == "写代码"
        # budget_max is optional but might be listed as missing by LLM
        # The extractor also cross-checks required slots
        assert "category" not in missing  # category was extracted

    def test_missing_slots_detection(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "slots": {},
            "missing_slots": ["category"],
        }))
        reg = _build_registry()
        extractor = SlotExtractor(llm, reg)
        slots, missing = extractor.extract("帮我推荐个东西", "product_recommendation")
        assert "category" in missing

    def test_unknown_intent_returns_empty(self):
        llm = MockLLMClient()
        reg = _build_registry()
        extractor = SlotExtractor(llm, reg)
        slots, missing = extractor.extract("test", "nonexistent")
        assert slots == {}
        assert missing == []

    def test_no_slots_defined(self):
        llm = MockLLMClient()
        reg = IntentRegistry()
        reg.register(IntentDefinition(name="simple", description="No slots"))
        extractor = SlotExtractor(llm, reg)
        slots, missing = extractor.extract("test", "simple")
        assert slots == {}
        assert missing == []

    def test_unparseable_response(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: "not json")
        reg = _build_registry()
        extractor = SlotExtractor(llm, reg)
        slots, missing = extractor.extract("test", "product_recommendation")
        # Should return required slots as missing
        assert "category" in missing


class TestConstraintExtractor:
    """Test hard/soft constraint extraction."""

    def test_hard_constraint_regex(self):
        extractor = ConstraintExtractor()
        constraints = extractor.extract("帮我推荐裤子，不超过100块")
        hard = [c for c in constraints if c.type == ConstraintType.HARD]
        assert any("price<100" in c.expression or "100" in c.expression for c in hard)

    def test_soft_constraint_regex(self):
        extractor = ConstraintExtractor()
        constraints = extractor.extract("最好是牛仔裤")
        soft = [c for c in constraints if c.type == ConstraintType.SOFT]
        assert len(soft) >= 1

    def test_mixed_constraints(self):
        extractor = ConstraintExtractor()
        constraints = extractor.extract("帮我推荐裤子，不超过100块，最好是牛仔裤")
        hard = [c for c in constraints if c.type == ConstraintType.HARD]
        soft = [c for c in constraints if c.type == ConstraintType.SOFT]
        assert len(hard) >= 1
        assert len(soft) >= 1

    def test_extract_separated(self):
        extractor = ConstraintExtractor()
        hard, soft = extractor.extract_separated("不超过100块，最好是牛仔裤")
        assert len(hard) >= 1
        assert len(soft) >= 1

    def test_no_constraints(self):
        extractor = ConstraintExtractor()
        constraints = extractor.extract("帮我推荐个手机")
        assert len(constraints) == 0

    def test_with_llm_extraction(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "hard_constraints": [{"expression": "price<200", "raw_text": "200以内"}],
            "soft_constraints": [{"expression": "color=black", "raw_text": "黑色更好"}],
        }))
        extractor = ConstraintExtractor(llm)
        constraints = extractor.extract("200以内，黑色更好")
        hard = [c for c in constraints if c.type == ConstraintType.HARD]
        soft = [c for c in constraints if c.type == ConstraintType.SOFT]
        # Regex + LLM constraints
        assert len(hard) >= 1
        assert len(soft) >= 1


class TestCrossTurnSlotMerger:
    """Test cross-turn slot accumulation."""

    def test_merge_across_turns(self):
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)

        # Turn 1: category only
        state = merger.merge("s1", {"category": "裤子"}, turn=1)
        assert state.slots["category"] == "裤子"
        assert "budget_max" not in state.slots

        # Turn 2: add budget
        state = merger.merge("s1", {"budget_max": 100}, turn=2)
        assert state.slots["category"] == "裤子"
        assert state.slots["budget_max"] == 100

    def test_latest_wins_conflict(self):
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)

        merger.merge("s1", {"budget_max": 100}, turn=1)
        state = merger.merge("s1", {"budget_max": 200}, turn=2)
        # Latest wins
        assert state.slots["budget_max"] == 200
        # Conflict logged
        conflicts = merger.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["key"] == "budget_max"
        assert conflicts[0]["old_value"] == 100
        assert conflicts[0]["new_value"] == 200

    def test_get_missing_required(self):
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)
        reg = _build_registry()

        state = merger.merge("s1", {"category": "手机"}, turn=1)
        missing = merger.get_missing_required(state, "product_recommendation", reg)
        # category is filled, budget_max is optional so not in required
        # Only "category" is required and it's filled
        assert "category" not in missing

    def test_update_missing_slots(self):
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)
        reg = _build_registry()

        state = merger.merge("s1", {"budget_max": 100}, turn=1, intent="product_recommendation")
        state = merger.update_missing_slots(state, "product_recommendation", reg)
        # category is required but not provided
        assert "category" in state.missing_slots

    def test_constraint_accumulation(self):
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)

        hard1 = [Constraint(type=ConstraintType.HARD, expression="price<100", raw_text="不超过100")]
        state = merger.merge("s1", {}, turn=1, hard_constraints=hard1)
        assert len(state.hard_constraints) == 1

        # Turn 2: add more constraints (should accumulate, not replace)
        soft1 = [Constraint(type=ConstraintType.SOFT, expression="type=牛仔裤", raw_text="最好是牛仔裤")]
        state = merger.merge("s1", {}, turn=2, soft_constraints=soft1)
        assert len(state.hard_constraints) == 1  # Still there from turn 1
        assert len(state.soft_constraints) == 1  # New from turn 2

    def test_constraint_persistence(self):
        """Constraints persist through entire conversation."""
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)

        hard = [Constraint(type=ConstraintType.HARD, expression="price<100", raw_text="不超过100")]
        merger.merge("s1", {"category": "裤子"}, turn=1, hard_constraints=hard)

        # Turn 2: no new constraints
        state = merger.merge("s1", {"budget_max": 100}, turn=2)
        assert len(state.hard_constraints) == 1  # Still persisted

        # Turn 3: still persisted
        state = merger.merge("s1", {"usage_scenario": "日常"}, turn=3)
        assert len(state.hard_constraints) == 1
        assert state.hard_constraints[0].expression == "price<100"

    def test_constraint_conflict_detection(self):
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)

        state = SlotState(session_id="s1")
        state.hard_constraints.append(
            Constraint(type=ConstraintType.HARD, expression="price<100", raw_text="")
        )
        # Same expression, different type -> conflict
        new_soft = [Constraint(type=ConstraintType.SOFT, expression="price<100", raw_text="")]
        assert CrossTurnSlotMerger.has_constraint_conflict(state, new_soft) is True

    def test_no_conflict_same_type(self):
        state = SlotState(session_id="s1")
        state.hard_constraints.append(
            Constraint(type=ConstraintType.HARD, expression="price<100", raw_text="")
        )
        # Same expression, same type -> no conflict (duplicate, but not conflicting)
        new_hard = [Constraint(type=ConstraintType.HARD, expression="price<100", raw_text="")]
        assert CrossTurnSlotMerger.has_constraint_conflict(state, new_hard) is False

    def test_clear_conflicts(self):
        store = MemorySlotStateStore()
        merger = CrossTurnSlotMerger(store)
        merger.merge("s1", {"budget": 100}, turn=1)
        merger.merge("s1", {"budget": 200}, turn=2)
        assert len(merger.get_conflicts()) == 1
        merger.clear_conflicts()
        assert len(merger.get_conflicts()) == 0


# ---------------------------------------------------------------------------
# D26: Evidence grading tests (tasks 10.2, 10.3, 10.4)
# ---------------------------------------------------------------------------


def _build_d26_registry() -> IntentRegistry:
    """Build a registry with a high-risk intent for D26 hard-op tests."""
    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="product_recommendation",
        description="推荐商品",
        slots=[
            SlotDefinition(name="category", required=True, description="商品类目"),
            SlotDefinition(name="budget_max", required=False, description="最高预算"),
        ],
        positive_examples=["推荐手机", "有什么耳机推荐"],
    ))
    reg.register(IntentDefinition(
        name="refund",
        description="申请退款（高风险）",
        slots=[
            SlotDefinition(name="order_id", required=True, description="订单号"),
            SlotDefinition(name="reason", required=True, description="退款原因"),
        ],
        positive_examples=["我要退款", "申请退货"],
    ))
    return reg


def _build_d26_pipeline(
    config: IntentRecognitionConfig | None = None,
    llm: MockLLMClient | None = None,
) -> IntentRecognitionPipeline:
    llm = llm or MockLLMClient()
    reg = _build_d26_registry()
    cfg = config or IntentRecognitionConfig()
    return IntentRecognitionPipeline(llm, reg, cfg)


class TestD26EvidenceGrading:
    """Tests for D26 evidence grading (tasks 10.2, 10.3, 10.4)."""

    def test_evidence_collected_for_verified_slots(self):
        """Task 10.2: slots from current input are tagged verified."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_d26_pipeline(llm=llm)
        result = pipe.recognize("帮我推荐一个手机")
        # verified_evidence should include the current input + the slot
        assert any(ev.grade == EvidenceGrade.VERIFIED for ev in result.verified_evidence)
        assert any("当前输入" in ev.content for ev in result.verified_evidence)
        assert all(ev.grade == EvidenceGrade.VERIFIED for ev in result.verified_evidence)

    def test_evidence_collected_for_provisional_slots(self):
        """Task 10.2: slots tagged provisional are collected as such."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": {"value": "手机", "evidence_grade": "provisional"}},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_d26_pipeline(llm=llm)
        result = pipe.recognize("帮我推荐一个手机")
        # provisional_evidence should be non-empty (slot dict tagged provisional)
        assert len(result.provisional_evidence) >= 1
        assert all(ev.grade == EvidenceGrade.PROVISIONAL for ev in result.provisional_evidence)

    def test_hard_op_check_triggers_clarification(self):
        """Task 10.3: high-risk intent with provisional required slot triggers clarify."""
        cfg = IntentRecognitionConfig()
        cfg.evidence.high_risk_intents = ["refund"]
        cfg.evidence.require_verified_for_hard_ops = True
        # Refund has required slots order_id, reason - mark them provisional
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "refund",
            "confidence": 0.92,
            "slots": {
                "order_id": {"value": "12345", "evidence_grade": "provisional"},
                "reason": {"value": "质量问题", "evidence_grade": "provisional"},
            },
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_d26_pipeline(config=cfg, llm=llm)
        result = pipe.recognize("我要退款")
        assert result.need_clarification is True
        assert result.clarification_question is not None

    def test_hard_op_check_discloses_assumptions_when_max_clarify(self):
        """Task 10.3: when max clarifications exceeded, assumptions are disclosed."""
        cfg = IntentRecognitionConfig()
        cfg.evidence.high_risk_intents = ["refund"]
        cfg.evidence.require_verified_for_hard_ops = True
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "refund",
            "confidence": 0.92,
            "slots": {
                "order_id": {"value": "12345", "evidence_grade": "provisional"},
                "reason": {"value": "质量问题", "evidence_grade": "provisional"},
            },
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_d26_pipeline(config=cfg, llm=llm)
        # Exceed max clarifications by repeatedly recording unresolved attempts
        for t in range(cfg.clarification.max_consecutive_clarifications + 1):
            pipe._reject_clarify.track_convergence("session_hard_op", t, resolved=False)
        result = pipe.recognize("我要退款", session_id="session_hard_op")
        assert result.assumptions_disclosed is True
        assert len(result.assumptions) >= 1

    def test_hard_op_check_disabled(self):
        """Task 10.3: when require_verified_for_hard_ops=False, no clarify triggers."""
        cfg = IntentRecognitionConfig()
        cfg.evidence.high_risk_intents = ["refund"]
        cfg.evidence.require_verified_for_hard_ops = False  # disabled
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "refund",
            "confidence": 0.92,
            "slots": {
                "order_id": {"value": "12345", "evidence_grade": "provisional"},
                "reason": {"value": "质量问题", "evidence_grade": "provisional"},
            },
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_d26_pipeline(config=cfg, llm=llm)
        result = pipe.recognize("我要退款")
        # Without hard-op check, no clarification triggered by D26
        # (may still be triggered by other checks, but not by hard-op)
        # Verify assumptions are NOT disclosed
        assert result.assumptions_disclosed is False

    def test_upgrade_provisional_to_verified(self):
        """Task 10.4: provisional slot upgraded to verified via user confirmation."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": {"value": "手机", "evidence_grade": "provisional"}},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_d26_pipeline(llm=llm)
        result = pipe.recognize("帮我推荐一个手机")
        # Before upgrade, slot is provisional
        assert len(result.provisional_evidence) >= 1
        # Upgrade the slot
        pipe._upgrade_provisional_to_verified(
            result, "category", "手机", source="user_confirmation"
        )
        # After upgrade, the slot is verified
        sv = result.get_slot_evidence("category")
        assert sv is not None
        assert sv.evidence_grade == EvidenceGrade.VERIFIED
        # And the evidence moved from provisional to verified
        assert any("用户确认" in ev.content for ev in result.verified_evidence)

    def test_cannot_downgrade_verified_to_provisional(self):
        """Task 10.4: verified evidence cannot be downgraded."""
        # There is no downgrade method; this test documents the contract by
        # ensuring _upgrade_provisional_to_verified only upgrades. A verified
        # slot passed through _upgrade_provisional_to_verified stays verified.
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": "手机"},  # plain value -> verified
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_d26_pipeline(llm=llm)
        result = pipe.recognize("帮我推荐一个手机")
        sv_before = result.get_slot_evidence("category")
        assert sv_before.evidence_grade == EvidenceGrade.VERIFIED
        # Running upgrade on an already-verified slot is a no-op for grade
        pipe._upgrade_provisional_to_verified(result, "category", "手机")
        sv_after = result.get_slot_evidence("category")
        assert sv_after.evidence_grade == EvidenceGrade.VERIFIED
