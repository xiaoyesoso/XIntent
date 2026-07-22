"""Tests for the lightweight LLM layer (Layer 2) - D4, D5, D12.

Uses MockLLMClient from user_input_normalization.llm.mock to avoid real
API calls. Covers prompt building, confidence routing, multi-signal
fusion, and the LightweightLLMClassifier end-to-end.
"""

import json

from intent_recognition import (
    IntentDefinition,
    IntentRecognitionConfig,
    IntentRegistry,
    RecognitionSource,
    SlotDefinition,
)
from intent_recognition.config import ArbitrationConfig, ConfidenceConfig
from intent_recognition.lightweight_llm import (
    CAN_DO,
    CANNOT_DO,
    ArbitrationInput,
    ConfidenceRouter,
    LightweightLLMClassifier,
    MultiSignalFuser,
    build_system_prompt,
    build_user_prompt,
)
from user_input_normalization.llm.mock import MockLLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_registry() -> IntentRegistry:
    """Build a small registry with two intents for testing."""
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
    return reg


def _llm_response(
    intent: str | None = "product_recommendation",
    confidence: float = 0.9,
    slots: dict | None = None,
    missing_slots: list[str] | None = None,
    need_clarification: bool = False,
    clarification_question: str | None = None,
    rejection_reason: str | None = None,
    hard_constraints: list | None = None,
    soft_constraints: list | None = None,
) -> str:
    """Build a canned LLM JSON response string."""
    payload = {
        "intent": intent,
        "confidence": confidence,
        "slots": slots or {},
        "missing_slots": missing_slots or [],
        "need_clarification": need_clarification,
        "clarification_question": clarification_question,
        "hard_constraints": hard_constraints or [],
        "soft_constraints": soft_constraints or [],
        "rejection_reason": rejection_reason,
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt building (D12)
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Prompt construction includes intent boundaries and slot definitions."""

    def test_system_prompt_contains_intent_names(self):
        reg = _build_registry()
        prompt = build_system_prompt(reg)
        assert "product_recommendation" in prompt
        assert "order_query" in prompt

    def test_system_prompt_contains_descriptions(self):
        reg = _build_registry()
        prompt = build_system_prompt(reg)
        assert "用户希望推荐商品" in prompt
        assert "用户希望查询订单" in prompt

    def test_system_prompt_contains_positive_examples(self):
        """Positive examples form the intent's matching boundary."""
        reg = _build_registry()
        prompt = build_system_prompt(reg)
        assert "推荐手机" in prompt
        assert "我的订单到哪了" in prompt

    def test_system_prompt_contains_negative_examples(self):
        """Negative examples form the intent's rejection boundary."""
        reg = _build_registry()
        prompt = build_system_prompt(reg)
        assert "查询订单状态 -> order_query" in prompt

    def test_system_prompt_contains_slot_definitions(self):
        reg = _build_registry()
        prompt = build_system_prompt(reg)
        assert "category" in prompt
        assert "budget_max" in prompt
        assert "必选" in prompt
        assert "可选" in prompt

    def test_system_prompt_contains_can_do_boundary(self):
        reg = _build_registry()
        prompt = build_system_prompt(reg)
        assert CAN_DO in prompt
        assert "识别用户输入对应的意图" in prompt

    def test_system_prompt_contains_cannot_do_boundary(self):
        reg = _build_registry()
        prompt = build_system_prompt(reg)
        assert CANNOT_DO in prompt
        assert "不直接回答用户问题" in prompt

    def test_user_prompt_includes_input_text(self):
        prompt = build_user_prompt("帮我推荐手机")
        assert "帮我推荐手机" in prompt

    def test_user_prompt_with_few_shot_examples(self):
        examples = [
            {"input": "推荐手机", "output": {"intent": "product_recommendation"}},
            {"input": "查订单", "output": {"intent": "order_query"}},
        ]
        prompt = build_user_prompt("帮我推荐手机", few_shot_examples=examples)
        assert "参考例子" in prompt
        assert "推荐手机" in prompt
        assert "product_recommendation" in prompt

    def test_user_prompt_without_few_shot(self):
        prompt = build_user_prompt("hello", few_shot_examples=None)
        assert "参考例子" not in prompt
        assert "hello" in prompt


# ---------------------------------------------------------------------------
# Confidence router (D4)
# ---------------------------------------------------------------------------


class TestConfidenceRouter:
    """D4: confidence-based routing."""

    def test_high_confidence_accepts(self):
        router = ConfidenceRouter()
        assert router.route(0.9) == "accept"

    def test_at_accept_threshold_accepts(self):
        """>= 0.85 -> accept (boundary inclusive)."""
        router = ConfidenceRouter()
        assert router.route(0.85) == "accept"

    def test_medium_confidence_clarifies(self):
        router = ConfidenceRouter()
        assert router.route(0.7) == "clarify"

    def test_at_clarify_threshold_clarifies(self):
        """>= 0.6 and < 0.85 -> clarify (boundary inclusive)."""
        router = ConfidenceRouter()
        assert router.route(0.6) == "clarify"

    def test_low_confidence_escalates(self):
        router = ConfidenceRouter()
        assert router.route(0.3) == "escalate"

    def test_zero_confidence_escalates(self):
        router = ConfidenceRouter()
        assert router.route(0.0) == "escalate"

    def test_custom_thresholds(self):
        from intent_recognition.config import ConfidenceConfig
        cfg = ConfidenceConfig(accept_threshold=0.9, clarify_threshold=0.5)
        router = ConfidenceRouter(cfg)
        assert router.route(0.85) == "clarify"
        assert router.route(0.92) == "accept"
        assert router.route(0.4) == "escalate"


# ---------------------------------------------------------------------------
# Multi-signal fuser (D5)
# ---------------------------------------------------------------------------


class TestMultiSignalFuser:
    """D5: multi-signal confidence fusion."""

    def test_basic_fusion_returns_score_and_breakdown(self):
        fuser = MultiSignalFuser()
        score, breakdown = fuser.fuse(
            llm_confidence=0.8,
            rule_match=False,
            vector_sim=0.7,
            historical_acc=0.9,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert "llm_confidence" in breakdown
        assert "rule_match" in breakdown
        assert "vector_similarity" in breakdown
        assert "historical_accuracy" in breakdown
        assert breakdown["llm_confidence"] == 0.8
        assert breakdown["rule_match"] == 0.0

    def test_rule_match_boosts_confidence(self):
        """A rule match should increase fused confidence over no rule match."""
        fuser = MultiSignalFuser()
        without_rule, _ = fuser.fuse(
            llm_confidence=0.7,
            rule_match=False,
            vector_sim=0.5,
            historical_acc=0.5,
        )
        with_rule, _ = fuser.fuse(
            llm_confidence=0.7,
            rule_match=True,
            vector_sim=0.5,
            historical_acc=0.5,
        )
        assert with_rule > without_rule

    def test_weights_sum_to_one_default(self):
        """Default weights should sum to 1.0."""
        cfg = IntentRecognitionConfig()
        total = (
            cfg.confidence.weight_llm_confidence
            + cfg.confidence.weight_rule_match
            + cfg.confidence.weight_vector_similarity
            + cfg.confidence.weight_historical_accuracy
        )
        assert abs(total - 1.0) < 0.01

    def test_default_weights_match_config(self):
        """weight_llm=0.5, weight_rule=0.2, weight_vector=0.2, weight_hist=0.1."""
        fuser = MultiSignalFuser()
        _, breakdown = fuser.fuse(0.5, False, 0.5, 0.5)
        assert breakdown["weight_llm"] == 0.5
        assert breakdown["weight_rule"] == 0.2
        assert breakdown["weight_vector"] == 0.2
        assert breakdown["weight_historical"] == 0.1

    def test_fused_value_matches_weighted_average(self):
        fuser = MultiSignalFuser()
        score, _ = fuser.fuse(
            llm_confidence=1.0,
            rule_match=True,
            vector_sim=1.0,
            historical_acc=1.0,
        )
        # All signals at 1.0 -> fused should be 1.0
        assert abs(score - 1.0) < 0.001

    def test_inputs_clamped_to_unit_interval(self):
        fuser = MultiSignalFuser()
        score, breakdown = fuser.fuse(
            llm_confidence=1.5,  # over clamp
            rule_match=True,
            vector_sim=-0.2,  # under clamp
            historical_acc=0.5,
        )
        assert breakdown["llm_confidence"] == 1.0
        assert breakdown["vector_similarity"] == 0.0
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# LightweightLLMClassifier end-to-end (D4, D5, D12)
# ---------------------------------------------------------------------------


class TestLightweightLLMClassifier:
    """End-to-end L2 tests using MockLLMClient."""

    def test_high_confidence_accepted(self):
        """Confidence 0.9 -> accept (no clarification flag set by router)."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="product_recommendation",
                confidence=0.9,
                slots={"category": "手机"},
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("推荐手机")
        assert result.intent == "product_recommendation"
        assert result.source == RecognitionSource.LIGHTWEIGHT_LLM
        assert result.layer_reached == 2
        # 0.9 -> router says accept, so clarification should be False
        assert result.need_clarification is False

    def test_medium_confidence_clarifies(self):
        """Confidence 0.7 -> clarify."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="product_recommendation",
                confidence=0.7,
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("推荐个东西")
        assert result.intent == "product_recommendation"
        # 0.7 -> router says clarify
        assert result.need_clarification is True
        assert result.clarification_question is not None

    def test_low_confidence_escalates(self):
        """Confidence 0.3 -> escalate (clarification flag cleared)."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="product_recommendation",
                confidence=0.3,
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("嗯")
        # Escalate path: do not set clarification (Layer 3 will handle)
        assert result.need_clarification is False
        # Confidence is still the fused value (low)
        assert result.confidence < 0.6

    def test_slots_extracted_from_llm_response(self):
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="product_recommendation",
                confidence=0.9,
                slots={"category": "手机", "budget_max": 3000},
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("推荐手机预算3000")
        assert result.slots["category"] == "手机"
        assert result.slots["budget_max"] == 3000

    def test_missing_required_slots_detected(self):
        """Required slot 'category' missing -> added to missing_slots."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="product_recommendation",
                confidence=0.9,
                slots={},  # no category provided
                missing_slots=[],
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("推荐个东西")
        assert "category" in result.missing_slots

    def test_constraints_parsed(self):
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="product_recommendation",
                confidence=0.9,
                slots={"category": "手机"},
                hard_constraints=[
                    {"type": "hard", "expression": "price<1000", "raw_text": "1000以内"}
                ],
                soft_constraints=[
                    {"type": "soft", "expression": "brand=华为", "raw_text": "华为"}
                ],
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("推荐1000以内的华为手机")
        assert len(result.hard_constraints) == 1
        assert result.hard_constraints[0].expression == "price<1000"
        assert len(result.soft_constraints) == 1
        assert result.soft_constraints[0].expression == "brand=华为"

    def test_unknown_intent_rejected(self):
        """LLM returns an intent not in registry -> treated as unsupported."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="nonexistent_intent",
                confidence=0.9,
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("whatever")
        assert result.intent is None
        assert result.rejection_reason is not None
        assert "nonexistent_intent" in result.rejection_reason

    def test_json_in_fenced_block_parsed(self):
        """LLM may return JSON inside ```json code blocks (pitfall guard)."""
        mock = MockLLMClient()
        fenced = "```json\n" + _llm_response(
            intent="order_query", confidence=0.9
        ) + "\n```"
        mock.set_default_handler(lambda sys_p, usr_p: fenced)
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("查询订单")
        assert result.intent == "order_query"

    def test_invalid_json_response_escalates(self):
        """When LLM returns non-JSON, classifier escalates."""
        mock = MockLLMClient()
        mock.set_default_handler(lambda sys_p, usr_p: "this is not json at all")
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("blah")
        assert result.intent is None
        assert result.rejection_reason is not None
        assert result.confidence == 0.0

    def test_few_shot_injection_in_user_prompt(self):
        """Few-shot examples are included in the user prompt sent to LLM."""
        mock = MockLLMClient()
        captured: dict = {}

        def capture_handler(sys_p, usr_p):
            captured["user"] = usr_p
            return _llm_response(intent="order_query", confidence=0.9)

        mock.set_default_handler(capture_handler)
        clf = LightweightLLMClassifier(mock, _build_registry())
        clf.classify(
            "查订单",
            few_shot_examples=[
                {"input": "我的订单", "output": {"intent": "order_query"}},
            ],
        )
        assert "参考例子" in captured["user"]
        assert "我的订单" in captured["user"]

    def test_multi_signal_rule_match_boosts_confidence(self):
        """A rule match should boost fused confidence over no rule match."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(
                intent="product_recommendation",
                confidence=0.7,
            )
        )
        clf = LightweightLLMClassifier(mock, _build_registry())

        without_rule = clf.classify("推荐手机", rule_matched=False)
        # Reuse same mock (stateless handler)
        with_rule = clf.classify("推荐手机", rule_matched=True)
        assert with_rule.confidence > without_rule.confidence
        # Breakdown should be present in signals
        assert "rule_match" in with_rule.signals
        assert with_rule.signals["rule_match"] == 1.0

    def test_signals_breakdown_attached_to_result(self):
        """Result.signals contains the multi-signal breakdown."""
        mock = MockLLMClient()
        mock.set_default_handler(
            lambda sys_p, usr_p: _llm_response(confidence=0.8)
        )
        clf = LightweightLLMClassifier(mock, _build_registry())
        result = clf.classify("推荐手机")
        assert "llm_confidence" in result.signals
        assert "fused" in result.signals

    def test_system_prompt_cached_and_sent(self):
        """The system prompt is built once and passed to the LLM."""
        mock = MockLLMClient()
        captured: dict = {}

        def capture_handler(sys_p, usr_p):
            captured["system"] = sys_p
            return _llm_response(confidence=0.9)

        mock.set_default_handler(capture_handler)
        clf = LightweightLLMClassifier(mock, _build_registry())
        clf.classify("推荐手机")
        assert "product_recommendation" in captured["system"]
        assert "必选" in captured["system"]  # slot definition marker

    def test_confusing_intents_with_few_shot(self):
        """Few-shot examples disambiguate confusing intents.

        Scenario: user input "我要退款" could be confusing, but few-shot
        examples steer the LLM toward the correct intent.
        """
        # Build a registry that also has a refund intent
        reg = _build_registry()
        reg.register(IntentDefinition(
            name="refund",
            description="用户希望退款",
            positive_examples=["我要退款", "退货退款"],
        ))

        mock = MockLLMClient()
        # Handler responds based on few-shot presence
        def handler(sys_p, usr_p):
            if "退款" in usr_p:
                return _llm_response(intent="refund", confidence=0.88)
            return _llm_response(intent="order_query", confidence=0.5)

        mock.set_default_handler(handler)
        clf = LightweightLLMClassifier(mock, reg)
        result = clf.classify(
            "我要退款",
            few_shot_examples=[
                {"input": "我要退款", "output": {"intent": "refund"}},
            ],
        )
        assert result.intent == "refund"
        assert result.confidence >= 0.85


# ---------------------------------------------------------------------------
# D28: Five-factor arbitration tests (tasks 10.6, 10.7)
# ---------------------------------------------------------------------------


def _arb_config(
    high_risk_intents: list[str] | None = None,
    enable_five_factor: bool = True,
    risk_aware_clarify: bool = True,
) -> IntentRecognitionConfig:
    """Build a config with D28 arbitration enabled for testing."""
    cfg = IntentRecognitionConfig()
    cfg.arbitration.enable_five_factor = enable_five_factor
    cfg.arbitration.risk_aware_clarify = risk_aware_clarify
    if high_risk_intents is not None:
        cfg.arbitration.high_risk_intents = high_risk_intents
    return cfg


class TestD28FiveFactorArbitration:
    """Tests for D28 five-factor arbitration (task 10.6)."""

    def test_factor2_missing_slots_forces_clarify(self):
        """Factor 2: missing required slots -> Clarify."""
        router = ConfidenceRouter(config=ConfidenceConfig(), arbitration_config=ArbitrationConfig())
        router._arb_config.enable_five_factor = True
        decision = router.arbitrate(ArbitrationInput(
            confidence=0.7,
            intent="product_recommendation",
            missing_slots=["category"],
        ))
        assert decision.decision == "clarify"
        assert decision.factor_results["slot_completeness"] == "fail"
        assert "missing required slots" in decision.reason

    def test_factor3_high_risk_provisional_forces_clarify(self):
        """Factor 3: high-risk intent + provisional required slot -> Clarify."""
        cfg = _arb_config(high_risk_intents=["refund"])
        router = ConfidenceRouter(
            config=cfg.confidence, arbitration_config=cfg.arbitration,
        )
        decision = router.arbitrate(ArbitrationInput(
            confidence=0.75,
            intent="refund",
            missing_slots=[],
            provisional_required_slots=["order_id"],
            high_risk_intents=["refund"],
        ))
        assert decision.decision == "clarify"
        assert decision.factor_results["hard_constraint_risk"] == "fail"
        assert "high-risk" in decision.reason

    def test_factor4_candidate_gap_forces_escalate(self):
        """Factor 4: Top1-Top2 gap below threshold -> Escalate to L3."""
        cfg = _arb_config()
        router = ConfidenceRouter(
            config=cfg.confidence, arbitration_config=cfg.arbitration,
        )
        decision = router.arbitrate(ArbitrationInput(
            confidence=0.72,
            intent="product_recommendation",
            missing_slots=[],
            candidate_gap=0.03,  # < 0.1 threshold
        ))
        assert decision.decision == "escalate"
        assert decision.factor_results["candidate_gap"] == "fail"

    def test_factor5_low_history_low_confidence_forces_clarify(self):
        """Factor 5: poor history + low adjusted confidence -> Clarify."""
        cfg = _arb_config()
        router = ConfidenceRouter(
            config=cfg.confidence, arbitration_config=cfg.arbitration,
        )
        decision = router.arbitrate(ArbitrationInput(
            confidence=0.65,
            intent="product_recommendation",
            missing_slots=[],
            historical_acc=0.3,  # < 0.5
        ))
        assert decision.decision == "clarify"
        assert decision.factor_results["confidence_history"] == "fail"

    def test_all_factors_pass_returns_accept(self):
        """When all factors pass (or skip), arbitration accepts."""
        cfg = _arb_config()
        router = ConfidenceRouter(
            config=cfg.confidence, arbitration_config=cfg.arbitration,
        )
        decision = router.arbitrate(ArbitrationInput(
            confidence=0.8,
            intent="product_recommendation",
            missing_slots=[],
            rule_matched=True,
            candidate_gap=0.3,
            historical_acc=0.9,
        ))
        assert decision.decision == "accept"
        assert decision.factor_results["rule_validation"] == "pass"
        assert decision.factor_results["slot_completeness"] == "pass"
        assert decision.factor_results["hard_constraint_risk"] == "pass"
        assert decision.factor_results["candidate_gap"] == "pass"
        assert decision.factor_results["confidence_history"] == "pass"

    def test_factor1_rule_fail_lowers_adjusted_confidence(self):
        """Factor 1: rule_matched=False lowers adjusted confidence but doesn't reject."""
        cfg = _arb_config()
        router = ConfidenceRouter(
            config=cfg.confidence, arbitration_config=cfg.arbitration,
        )
        decision = router.arbitrate(ArbitrationInput(
            confidence=0.8,
            intent="product_recommendation",
            missing_slots=[],
            rule_matched=False,
            historical_acc=0.9,
        ))
        # Rule fail lowers confidence; factor 5 still passes (high history)
        # so decision is accept, but adjusted < original
        assert decision.factor_results["rule_validation"] == "fail"
        assert decision.adjusted_confidence < 0.8

    def test_arbitrate_l2_l3_accepts_high_conf_l3(self):
        """L2/L3 disagreement: L3 high-conf + complete slots -> accept L3."""
        from intent_recognition.models import IntentRecognitionResult, RecognitionSource
        cfg = _arb_config()
        router = ConfidenceRouter(
            config=cfg.confidence, arbitration_config=cfg.arbitration,
        )
        l2 = IntentRecognitionResult(
            intent="product_recommendation", confidence=0.5,
            source=RecognitionSource.LIGHTWEIGHT_LLM,
        )
        l3 = IntentRecognitionResult(
            intent="order_query", confidence=0.92,
            source=RecognitionSource.DEEP_LLM,
            missing_slots=[],
        )
        decision = router.arbitrate_l2_l3(l2, l3)
        assert decision.decision == "accept"
        assert "L3 high-confidence" in decision.reason

    def test_arbitrate_l2_l3_both_low_forces_clarify(self):
        """L2/L3 disagreement: both below clarify threshold -> force Clarify."""
        from intent_recognition.models import IntentRecognitionResult, RecognitionSource
        cfg = _arb_config()
        router = ConfidenceRouter(
            config=cfg.confidence, arbitration_config=cfg.arbitration,
        )
        l2 = IntentRecognitionResult(
            intent="product_recommendation", confidence=0.4,
            source=RecognitionSource.LIGHTWEIGHT_LLM,
        )
        l3 = IntentRecognitionResult(
            intent="order_query", confidence=0.5,
            source=RecognitionSource.DEEP_LLM,
        )
        decision = router.arbitrate_l2_l3(l2, l3)
        assert decision.decision == "clarify"
        assert "three-layer ambiguity" in decision.reason


class TestD28RiskAwareClarify:
    """Tests for D28 risk-aware Clarify at the classifier level (task 10.7)."""

    def test_high_risk_intent_triggers_clarify_in_ambiguous_zone(self):
        """High-risk intent in ambiguous zone triggers Clarify via factor 3."""
        reg = _build_registry()
        reg.register(IntentDefinition(
            name="refund",
            description="用户希望退款",
            slots=[SlotDefinition(name="order_id", required=True)],
            positive_examples=["我要退款"],
        ))
        cfg = _arb_config(high_risk_intents=["refund"])
        mock = MockLLMClient()
        # L2 returns ambiguous-zone confidence for high-risk intent
        def handler(_sys_p, _usr_p):
            return json.dumps({
                "intent": "refund",
                "confidence": 0.7,
                "slots": {"order_id": {"value": "123", "evidence_grade": "provisional"}},
                "missing_slots": [],
                "hard_constraints": [],
                "soft_constraints": [],
            })
        mock.set_default_handler(handler)
        clf = LightweightLLMClassifier(mock, reg, config=cfg)
        result = clf.classify("我要退款")
        # Arbitration should force clarify due to provisional required slot on high-risk intent
        assert result.need_clarification is True
        assert result.arbitration_breakdown.get("factors", {}).get("hard_constraint_risk") == "fail"

    def test_non_high_risk_intent_does_not_trigger_risk_clarify(self):
        """Non-high-risk intent in ambiguous zone does NOT trigger factor-3 Clarify."""
        cfg = _arb_config(high_risk_intents=["refund"])  # product_recommendation NOT in list
        mock = MockLLMClient()
        mock.set_default_handler(lambda sys_p, usr_p: _llm_response(
            intent="product_recommendation",
            confidence=0.75,
            slots={"category": "手机"},
            missing_slots=[],
        ))
        clf = LightweightLLMClassifier(mock, _build_registry(), config=cfg)
        result = clf.classify("推荐个手机")
        # No missing slots, no high-risk -> arbitration accepts
        assert result.need_clarification is False
        assert result.arbitration_breakdown.get("factors", {}).get("hard_constraint_risk") == "pass"

    def test_risk_aware_clarify_disabled(self):
        """When risk_aware_clarify=False, factor 3 always passes."""
        cfg = _arb_config(high_risk_intents=["refund"], risk_aware_clarify=False)
        reg = _build_registry()
        reg.register(IntentDefinition(
            name="refund",
            description="用户希望退款",
            slots=[SlotDefinition(name="order_id", required=True)],
            positive_examples=["我要退款"],
        ))
        mock = MockLLMClient()
        mock.set_default_handler(lambda sys_p, usr_p: json.dumps({
            "intent": "refund",
            "confidence": 0.75,
            "slots": {"order_id": {"value": "123", "evidence_grade": "provisional"}},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        clf = LightweightLLMClassifier(mock, reg, config=cfg)
        result = clf.classify("我要退款")
        # Factor 3 disabled -> no risk clarify
        assert result.need_clarification is False
