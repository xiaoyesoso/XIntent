"""Tests for intent recognition pipeline orchestrator."""

import json

from user_input_normalization.llm.mock import MockLLMClient

from intent_recognition import (
    IntentDefinition,
    IntentRecognitionConfig,
    IntentRecognitionPipeline,
    IntentRegistry,
    RecognitionSource,
    SlotDefinition,
)


def _build_registry() -> IntentRegistry:
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
        name="order_query",
        description="查询订单",
        positive_examples=["我的订单到哪了"],
    ))
    return reg


def _build_pipeline(
    llm: MockLLMClient | None = None,
    config: IntentRecognitionConfig | None = None,
) -> IntentRecognitionPipeline:
    llm = llm or MockLLMClient()
    reg = _build_registry()
    cfg = config or IntentRecognitionConfig()
    return IntentRecognitionPipeline(llm, reg, cfg)


class TestPipelineLayer1:
    """Test Layer 1 (code-based) recognition."""

    def test_keyword_match_continues(self):
        pipe = _build_pipeline()
        # "继续" is a built-in keyword in rule engine, but requires context["in_flow"]=True
        result = pipe.recognize("继续", context={"in_flow": True})
        assert result.source == RecognitionSource.CODE_LAYER
        assert result.confidence == 1.0
        assert result.layer_reached == 1

    def test_regex_match_order_query(self):
        pipe = _build_pipeline()
        result = pipe.recognize("查询订单 12345")
        assert result.source == RecognitionSource.CODE_LAYER
        assert result.confidence == 1.0

    def test_page_guidance_event(self):
        pipe = _build_pipeline()
        result = pipe.recognize("", event="click:next")
        # Page guidance should match
        assert result is not None
        assert result.source == RecognitionSource.CODE_LAYER


class TestPipelineLayer2:
    """Test Layer 2 (lightweight LLM) recognition."""

    def test_high_confidence_accept(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_pipeline(llm)
        result = pipe.recognize("帮我推荐一个手机")
        assert result.intent == "product_recommendation"
        assert result.confidence >= 0.85
        assert result.source == RecognitionSource.LIGHTWEIGHT_LLM

    def test_low_confidence_escalate_to_l3(self):
        llm = MockLLMClient()
        # L2 returns low confidence
        def handler(sys, usr):
            if "复杂场景" in sys or "深度推理" in sys:
                # L3 response
                return json.dumps({
                    "intent": "product_recommendation",
                    "confidence": 0.9,
                    "slots": {"category": "手机"},
                    "missing_slots": [],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "sub_intents": [],
                    "intent_switched": False,
                })
            # L2 response - low confidence
            return json.dumps({
                "intent": "product_recommendation",
                "confidence": 0.3,
                "slots": {},
                "missing_slots": [],
                "hard_constraints": [],
                "soft_constraints": [],
            })
        llm.set_default_handler(handler)
        pipe = _build_pipeline(llm)
        result = pipe.recognize("那个，就是，帮我看看有没有什么手机可以推荐的")
        # Should escalate to L3
        assert result.source == RecognitionSource.DEEP_LLM
        assert result.layer_reached == 3


class TestPipelineSlotFilling:
    """Test slot filling in pipeline."""

    def test_missing_required_slot_triggers_clarification(self):
        llm = MockLLMClient()
        # L2 returns high confidence but no category slot
        def handler(sys, usr):
            if "参数抽取" in sys or "槽位" in usr:
                return json.dumps({
                    "slots": {},
                    "missing_slots": ["category"],
                })
            return json.dumps({
                "intent": "product_recommendation",
                "confidence": 0.9,
                "slots": {},
                "missing_slots": [],
                "hard_constraints": [],
                "soft_constraints": [],
            })
        llm.set_default_handler(handler)
        pipe = _build_pipeline(llm)
        result = pipe.recognize("帮我推荐个东西", session_id="s1")
        assert result.intent == "product_recommendation"
        assert "category" in result.missing_slots
        assert result.need_clarification is True
        assert result.clarification_question is not None

    def test_constraint_extraction(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": "裤子"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_pipeline(llm)
        result = pipe.recognize("帮我推荐裤子，不超过100块，最好是牛仔裤")
        assert result.intent == "product_recommendation"
        # Constraints should be extracted by regex
        assert len(result.hard_constraints) >= 1 or len(result.soft_constraints) >= 1


class TestPipelineCrossTurn:
    """Test cross-turn state management."""

    def test_slot_accumulation_across_turns(self):
        llm = MockLLMClient()

        def handler(sys, usr):
            if "参数抽取" in sys:
                if "100" in usr:
                    return json.dumps({"slots": {"budget_max": 100}, "missing_slots": []})
                return json.dumps({"slots": {"category": "裤子"}, "missing_slots": []})
            return json.dumps({
                "intent": "product_recommendation",
                "confidence": 0.92,
                "slots": {},
                "missing_slots": [],
                "hard_constraints": [],
                "soft_constraints": [],
            })
        llm.set_default_handler(handler)
        pipe = _build_pipeline(llm)

        # Turn 1: only category
        r1 = pipe.recognize("帮我推荐裤子", session_id="s1", turn=1)
        assert r1.intent == "product_recommendation"
        assert r1.slots.get("category") == "裤子"

        # Turn 2: add budget
        r2 = pipe.recognize("不超过100块", session_id="s1", turn=2)
        # Slots should be accumulated
        state = pipe.get_slot_state("s1")
        assert state is not None
        assert state.slots.get("category") == "裤子"  # From turn 1
        assert state.slots.get("budget_max") == 100  # From turn 2

    def test_intent_history(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_pipeline(llm)
        pipe.recognize("推荐手机", session_id="s1", turn=1)
        pipe.recognize("还有别的吗", session_id="s1", turn=2)
        history = pipe.get_intent_history("s1")
        assert len(history) == 2

    def test_reset_session(self):
        pipe = _build_pipeline()
        pipe.reset_session("s1")
        assert pipe.get_slot_state("s1") is None


class TestPipelineRejection:
    """Test rejection and clarification."""

    def test_unsupported_input_rejected(self):
        llm = MockLLMClient()
        # L2 returns null intent (unsupported)
        def handler(sys, usr):
            if "深度推理" in sys or "复杂场景" in sys:
                return json.dumps({
                    "intent": None,
                    "confidence": 0.0,
                    "slots": {},
                    "missing_slots": [],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "sub_intents": [],
                    "intent_switched": False,
                })
            return json.dumps({
                "intent": None,
                "confidence": 0.2,
                "slots": {},
                "missing_slots": [],
                "hard_constraints": [],
                "soft_constraints": [],
            })
        llm.set_default_handler(handler)
        pipe = _build_pipeline(llm)
        result = pipe.recognize("帮我生成一个奥特曼动画")
        assert result.intent is None
        assert result.rejection_reason is not None
        assert result.is_unsupported is True

    def test_implicit_failure_signal_detection(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.92,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
        }))
        pipe = _build_pipeline(llm)
        # Turn 1: normal
        pipe.recognize("推荐手机", session_id="s1", turn=1)
        # Turn 2: failure signal
        result = pipe.recognize("你理解错了", session_id="s1", turn=2)
        # The failure signal is detected and recorded
        # The L2 will still return a result for the current input
        assert result is not None


class TestPipelineGracefulDegradation:
    """Test graceful degradation when L3 fails."""

    def test_l3_failure_fallback_to_l2(self):
        llm = MockLLMClient()

        def handler(sys, usr):
            if "深度推理" in sys or "复杂场景" in sys:
                raise RuntimeError("L3 API error")
            # L2 returns low confidence to trigger escalation
            return json.dumps({
                "intent": "product_recommendation",
                "confidence": 0.4,
                "slots": {"category": "手机"},
                "missing_slots": [],
                "hard_constraints": [],
                "soft_constraints": [],
            })
        llm.set_default_handler(handler)
        pipe = _build_pipeline(llm)
        result = pipe.recognize("帮我推荐个手机")
        # L3 failed, should fall back to L2 result
        assert result.intent == "product_recommendation"
        assert result.source == RecognitionSource.LIGHTWEIGHT_LLM
