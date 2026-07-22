"""Tests for deep reasoning LLM classifier (Layer 3)."""

import json

from user_input_normalization.llm.mock import MockLLMClient

from intent_recognition import (
    IntentDefinition,
    IntentRegistry,
    RecognitionSource,
    SlotDefinition,
)
from intent_recognition.deep_llm import DeepLLMClassifier


def _build_registry() -> IntentRegistry:
    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="product_recommendation",
        description="推荐商品",
        slots=[
            SlotDefinition(name="category", required=True, description="商品类目"),
            SlotDefinition(name="budget_max", required=False, description="最高预算"),
        ],
    ))
    reg.register(IntentDefinition(
        name="order_query",
        description="查询订单",
    ))
    reg.register(IntentDefinition(
        name="refund",
        description="申请退款",
        parent_intent="order_query",
    ))
    return reg


class TestDeepLLMClassifier:
    """Test deep LLM classifier."""

    def test_complex_expression_recognition(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.9,
            "slots": {"category": "手机", "budget_max": 3000},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": [],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("那个，就是，帮我看看有没有什么手机可以推荐的")
        assert result.intent == "product_recommendation"
        assert result.confidence == 0.9
        assert result.source == RecognitionSource.DEEP_LLM
        assert result.layer_reached == 3

    def test_cross_turn_context_dependency(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.85,
            "slots": {"category": "手机"},
            "missing_slots": ["budget_max"],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": [],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify(
            "帮我推荐个手机",
            dialogue_history=[{"role": "user", "content": "我想换手机"}],
        )
        assert result.intent == "product_recommendation"
        assert "budget_max" in result.missing_slots

    def test_intent_switch_detection(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "refund",
            "confidence": 0.88,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": [],
            "intent_switched": True,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify(
            "算了不推荐了，我之前那个订单要退款",
            previous_intent="product_recommendation",
        )
        assert result.intent == "refund"
        assert result.intent_switched is True
        assert result.previous_intent == "product_recommendation"

    def test_multi_intent_decomposition(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.82,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": ["product_recommendation", "product_query"],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我推荐手机，顺便查一下iPhone 16的价格")
        assert len(result.sub_intents) == 2
        assert "product_recommendation" in result.sub_intents
        assert "product_query" in result.sub_intents

    def test_implicit_info_completion(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.9,
            "slots": {"category": "手机", "budget_max": 3000},
            "missing_slots": [],
            "hard_constraints": ["price<3000"],
            "soft_constraints": [],
            "sub_intents": [],
            "intent_switched": False,
            "implicit_completion": "从前文推断预算3000",
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我推荐个手机")
        assert result.slots.get("budget_max") == 3000
        assert len(result.hard_constraints) == 1
        assert result.hard_constraints[0].expression == "price<3000"

    def test_unregistered_intent_returns_null(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "nonexistent_intent",
            "confidence": 0.5,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": [],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("some random text")
        assert result.intent is None
        assert result.confidence == 0.0

    def test_unparseable_response(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: "not json at all")
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("test")
        assert result.intent is None
        assert result.rejection_reason is not None

    def test_hierarchical_intent_dispatch(self):
        """Test that hierarchical intents (parent-child) are recognized."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "refund",
            "confidence": 0.92,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": [],
            "intent_switched": False,
        }))
        reg = _build_registry()
        assert reg.list_children("order_query")  # refund is child of order_query
        clf = DeepLLMClassifier(llm, reg)
        result = clf.classify("我要退款")
        assert result.intent == "refund"

    def test_constraint_parsing(self):
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.9,
            "slots": {"category": "裤子"},
            "missing_slots": [],
            "hard_constraints": [{"expression": "price<100", "raw_text": "不超过100块"}],
            "soft_constraints": [{"expression": "type=牛仔裤", "raw_text": "最好是牛仔裤"}],
            "sub_intents": [],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我推荐裤子，不超过100块，最好是牛仔裤")
        assert len(result.hard_constraints) == 1
        assert result.hard_constraints[0].expression == "price<100"
        assert len(result.soft_constraints) == 1
        assert result.soft_constraints[0].expression == "type=牛仔裤"


# ---------------------------------------------------------------------------
# D27: sub_tasks vs independent_intents boundary tests (task 10.5)
# ---------------------------------------------------------------------------


class TestD27BoundaryDetection:
    """Tests for D27 sub_tasks vs independent_intents boundary (task 10.5)."""

    def test_independent_intents_parsed_separately(self):
        """D27: independent_intents parsed from new field, not sub_tasks."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.85,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "independent_intents": ["product_recommendation", "order_query"],
            "sub_tasks": [],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我推荐手机，顺便查一下我的订单")
        assert result.independent_intents == ["product_recommendation", "order_query"]
        assert result.sub_tasks == []

    def test_sub_tasks_parsed_and_kept_separate(self):
        """D27: sub_tasks (intra-flow steps) parsed and NOT mixed into independent_intents."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.88,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "independent_intents": ["product_recommendation"],
            "sub_tasks": ["比价", "查配送时间", "看评价"],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我推荐手机，顺便比价、查配送时间、看评价")
        # sub_tasks captured
        assert result.sub_tasks == ["比价", "查配送时间", "看评价"]
        # independent_intents does NOT include sub_tasks
        assert result.independent_intents == ["product_recommendation"]
        assert "比价" not in result.independent_intents

    def test_legacy_sub_intents_falls_back_to_independent(self):
        """D27: backward compat — legacy `sub_intents` is treated as independent_intents."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.82,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": ["product_recommendation", "order_query"],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我推荐手机，顺便查一下订单")
        # legacy sub_intents -> independent_intents (alias sync)
        assert "product_recommendation" in result.independent_intents
        assert "order_query" in result.independent_intents
        # alias: sub_intents property mirrors independent_intents
        assert result.sub_intents == result.independent_intents

    def test_sub_tasks_not_in_relations_or_pending(self):
        """D27: sub_tasks must NOT enter D20 governance (relations, pending_intents)."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.85,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "independent_intents": ["product_recommendation", "order_query"],
            "sub_tasks": ["比价", "查配送"],
            "relations": [
                {"src": "order_query", "dst": "product_recommendation", "constraints": []}
            ],
            "pending_intents": [],
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我推荐手机，顺便查订单，还要比价")
        # sub_tasks present
        assert result.sub_tasks == ["比价", "查配送"]
        # relations only reference independent_intents, not sub_tasks
        for rel in result.relations:
            assert rel.src in result.independent_intents
            assert rel.dst in result.independent_intents
            assert rel.src not in result.sub_tasks
            assert rel.dst not in result.sub_tasks
        # pending_intents only contains independent_intents
        for p in result.pending_intents:
            assert p not in result.sub_tasks

    def test_process_description_filter_only_affects_independent(self):
        """D27: D20 process-description filter clears independent_intents but keeps sub_tasks."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "product_recommendation",
            "confidence": 0.7,
            "slots": {"category": "手机"},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "independent_intents": ["product_recommendation", "order_query"],
            "sub_tasks": ["比价"],
            "relations": [],
            "pending_intents": [],
            "is_process_description": True,
            "intent_switched": False,
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("先推荐再查订单再比价，整个流程是怎样的")
        # D20 filter collapses independent_intents to []
        assert result.independent_intents == []
        # sub_tasks are NOT filtered
        assert result.sub_tasks == ["比价"]
