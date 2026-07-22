"""Tests for D20 true multi-intent detection (DeepLLMClassifier extensions)."""

import json

from user_input_normalization.llm.mock import MockLLMClient

from intent_recognition import (
    IntentDefinition,
    IntentRecognitionConfig,
    IntentRegistry,
    MultiIntentRelation,
    SlotDefinition,
)
from intent_recognition.config import MultiIntentConfig
from intent_recognition.deep_llm import DeepLLMClassifier
from intent_recognition.deep_llm.classifier import _topological_sort


def _build_registry() -> IntentRegistry:
    """Build a registry with multiple intents for multi-intent tests."""
    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="product_recommendation",
        description="推荐商品",
        slots=[
            SlotDefinition(name="category", required=True, description="商品类目"),
        ],
    ))
    reg.register(IntentDefinition(
        name="order_query",
        description="查询订单",
    ))
    reg.register(IntentDefinition(
        name="refund",
        description="申请退款",
    ))
    reg.register(IntentDefinition(
        name="logistics_query",
        description="查询物流",
    ))
    return reg


class TestTopologicalSort:
    """Tests for the _topological_sort helper (D20)."""

    def test_topological_sort_no_relations_returns_original_order(self):
        result = _topological_sort(["a", "b", "c"], [])
        assert result == ["a", "b", "c"]

    def test_topological_sort_simple_chain(self):
        """Chain: a depends on b, b depends on c -> order: c, b, a.

        Semantics: src executes after dst.
        - {src: a, dst: b}: b before a
        - {src: b, dst: c}: c before b
        Expected topological order: [c, b, a]
        """
        relations = [
            MultiIntentRelation(src="a", dst="b"),
            MultiIntentRelation(src="b", dst="c"),
        ]
        result = _topological_sort(["a", "b", "c"], relations)
        assert result == ["c", "b", "a"]

    def test_topological_sort_cycle_falls_back_to_original_order(self):
        """Cycle: a -> b -> a should fall back to original order."""
        relations = [
            MultiIntentRelation(src="a", dst="b"),  # b before a
            MultiIntentRelation(src="b", dst="a"),  # a before b  (cycle!)
        ]
        result = _topological_sort(["a", "b"], relations)
        assert result == ["a", "b"]


class TestMultiIntentClassification:
    """Tests for DeepLLMClassifier multi-intent handling (D20)."""

    def test_process_description_filtering(self):
        """When is_process_description=True and filter is enabled, sub_intents cleared."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "order_query",
            "confidence": 0.85,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": ["order_query", "logistics_query"],
            "intent_switched": False,
            "is_process_description": True,
            "relations": [{"src": "logistics_query", "dst": "order_query"}],
            "pending_intents": ["logistics_query"],
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("查完订单顺便看看物流")
        # Process-description filter clears sub_intents and relations
        assert result.sub_intents == []
        assert result.relations == []
        assert result.pending_intents == []

    def test_true_multi_intent_with_relations_output(self):
        """True multi-intent returns non-empty relations list."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "order_query",
            "confidence": 0.88,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": ["order_query", "logistics_query"],
            "intent_switched": False,
            "is_process_description": False,
            "relations": [{"src": "logistics_query", "dst": "order_query",
                           "constraints": ["订单查询完成后再查物流"]}],
            "pending_intents": [],
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我查订单，然后查一下物流")
        assert len(result.relations) == 1
        assert result.relations[0].src == "logistics_query"
        assert result.relations[0].dst == "order_query"
        assert result.relations[0].constraints == ["订单查询完成后再查物流"]

    def test_sequential_execution_sets_first_intent_as_main(self):
        """With sequential_execution, main intent = topological-first sub-intent."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "logistics_query",  # LLM nominally picks logistics_query
            "confidence": 0.8,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": ["order_query", "logistics_query"],
            "intent_switched": False,
            "is_process_description": False,
            # logistics_query (src) executes after order_query (dst)
            # => order_query should be first in topological order
            "relations": [{"src": "logistics_query", "dst": "order_query"}],
            "pending_intents": [],
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("查订单和物流")
        # Topological first should be order_query (dst before src)
        assert result.intent == "order_query"
        assert result.pending_intents == ["logistics_query"]
        # sub_intents preserved as full list
        assert set(result.sub_intents) == {"order_query", "logistics_query"}

    def test_pending_intents_in_topological_order(self):
        """pending_intents should be in topological order (after the main one)."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "order_query",
            "confidence": 0.8,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            # Three sub-intents with a chain: refund -> logistics_query -> order_query
            # i.e. src=refund, dst=logistics_query (logistics before refund)
            #      src=logistics_query, dst=order_query (order before logistics)
            # Topological order: order_query, logistics_query, refund
            "sub_intents": ["order_query", "logistics_query", "refund"],
            "intent_switched": False,
            "is_process_description": False,
            "relations": [
                {"src": "refund", "dst": "logistics_query"},
                {"src": "logistics_query", "dst": "order_query"},
            ],
            "pending_intents": [],
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("查订单 然后查物流 然后退款")
        assert result.intent == "order_query"
        # Pending should be in topological order: logistics_query, then refund
        assert result.pending_intents == ["logistics_query", "refund"]

    def test_multi_intent_enable_false_passes_through(self):
        """When multi_intent.enable=False, D20 governance is bypassed (pass-through)."""
        cfg = IntentRecognitionConfig()
        cfg.multi_intent = MultiIntentConfig(
            enable=False,
            sequential_execution=True,
            filter_process_description=True,
        )
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "order_query",
            "confidence": 0.8,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": ["order_query", "logistics_query"],
            "intent_switched": False,
            "is_process_description": True,  # would be filtered if enable=True
            "relations": [{"src": "logistics_query", "dst": "order_query"}],
            "pending_intents": ["custom_pending"],  # arbitrary, should pass through
        }))
        clf = DeepLLMClassifier(llm, _build_registry(), config=cfg)
        result = clf.classify("查订单和物流")
        # Pass-through: sub_intents NOT cleared, intent NOT overridden
        assert result.sub_intents == ["order_query", "logistics_query"]
        assert result.intent == "order_query"  # unchanged from LLM output
        assert result.relations == [
            MultiIntentRelation(src="logistics_query", dst="order_query"),
        ]
        assert result.pending_intents == ["custom_pending"]

    def test_no_sub_intents_does_not_trigger_sequential_logic(self):
        """Single intent (no sub_intents) should not be touched by sequential logic."""
        llm = MockLLMClient()
        llm.set_default_handler(lambda sys, usr: json.dumps({
            "intent": "order_query",
            "confidence": 0.9,
            "slots": {},
            "missing_slots": [],
            "hard_constraints": [],
            "soft_constraints": [],
            "sub_intents": [],
            "intent_switched": False,
            "is_process_description": False,
            "relations": [],
            "pending_intents": [],
        }))
        clf = DeepLLMClassifier(llm, _build_registry())
        result = clf.classify("帮我查订单")
        assert result.intent == "order_query"
        assert result.pending_intents == []
        assert result.sub_intents == []

    def test_topological_sort_with_isolated_node_preserved(self):
        """An isolated node (no relations) should still appear in the topological order."""
        # sub_intents: [a, b, c]; relation: b -> a (a before b)
        relations = [MultiIntentRelation(src="b", dst="a")]
        result = _topological_sort(["a", "b", "c"], relations)
        # a should come before b; c can be anywhere (isolated, keeps original order)
        assert result.index("a") < result.index("b")
        assert set(result) == {"a", "b", "c"}

    def test_topological_sort_extra_relation_not_in_intents_ignored(self):
        """Relations referencing intents not in the list should be ignored."""
        relations = [
            MultiIntentRelation(src="a", dst="b"),
            MultiIntentRelation(src="a", dst="nonexistent"),  # should be ignored
        ]
        result = _topological_sort(["a", "b"], relations)
        # b before a (only valid relation)
        assert result == ["b", "a"]
