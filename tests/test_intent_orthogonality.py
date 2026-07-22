"""Tests for D19 intent orthogonality governance (IntentRegistry extensions)."""

from intent_recognition import (
    IntentDefinition,
    IntentRegistry,
    SlotDefinition,
)


def _build_overlapping_registry() -> IntentRegistry:
    """Build a registry with two partially overlapping intents (sim ~= 0.667).

    A: desc="apple banana", ex=["apple banana cherry", "unique x"]
    B: desc="apple banana cherry", ex=["apple banana cherry", "unique y"]
    Tokens A: {apple, banana, cherry, unique, x} (5)
    Tokens B: {apple, banana, cherry, unique, y} (5)
    Intersection: {apple, banana, cherry, unique} (4)
    Union: {apple, banana, cherry, unique, x, y} (6)
    Jaccard = 4/6 = 0.667
    """
    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="product_recommendation",
        description="apple banana",
        positive_examples=["apple banana cherry", "unique x"],
    ))
    reg.register(IntentDefinition(
        name="product_query",
        description="apple banana cherry",
        positive_examples=["apple banana cherry", "unique y"],
    ))
    return reg


def _build_disjoint_registry() -> IntentRegistry:
    """Build a registry with two completely disjoint intents."""
    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="weather_query",
        description="weather forecast temperature rain",
        positive_examples=["what is the weather today"],
    ))
    reg.register(IntentDefinition(
        name="order_refund",
        description="refund order cancel return money",
        positive_examples=["I want to refund my order"],
    ))
    return reg


class TestDetectOverlap:
    """Tests for IntentRegistry.detect_overlap (D19)."""

    def test_detect_overlap_finds_overlapping_intents(self):
        """Two intents sharing tokens should be flagged as overlapping."""
        reg = _build_overlapping_registry()
        reports = reg.detect_overlap(threshold=0.3)
        assert len(reports) >= 1
        names = {r.intent_a for r in reports} | {r.intent_b for r in reports}
        assert "product_recommendation" in names
        assert "product_query" in names
        match = next(
            r for r in reports
            if {r.intent_a, r.intent_b} == {"product_recommendation", "product_query"}
        )
        assert 0.0 < match.similarity <= 1.0
        assert match.suggestion in {"split", "merge_with_param"}

    def test_detect_overlap_default_threshold_finds_high_similarity(self):
        """With near-identical intents, default threshold (0.7) should fire."""
        reg = IntentRegistry()
        reg.register(IntentDefinition(
            name="a",
            description="推荐 手机",
            positive_examples=["帮我推荐手机"],
        ))
        reg.register(IntentDefinition(
            name="b",
            description="推荐 手机",
            positive_examples=["帮我推荐手机"],
        ))
        reports = reg.detect_overlap()  # default threshold 0.7
        assert len(reports) == 1
        assert reports[0].similarity == 1.0
        assert reports[0].suggestion == "split"  # > 0.85

    def test_detect_overlap_returns_empty_when_no_overlap(self):
        reg = _build_disjoint_registry()
        reports = reg.detect_overlap(threshold=0.1)
        assert reports == []

    def test_detect_overlap_with_custom_threshold(self):
        reg = _build_overlapping_registry()
        # Very high threshold -> no pairs reported
        high_reports = reg.detect_overlap(threshold=0.99)
        assert high_reports == []
        # Low threshold -> the overlap pair is reported
        low_reports = reg.detect_overlap(threshold=0.05)
        assert len(low_reports) >= 1

    def test_detect_overlap_suggestion_split_for_high_similarity(self):
        """When similarity > 0.85, suggestion should be 'split'."""
        reg = IntentRegistry()
        reg.register(IntentDefinition(
            name="a",
            description="推荐手机",
            positive_examples=["帮我推荐手机", "推荐一个手机"],
        ))
        reg.register(IntentDefinition(
            name="b",
            description="推荐手机",
            positive_examples=["帮我推荐手机", "推荐一个手机"],
        ))
        reports = reg.detect_overlap(threshold=0.5)
        assert len(reports) == 1
        assert reports[0].suggestion == "split"
        assert reports[0].similarity > 0.85

    def test_detect_overlap_suggestion_merge_for_mid_similarity(self):
        """When 0.7 <= similarity <= 0.85, suggestion should be 'merge_with_param'.

        A: desc="apple banana cherry", ex=["apple banana"]
           tokens: {apple, banana, cherry}
        B: desc="apple banana cherry", ex=["apple date"]
           tokens: {apple, banana, cherry, date}
        Intersection: {apple, banana, cherry} = 3
        Union: {apple, banana, cherry, date} = 4
        Jaccard = 3/4 = 0.75
        """
        reg = IntentRegistry()
        reg.register(IntentDefinition(
            name="a",
            description="apple banana cherry",
            positive_examples=["apple banana"],
        ))
        reg.register(IntentDefinition(
            name="b",
            description="apple banana cherry",
            positive_examples=["apple date"],
        ))
        reports = reg.detect_overlap(threshold=0.5)
        ab = next(
            (r for r in reports if {r.intent_a, r.intent_b} == {"a", "b"}),
            None,
        )
        assert ab is not None, "Expected an overlap report for (a, b)"
        assert 0.7 <= ab.similarity <= 0.85, f"sim={ab.similarity} not in merge band"
        assert ab.suggestion == "merge_with_param"


class TestSplitIntent:
    """Tests for IntentRegistry.split_intent (D19)."""

    def test_split_intent_creates_new_intent_and_removes_overlapping_examples(self):
        """Splitting two intents extracts shared examples into a new intent.

        Setup:
            A: desc="apple banana", ex=["apple banana cherry" (overlap), "unique x"]
            B: desc="apple banana cherry", ex=["apple banana cherry" (overlap), "unique y"]

        Example "apple banana cherry" tokens: {apple, banana, cherry}
        B text tokens: {apple, banana, cherry, unique, y}
        Jaccard(example, B) = 3/5 = 0.6 > 0.5 -> flagged as overlapping

        Example "unique x" tokens: {unique, x}
        B text tokens: {apple, banana, cherry, unique, y}
        Jaccard(example, B) = 1/6 ~= 0.167 -> stays in A
        """
        reg = IntentRegistry()
        reg.register(IntentDefinition(
            name="product_recommendation",
            description="apple banana",
            positive_examples=["apple banana cherry", "unique x"],
        ))
        reg.register(IntentDefinition(
            name="product_query",
            description="apple banana cherry",
            positive_examples=["apple banana cherry", "unique y"],
        ))
        new_intent = reg.split_intent(
            intent_a="product_recommendation",
            intent_b="product_query",
            new_intent_name="product_overlap",
            new_intent_description="overlap intent",
        )
        # New intent should be registered
        assert reg.has_intent("product_overlap")
        assert new_intent.name == "product_overlap"
        assert new_intent.description == "overlap intent"
        # The new intent should have absorbed the overlapping example
        assert "apple banana cherry" in new_intent.positive_examples
        assert len(new_intent.positive_examples) >= 1
        # The original intents should still exist
        assert reg.has_intent("product_recommendation")
        assert reg.has_intent("product_query")
        # The overlapping example should be removed from both originals
        a = reg.get("product_recommendation")
        b = reg.get("product_query")
        assert "apple banana cherry" not in a.positive_examples
        assert "apple banana cherry" not in b.positive_examples
        # Unique examples should be retained
        assert "unique x" in a.positive_examples
        assert "unique y" in b.positive_examples


class TestMergeWithParam:
    """Tests for IntentRegistry.merge_with_param (D19)."""

    def test_merge_with_param_merges_intents_and_adds_dispatch_param_slot(self):
        reg = IntentRegistry()
        reg.register(IntentDefinition(
            name="intent_a",
            description="first intent",
            positive_examples=["do a thing", "do a stuff"],
            slots=[SlotDefinition(name="category", required=True, description="cat")],
        ))
        reg.register(IntentDefinition(
            name="intent_b",
            description="second intent",
            positive_examples=["do b thing"],
            slots=[SlotDefinition(name="budget", required=False, description="budget")],
        ))
        merged = reg.merge_with_param(
            intents=["intent_a", "intent_b"],
            new_intent_name="merged_intent",
            dispatch_param="source_intent",
        )
        # New intent registered
        assert reg.has_intent("merged_intent")
        assert merged.name == "merged_intent"
        # Union of examples
        assert set(merged.positive_examples) == {"do a thing", "do a stuff", "do b thing"}
        # Union of slots + dispatch_param
        slot_names = {s.name for s in merged.slots}
        assert "category" in slot_names
        assert "budget" in slot_names
        assert "source_intent" in slot_names

    def test_merge_with_param_removes_original_intents(self):
        reg = IntentRegistry()
        reg.register(IntentDefinition(
            name="intent_a",
            description="first intent",
            positive_examples=["ex1"],
        ))
        reg.register(IntentDefinition(
            name="intent_b",
            description="second intent",
            positive_examples=["ex2"],
        ))
        reg.merge_with_param(
            intents=["intent_a", "intent_b"],
            new_intent_name="merged",
            dispatch_param="source",
        )
        assert not reg.has_intent("intent_a")
        assert not reg.has_intent("intent_b")
        assert reg.has_intent("merged")


class TestSimilarityHelpers:
    """Tests for the _compute_similarity / _tokenize helpers (D19)."""

    def test_compute_similarity_identical_texts_returns_one(self):
        # Identical texts -> Jaccard = 1.0
        sim = IntentRegistry._compute_similarity("推荐手机", "推荐手机")
        assert sim == 1.0

    def test_compute_similarity_disjoint_texts_returns_zero(self):
        # Disjoint token sets -> Jaccard = 0.0
        sim = IntentRegistry._compute_similarity("abc", "xyz")
        assert sim == 0.0
