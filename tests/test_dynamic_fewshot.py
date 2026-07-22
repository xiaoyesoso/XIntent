"""Tests for D18: dynamic few-shot injection.

Covers DynamicFewShotInjector behavior:
* static examples are always injected
* dynamic examples are retrieved when dynamic_enabled=True
* dynamic_enabled=False returns only static examples
* two-section rendering (固定案例 / 动态案例)
* empty store returns empty string
* get_examples() returns the combined list

Also covers the build_user_prompt_with_sections helper added to prompts.py.
"""

import json

from intent_recognition import FewShotExample, FewShotKind
from intent_recognition.config import DynamicFewShotConfig
from intent_recognition.lightweight_llm import (
    DynamicFewShotInjector,
    build_user_prompt_with_sections,
)
from intent_recognition.storage import MemoryFewShotStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_static(text: str, intent: str | None = None, output: dict | None = None) -> FewShotExample:
    """Helper: build a STATIC example."""
    return FewShotExample(
        text=text,
        intent=intent,
        output=output,
        kind=FewShotKind.STATIC,
    )


def _make_dynamic(text: str, intent: str | None = None, output: dict | None = None) -> FewShotExample:
    """Helper: build a DYNAMIC example."""
    return FewShotExample(
        text=text,
        intent=intent,
        output=output,
        kind=FewShotKind.DYNAMIC,
    )


def _build_store_with_examples() -> MemoryFewShotStore:
    """Build a store with 2 static + 3 dynamic examples."""
    store = MemoryFewShotStore()
    store.bulk_add([
        _make_static("我要退款", intent="refund"),
        _make_static("推荐手机", output={"intent": "product_recommendation"}),
        _make_dynamic("推荐便宜手机", output={"intent": "product_recommendation"}),
        _make_dynamic("查询我的订单", intent="order_query"),
        _make_dynamic("今天天气如何", intent="weather_query"),
    ])
    return store


# ---------------------------------------------------------------------------
# DynamicFewShotInjector - static behavior
# ---------------------------------------------------------------------------


class TestStaticInjection:
    """D18: STATIC examples are always injected."""

    def test_static_examples_always_injected(self):
        """Even when dynamic_enabled=False, static examples are returned."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=False)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("any query")
        # Two static examples should be in the output
        assert "我要退款" in text
        assert "推荐手机" in text
        # Static section header should always be present
        assert "## 固定案例（始终注入）" in text

    def test_dynamic_section_header_present_even_when_disabled(self):
        """When dynamic is disabled, the section header still renders (empty)."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=False)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("any query")
        assert "## 动态案例（按当前输入检索）" in text


# ---------------------------------------------------------------------------
# DynamicFewShotInjector - dynamic behavior
# ---------------------------------------------------------------------------


class TestDynamicInjection:
    """D18: DYNAMIC examples are retrieved by similarity to the query."""

    def test_dynamic_examples_retrieved_by_query(self):
        """dynamic_enabled=True + query about phones -> phone-related dynamic example."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=True, dynamic_top_k=1)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("推荐便宜手机")
        # The phone-related dynamic example should appear
        assert "推荐便宜手机" in text

    def test_dynamic_top_k_limits_dynamic_examples(self):
        """dynamic_top_k=1 means at most 1 dynamic example is injected."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=True, dynamic_top_k=1)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("手机 订单 天气")
        # Count "## 例子" occurrences inside the dynamic section only
        # (split on the dynamic section header to isolate it)
        parts = text.split("## 动态案例（按当前输入检索）")
        assert len(parts) == 2
        dynamic_section = parts[1]
        # At most 1 "## 例子" inside the dynamic section
        assert dynamic_section.count("## 例子") <= 1

    def test_dynamic_disabled_returns_only_static_examples(self):
        """dynamic_enabled=False -> get_examples returns ONLY static."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=False)
        injector = DynamicFewShotInjector(store, cfg)
        examples = injector.get_examples("推荐便宜手机")
        # Only the 2 static examples should be present
        assert len(examples) == 2
        for ex in examples:
            assert ex.kind == FewShotKind.STATIC

    def test_dynamic_enabled_returns_combined_list(self):
        """dynamic_enabled=True -> get_examples returns static + dynamic."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=True, dynamic_top_k=2)
        injector = DynamicFewShotInjector(store, cfg)
        examples = injector.get_examples("推荐便宜手机")
        # 2 static + up to 2 dynamic = up to 4
        assert len(examples) >= 2  # at least the static ones
        # First two should be static
        assert examples[0].kind == FewShotKind.STATIC
        assert examples[1].kind == FewShotKind.STATIC
        # Verify the order: static first, then dynamic
        kinds = [ex.kind for ex in examples]
        static_count = sum(1 for k in kinds if k == FewShotKind.STATIC)
        assert static_count == 2


# ---------------------------------------------------------------------------
# Two-section rendering
# ---------------------------------------------------------------------------


class TestSectionRendering:
    """D18: two-section format with proper headers."""

    def test_two_section_headers_present(self):
        """Both 固定案例 and 动态案例 headers should be present."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=True, dynamic_top_k=2)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("推荐便宜手机")
        assert "## 固定案例（始终注入）" in text
        assert "## 动态案例（按当前输入检索）" in text

    def test_example_block_format(self):
        """Each example renders as ## 例子 N + 输入： + 输出：."""
        store = MemoryFewShotStore()
        store.add(_make_static("hello", intent="greeting"))
        cfg = DynamicFewShotConfig(dynamic_enabled=False)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("any")
        assert "## 例子 1" in text
        assert "输入：hello" in text
        assert "输出：greeting" in text

    def test_example_with_output_dict_renders_json(self):
        """When output is a dict, it renders as JSON in 输出： line."""
        store = MemoryFewShotStore()
        store.add(_make_static(
            "推荐手机", output={"intent": "product_recommendation", "slots": {"category": "手机"}}
        ))
        cfg = DynamicFewShotConfig(dynamic_enabled=False)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("any")
        # Output should be a JSON dict (contains intent: product_recommendation)
        assert "输出：" in text
        assert "product_recommendation" in text

    def test_example_with_no_output_renders_null(self):
        """When neither output nor intent is set, 输出：null."""
        store = MemoryFewShotStore()
        store.add(FewShotExample(text="unclear input", kind=FewShotKind.STATIC))
        cfg = DynamicFewShotConfig(dynamic_enabled=False)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("any")
        assert "输出：null" in text


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------


class TestEmptyStore:
    """D18: empty store -> empty string from inject()."""

    def test_empty_store_returns_empty_string(self):
        store = MemoryFewShotStore()
        cfg = DynamicFewShotConfig(dynamic_enabled=True, dynamic_top_k=3)
        injector = DynamicFewShotInjector(store, cfg)
        text = injector.inject("any query")
        assert text == ""

    def test_empty_store_get_examples_returns_empty_list(self):
        store = MemoryFewShotStore()
        cfg = DynamicFewShotConfig(dynamic_enabled=True, dynamic_top_k=3)
        injector = DynamicFewShotInjector(store, cfg)
        assert injector.get_examples("any query") == []


# ---------------------------------------------------------------------------
# get_examples
# ---------------------------------------------------------------------------


class TestGetExamples:
    """D18: get_examples returns combined list."""

    def test_get_examples_returns_combined_list(self):
        """Static + dynamic examples combined into one list."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=True, dynamic_top_k=1)
        injector = DynamicFewShotInjector(store, cfg)
        examples = injector.get_examples("推荐便宜手机")
        # 2 static + at most 1 dynamic
        assert 2 <= len(examples) <= 3
        # Static examples come first
        assert examples[0].kind == FewShotKind.STATIC
        assert examples[1].kind == FewShotKind.STATIC

    def test_get_examples_dynamic_disabled_skips_search(self):
        """When dynamic is disabled, search() is never called - verify by
        using a query that would otherwise return results."""
        store = _build_store_with_examples()
        cfg = DynamicFewShotConfig(dynamic_enabled=False)
        injector = DynamicFewShotInjector(store, cfg)
        # Query that strongly matches a dynamic example
        examples = injector.get_examples("推荐便宜手机")
        # No dynamic examples should be returned
        assert all(ex.kind == FewShotKind.STATIC for ex in examples)


# ---------------------------------------------------------------------------
# build_user_prompt_with_sections (in prompts.py)
# ---------------------------------------------------------------------------


class TestBuildUserPromptWithSections:
    """D18: prompts.py helper for two-section user prompt."""

    def test_static_and_dynamic_sections_rendered(self):
        """Both sections render when examples are provided."""
        static = [_make_static("hello", intent="greeting")]
        dynamic = [_make_dynamic("hi there", intent="greeting")]
        prompt = build_user_prompt_with_sections(
            text="用户输入",
            static_examples=static,
            dynamic_examples=dynamic,
        )
        assert "# few-shot" in prompt
        assert "## 固定案例（始终注入）" in prompt
        assert "## 动态案例（按当前输入检索）" in prompt
        assert "hello" in prompt
        assert "hi there" in prompt
        assert "# 用户输入" in prompt
        assert "用户输入" in prompt
        # Separator before user input
        assert "---" in prompt

    def test_no_examples_omits_few_shot_section(self):
        """When both lists are empty/None, only the user input is rendered."""
        prompt = build_user_prompt_with_sections(text="hello", static_examples=None, dynamic_examples=None)
        assert "# few-shot" not in prompt
        assert "# 用户输入" in prompt
        assert "hello" in prompt

    def test_static_only_renders_empty_dynamic_section(self):
        """Static examples but no dynamic examples - dynamic section still renders."""
        static = [_make_static("hello", intent="greeting")]
        prompt = build_user_prompt_with_sections(
            text="input",
            static_examples=static,
            dynamic_examples=None,
        )
        assert "## 固定案例（始终注入）" in prompt
        assert "## 动态案例（按当前输入检索）" in prompt
        assert "hello" in prompt

    def test_backward_compat_build_user_prompt_unchanged(self):
        """The original build_user_prompt function must still work as before."""
        from intent_recognition.lightweight_llm import build_user_prompt

        prompt = build_user_prompt("test input", few_shot_examples=None)
        assert "test input" in prompt
        # Original function uses 参考例子 wording, NOT the new 固定案例 wording
        assert "固定案例" not in prompt

    def test_example_indexing_continues_across_sections(self):
        """Example indices should continue across static + dynamic sections (1, 2, 3...)."""
        static = [
            _make_static("static1", intent="a"),
            _make_static("static2", intent="b"),
        ]
        dynamic = [_make_dynamic("dynamic1", intent="c")]
        prompt = build_user_prompt_with_sections(
            text="input",
            static_examples=static,
            dynamic_examples=dynamic,
        )
        # Indices should be 1, 2, 3 across both sections
        assert "## 例子 1" in prompt
        assert "## 例子 2" in prompt
        assert "## 例子 3" in prompt


# ---------------------------------------------------------------------------
# Backward compat / sanity
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """D18 must not break existing import paths."""

    def test_dynamic_fewshot_exported_from_package(self):
        from intent_recognition.lightweight_llm import (
            DynamicFewShotInjector as DFSI,
            build_user_prompt_with_sections as bups,
        )
        assert DFSI is DynamicFewShotInjector
        assert bups is build_user_prompt_with_sections
