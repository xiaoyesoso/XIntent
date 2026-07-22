"""D18: Dynamic few-shot injection.

Retrieves and formats few-shot examples for prompt injection.  Always
injects ``STATIC`` examples (boundary cases, unsupported/unclear hints).
When ``dynamic_enabled=True``, additionally retrieves ``Top-K`` dynamic
examples via ``FewShotStore.search()`` using the current user input as
the query.

The injector is a pure formatting layer over ``FewShotStore`` - it does
not call any LLM and does not modify the store.  It is fully opt-in:
when ``dynamic_enabled=False`` only the static examples are returned.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import DynamicFewShotConfig
from ..models import FewShotExample, FewShotKind
from ..storage.base import FewShotStore


# ---------------------------------------------------------------------------
# Section headers (single source of truth - same wording as
# build_user_prompt_with_sections in prompts.py)
# ---------------------------------------------------------------------------

STATIC_SECTION_HEADER = "## 固定案例（始终注入）"
DYNAMIC_SECTION_HEADER = "## 动态案例（按当前输入检索）"


class DynamicFewShotInjector:
    """D18: Retrieve + format few-shot examples for prompt injection.

    The injector is independent of any retriever: it pulls STATIC examples
    via ``store.list_all(kind=STATIC)`` (always) and DYNAMIC examples via
    ``store.search(text, top_k, kind=DYNAMIC)`` (only when
    ``config.dynamic_enabled``).
    """

    def __init__(
        self,
        fewshot_store: FewShotStore,
        config: DynamicFewShotConfig,
    ) -> None:
        self._store = fewshot_store
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_examples(self, text: str) -> list[FewShotExample]:
        """Return the combined list of static (+ dynamic) examples.

        The static portion is always included.  Dynamic examples are appended
        when ``config.dynamic_enabled`` is True; otherwise an empty list is
        appended (so the return value is just the static examples).
        """
        static = self._store.list_all(kind=FewShotKind.STATIC)
        if self._config.dynamic_enabled:
            dynamic = self._store.search(
                text,
                top_k=self._config.dynamic_top_k,
                kind=FewShotKind.DYNAMIC,
            )
        else:
            dynamic = []
        # Combine - order: static first, then dynamic
        combined: list[FewShotExample] = list(static)
        combined.extend(dynamic)
        return combined

    def inject(self, text: str) -> str:
        """Return a formatted few-shot section string for prompt injection.

        Layout:
        ```
        ## 固定案例（始终注入）
        ## 例子 1
        输入：...
        输出：...

        ## 例子 2
        ...

        ## 动态案例（按当前输入检索）
        ## 例子 1
        ...
        ```

        Returns an empty string when the store has no static and no dynamic
        examples (so callers can simply concatenate without guarding).
        """
        static = self._store.list_all(kind=FewShotKind.STATIC)
        if self._config.dynamic_enabled:
            dynamic = self._store.search(
                text,
                top_k=self._config.dynamic_top_k,
                kind=FewShotKind.DYNAMIC,
            )
        else:
            dynamic = []
        if not static and not dynamic:
            return ""
        lines: list[str] = [STATIC_SECTION_HEADER]
        for i, ex in enumerate(static, start=1):
            lines.append(self._format_example(i, ex))
        # Empty section still renders the header so the consumer (and tests)
        # can rely on a stable shape.
        lines.append("")
        lines.append(DYNAMIC_SECTION_HEADER)
        for i, ex in enumerate(dynamic, start=1):
            lines.append(self._format_example(i, ex))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_example(index: int, ex: FewShotExample) -> str:
        """Format a single example as ``## 例子 N`` block.

        Renders as:
        ```
        ## 例子 {index}
        输入：{text}
        输出：{output_json or intent or null}
        ```
        """
        parts: list[str] = [f"## 例子 {index}"]
        parts.append(f"输入：{ex.text}")
        if ex.output is not None:
            # Render the full output dict as JSON for max fidelity
            try:
                output_str = json.dumps(ex.output, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                output_str = str(ex.output)
            parts.append(f"输出：{output_str}")
        elif ex.intent is not None:
            parts.append(f"输出：{ex.intent}")
        else:
            parts.append("输出：null")
        return "\n".join(parts)
