"""Prompts for the lightweight LLM layer (D4, D5, D12, D18).

Builds system and user prompts for the Flash/Mini model used in Layer 2.
Includes candidate intents, boundaries, slot definitions, and optional
few-shot examples. Follows the user-input-normalization CAN_DO / CANNOT_DO
responsibility-boundary pattern.

D18 adds ``build_user_prompt_with_sections`` which renders STATIC + DYNAMIC
few-shot examples into two clearly-labelled sections. The original
``build_user_prompt`` is preserved unchanged for backward compatibility.
"""

from __future__ import annotations

import json
from typing import Any

from ..intent_registry import IntentRegistry
from ..models import FewShotExample


# ---------------------------------------------------------------------------
# Responsibility boundary (D3-style can do / cannot do list)
# ---------------------------------------------------------------------------

CAN_DO = """
你能做（且仅做）以下事情：
1. 识别用户输入对应的意图（从候选意图列表中选择）
2. 提取意图相关的槽位（slot）值
3. 提取硬约束（must）与软约束（try）
4. 输出 0-1 之间的置信度（confidence）
5. 当意图不明确或槽位缺失时，标记需要澄清（need_clarification）
6. 当输入超出候选意图范围时，标记为不支持（intent=null, rejection_reason）
""".strip()

CANNOT_DO = """
你不能做以下事情（严格禁止）：
1. 不直接回答用户问题（你只做意图识别，不回答问题本身）
2. 不执行工具（不调用任何外部工具或 API）
3. 不做最终推荐（不推荐任何方案或产品）
4. 不生成完整方案（不生成任何解决方案）
5. 不擅自补充事实（不编造上下文中不存在的信息）
6. 不伪造实时信息（不编造价格、日期、实时数据等）
7. 不输出候选意图列表之外的意图（除非标记为不支持）
""".strip()


# ---------------------------------------------------------------------------
# System prompt template (D12)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """你是一个轻量级意图识别助手（Layer 2 - Flash/Mini 模型）。
你的职责是根据用户输入，从候选意图列表中识别最匹配的意图，并提取相关槽位与约束。

# 职责边界

{can_do}

{cannot_do}

# 候选意图（含边界与槽位定义）

{intent_descriptions}

# 输出要求

你必须输出 JSON 格式的结构化结果，包含以下字段：

{{
  "intent": "意图名称（必须来自候选列表），不支持时为 null",
  "confidence": 0.0到1.0之间的置信度,
  "slots": {{
    "slot_name": "槽位值（可为 null 表示未提供）"
  }},
  "missing_slots": ["未提供的必选槽位名称"],
  "need_clarification": false,
  "clarification_question": null,
  "hard_constraints": [
    {{
      "type": "hard",
      "expression": "约束表达式，如 'price<100'",
      "raw_text": "原始用户文本片段"
    }}
  ],
  "soft_constraints": [
    {{
      "type": "soft",
      "expression": "约束表达式",
      "raw_text": "原始用户文本片段"
    }}
  ],
  "rejection_reason": "不支持时填写原因，否则为 null"
}}

# 判定原则

- 高置信度（>=0.85）：意图明确、槽位完整 -> 直接接受
- 中置信度（>=0.6 且 <0.85）：意图可能正确但存在歧义 -> 标记需要澄清
- 低置信度（<0.6）：意图不明或超出候选范围 -> 标记拒绝或交由深层 LLM 处理
- 硬约束（must）：必须满足的条件（如价格上限）
- 软约束（try）：尽量满足的偏好（如品牌偏好）
""".strip()


def build_system_prompt(registry: IntentRegistry) -> str:
    """Build the system prompt from the registry's intent definitions.

    Includes candidate intent names, descriptions, positive/negative examples
    (boundaries), and slot definitions.
    """
    lines: list[str] = []
    for d in registry.list_all():
        lines.append(f"- {d.name}: {d.description}")
        if d.positive_examples:
            lines.append(f"  正例: {'; '.join(d.positive_examples)}")
        if d.negative_examples:
            lines.append(f"  反例: {'; '.join(d.negative_examples)}")
        slot_desc = registry.build_slot_description(d.name)
        if slot_desc:
            lines.append(slot_desc)
    intent_descriptions = "\n".join(lines) if lines else "(无候选意图)"

    return SYSTEM_PROMPT_TEMPLATE.format(
        can_do=CAN_DO,
        cannot_do=CANNOT_DO,
        intent_descriptions=intent_descriptions,
    )


def build_user_prompt(
    text: str, few_shot_examples: list[dict[str, Any]] | None = None
) -> str:
    """Build the user prompt with optional few-shot examples.

    Few-shot examples are dicts with at least 'input' and 'output' keys.
    The output value may be a dict or a JSON string.
    """
    parts: list[str] = []

    if few_shot_examples:
        parts.append("# 参考例子（few-shot）\n")
        for i, ex in enumerate(few_shot_examples, 1):
            parts.append(f"## 例子 {i}")
            parts.append(f"输入：{ex.get('input', '')}")
            output = ex.get("output", {})
            if isinstance(output, (dict, list)):
                output_str = json.dumps(output, ensure_ascii=False, indent=2)
            else:
                output_str = str(output)
            parts.append(f"输出：{output_str}")
            parts.append("")
        parts.append("---\n")

    parts.append(f"# 用户输入\n{text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Two-section few-shot prompt (D18)
# ---------------------------------------------------------------------------

# Section headers shared with DynamicFewShotInjector - keep wording in sync.
_STATIC_SECTION_HEADER = "## 固定案例（始终注入）"
_DYNAMIC_SECTION_HEADER = "## 动态案例（按当前输入检索）"


def _format_fewshot_example(index: int, ex: FewShotExample) -> str:
    """Render a single FewShotExample as ``## 例子 N`` block (D18).

    Layout:
    ```
    ## 例子 {index}
    输入：{text}
    输出：{output_json or intent or null}
    ```
    """
    parts: list[str] = [f"## 例子 {index}"]
    parts.append(f"输入：{ex.text}")
    if ex.output is not None:
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


def build_user_prompt_with_sections(
    text: str,
    static_examples: list[FewShotExample] | None = None,
    dynamic_examples: list[FewShotExample] | None = None,
) -> str:
    """Build the user prompt with two-section few-shot format (D18).

    Renders as:
    ```
    # few-shot
    ## 固定案例（始终注入）
    ## 例子 1
    输入：...
    输出：...

    ## 例子 2
    ...

    ## 动态案例（按当前输入检索）
    ## 例子 1
    ...

    ---

    # 用户输入
    {text}
    ```

    When neither static nor dynamic examples are provided, the few-shot
    section is omitted entirely and only the user input is returned (matching
    the behavior of ``build_user_prompt`` with ``few_shot_examples=None``).
    """
    static = static_examples or []
    dynamic = dynamic_examples or []
    parts: list[str] = []
    if static or dynamic:
        parts.append("# few-shot")
        parts.append(_STATIC_SECTION_HEADER)
        idx = 1
        for ex in static:
            parts.append(_format_fewshot_example(idx, ex))
            idx += 1
        parts.append("")
        parts.append(_DYNAMIC_SECTION_HEADER)
        for ex in dynamic:
            parts.append(_format_fewshot_example(idx, ex))
            idx += 1
        parts.append("")
        parts.append("---")
        parts.append("")
    parts.append(f"# 用户输入\n{text}")
    return "\n".join(parts)
