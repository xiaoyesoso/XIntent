"""pre-normalization responsibility boundary prompts (corresponds to D3).

Defines the "can do / cannot do" list to prevent the large model from overstepping its authority.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Responsibility boundary (D3 can do / cannot do list)
# ---------------------------------------------------------------------------

CAN_DO = """
你能做（且仅做）以下事情：
1. 指代消解：将"这个、那个、第一个、刚才那个"等代词替换为具体实体
2. 省略句补全：补全缺失的主语、宾语、动作对象、约束条件
3. 病句修正：修正口语化、跳跃、语序混乱、临时改口的输入，保留原意
4. 术语标准化：将非标准术语、同义词标准化为统一术语
5. 黑话解释：将黑话、内部代号解释为标准表达
6. 多义词消歧：根据上下文对多义词进行消歧
7. 判断是否需要澄清：当代词消解不出、词义不明时，标记需要澄清
8. 判断是否需要搜索：当输入涉及外部事实时，标记需要搜索
""".strip()

CANNOT_DO = """
你不能做以下事情（严格禁止）：
1. 不直接回答用户问题（这是最容易犯的错误！你只做输入规范化，不回答问题本身）
2. 不执行工具（不调用任何外部工具或 API）
3. 不做最终推荐（不推荐任何方案或产品）
4. 不判断业务意图（意图识别是下游模块的事）
5. 不生成完整方案（不生成任何解决方案）
6. 不擅自补充事实（不编造上下文中不存在的信息）
7. 不强行猜测低置信内容（置信度低时标记需澄清，而非瞎猜）
8. 不伪造实时信息（不编造价格、日期、实时数据等）
""".strip()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""你是一个用户输入规范化助手。你的职责是将用户随意的、不规范的输入转化为规范化的、结构化的表达。

# 职责边界

{CAN_DO}

{CANNOT_DO}

# 输出要求

你必须输出 JSON 格式的结构化结果，包含以下字段：

{{
  "normalized_input": "规范化后的完整句子",
  "spo": {{
    "subject": "主语（可为 null）",
    "subject_source": "主语来源：原文 / 补全 / 对话历史第N轮",
    "predicate": "谓语",
    "obj": "宾语（可为 null）",
    "obj_source": "宾语来源"
  }},
  "modifiers": {{
    "attributive": "定语解释（可为 null）",
    "adverbial": "状语解释（可为 null）",
    "complement": "补语解释（可为 null）"
  }},
  "pronoun_resolutions": [
    {{
      "pronoun": "代词，如'第二个'",
      "resolved_to": "指代对象，如'TCC方案'",
      "confidence": 0.92,
      "evidence_source": "证据来源，如'对话历史第3轮'",
      "named_entity": "语义化名称（按名索引），如'TCC方案'（可为 null）"
    }}
  ],
  "quantifiable_adjectives": [
    {{
      "adjective": "形容词，如'最有性价比'",
      "quantified": false,
      "quantified_value": null,
      "route_to": "deep"
    }}
  ],
  "term_mappings": [
    {{
      "original": "原始术语，如'RAG'",
      "standard": "标准术语，如'检索增强生成'",
      "source": "来源，如'vocabulary-table'"
    }}
  ],
  "completions": [
    {{
      "field": "补全字段，如'主语'",
      "content": "补全内容",
      "source": "补全来源，如'对话历史第2轮'"
    }}
  ]
}}

# 完整性要求

- 检查主谓宾是否完整
- 检查代词是否完全消解（输出代词消解表格覆盖全部代词）
- 形容词、极值等是否已输出"可量化"内容（如未量化，标记 route_to: "deep"）
- 代词消解表格需作为关键事实贯穿本轮对话

# 命名规范

为消解后的实体赋予语义化名称（如"TCC方案"而非"方案一"），以便用户后续按名引用。
""".strip()


# ---------------------------------------------------------------------------
# few-shot example formatting
# ---------------------------------------------------------------------------

def format_fewshot(examples: list) -> str:
    """Format few-shot examples as a prompt fragment."""
    if not examples:
        return ""
    lines = ["# 参考例子（few-shot）\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"## 例子 {i}")
        lines.append(f"输入：{ex.input}")
        if ex.context_summary:
            lines.append(f"上下文：{ex.context_summary}")
        import json
        lines.append(f"输出：{json.dumps(ex.normalized_output, ensure_ascii=False, indent=2)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(bundle) -> str:
    """Format context as a prompt fragment."""
    lines = ["# 上下文信息\n"]

    # User profile
    if bundle and getattr(bundle, "user_profile", None):
        profile = bundle.user_profile
        lines.append(f"用户行业：{profile.industry}")
        if profile.topic_tendencies:
            tendencies = "、".join(f"{k}->{v}" for k, v in profile.topic_tendencies.items())
            lines.append(f"话题倾向：{tendencies}")
        if profile.preferences:
            import json
            lines.append(f"用户偏好：{json.dumps(profile.preferences, ensure_ascii=False)}")
        lines.append("")

    # Key facts
    if bundle and getattr(bundle, "key_facts", None):
        lines.append("关键事实（跨轮复用）：")
        for fact in bundle.key_facts:
            import json
            lines.append(f"  - [{fact.fact_type.value}] {json.dumps(fact.content, ensure_ascii=False)}")
        lines.append("")

    # Dialogue history summary
    if bundle and getattr(bundle, "dialogue_summary", None):
        lines.append(f"对话摘要：{bundle.dialogue_summary}")
        lines.append("")

    # Recalled dialogue details
    if bundle and getattr(bundle, "recalled_details", None):
        lines.append("相关对话细节：")
        for detail in bundle.recalled_details:
            lines.append(f"  - [第{detail.turn}轮] {detail.role}: {detail.content[:200]}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vocabulary injection formatting
# ---------------------------------------------------------------------------

def format_vocab(entries: list) -> str:
    """Format vocabulary entries as a prompt fragment."""
    if not entries:
        return ""
    lines = ["# 词汇表（术语 / 黑话 / 缩写）\n"]
    for entry in entries:
        alt = ""
        if entry.alternative_meanings:
            alt = "（备选含义：" + "、".join(f"{a.industry}:{a.meaning}" for a in entry.alternative_meanings) + "）"
        lines.append(f"- {entry.term} => {entry.standard_meaning} [行业:{entry.industry}]{alt}")
    lines.append("")
    return "\n".join(lines)
