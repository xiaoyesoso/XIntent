"""Constraint extractor - identifies hard/soft constraints (D8)."""

from __future__ import annotations

import json
import re
from typing import Any

from user_input_normalization.llm.base import LLMClient

from ..models import Constraint, ConstraintType

CONSTRAINT_PROMPT = """从用户输入中识别硬约束和软约束。

# 硬约束（hard_constraints）
用户明确要求必须满足的条件，不满足则失败。
例如：
- "不超过100块" -> {{"expression": "price<100", "raw_text": "不超过100块"}}
- "不要修改A接口" -> {{"expression": "file!=A接口", "raw_text": "不要修改A接口"}}
- "必须是牛仔裤" -> {{"expression": "type=牛仔裤", "raw_text": "必须是牛仔裤"}}

# 软约束（soft_constraints）
用户表达的偏好，尽力满足但可以不满足。
例如：
- "最好是牛仔裤" -> {{"expression": "type=牛仔裤", "raw_text": "最好是牛仔裤"}}
- "轻一点更好" -> {{"expression": "weight=light", "raw_text": "轻一点更好"}}

# 用户输入
{user_input}

# 输出格式（JSON）
{{
  "hard_constraints": [{{"expression": "...", "raw_text": "..."}}],
  "soft_constraints": [{{"expression": "...", "raw_text": "..."}}]
}}

只识别用户明确表达的约束，不要猜测。
"""

# Built-in regex patterns for common constraints
HARD_CONSTRAINT_PATTERNS = [
    (r"不超过\s*(\d+)\s*块?", lambda m: Constraint(
        type=ConstraintType.HARD, expression=f"price<{m.group(1)}", raw_text=m.group(0)
    )),
    (r"必须\s*(.+?)(?:[，。！？\s]|$)", lambda m: Constraint(
        type=ConstraintType.HARD, expression=m.group(1).strip(), raw_text=m.group(0)
    )),
    (r"不要\s*(.+?)(?:[，。！？\s]|$)", lambda m: Constraint(
        type=ConstraintType.HARD, expression=f"!{m.group(1).strip()}", raw_text=m.group(0)
    )),
    (r"不能\s*(.+?)(?:[，。！？\s]|$)", lambda m: Constraint(
        type=ConstraintType.HARD, expression=f"!{m.group(1).strip()}", raw_text=m.group(0)
    )),
]

SOFT_CONSTRAINT_PATTERNS = [
    (r"最好(?:是)?\s*(.+?)(?:[，。！？\s]|$)", lambda m: Constraint(
        type=ConstraintType.SOFT, expression=m.group(1).strip(), raw_text=m.group(0)
    )),
    (r"优先\s*(.+?)(?:[，。！？\s]|$)", lambda m: Constraint(
        type=ConstraintType.SOFT, expression=m.group(1).strip(), raw_text=m.group(0)
    )),
    (r".*更好", lambda m: Constraint(
        type=ConstraintType.SOFT, expression="better", raw_text=m.group(0)
    )),
]


class ConstraintExtractor:
    """Extracts hard and soft constraints from user input."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def extract(self, text: str) -> list[Constraint]:
        """Extract all constraints from text.

        Uses regex patterns first (fast), then LLM if available (comprehensive).
        """
        constraints: list[Constraint] = []

        # Fast path: regex patterns
        for pattern, builder in HARD_CONSTRAINT_PATTERNS:
            for match in re.finditer(pattern, text):
                constraints.append(builder(match))

        for pattern, builder in SOFT_CONSTRAINT_PATTERNS:
            for match in re.finditer(pattern, text):
                constraints.append(builder(match))

        # If LLM available, use it for comprehensive extraction
        if self._llm:
            llm_constraints = self._extract_with_llm(text)
            # Merge: LLM results may catch what regex missed
            existing_exprs = {c.expression for c in constraints}
            for c in llm_constraints:
                if c.expression not in existing_exprs:
                    constraints.append(c)

        return constraints

    def extract_separated(self, text: str) -> tuple[list[Constraint], list[Constraint]]:
        """Return (hard_constraints, soft_constraints) separately."""
        all_constraints = self.extract(text)
        hard = [c for c in all_constraints if c.type == ConstraintType.HARD]
        soft = [c for c in all_constraints if c.type == ConstraintType.SOFT]
        return hard, soft

    def _extract_with_llm(self, text: str) -> list[Constraint]:
        """Use LLM for comprehensive constraint extraction."""
        if not self._llm:
            return []

        prompt = CONSTRAINT_PROMPT.format(user_input=text)
        response = self._llm.chat(
            system_prompt="你是一个约束识别专家，只输出 JSON。",
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        data = self._extract_json(response)
        if not data:
            return []

        result: list[Constraint] = []
        for item in data.get("hard_constraints", []):
            if isinstance(item, dict):
                result.append(Constraint(
                    type=ConstraintType.HARD,
                    expression=item.get("expression", ""),
                    raw_text=item.get("raw_text", ""),
                ))
            elif isinstance(item, str):
                result.append(Constraint(
                    type=ConstraintType.HARD, expression=item, raw_text=item
                ))

        for item in data.get("soft_constraints", []):
            if isinstance(item, dict):
                result.append(Constraint(
                    type=ConstraintType.SOFT,
                    expression=item.get("expression", ""),
                    raw_text=item.get("raw_text", ""),
                ))
            elif isinstance(item, str):
                result.append(Constraint(
                    type=ConstraintType.SOFT, expression=item, raw_text=item
                ))

        return result

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON from text."""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
