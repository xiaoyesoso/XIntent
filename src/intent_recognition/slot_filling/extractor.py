"""Slot extractor - extracts slots from user input (D7)."""

from __future__ import annotations

import json
import re
from typing import Any

from user_input_normalization.llm.base import LLMClient

from ..intent_registry import IntentRegistry

SLOT_EXTRACTION_PROMPT = """你是一个参数抽取专家。从用户输入中抽取指定意图的参数（槽位）。

# 意图：{intent_name}
{intent_description}

# 需要抽取的槽位
{slot_definitions}

# 用户输入
{user_input}

# 输出格式（JSON）
{{
  "slots": {{}},
  "missing_slots": []
}}

- slots: 从用户输入中抽取到的槽位值
- missing_slots: 必选槽位中，用户输入未提及的槽位名称

只抽取用户明确表达的信息，不要猜测或编造。
"""


class SlotExtractor:
    """Extracts slots from user input based on intent definition."""

    def __init__(
        self,
        llm_client: LLMClient,
        registry: IntentRegistry,
    ) -> None:
        self._llm = llm_client
        self._registry = registry

    def extract(
        self, text: str, intent_name: str
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract slots from text for the given intent.

        Returns (extracted_slots, missing_slots).
        """
        definition = self._registry.get(intent_name)
        if not definition:
            return {}, []

        # If no slots defined, nothing to extract
        if not definition.slots:
            return {}, []

        slot_desc = self._registry.build_slot_description(intent_name)
        prompt = SLOT_EXTRACTION_PROMPT.format(
            intent_name=intent_name,
            intent_description=definition.description,
            slot_definitions=slot_desc,
            user_input=text,
        )

        response = self._llm.chat(
            system_prompt="你是一个参数抽取专家，只输出 JSON。",
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        data = self._extract_json(response)
        if not data:
            return {}, self._registry.get_required_slots(intent_name)

        slots = data.get("slots", {}) or {}
        missing = data.get("missing_slots", []) or []

        # Cross-check: verify required slots are in missing list if not extracted
        required = self._registry.get_required_slots(intent_name)
        for req in required:
            if req not in slots and req not in missing:
                missing.append(req)

        return slots, missing

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
