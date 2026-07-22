"""Keyword and regex matcher (D3).

Matches simple, unambiguous inputs like "继续", "下一步" against configurable
keyword-to-intent and regex-to-intent mappings.
"""

from __future__ import annotations

import re

from ..models import IntentRecognitionResult, RecognitionSource


class KeywordMatcher:
    """Matches simple keyword and regex patterns to intents.

    Keyword matches are exact (after strip). Regex matches use search.
    The first matching keyword (in insertion order) wins; if no keyword
    matches, the first matching regex wins.
    """

    # Default keyword -> intent mapping (ordered)
    DEFAULT_KEYWORD_MAP: dict[str, str] = {
        "继续": "continue",
        "下一步": "continue",
        "上一步": "go_back",
        "返回": "go_back",
        "取消": "cancel",
        "确认": "confirm",
        "是": "confirm",
        "否": "cancel",
    }

    # Default regex pattern -> intent mapping (ordered)
    DEFAULT_REGEX_MAP: list[tuple[str, str]] = [
        (r"查询订单.*", "order_query"),
        (r"我的订单.*", "order_query"),
        (r"退款.*", "refund"),
        (r"退货.*", "refund"),
    ]

    def __init__(
        self,
        keyword_map: dict[str, str] | None = None,
        regex_map: list[tuple[str, str]] | None = None,
    ) -> None:
        self._keyword_map: dict[str, str] = dict(
            keyword_map or self.DEFAULT_KEYWORD_MAP
        )
        self._regex_map: list[tuple[str, str]] = list(
            regex_map or self.DEFAULT_REGEX_MAP
        )
        # Pre-compile regex patterns for performance (< 10ms target)
        self._compiled: list[tuple[re.Pattern[str], str]] = [
            (re.compile(p), intent) for p, intent in self._regex_map
        ]

    def register_keyword(self, keyword: str, intent: str) -> None:
        """Register a keyword -> intent mapping."""
        self._keyword_map[keyword] = intent

    def register_regex(self, pattern: str, intent: str) -> None:
        """Register a regex pattern -> intent mapping."""
        self._regex_map.append((pattern, intent))
        self._compiled.append((re.compile(pattern), intent))

    def match(self, text: str) -> IntentRecognitionResult | None:
        """Match text against keyword and regex maps.

        Returns the first match, or None when no keyword/regex matches.
        """
        if not text:
            return None

        stripped = text.strip()

        # Keyword exact match first (highest priority)
        intent = self._keyword_map.get(stripped)
        if intent is not None:
            return IntentRecognitionResult(
                intent=intent,
                confidence=1.0,
                source=RecognitionSource.CODE_LAYER,
                layer_reached=1,
                signals={"keyword_match": 1.0},
            )

        # Regex search (in order)
        for pattern, intent_name in self._compiled:
            if pattern.search(stripped):
                return IntentRecognitionResult(
                    intent=intent_name,
                    confidence=1.0,
                    source=RecognitionSource.CODE_LAYER,
                    layer_reached=1,
                    signals={"regex_match": 1.0},
                )

        return None
