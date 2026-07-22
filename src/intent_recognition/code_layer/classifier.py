"""Code-layer classifier (D3).

Orchestrates PageGuidanceMatcher -> KeywordMatcher -> RuleEngine.
Zero LLM calls; target < 10ms latency. All matches set source=CODE_LAYER,
confidence=1.0, layer_reached=1. Returns None when unmatched (signal to
escalate to the lightweight LLM layer).
"""

from __future__ import annotations

from typing import Any

from ..models import IntentRecognitionResult
from .keyword_matcher import KeywordMatcher
from .page_guidance import PageGuidanceMatcher
from .rule_engine import RuleEngine


class CodeLayerClassifier:
    """Layer 1 classifier: deterministic, code-based intent recognition.

    Pipeline order (first match wins):
      1. PageGuidanceMatcher - explicit UI events
      2. KeywordMatcher - exact keywords and regex patterns
      3. RuleEngine - context-aware rules on normalized input

    Returns None when no matcher produces a result, signalling the caller
    to escalate to Layer 2 (lightweight LLM).
    """

    def __init__(
        self,
        page_guidance: PageGuidanceMatcher | None = None,
        keyword_matcher: KeywordMatcher | None = None,
        rule_engine: RuleEngine | None = None,
    ) -> None:
        self.page_guidance = page_guidance or PageGuidanceMatcher()
        self.keyword_matcher = keyword_matcher or KeywordMatcher()
        self.rule_engine = rule_engine or RuleEngine()

    def classify(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        event: str | None = None,
    ) -> IntentRecognitionResult | None:
        """Run the three matchers in order; return first match or None."""
        # 1. Page guidance (UI events)
        result = self.page_guidance.match(event, context)
        if result is not None:
            return result

        # 2. Keyword / regex
        result = self.keyword_matcher.match(text)
        if result is not None:
            return result

        # 3. Rule engine (context-aware, works on normalized text)
        result = self.rule_engine.match(text, context)
        if result is not None:
            return result

        # Unmatched: signal escalation to L2 (caller checks for None)
        return None
