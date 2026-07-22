"""Page guidance matcher (D3).

Maps UI events (button clicks, page navigation) to intents.
When no UI event is provided, returns None (pass-through to next matcher).
"""

from __future__ import annotations

from typing import Any

from ..models import IntentRecognitionResult, RecognitionSource


class PageGuidanceMatcher:
    """Matches UI events to intents.

    UI events are explicit user actions that strongly indicate intent
    (e.g., clicking a "next" button, navigating to the order list page).
    These matches bypass LLM layers because the intent is unambiguous.
    """

    # Default event -> intent mapping
    DEFAULT_EVENT_MAP: dict[str, str] = {
        "click:next": "continue",
        "click:previous": "go_back",
        "click:submit": "submit",
        "click:cancel": "cancel",
        "click:confirm": "confirm",
        "page:order_list": "order_query",
        "page:product_detail": "product_view",
        "page:cart": "view_cart",
        "page:checkout": "checkout",
    }

    def __init__(self, event_map: dict[str, str] | None = None) -> None:
        # Copy to avoid mutating class-level default
        self._event_map: dict[str, str] = dict(event_map or self.DEFAULT_EVENT_MAP)

    def register_event(self, event: str, intent: str) -> None:
        """Register or update an event -> intent mapping."""
        self._event_map[event] = intent

    def match(
        self, event: str | None, context: dict[str, Any] | None = None
    ) -> IntentRecognitionResult | None:
        """Match an event to an intent.

        Returns None when no event is provided (pass-through to next matcher),
        or when the event is not in the map.
        """
        if not event:
            return None
        intent = self._event_map.get(event)
        if intent is None:
            return None
        return IntentRecognitionResult(
            intent=intent,
            confidence=1.0,
            source=RecognitionSource.CODE_LAYER,
            layer_reached=1,
            signals={"page_guidance": 1.0},
        )
