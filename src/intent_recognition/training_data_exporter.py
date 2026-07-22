"""Training-data exporter (D24).

Exports historical dialogue from an :class:`IntentHistoryStore` as JSONL
training records suitable for supervised fine-tuning of a lightweight
intent-recognition model.

Each output line is a chat-style record:

    {"messages": [
        {"role": "system", "content": "<system_prompt>"},
        {"role": "user", "content": "<original user text>"},
        {"role": "assistant", "content": "<JSON of IntentRecognitionResult>"}
    ]}

Design note: this module only produces the *integration point* (data export
+ before/after comparison). The actual fine-tuning job is run offline by
the platform; the resulting model is wired back in via
``OpenAICompatibleClient(model_tier="fine_tuned")``.
"""

from __future__ import annotations

import json
from typing import Any

from .models import IntentRecognitionResult
from .storage.base import IntentHistoryStore


class TrainingDataExporter:
    """Export historical recognition results as JSONL training data.

    Parameters
    ----------
    history_store:
        Store containing past :class:`IntentRecognitionResult` records.
    system_prompt_template:
        System prompt prepended to every exported record. May contain
        ``{intent_list}`` as a placeholder (left as-is when not present).
    """

    def __init__(
        self,
        history_store: IntentHistoryStore,
        system_prompt_template: str = "",
    ) -> None:
        self._history_store = history_store
        self._system_prompt_template = system_prompt_template

    def export(
        self,
        output_path: str,
        session_ids: list[str] | None = None,
    ) -> int:
        """Export history to ``output_path`` as JSONL.

        Parameters
        ----------
        output_path:
            Destination file path. Each line is one JSON training record.
        session_ids:
            Optional allow-list of session IDs to export. When ``None``,
            all sessions are exported via the store's ``list_sessions``
            method (if available).

        Returns
        -------
        int
            Number of records written.

        Raises
        ------
        NotImplementedError
            When ``session_ids is None`` and the store does not implement
            ``list_sessions()``.
        """
        if session_ids is None:
            session_ids = self._resolve_all_session_ids()

        system_prompt = self._system_prompt_template
        count = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for session_id in session_ids:
                history = self._history_store.get_history(session_id)
                for result in history:
                    record = self._build_record(result, system_prompt)
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_all_session_ids(self) -> list[str]:
        """Return all session IDs known to the store.

        Falls back to the store's ``list_sessions()`` method when present.
        Production stores (Redis / MySQL) should implement this method;
        the in-memory store shipped for dev/test does not, so callers
        must pass ``session_ids`` explicitly.
        """
        list_sessions = getattr(self._history_store, "list_sessions", None)
        if not callable(list_sessions):
            raise NotImplementedError(
                "IntentHistoryStore does not implement list_sessions(); "
                "pass session_ids explicitly or extend the store with a "
                "list_sessions() -> list[str] method."
            )
        return list(list_sessions())

    def _build_record(
        self,
        result: IntentRecognitionResult,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Build a single JSONL record from an ``IntentRecognitionResult``."""
        # IntentHistoryStore persists IntentRecognitionResult but not the
        # original user text. We fall back to a ``_raw_text`` slot if the
        # pipeline stashed it there; otherwise we emit only the assistant
        # turn.
        # TODO(production): extend IntentHistoryStore to store raw user text
        # alongside the result so the user turn is always populated.
        raw_text = result.slots.get("_raw_text", "") if result.slots else ""
        assistant_content = json.dumps(
            result.model_dump(), ensure_ascii=False
        )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if raw_text:
            messages.append({"role": "user", "content": str(raw_text)})
        messages.append({"role": "assistant", "content": assistant_content})

        return {"messages": messages}
