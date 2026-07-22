"""D17: Retrieval-based candidate narrowing.

Narrows the candidate intent set before the lightweight LLM (Layer 2) runs.
Three implementations share a common ``CandidateRetriever`` interface:

* ``VectorCandidateRetriever`` - pure-text Jaccard similarity (no LLM call)
* ``LLMCoarseRetriever``       - one cheap LLM call that returns a JSON list
* ``HybridCandidateRetriever`` - vector Top-2N then LLM Top-N

A dynamic-N strategy widens the candidate set when the context window has
plenty of free space or when a previous L2/L3 attempt failed and is being
retried.  Retriever modules are entirely opt-in (``RetrievalConfig.enable``
defaults to ``False``), preserving the existing three-layer behavior.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from user_input_normalization.llm.base import LLMClient

from ..config import RetrievalConfig
from ..models import IntentDefinition


# ---------------------------------------------------------------------------
# Tokenization helpers (mirror MemoryFewShotStore so vector retrieval uses the
# exact same notion of similarity as few-shot retrieval)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Char-level tokenization for CJK + whitespace split for Latin tokens."""
    tokens: set[str] = set()
    if not text:
        return tokens
    for part in text.split():
        tokens.add(part.lower())
        for ch in part:
            if "\u4e00" <= ch <= "\u9fff":
                tokens.add(ch)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard token-overlap similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _intent_corpus(intent: IntentDefinition) -> str:
    """Concatenate description + positive examples for an intent."""
    parts: list[str] = [intent.description]
    parts.extend(intent.positive_examples)
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CandidateRetriever(ABC):
    """Abstract base class for D17 candidate narrowing.

    Implementations narrow ``all_intents`` to a smaller ranked candidate set
    before Layer 2 (lightweight LLM) runs.  Returning the full input list is
    a valid no-op narrowing, so a disabled or empty retriever can simply
    return ``list(all_intents)``.
    """

    @abstractmethod
    def retrieve(
        self,
        text: str,
        all_intents: list[IntentDefinition],
        context: dict[str, Any] | None = None,
    ) -> list[IntentDefinition]:
        """Return a narrowed candidate list (subset of ``all_intents``)."""

    # ------------------------------------------------------------------
    # Dynamic N strategy (shared by all implementations)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dynamic_n(
        base_n: int,
        context: dict[str, Any] | None,
        retry_count: int,
    ) -> int:
        """D17: widen N based on context headroom or retry pressure.

        * When ``context["context_window_free_pct"] > 0.5`` we have plenty of
          token budget - return ``int(base_n * 1.5)`` so the LLM sees more
          candidates and can pick a better one.
        * When ``retry_count > 0`` (L2 escalated + L3 failed retry) we widen
          aggressively but cap at 100 to avoid runaway prompts - return
          ``min(base_n * 2, 100)``.
        * Otherwise return ``base_n`` unchanged.
        """
        if context:
            free_pct = context.get("context_window_free_pct")
            if isinstance(free_pct, (int, float)) and free_pct > 0.5:
                return int(base_n * 1.5)
        if retry_count > 0:
            return min(base_n * 2, 100)
        return base_n

    @staticmethod
    def _retry_count_from_context(context: dict[str, Any] | None) -> int:
        """Read ``retry_count`` from a context dict (0 when absent/invalid)."""
        if not context:
            return 0
        v = context.get("retry_count", 0)
        if isinstance(v, bool):
            # bool is a subclass of int - guard against truthy booleans
            return 0
        if isinstance(v, int) and v > 0:
            return v
        return 0

    def _effective_n(
        self,
        config: RetrievalConfig,
        context: dict[str, Any] | None,
    ) -> int:
        """Return the candidate count to retrieve given config + context."""
        base_n = config.top_n
        if not config.dynamic_n:
            return base_n
        retry_count = self._retry_count_from_context(context)
        return self._compute_dynamic_n(base_n, context, retry_count)


# ---------------------------------------------------------------------------
# Vector retriever (no LLM)
# ---------------------------------------------------------------------------


class VectorCandidateRetriever(CandidateRetriever):
    """D17: Vector retriever using Jaccard token-overlap similarity.

    No external vector DB required - uses the same Jaccard approach as
    ``MemoryFewShotStore`` so behavior is consistent across dev/test.  Each
    intent is scored against the query by Jaccard similarity of its
    ``description + positive_examples`` tokens.
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        # Caller may pass a config with enable=False; we still construct so the
        # factory pattern can flip it on.  Internal methods read top_n etc.
        self._config = config or RetrievalConfig(enable=True)

    def retrieve(
        self,
        text: str,
        all_intents: list[IntentDefinition],
        context: dict[str, Any] | None = None,
    ) -> list[IntentDefinition]:
        if not all_intents:
            return []
        n = self._effective_n(self._config, context)
        # If N covers everything, no narrowing needed
        if n >= len(all_intents):
            return list(all_intents)
        query_tokens = _tokenize(text)
        scored: list[tuple[IntentDefinition, float]] = []
        for intent in all_intents:
            corpus_tokens = _tokenize(_intent_corpus(intent))
            score = _jaccard(query_tokens, corpus_tokens)
            scored.append((intent, score))
        # Stable sort by score desc; Python's sort is stable so ties keep
        # registry order, which is deterministic for tests.
        scored.sort(key=lambda x: x[1], reverse=True)
        # Keep only the top-N (drop zero-score tail only if N is smaller
        # than the candidate pool; that is already guaranteed above).
        return [intent for intent, _ in scored[:n]]


# ---------------------------------------------------------------------------
# LLM coarse retriever
# ---------------------------------------------------------------------------


_COARSE_PROMPT_TEMPLATE = """你是一个意图候选筛选助手（D17）。

任务：从下面的候选意图列表中，挑选出可能与用户输入匹配的意图，按可能性从高到低排序。

候选意图列表（名称: 描述）：
{intent_list}

用户输入：
{user_text}

请输出一个 JSON 数组，仅包含最可能匹配的意图名称，最多 {top_n} 个。
格式示例：["intent_a", "intent_b"]
不要输出其他内容。
""".strip()


class LLMCoarseRetriever(CandidateRetriever):
    """D17: LLM-based coarse retriever.

    Sends a single cheap LLM call with a minimal prompt that lists all
    candidate intent names + descriptions, asks the model to return a JSON
    array of the most likely names, then maps those names back to
    ``IntentDefinition`` objects.

    Falls back to the full candidate list (truncated to ``top_n``) when the
    LLM response cannot be parsed or no names match - this prevents a
    broken LLM call from zeroing out the candidate set.
    """

    def __init__(self, llm_client: LLMClient, config: RetrievalConfig) -> None:
        self._llm = llm_client
        self._config = config

    def retrieve(
        self,
        text: str,
        all_intents: list[IntentDefinition],
        context: dict[str, Any] | None = None,
    ) -> list[IntentDefinition]:
        if not all_intents:
            return []
        n = self._effective_n(self._config, context)
        if n >= len(all_intents):
            return list(all_intents)

        intent_list = "\n".join(
            f"- {d.name}: {d.description}" for d in all_intents
        )
        prompt = _COARSE_PROMPT_TEMPLATE.format(
            intent_list=intent_list,
            user_text=text,
            top_n=n,
        )
        raw = self._llm.chat(
            system_prompt=(
                "You are a coarse intent candidate retriever for D17. "
                "Return ONLY a JSON array of intent names."
            ),
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=500,
        )
        names = self._extract_json_list(raw)
        by_name = {d.name: d for d in all_intents}
        ordered: list[IntentDefinition] = []
        for name in names:
            intent = by_name.get(name)
            if intent is not None and intent not in ordered:
                ordered.append(intent)
        if not ordered:
            # Could not parse or no names matched - fall back to the first N
            # candidates so L2 still gets something to work with.
            return list(all_intents)[:n]
        # Truncate to N (defensive - LLM may return more than asked)
        return ordered[:n]

    @staticmethod
    def _extract_json_list(text: str) -> list[str]:
        """Parse a JSON list of strings from an LLM response.

        Mirrors the 3-strategy extraction pattern from
        ``lightweight_llm.classifier._extract_json`` but targets JSON arrays
        instead of objects.
        """
        if not text:
            return []
        # Strategy 1: direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
        # Strategy 2: ```json fenced block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return [str(x) for x in data]
            except json.JSONDecodeError:
                pass
        # Strategy 3: first [ ... ] block (greedy)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(x) for x in data]
            except json.JSONDecodeError:
                pass
        return []


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridCandidateRetriever(CandidateRetriever):
    """D17: Hybrid retriever = vector Top-2N then LLM Top-N.

    The vector stage is cheap (no LLM) and acts as a recall step that
    narrows the full registry down to ``2 * N`` candidates.  The LLM stage
    then does the expensive precision pass over this smaller pool, returning
    the final ``N`` candidates.
    """

    def __init__(
        self,
        vector_retriever: VectorCandidateRetriever,
        llm_retriever: LLMCoarseRetriever,
    ) -> None:
        self._vector = vector_retriever
        self._llm = llm_retriever

    def retrieve(
        self,
        text: str,
        all_intents: list[IntentDefinition],
        context: dict[str, Any] | None = None,
    ) -> list[IntentDefinition]:
        if not all_intents:
            return []
        # Stage 1: vector recall over 2 * N
        vector_n = self._vector._effective_n(self._vector._config, context)
        recall_n = min(max(vector_n * 2, vector_n), len(all_intents))
        # Temporarily widen the vector retriever's effective N for the recall
        # pass.  We do this by reading its config and constructing a widened
        # snapshot, since the effective N also depends on dynamic_n.
        original_top_n = self._vector._config.top_n
        try:
            self._vector._config.top_n = recall_n
            recall_pool = self._vector.retrieve(text, all_intents, context)
        finally:
            # Restore original top_n even if retrieve raises
            self._vector._config.top_n = original_top_n
        if not recall_pool:
            return []
        # Stage 2: LLM precision over the recall pool (final N)
        return self._llm.retrieve(text, recall_pool, context)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_retriever(
    config: RetrievalConfig,
    llm_client: LLMClient | None = None,
) -> CandidateRetriever | None:
    """Build a retriever from config.

    Returns ``None`` when retrieval is disabled (``config.enable=False``)
    or when a method that requires an LLM client is selected but no client
    was passed.  Callers should treat ``None`` as "no narrowing" and feed
    the full intent list to Layer 2 unchanged.
    """
    if not config.enable:
        return None
    if config.method == "vector":
        return VectorCandidateRetriever(config)
    if config.method == "llm_coarse":
        if llm_client is None:
            return None
        return LLMCoarseRetriever(llm_client, config)
    if config.method == "hybrid":
        if llm_client is None:
            return None
        vector_r = VectorCandidateRetriever(config)
        llm_r = LLMCoarseRetriever(llm_client, config)
        return HybridCandidateRetriever(vector_r, llm_r)
    # Unknown method: disable
    return None
