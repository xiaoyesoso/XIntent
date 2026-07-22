"""Offline analyzer (corresponding to tasks 8.3, 8.4, 8.5 / D11 self-iteration mechanism).

Scans multi-turn dialogue history, marks suspected terms, records occurrence counts,
and promotes them to official vocabulary entries once thresholds are met.

Flow:
1. analyze_session: scan the session, extract suspected terms, call increment_count to record
2. check_promotion: check whether all candidate terms have reached the threshold, trigger promotion
3. run: full offline analysis flow, returns a summary report
"""

from __future__ import annotations

import re
from typing import Any

from ..config import Config, get_config
from ..models import VocabEntry, VocabLevel, VocabStatus
from ..storage.base import DialogueHistoryStore, VocabStore


# ---------------------------------------------------------------------------
# Suspected term extraction patterns
# ---------------------------------------------------------------------------

# English abbreviations: 2-6 uppercase letters (e.g. RAG, CRM, XQR)
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")

# Chinese "jargon" pattern: 2-4 Chinese characters + common suffix (e.g. "抓手", "对齐", "赋能")
_SLANG_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,4}")

# Known stopwords (avoid marking common words as candidates)
_STOPWORDS = {
    "这个", "那个", "什么", "怎么", "为什么", "可以", "能够",
    "我们", "你们", "他们", "一个", "一些", "已经", "现在",
    "今天", "明天", "昨天", "目前", "之后", "之前", "然后",
    "但是", "因为", "所以", "如果", "虽然", "不过", "而且",
    "非常", "特别", "比较", "稍微", "一点", "一些",
    "我", "你", "他", "她", "它", "的", "了", "是", "在", "有",
    "和", "与", "或", "及", "等", "之", "其",
}


def extract_potential_terms(text: str) -> list[str]:
    """Extract potential non-standard terms from text.

    Extraction rules:
    - English abbreviations (2-6 uppercase letters)
    - Chinese 2-4 character combinations (filtering stopwords)

    Args:
        text: Text to extract from

    Returns:
        List of extracted potential terms (deduplicated, order-preserving)
    """
    if not text:
        return []

    terms: list[str] = []
    seen: set[str] = set()

    # 1. English abbreviations
    for match in _ACRONYM_PATTERN.finditer(text):
        term = match.group(0)
        if term not in seen:
            terms.append(term)
            seen.add(term)

    # 2. Chinese 2-4 character combinations (simple sliding window)
    # Here we only extract consecutive 2-4 Chinese characters as a candidate term
    chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
    for chunk in chinese_chars:
        # Extract 2-4 character substrings
        n = len(chunk)
        for size in (2, 3, 4):
            if n < size:
                continue
            for i in range(n - size + 1):
                term = chunk[i : i + size]
                if term in _STOPWORDS:
                    continue
                # Filter overly common structures (e.g. "的话", "不了")
                if term.endswith(("的", "了", "着", "过", "们")):
                    continue
                if term not in seen:
                    terms.append(term)
                    seen.add(term)

    return terms


def infer_basic_meaning(term: str, context: str) -> str:
    """Infer the basic meaning of a term based on context (simple implementation).

    In production, an LLM should be called for inference. Here we use a simple context snippet extraction.

    Args:
        term: The term
        context: The context in which the term appears

    Returns:
        Inferred basic meaning (string)
    """
    if not context:
        return ""

    # Find the position of term in context, extract surrounding snippet
    idx = context.find(term)
    if idx < 0:
        return ""

    start = max(0, idx - 10)
    end = min(len(context), idx + len(term) + 20)
    snippet = context[start:end].replace("\n", " ").strip()
    return f"上下文片段：...{snippet}..."


class OfflineAnalyzer:
    """Offline analyzer (corresponds to D11 self-iteration mechanism).

    Scans multi-turn dialogue history, marks suspected terms, records occurrence counts
    and basic meaning inferences, and promotes them to official vocabulary once thresholds are met.
    """

    def __init__(
        self,
        vocab_store: VocabStore,
        dialogue_store: DialogueHistoryStore,
        config: Config | None = None,
    ) -> None:
        self.vocab_store = vocab_store
        self.dialogue_store = dialogue_store
        self.config = config or get_config()

    # ------------------------------------------------------------------
    # Offline analysis: mark suspected terms (task 8.3)
    # ------------------------------------------------------------------

    def analyze_session(self, session_id: str) -> list[VocabEntry]:
        """Scan multi-turn dialogue, mark suspected terms, record occurrence counts.

        Args:
            session_id: Session ID

        Returns:
            List of candidate vocabulary entries hit in this scan
        """
        turns = self.dialogue_store.get_recent(session_id, n=1000)
        if not turns:
            return []

        # Extract user_id (inferred from the first turn, simplified implementation)
        user_id = self._infer_user_id(session_id, turns)

        found_entries: list[VocabEntry] = []
        seen_ids: set[str] = set()

        for turn in turns:
            if turn.role != "user":
                continue
            terms = extract_potential_terms(turn.content)
            for term in terms:
                # Call increment_count to record occurrence count
                entry = self.vocab_store.increment_count(term, user_id=user_id)
                if entry is not None and entry.vocab_id not in seen_ids:
                    # Infer basic meaning (if standard meaning is empty)
                    if not entry.standard_meaning:
                        meaning = infer_basic_meaning(term, turn.content)
                        if meaning:
                            entry.standard_meaning = meaning
                    found_entries.append(entry)
                    seen_ids.add(entry.vocab_id)

        return found_entries

    # ------------------------------------------------------------------
    # Threshold promotion check (task 8.4, 8.5)
    # ------------------------------------------------------------------

    def check_promotion(self, session_id: str) -> list[VocabEntry]:
        """Check whether all candidate terms have reached the threshold, trigger promotion.

        Promotion conditions (any one):
        - Total occurrence count > min_total_count (100)
        - Number of discussants > min_discussant_count (3) -> public vocabulary
        - Consecutive discussion count > min_consecutive_count (10) -> personal vocabulary

        If require_human_review=True, promote to DRAFT for review;
        otherwise promote directly to PROMOTED.

        Args:
            session_id: Session ID (used for audit records; this implementation does not strictly depend on it)

        Returns:
            List of vocabulary entries promoted in this run
        """
        candidates = self.vocab_store.list_candidates(status="candidate")
        promo_cfg = self.config.vocab_promotion

        promoted: list[VocabEntry] = []
        for entry in candidates:
            met = (
                entry.occurrence_count > promo_cfg.min_total_count
                or entry.discussant_count > promo_cfg.min_discussant_count
                or entry.consecutive_count > promo_cfg.min_consecutive_count
            )
            if not met:
                continue

            # Determine level based on number of discussants (task 8.5)
            new_level = (
                VocabLevel.PUBLIC
                if entry.discussant_count > promo_cfg.min_discussant_count
                else VocabLevel.PERSONAL
            )
            if entry.level != new_level:
                entry.level = new_level

            # Determine post-promotion status
            if promo_cfg.require_human_review:
                new_status = VocabStatus.DRAFT
            else:
                new_status = VocabStatus.PROMOTED

            self.vocab_store.update_status(entry.vocab_id, new_status.value)
            entry.status = new_status
            promoted.append(entry)

        return promoted

    # ------------------------------------------------------------------
    # Full offline analysis flow
    # ------------------------------------------------------------------

    def run(self, session_id: str) -> dict[str, Any]:
        """Full offline analysis flow, returns a summary report.

        Args:
            session_id: Session ID

        Returns:
            Summary report dict:
            - terms_found: list of suspected terms found in this run
            - promoted: list of vocabulary entries promoted in this run
            - pending_review: list of vocabulary entries currently pending human review
        """
        found = self.analyze_session(session_id)
        promoted = self.check_promotion(session_id)
        pending = self.vocab_store.list_candidates(status="draft")

        return {
            "terms_found": found,
            "promoted": promoted,
            "pending_review": pending,
            "summary": {
                "terms_found_count": len(found),
                "promoted_count": len(promoted),
                "pending_review_count": len(pending),
                "session_id": session_id,
            },
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_user_id(session_id: str, turns: list) -> str | None:
        """Infer user_id from dialogue history (simplified implementation).

        In production, should be obtained from session metadata.
        """
        # Simplified: take the content hash of the first user turn as a pseudo user_id
        for turn in turns:
            if turn.role == "user":
                return f"u_{hash(turn.content) % 100000:05d}"
        return None
