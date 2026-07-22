"""Vocabulary service (corresponding to Group 8 / D10 / D11 / D12).

Provides query, add, promote, rollback and other capabilities for a two-level vocabulary (public + personal).
- D10: Two-level vocabulary architecture (public + personal)
- D11: Self-iteration mechanism (candidate -> threshold promotion -> official)
- D12: Industry profiling (same term, different meanings across industries)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import Config, get_config
from ..models import VocabEntry, VocabLevel, VocabStatus
from ..storage.base import VocabStore


class VocabularyTable:
    """Two-level vocabulary service (corresponds to D10/D11/D12).

    Wraps VocabStore to provide business-level API:
    - lookup / lookup_semantic: query public + personal vocabulary
    - add_entry: add a new entry
    - promote_candidate: promote a candidate term
    - determine_level: decide level based on number of discussants
    - inject_context: retrieve and prepare entries for injecting into the context window
    - get_by_industry: filter by industry
    - rollback: rollback a promoted entry
    - review_pending: list entries pending human review
    """

    def __init__(
        self,
        vocab_store: VocabStore,
        config: Config | None = None,
    ) -> None:
        self.store = vocab_store
        self.config = config or get_config()

    # ------------------------------------------------------------------
    # Query (task 8.1, 8.2)
    # ------------------------------------------------------------------

    def lookup(
        self,
        term: str,
        industry: str = "通用",
        user_id: str | None = None,
    ) -> list[VocabEntry]:
        """Query the vocabulary (public + personal).

        - Public vocabulary: shared across users
        - Personal vocabulary: only returns entries belonging to the current user_id
        - Industry first: entries with an exact industry match rank higher

        Args:
            term: Term to query
            industry: Industry of the current user (D12 industry profiling)
            user_id: Current user ID (for querying personal vocabulary)

        Returns:
            List of matched vocabulary entries; exact industry matches rank first
        """
        # Query public vocabulary
        public_results = self.store.search(
            term=term,
            level=VocabLevel.PUBLIC,
            industry=industry,
        )
        # Query personal vocabulary
        personal_results: list[VocabEntry] = []
        if user_id is not None:
            personal_results = self.store.search(
                term=term,
                level=VocabLevel.PERSONAL,
                industry=industry,
                user_id=user_id,
            )

        # Merge: personal vocabulary first (D10 joint query priority)
        combined = personal_results + public_results

        # Industry-priority sort: exact industry match > 通用
        combined.sort(key=lambda e: 0 if e.industry == industry else 1)
        return combined

    def lookup_semantic(
        self,
        query: str,
        industry: str = "通用",
        top_k: int = 5,
    ) -> list[VocabEntry]:
        """Semantic search of the vocabulary (vector database).

        Even if the user does not use exact terms, semantic matching can still hit.

        Args:
            query: Query text entered by the user
            industry: Industry of the current user
            top_k: Return the top K entries

        Returns:
            List of matched vocabulary entries
        """
        return self.store.search_semantic(
            query=query,
            top_k=top_k,
            industry=industry,
        )

    # ------------------------------------------------------------------
    # Add (task 8.1)
    # ------------------------------------------------------------------

    def add_entry(self, entry: VocabEntry) -> None:
        """Add a new vocabulary entry (public or personal).

        Args:
            entry: VocabEntry instance
        """
        self.store.save(entry)

    # ------------------------------------------------------------------
    # Threshold promotion (task 8.4)
    # ------------------------------------------------------------------

    def promote_candidate(
        self, vocab_id: str, force: bool = False
    ) -> bool:
        """Promote a candidate term to an official entry.

        Promotion conditions (any one):
        - Total occurrence count > min_total_count (100)
        - Number of discussants > min_discussant_count (3) -> promote to public vocabulary
        - Consecutive discussion count > min_consecutive_count (10) -> promote to personal vocabulary

        If require_human_review=True, promote to DRAFT status pending human review;
        otherwise promote directly to PROMOTED.

        Args:
            vocab_id: Vocabulary entry ID
            force: Whether to force promotion (skip threshold check)

        Returns:
            True means promotion was triggered, False means threshold not reached
        """
        entry = self.store.get(vocab_id)
        if entry is None:
            return False

        if entry.status == VocabStatus.PROMOTED:
            # Already promoted, idempotent return
            return True

        # Threshold check
        if not force:
            promo_cfg = self.config.vocab_promotion
            met = (
                entry.occurrence_count > promo_cfg.min_total_count
                or entry.discussant_count > promo_cfg.min_discussant_count
                or entry.consecutive_count > promo_cfg.min_consecutive_count
            )
            if not met:
                return False

        # Determine level based on number of discussants (task 8.5)
        new_level = self.determine_level(entry.discussant_count)
        if entry.level != new_level:
            # Note: VocabEntry is a pydantic model, direct assignment works
            entry.level = new_level

        # Determine post-promotion status
        if self.config.vocab_promotion.require_human_review and not force:
            new_status = VocabStatus.DRAFT
        else:
            new_status = VocabStatus.PROMOTED

        self.store.update_status(vocab_id, new_status.value)
        # Sync the entry object itself (avoid inconsistency with storage)
        entry.status = new_status
        entry.updated_at = datetime.now()
        return True

    # ------------------------------------------------------------------
    # Level determination (task 8.5)
    # ------------------------------------------------------------------

    def determine_level(self, discussant_count: int) -> VocabLevel:
        """Determine the vocabulary level based on the number of discussants.

        - Discussants > 3: public vocabulary (multi-user discussion, cross-user universality)
        - Otherwise: personal vocabulary

        Args:
            discussant_count: Number of distinct users who discussed the term

        Returns:
            VocabLevel.PUBLIC or VocabLevel.PERSONAL
        """
        if discussant_count > self.config.vocab_promotion.min_discussant_count:
            return VocabLevel.PUBLIC
        return VocabLevel.PERSONAL

    # ------------------------------------------------------------------
    # Retrieval injection (task 8.2)
    # ------------------------------------------------------------------

    def inject_context(
        self,
        query: str,
        industry: str,
        user_id: str | None,
    ) -> list[VocabEntry]:
        """Retrieve and prepare vocabulary entries for injection into the context window.

        pre-normalization / deep-normalization queries the vocabulary before calling the LLM,
        and injects matched vocabulary entries into the context window (D10 retrieval injection mechanism).

        Args:
            query: User input text
            industry: Industry of the current user
            user_id: Current user ID

        Returns:
            List of vocabulary entries to inject, including term, standard meaning, industry context, source
        """
        # 1. Exact match
        # Extract candidate terms from query (simple implementation: try whole-match + known vocabulary scan)
        entries: list[VocabEntry] = []
        seen_ids: set[str] = set()

        # Query with the whole string as a term
        for entry in self.lookup(query, industry=industry, user_id=user_id):
            if entry.vocab_id not in seen_ids:
                entries.append(entry)
                seen_ids.add(entry.vocab_id)

        # 2. Semantic search supplement (more reliant on semantics when no exact match)
        semantic_results = self.lookup_semantic(
            query=query, industry=industry, top_k=5
        )
        for entry in semantic_results:
            if entry.vocab_id not in seen_ids:
                entries.append(entry)
                seen_ids.add(entry.vocab_id)

        return entries

    # ------------------------------------------------------------------
    # Industry filter (task 8.6)
    # ------------------------------------------------------------------

    def get_by_industry(self, industry: str) -> list[VocabEntry]:
        """Filter vocabulary entries by industry.

        Args:
            industry: Industry name, e.g. "IT", "医疗"

        Returns:
            All promoted (PROMOTED) vocabulary entries belonging to that industry
        """
        # VocabStore.search only returns PROMOTED entries
        # Here we iterate all candidates and filter (simplified implementation; the storage layer should support this)
        all_candidates: list[VocabEntry] = []
        # List all promoted entries: list_candidates is not suitable (it only lists candidates)
        # Use search_semantic to scan instead (only promoted entries are retrieved)
        # Here we iterate all PROMOTED status entries
        for status in ("promoted",):
            try:
                # MemoryVocabStore.list_candidates supports passing in status
                entries = self.store.list_candidates(status=status)
                all_candidates.extend(entries)
            except Exception:
                pass

        return [e for e in all_candidates if e.industry == industry]

    # ------------------------------------------------------------------
    # Rollback (task 8.8)
    # ------------------------------------------------------------------

    def rollback(self, vocab_id: str) -> None:
        """Rollback a promoted vocabulary entry to candidate status.

        Used for version rollback when an erroneous entry is discovered (D versioning and audit).

        Args:
            vocab_id: Vocabulary entry ID
        """
        self.store.update_status(vocab_id, VocabStatus.CANDIDATE.value)
        entry = self.store.get(vocab_id)
        if entry is not None:
            entry.updated_at = datetime.now()

    # ------------------------------------------------------------------
    # Pending human review (task 8.4 supporting)
    # ------------------------------------------------------------------

    def review_pending(self) -> list[VocabEntry]:
        """List vocabulary entries pending human review (status=draft).

        Returns:
            List of vocabulary entries in DRAFT status
        """
        return self.store.list_candidates(status="draft")
