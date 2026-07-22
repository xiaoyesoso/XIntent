"""Vocabulary tests (corresponding to task 8.9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from user_input_normalization.config import Config, VocabPromotionConfig
from user_input_normalization.models import (
    AlternativeMeaning,
    DialogueTurn,
    VocabEntry,
    VocabLevel,
    VocabStatus,
)
from user_input_normalization.storage.memory import (
    MemoryDialogueHistoryStore,
    MemoryVocabStore,
)
from user_input_normalization.vocabulary import (
    OfflineAnalyzer,
    VocabularyTable,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vocab_store() -> MemoryVocabStore:
    return MemoryVocabStore()


@pytest.fixture
def dialogue_store() -> MemoryDialogueHistoryStore:
    return MemoryDialogueHistoryStore()


@pytest.fixture
def vocab_table(vocab_store: MemoryVocabStore) -> VocabularyTable:
    return VocabularyTable(vocab_store=vocab_store)


@pytest.fixture
def no_review_config() -> Config:
    """Configuration with human review disabled, convenient for directly testing promotion to PROMOTED."""
    cfg = Config()
    cfg.vocab_promotion = VocabPromotionConfig(require_human_review=False)
    return cfg


@pytest.fixture
def vocab_table_no_review(
    vocab_store: MemoryVocabStore, no_review_config: Config
) -> VocabularyTable:
    return VocabularyTable(vocab_store=vocab_store, config=no_review_config)


@pytest.fixture
def analyzer(
    vocab_store: MemoryVocabStore,
    dialogue_store: MemoryDialogueHistoryStore,
    no_review_config: Config,
) -> OfflineAnalyzer:
    return OfflineAnalyzer(
        vocab_store=vocab_store,
        dialogue_store=dialogue_store,
        config=no_review_config,
    )


def _make_entry(
    term: str = "RAG",
    level: VocabLevel = VocabLevel.PUBLIC,
    industry: str = "通用",
    meaning: str = "检索增强生成",
    status: VocabStatus = VocabStatus.PROMOTED,
    user_id: str | None = None,
    occurrence_count: int = 0,
    discussant_count: int = 0,
    consecutive_count: int = 0,
    alternatives: list[AlternativeMeaning] | None = None,
) -> VocabEntry:
    """Vocabulary entry helper function."""
    return VocabEntry(
        vocab_id=f"vocab_{term}_{level.value}_{industry}",
        term=term,
        level=level,
        industry=industry,
        standard_meaning=meaning,
        status=status,
        user_id=user_id,
        occurrence_count=occurrence_count,
        discussant_count=discussant_count,
        consecutive_count=consecutive_count,
        alternative_meanings=alternatives or [],
    )


# ---------------------------------------------------------------------------
# Two-level vocabulary CRUD (task 8.1)
# ---------------------------------------------------------------------------


class TestTwoLevelCRUD:
    def test_add_and_lookup_public_entry(self, vocab_table: VocabularyTable) -> None:
        """Add and query public vocabulary."""
        entry = _make_entry(term="RAG", meaning="检索增强生成")
        vocab_table.add_entry(entry)

        results = vocab_table.lookup("RAG")
        assert len(results) == 1
        assert results[0].term == "RAG"
        assert results[0].standard_meaning == "检索增强生成"
        assert results[0].level == VocabLevel.PUBLIC

    def test_add_and_lookup_personal_entry(self, vocab_table: VocabularyTable) -> None:
        """Add and query personal vocabulary (only returns entries belonging to the current user)."""
        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目代号",
            user_id="u1",
        )
        vocab_table.add_entry(entry)

        # Belonging to u1 should be found
        results = vocab_table.lookup("XQR", user_id="u1")
        assert len(results) == 1
        assert results[0].standard_meaning == "某内部项目代号"

        # Belonging to other users should not be found (personal vocabulary isolation)
        results_other = vocab_table.lookup("XQR", user_id="u2")
        assert len(results_other) == 0

    def test_personal_takes_priority_over_public(
        self, vocab_table: VocabularyTable
    ) -> None:
        """In joint queries, personal vocabulary takes priority."""
        public_entry = _make_entry(
            term="CRM",
            level=VocabLevel.PUBLIC,
            meaning="Customer Relationship Management",
        )
        personal_entry = _make_entry(
            term="CRM",
            level=VocabLevel.PERSONAL,
            meaning="我的客户关系管理笔记",
            user_id="u1",
            # Use a different ID to avoid overwriting
        )
        personal_entry.vocab_id = "vocab_crm_personal_u1"
        vocab_table.add_entry(public_entry)
        vocab_table.add_entry(personal_entry)

        results = vocab_table.lookup("CRM", user_id="u1")
        assert len(results) == 2
        # Personal vocabulary should rank first
        assert results[0].level == VocabLevel.PERSONAL
        assert results[1].level == VocabLevel.PUBLIC

    def test_lookup_nonexistent_returns_empty(
        self, vocab_table: VocabularyTable
    ) -> None:
        """Querying a non-existent term should return an empty list."""
        assert vocab_table.lookup("不存在的词") == []


# ---------------------------------------------------------------------------
# Industry profiling (task 8.6 / D12)
# ---------------------------------------------------------------------------


class TestIndustryProfile:
    def test_crm_different_meaning_in_it_vs_medical(
        self, vocab_table: VocabularyTable
    ) -> None:
        """CRM has different meanings in IT and medical industries."""
        it_entry = _make_entry(
            term="CRM",
            level=VocabLevel.PUBLIC,
            industry="IT",
            meaning="Customer Relationship Management",
        )
        medical_entry = _make_entry(
            term="CRM",
            level=VocabLevel.PUBLIC,
            industry="医疗",
            meaning="Clinical Research Management",
        )
        # Use a different ID
        medical_entry.vocab_id = "vocab_crm_medical"
        vocab_table.add_entry(it_entry)
        vocab_table.add_entry(medical_entry)

        # IT industry query
        it_results = vocab_table.lookup("CRM", industry="IT")
        assert len(it_results) >= 1
        assert any(r.standard_meaning == "Customer Relationship Management" for r in it_results)

        # Medical industry query
        med_results = vocab_table.lookup("CRM", industry="医疗")
        assert len(med_results) >= 1
        assert any(r.standard_meaning == "Clinical Research Management" for r in med_results)

    def test_industry_priority_ordering(self, vocab_table: VocabularyTable) -> None:
        """Entries with exact industry matches should rank before 通用 entries."""
        general_entry = _make_entry(
            term="抓手",
            level=VocabLevel.PUBLIC,
            industry="通用",
            meaning="通用释义：切入点",
        )
        it_entry = _make_entry(
            term="抓手",
            level=VocabLevel.PUBLIC,
            industry="IT",
            meaning="IT 行业：核心着力点",
        )
        it_entry.vocab_id = "vocab_zhuashou_it"
        vocab_table.add_entry(general_entry)
        vocab_table.add_entry(it_entry)

        results = vocab_table.lookup("抓手", industry="IT")
        assert len(results) == 2
        # IT industry exact match should rank first
        assert results[0].industry == "IT"
        assert results[1].industry == "通用"

    def test_get_by_industry(self, vocab_table: VocabularyTable) -> None:
        """Filter vocabulary by industry."""
        it_entry = _make_entry(
            term="RAG",
            industry="IT",
            meaning="检索增强生成",
        )
        medical_entry = _make_entry(
            term="EMR",
            industry="医疗",
            meaning="电子病历",
        )
        medical_entry.vocab_id = "vocab_emr_medical"
        vocab_table.add_entry(it_entry)
        vocab_table.add_entry(medical_entry)

        it_entries = vocab_table.get_by_industry("IT")
        assert any(e.term == "RAG" for e in it_entries)
        assert not any(e.term == "EMR" for e in it_entries)


# ---------------------------------------------------------------------------
# Semantic search (task 8.2)
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    def test_lookup_semantic_finds_related(self, vocab_table: VocabularyTable) -> None:
        """Semantic search should be able to hit related vocabulary."""
        entry = _make_entry(
            term="RAG",
            meaning="检索增强生成",
        )
        vocab_table.add_entry(entry)

        # Semantic search using "那个检索增强的东西"
        results = vocab_table.lookup_semantic("那个检索增强的东西", top_k=5)
        # MemoryVocabStore uses Jaccard similarity; should at least hit entries containing "检索增强"
        assert any(r.term == "RAG" for r in results)

    def test_lookup_semantic_respects_top_k(self, vocab_table: VocabularyTable) -> None:
        """Semantic search should respect the top_k limit."""
        for i, term in enumerate(["RAG", "RAG2", "RAG3"]):
            entry = _make_entry(term=term, meaning="检索增强生成")
            entry.vocab_id = f"vocab_{term}_{i}"
            vocab_table.add_entry(entry)

        results = vocab_table.lookup_semantic("检索增强", top_k=2)
        assert len(results) <= 2

    def test_inject_context_combines_exact_and_semantic(
        self, vocab_table: VocabularyTable
    ) -> None:
        """inject_context should combine exact and semantic search."""
        entry = _make_entry(term="RAG", meaning="检索增强生成")
        vocab_table.add_entry(entry)

        # Exact query
        results = vocab_table.inject_context("RAG", industry="通用", user_id=None)
        assert any(r.term == "RAG" for r in results)

        # Semantic query (non-exact term)
        results_semantic = vocab_table.inject_context(
            "那个检索增强的东西", industry="通用", user_id=None
        )
        assert any(r.term == "RAG" for r in results_semantic)


# ---------------------------------------------------------------------------
# Threshold promotion (task 8.4, 8.5)
# ---------------------------------------------------------------------------


class TestPromotion:
    def test_determine_level_public_when_many_discussants(
        self, vocab_table: VocabularyTable
    ) -> None:
        """Discussant count > 3 should be classified as public vocabulary."""
        assert vocab_table.determine_level(4) == VocabLevel.PUBLIC
        assert vocab_table.determine_level(10) == VocabLevel.PUBLIC

    def test_determine_level_personal_when_few_discussants(
        self, vocab_table: VocabularyTable
    ) -> None:
        """Discussant count <= 3 should be classified as personal vocabulary."""
        assert vocab_table.determine_level(1) == VocabLevel.PERSONAL
        assert vocab_table.determine_level(3) == VocabLevel.PERSONAL

    def test_promote_by_total_count(
        self, vocab_table_no_review: VocabularyTable
    ) -> None:
        """Total occurrence count > 100 triggers promotion."""
        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目",
            status=VocabStatus.CANDIDATE,
            occurrence_count=101,
        )
        vocab_table_no_review.add_entry(entry)

        ok = vocab_table_no_review.promote_candidate(entry.vocab_id)
        assert ok is True
        updated = vocab_table_no_review.store.get(entry.vocab_id)
        assert updated is not None
        assert updated.status == VocabStatus.PROMOTED

    def test_promote_by_discussant_count_becomes_public(
        self, vocab_table_no_review: VocabularyTable
    ) -> None:
        """Discussant count > 3 should be promoted to public vocabulary."""
        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目",
            status=VocabStatus.CANDIDATE,
            occurrence_count=10,
            discussant_count=4,
        )
        vocab_table_no_review.add_entry(entry)

        ok = vocab_table_no_review.promote_candidate(entry.vocab_id)
        assert ok is True
        updated = vocab_table_no_review.store.get(entry.vocab_id)
        assert updated is not None
        assert updated.status == VocabStatus.PROMOTED
        assert updated.level == VocabLevel.PUBLIC

    def test_promote_by_consecutive_count_stays_personal(
        self, vocab_table_no_review: VocabularyTable
    ) -> None:
        """Single-user consecutive discussion > 10 times promotes to personal vocabulary."""
        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目",
            status=VocabStatus.CANDIDATE,
            occurrence_count=15,
            consecutive_count=11,
            discussant_count=1,
            user_id="u1",
        )
        vocab_table_no_review.add_entry(entry)

        ok = vocab_table_no_review.promote_candidate(entry.vocab_id)
        assert ok is True
        updated = vocab_table_no_review.store.get(entry.vocab_id)
        assert updated is not None
        assert updated.status == VocabStatus.PROMOTED
        assert updated.level == VocabLevel.PERSONAL

    def test_promote_below_threshold_returns_false(
        self, vocab_table_no_review: VocabularyTable
    ) -> None:
        """Below threshold should return False."""
        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目",
            status=VocabStatus.CANDIDATE,
            occurrence_count=10,
            discussant_count=1,
            consecutive_count=2,
        )
        vocab_table_no_review.add_entry(entry)

        ok = vocab_table_no_review.promote_candidate(entry.vocab_id)
        assert ok is False

    def test_promote_force_bypasses_threshold(
        self, vocab_table_no_review: VocabularyTable
    ) -> None:
        """force=True should skip the threshold check."""
        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目",
            status=VocabStatus.CANDIDATE,
            occurrence_count=1,
        )
        vocab_table_no_review.add_entry(entry)

        ok = vocab_table_no_review.promote_candidate(entry.vocab_id, force=True)
        assert ok is True

    def test_promote_with_human_review_goes_to_draft(
        self, vocab_store: MemoryVocabStore
    ) -> None:
        """When require_human_review=True, should promote to DRAFT for review."""
        cfg = Config()
        cfg.vocab_promotion = VocabPromotionConfig(require_human_review=True)
        table = VocabularyTable(vocab_store=vocab_store, config=cfg)

        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目",
            status=VocabStatus.CANDIDATE,
            occurrence_count=101,
        )
        table.add_entry(entry)

        ok = table.promote_candidate(entry.vocab_id)
        assert ok is True
        updated = table.store.get(entry.vocab_id)
        assert updated is not None
        assert updated.status == VocabStatus.DRAFT

    def test_promote_nonexistent_returns_false(
        self, vocab_table: VocabularyTable
    ) -> None:
        """Promoting a non-existent vocab_id should return False."""
        assert vocab_table.promote_candidate("nonexistent") is False

    def test_review_pending_returns_draft_entries(
        self, vocab_store: MemoryVocabStore
    ) -> None:
        """review_pending should return DRAFT-status entries."""
        cfg = Config()
        cfg.vocab_promotion = VocabPromotionConfig(require_human_review=True)
        table = VocabularyTable(vocab_store=vocab_store, config=cfg)

        entry = _make_entry(
            term="XQR",
            status=VocabStatus.CANDIDATE,
            occurrence_count=101,
        )
        table.add_entry(entry)
        table.promote_candidate(entry.vocab_id)

        pending = table.review_pending()
        assert len(pending) >= 1
        assert all(e.status == VocabStatus.DRAFT for e in pending)


# ---------------------------------------------------------------------------
# Rollback (task 8.8)
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_promoted_to_candidate(
        self, vocab_table: VocabularyTable
    ) -> None:
        """Rollback a promoted entry to candidate status."""
        entry = _make_entry(
            term="XQR",
            status=VocabStatus.PROMOTED,
            meaning="某内部项目",
        )
        vocab_table.add_entry(entry)

        vocab_table.rollback(entry.vocab_id)
        updated = vocab_table.store.get(entry.vocab_id)
        assert updated is not None
        assert updated.status == VocabStatus.CANDIDATE

    def test_rollback_after_promotion(
        self, vocab_table_no_review: VocabularyTable
    ) -> None:
        """After promotion, rollback should restore to candidate status."""
        entry = _make_entry(
            term="XQR",
            status=VocabStatus.CANDIDATE,
            occurrence_count=101,
        )
        vocab_table_no_review.add_entry(entry)

        # Promote
        assert vocab_table_no_review.promote_candidate(entry.vocab_id) is True
        assert (
            vocab_table_no_review.store.get(entry.vocab_id).status
            == VocabStatus.PROMOTED
        )

        # Rollback
        vocab_table_no_review.rollback(entry.vocab_id)
        assert (
            vocab_table_no_review.store.get(entry.vocab_id).status
            == VocabStatus.CANDIDATE
        )


# ---------------------------------------------------------------------------
# Offline analysis (tasks 8.3, 8.4, 8.5)
# ---------------------------------------------------------------------------


class TestOfflineAnalyzer:
    def test_extract_potential_terms_finds_acronyms(self) -> None:
        """extract_potential_terms should be able to extract English abbreviations."""
        from user_input_normalization.vocabulary.offline_analyzer import (
            extract_potential_terms,
        )

        terms = extract_potential_terms("我们用 RAG 来做检索增强，CRM 系统也要升级")
        assert "RAG" in terms
        assert "CRM" in terms

    def test_extract_potential_terms_filters_stopwords(self) -> None:
        """Should filter stopwords."""
        from user_input_normalization.vocabulary.offline_analyzer import (
            extract_potential_terms,
        )

        terms = extract_potential_terms("这个那个什么")
        # Stopwords should be filtered
        assert "这个" not in terms
        assert "那个" not in terms

    def test_analyze_session_finds_terms(
        self,
        analyzer: OfflineAnalyzer,
        dialogue_store: MemoryDialogueHistoryStore,
    ) -> None:
        """analyze_session should be able to extract suspected terms from dialogue."""
        # Prepare dialogue history
        dialogue_store.append(
            "session_1",
            DialogueTurn(turn=1, role="user", content="帮我查一下 RAG 的资料"),
        )
        dialogue_store.append(
            "session_1",
            DialogueTurn(turn=2, role="assistant", content="好的，RAG 是检索增强生成"),
        )
        dialogue_store.append(
            "session_1",
            DialogueTurn(turn=3, role="user", content="CRM 系统的数据也要看一下"),
        )

        found = analyzer.analyze_session("session_1")
        # Should be able to find RAG and CRM
        terms_found = [e.term for e in found]
        assert "RAG" in terms_found
        assert "CRM" in terms_found

    def test_analyze_session_empty_history(
        self, analyzer: OfflineAnalyzer
    ) -> None:
        """Empty session should return an empty list."""
        found = analyzer.analyze_session("nonexistent_session")
        assert found == []

    def test_check_promotion_promotes_qualified(
        self,
        analyzer: OfflineAnalyzer,
        vocab_store: MemoryVocabStore,
    ) -> None:
        """check_promotion should promote candidate terms that reach the threshold."""
        # Prepare a candidate term that reaches the threshold
        entry = _make_entry(
            term="XQR",
            level=VocabLevel.PERSONAL,
            meaning="某内部项目",
            status=VocabStatus.CANDIDATE,
            occurrence_count=101,
            discussant_count=1,
        )
        vocab_store.save(entry)

        promoted = analyzer.check_promotion("session_1")
        assert len(promoted) == 1
        assert promoted[0].term == "XQR"
        assert promoted[0].status == VocabStatus.PROMOTED

    def test_check_promotion_skips_unqualified(
        self,
        analyzer: OfflineAnalyzer,
        vocab_store: MemoryVocabStore,
    ) -> None:
        """check_promotion should skip candidate terms that do not reach the threshold."""
        entry = _make_entry(
            term="XQR",
            status=VocabStatus.CANDIDATE,
            occurrence_count=10,
            discussant_count=1,
        )
        vocab_store.save(entry)

        promoted = analyzer.check_promotion("session_1")
        assert len(promoted) == 0

    def test_run_returns_summary(
        self,
        analyzer: OfflineAnalyzer,
        dialogue_store: MemoryDialogueHistoryStore,
    ) -> None:
        """run should return a complete summary report."""
        dialogue_store.append(
            "session_run",
            DialogueTurn(turn=1, role="user", content="RAG 是什么"),
        )
        dialogue_store.append(
            "session_run",
            DialogueTurn(turn=2, role="user", content="CRM 怎么用"),
        )

        report = analyzer.run("session_run")
        assert "terms_found" in report
        assert "promoted" in report
        assert "pending_review" in report
        assert "summary" in report
        assert report["summary"]["session_id"] == "session_run"
        assert report["summary"]["terms_found_count"] >= 0

    def test_increment_count_accumulates(
        self,
        analyzer: OfflineAnalyzer,
        vocab_store: MemoryVocabStore,
        dialogue_store: MemoryDialogueHistoryStore,
    ) -> None:
        """Multiple scans should accumulate occurrence counts."""
        dialogue_store.append(
            "session_acc",
            DialogueTurn(turn=1, role="user", content="RAG 资料"),
        )
        dialogue_store.append(
            "session_acc",
            DialogueTurn(turn=2, role="user", content="RAG 应用"),
        )

        analyzer.analyze_session("session_acc")
        # Second scan should accumulate
        analyzer.analyze_session("session_acc")

        # Find the RAG entry, occurrence_count should be 2
        entries = [
            e for e in vocab_store._entries.values() if e.term == "RAG"
        ]
        assert len(entries) >= 1
        assert entries[0].occurrence_count >= 2
