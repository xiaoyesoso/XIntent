"""Context integration regression tests (corresponding to task 7.7).

Coverage:
- Three-layer context joint assembly (task 7.1-7.3)
- Priority conflict resolution (task 7.4)
- Timeliness management (task 7.5)
- Observability (task 7.6)
- Context-missing degradation
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from user_input_normalization.context import ContextBundle, ContextIntegrator
from user_input_normalization.models import (
    DialogueTurn,
    FactType,
    KeyFact,
    UserProfile,
)
from user_input_normalization.storage.memory import (
    MemoryDialogueHistoryStore,
    MemoryKeyFactStore,
    MemoryUserProfileStore,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def integrator() -> ContextIntegrator:
    """Construct an integrator with empty storage."""
    return ContextIntegrator(
        profile_store=MemoryUserProfileStore(),
        key_fact_store=MemoryKeyFactStore(),
        dialogue_store=MemoryDialogueHistoryStore(),
    )


@pytest.fixture
def integrator_with_data() -> tuple[ContextIntegrator, dict]:
    """Construct an integrator prefilled with three-layer context data."""
    profile_store = MemoryUserProfileStore()
    key_fact_store = MemoryKeyFactStore()
    dialogue_store = MemoryDialogueHistoryStore()

    # User profile: IT industry, prefers "高端品牌", topic tendency 三国演义->电视剧
    profile = UserProfile(
        user_id="u1",
        industry="IT",
        preferences={"brand": "高端品牌", "price_sensitivity": "low"},
        topic_tendencies={"三国演义": "电视剧"},
    )
    profile_store.save(profile)

    # Key fact: user has acknowledged the TCC plan (conflicts with profile: profile prefers high-end, fact acknowledges mid-tier)
    ack_fact = KeyFact(
        fact_id="f1",
        session_id="s1",
        turn=2,
        fact_type=FactType.KEY_FACT,
        content={
            "topic": "brand",
            "value": "中端品牌方案",
            "acknowledged": True,
            "resolution": "TCC方案",
        },
        confidence=0.95,
    )
    # Key fact: pronoun resolution table (第一个方案 -> TCC方案)
    pron_fact = KeyFact(
        fact_id="f2",
        session_id="s1",
        turn=1,
        fact_type=FactType.PRONOUN_RESOLUTION,
        content={"pronoun": "第一个方案", "resolved_to": "TCC方案"},
        confidence=0.9,
    )
    key_fact_store.save(ack_fact)
    key_fact_store.save(pron_fact)

    # Dialogue history
    dialogue_store.append(
        "s1",
        DialogueTurn(
            turn=1,
            role="assistant",
            content="我推荐两个方案：TCC方案 和 轻量方案",
            summary="介绍了 TCC 与轻量两个方案",
        ),
    )
    dialogue_store.append(
        "s1",
        DialogueTurn(
            turn=2,
            role="user",
            content="那就用 TCC方案 吧",
            summary="用户认可 TCC方案",
        ),
    )
    dialogue_store.set_summary("s1", "用户已认可 TCC方案，需展开实施细节")

    integrator = ContextIntegrator(
        profile_store=profile_store,
        key_fact_store=key_fact_store,
        dialogue_store=dialogue_store,
    )
    return integrator, {
        "session_id": "s1",
        "user_id": "u1",
        "ack_fact": ack_fact,
        "pron_fact": pron_fact,
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# 7.1-7.3 Three-layer context joint assembly
# ---------------------------------------------------------------------------


class TestAssembleThreeLayers:
    """Three-layer context joint assembly."""

    def test_assemble_returns_bundle(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        integrator, data = integrator_with_data
        bundle = integrator.assemble(
            session_id=data["session_id"],
            user_id=data["user_id"],
            current_input="那个怎么实施？",
        )
        assert isinstance(bundle, ContextBundle)
        assert bundle.session_id == data["session_id"]
        assert bundle.user_id == data["user_id"]
        assert bundle.current_input == "那个怎么实施？"

    def test_user_profile_injected(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """User profile layer is injected."""
        integrator, data = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        assert bundle.user_profile is not None
        assert bundle.user_profile.industry == "IT"

    def test_key_facts_injected(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Key fact layer is injected."""
        integrator, data = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        assert len(bundle.key_facts) >= 2
        # Should include pronoun resolution table and acknowledged point
        fact_types = {f.fact_type for f in bundle.key_facts}
        assert FactType.PRONOUN_RESOLUTION in fact_types
        assert FactType.KEY_FACT in fact_types

    def test_dialogue_summary_injected(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Dialogue history summary is injected."""
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        assert bundle.dialogue_summary is not None
        assert "TCC" in bundle.dialogue_summary

    def test_recalled_details_injected(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Dialogue history recalled details are injected."""
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "TCC方案")
        assert len(bundle.recalled_details) >= 1

    def test_priority_order(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Priority order should be key_facts > user_profile > dialogue_history."""
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        assert bundle.priority_order == [
            "key_facts",
            "user_profile",
            "dialogue_history",
        ]


# ---------------------------------------------------------------------------
# Context-missing degradation
# ---------------------------------------------------------------------------


class TestDegradeOnMissing:
    """Context-missing degradation (spec: "context-missing degradation")."""

    def test_no_profile_degrades(
        self, integrator: ContextIntegrator
    ) -> None:
        """Degrade when user profile is empty."""
        bundle = integrator.assemble("s_new", "u_new", "test")
        assert bundle.user_profile is None
        assert "user_profile" in bundle.degraded_layers

    def test_no_facts_degrades(self, integrator: ContextIntegrator) -> None:
        """Degrade when key facts are empty."""
        bundle = integrator.assemble("s_new", "u_new", "test")
        assert len(bundle.key_facts) == 0
        assert "key_facts" in bundle.degraded_layers

    def test_no_dialogue_degrades(
        self, integrator: ContextIntegrator
    ) -> None:
        """Degrade when dialogue history is empty."""
        bundle = integrator.assemble("s_new", "u_new", "test")
        assert bundle.dialogue_summary is None
        assert "dialogue_history" in bundle.degraded_layers

    def test_degrade_does_not_break(
        self, integrator: ContextIntegrator
    ) -> None:
        """Missing a layer should not break processing; still returns a bundle."""
        bundle = integrator.assemble("s_new", "u_new", "test")
        assert bundle is not None
        assert isinstance(bundle, ContextBundle)


# ---------------------------------------------------------------------------
# 7.4 Priority conflict resolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    """Three-layer context conflict resolution (key facts > user profile > dialogue history)."""

    def test_key_facts_win_over_profile(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Key facts take priority over user profile."""
        integrator, data = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        result = integrator.resolve_conflict(
            bundle.key_facts, bundle.user_profile
        )
        # Should have a conflict: profile prefers "高端品牌", fact acknowledges "中端品牌方案"
        assert len(result["conflicts"]) >= 1
        conflict = result["conflicts"][0]
        assert conflict["winner"] == "key_facts"
        assert conflict["key_fact_value"] != conflict["profile_value"]

    def test_acknowledged_points_highest_priority(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Points acknowledged by the user have the highest priority."""
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        result = integrator.resolve_conflict(
            bundle.key_facts, bundle.user_profile
        )
        # Should recognize acknowledged points
        assert len(result["acknowledged_points"]) >= 1
        ack = result["acknowledged_points"][0]
        assert ack["value"] in ("中端品牌方案", "TCC方案")

    def test_no_conflict_when_values_agree(self) -> None:
        """No conflict when key facts agree with profile."""
        profile_store = MemoryUserProfileStore()
        key_fact_store = MemoryKeyFactStore()
        profile_store.save(
            UserProfile(
                user_id="u1",
                preferences={"brand": "高端品牌"},
            )
        )
        key_fact_store.save(
            KeyFact(
                fact_id="f1",
                session_id="s1",
                turn=1,
                fact_type=FactType.KEY_FACT,
                content={
                    "topic": "brand",
                    "preference": "高端品牌",
                    "acknowledged": True,
                },
            )
        )
        integrator = ContextIntegrator(
            profile_store=profile_store,
            key_fact_store=key_fact_store,
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        result = integrator.resolve_conflict(
            key_fact_store.get_by_session("s1"),
            profile_store.get("u1"),
        )
        assert len(result["conflicts"]) == 0

    def test_winner_when_only_profile_present(self) -> None:
        """When only profile is present, winner is user_profile."""
        profile_store = MemoryUserProfileStore()
        profile_store.save(
            UserProfile(user_id="u1", preferences={"k": "v"})
        )
        integrator = ContextIntegrator(
            profile_store=profile_store,
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        result = integrator.resolve_conflict([], profile_store.get("u1"))
        assert result["winner"] == "user_profile"


# ---------------------------------------------------------------------------
# 7.5 Timeliness management
# ---------------------------------------------------------------------------


class TestExpiryManagement:
    """Context timeliness management."""

    def test_no_expiry_when_expires_at_none(self) -> None:
        """Never expires when expires_at is None."""
        fact = KeyFact(
            fact_id="f1",
            session_id="s1",
            turn=1,
            fact_type=FactType.KEY_FACT,
            content={"k": "v"},
            expires_at=None,
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        assert integrator.is_expired(fact) is False

    def test_expired_when_expires_at_past(self) -> None:
        """Expired when expires_at is in the past."""
        fact = KeyFact(
            fact_id="f1",
            session_id="s1",
            turn=1,
            fact_type=FactType.KEY_FACT,
            content={"k": "v"},
            expires_at=datetime.now() - timedelta(days=1),
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        assert integrator.is_expired(fact) is True

    def test_not_expired_when_expires_at_future(self) -> None:
        """Not expired when expires_at is in the future."""
        fact = KeyFact(
            fact_id="f1",
            session_id="s1",
            turn=1,
            fact_type=FactType.KEY_FACT,
            content={"k": "v"},
            expires_at=datetime.now() + timedelta(days=30),
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        assert integrator.is_expired(fact) is False

    def test_expired_when_status_expired(self) -> None:
        """Treated as expired when status='expired'."""
        fact = KeyFact(
            fact_id="f1",
            session_id="s1",
            turn=1,
            fact_type=FactType.KEY_FACT,
            content={"k": "v"},
            status="expired",
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        assert integrator.is_expired(fact) is True

    def test_revoked_is_expired(self) -> None:
        """Treated as expired when status='revoked'."""
        fact = KeyFact(
            fact_id="f1",
            session_id="s1",
            turn=1,
            fact_type=FactType.KEY_FACT,
            content={"k": "v"},
            status="revoked",
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        assert integrator.is_expired(fact) is True

    def test_assemble_filters_expired_facts(self) -> None:
        """Assembly should filter out expired facts."""
        profile_store = MemoryUserProfileStore()
        key_fact_store = MemoryKeyFactStore()
        dialogue_store = MemoryDialogueHistoryStore()

        # One expired fact + one active fact
        key_fact_store.save(
            KeyFact(
                fact_id="expired",
                session_id="s1",
                turn=1,
                fact_type=FactType.KEY_FACT,
                content={"k": "expired"},
                expires_at=datetime.now() - timedelta(days=1),
            )
        )
        key_fact_store.save(
            KeyFact(
                fact_id="active",
                session_id="s1",
                turn=2,
                fact_type=FactType.KEY_FACT,
                content={"k": "active"},
                expires_at=datetime.now() + timedelta(days=30),
            )
        )
        integrator = ContextIntegrator(
            profile_store=profile_store,
            key_fact_store=key_fact_store,
            dialogue_store=dialogue_store,
        )
        bundle = integrator.assemble("s1", "u1", "test")
        fact_ids = [f.fact_id for f in bundle.key_facts]
        assert "active" in fact_ids
        assert "expired" not in fact_ids

    def test_mark_expired_facts_updates_status(self) -> None:
        """mark_expired_facts should update expired facts' status to expired."""
        profile_store = MemoryUserProfileStore()
        key_fact_store = MemoryKeyFactStore()
        dialogue_store = MemoryDialogueHistoryStore()
        key_fact_store.save(
            KeyFact(
                fact_id="f_expired",
                session_id="s1",
                turn=1,
                fact_type=FactType.KEY_FACT,
                content={"k": "v"},
                expires_at=datetime.now() - timedelta(days=1),
            )
        )
        integrator = ContextIntegrator(
            profile_store=profile_store,
            key_fact_store=key_fact_store,
            dialogue_store=dialogue_store,
        )
        marked = integrator.mark_expired_facts("s1")
        assert "f_expired" in marked
        fact = key_fact_store.get("f_expired")
        assert fact is not None
        assert fact.status == "expired"


# ---------------------------------------------------------------------------
# 7.6 Observability
# ---------------------------------------------------------------------------


class TestObservability:
    """Context injection observability."""

    def test_explain_influence_returns_dict(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        report = integrator.explain_influence(bundle)
        assert isinstance(report, dict)
        assert "layers" in report
        assert "degraded_layers" in report
        assert "conflict_resolution" in report
        assert "priority_order" in report

    def test_explain_influence_includes_all_layers(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Report should include three-layer information."""
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        report = integrator.explain_influence(bundle)
        assert "user_profile" in report["layers"]
        assert "key_facts" in report["layers"]
        assert "dialogue_history" in report["layers"]

    def test_explain_influence_records_injection(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Report should record whether each layer is injected."""
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        report = integrator.explain_influence(bundle)
        assert report["layers"]["user_profile"]["injected"] is True
        assert report["layers"]["key_facts"]["injected"] is True
        assert report["layers"]["dialogue_history"]["injected"] is True

    def test_explain_includes_acknowledged_points(
        self, integrator_with_data: tuple[ContextIntegrator, dict]
    ) -> None:
        """Report should include acknowledged_points."""
        integrator, _ = integrator_with_data
        bundle = integrator.assemble("s1", "u1", "那个怎么样")
        report = integrator.explain_influence(bundle)
        conflict = report["conflict_resolution"]
        assert conflict["acknowledged_points"]
        assert conflict["acknowledged_count"] >= 0 or len(
            conflict["acknowledged_points"]
        ) >= 1

    def test_explain_degraded_layers_when_empty(self) -> None:
        """Degraded layers should be recorded when context is empty."""
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        bundle = integrator.assemble("s_new", "u_new", "test")
        report = integrator.explain_influence(bundle)
        assert "user_profile" in report["degraded_layers"]
        assert "key_facts" in report["degraded_layers"]
        assert "dialogue_history" in report["degraded_layers"]


# ---------------------------------------------------------------------------
# Cross-session recall
# ---------------------------------------------------------------------------


class TestCrossSessionRecall:
    """Cross-session recall (spec: "cross-session recall")."""

    def test_recall_from_long_term_history(self) -> None:
        """Dialogue history recall should be able to hit historical details."""
        dialogue_store = MemoryDialogueHistoryStore()
        # Simulate historical dialogue: contains a number
        dialogue_store.append(
            "s1",
            DialogueTurn(
                turn=5,
                role="assistant",
                content="上次提到的数字是 42",
            ),
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=MemoryKeyFactStore(),
            dialogue_store=dialogue_store,
        )
        bundle = integrator.assemble("s1", "u1", "上次提到的数字是多少")
        # Should recall the dialogue containing "数字"
        assert any(
            "42" in t.content for t in bundle.recalled_details
        )


# ---------------------------------------------------------------------------
# Key facts influence anaphora resolution (spec scenario)
# ---------------------------------------------------------------------------


class TestKeyFactInfluencesResolution:
    """Key facts influence anaphora resolution (spec scenario: user acknowledged TCC plan -> '它' resolved as TCC)."""

    def test_acknowledged_fact_provides_resolution(self) -> None:
        """Acknowledged points in key facts should serve as anaphora resolution basis."""
        key_fact_store = MemoryKeyFactStore()
        key_fact_store.save(
            KeyFact(
                fact_id="f1",
                session_id="s1",
                turn=2,
                fact_type=FactType.KEY_FACT,
                content={
                    "topic": "selected_plan",
                    "value": "TCC方案",
                    "acknowledged": True,
                },
            )
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=key_fact_store,
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        bundle = integrator.assemble("s1", "u1", "那就用它吧")
        # Should be able to find acknowledged points from key facts
        ack_facts = [
            f for f in bundle.key_facts if f.content.get("acknowledged") is True
        ]
        assert len(ack_facts) >= 1
        assert ack_facts[0].content.get("value") == "TCC方案"

    def test_pronoun_resolution_table_reused(self) -> None:
        """Pronoun resolution table reused across turns (D6)."""
        key_fact_store = MemoryKeyFactStore()
        key_fact_store.save(
            KeyFact(
                fact_id="f_pron",
                session_id="s1",
                turn=1,
                fact_type=FactType.PRONOUN_RESOLUTION,
                content={
                    "pronoun": "第一个方案",
                    "resolved_to": "TCC方案",
                },
                confidence=0.95,
            )
        )
        integrator = ContextIntegrator(
            profile_store=MemoryUserProfileStore(),
            key_fact_store=key_fact_store,
            dialogue_store=MemoryDialogueHistoryStore(),
        )
        # Second turn: user mentions "第一个方案" again, should be able to reuse key facts
        bundle = integrator.assemble("s1", "u1", "第一个方案的细节")
        pron_facts = [
            f for f in bundle.key_facts
            if f.fact_type == FactType.PRONOUN_RESOLUTION
        ]
        assert len(pron_facts) >= 1
        assert pron_facts[0].content.get("resolved_to") == "TCC方案"
