"""Deep context integration module (corresponding to task 7 / D14).

Integrates three layers of context to support normalization inference:
1. User profile layer (UserProfile): industry, preferences, historical behavior, topic tendencies
2. Key fact layer (KeyFact): user preferences, opinions, attitudes, acknowledged points, anaphora resolution table
3. Dialogue history recall layer (DialogueHistoryStore): summary + details (RAG)

Three-layer priority (on conflict):
    Key facts > User profile > Dialogue history recall
Among them, "points acknowledged by the user" have the highest priority.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ..config import Config, get_config
from ..models import (
    DialogueTurn,
    FactType,
    KeyFact,
    UserProfile,
)
from ..storage.base import (
    DialogueHistoryStore,
    KeyFactStore,
    UserProfileStore,
)


# ---------------------------------------------------------------------------
# ContextBundle: context integration result
# ---------------------------------------------------------------------------


class ContextBundle(BaseModel):
    """Three-layer context integration result bundle (corresponds to D14).

    All three layers are exposed through the same structure; downstream
    pre-normalization / deep-normalization consumes this bundle directly,
    without separately accessing each store.
    """

    session_id: str = Field(description="会话 ID")
    user_id: str = Field(description="用户 ID")
    current_input: str = Field(description="当前用户输入")
    user_profile: UserProfile | None = Field(
        default=None, description="用户画像层（可能为空，表示降级）"
    )
    key_facts: list[KeyFact] = Field(
        default_factory=list, description="关键事实层（按 turn 排序）"
    )
    dialogue_summary: str | None = Field(
        default=None, description="对话历史摘要（常驻）"
    )
    recalled_details: list[DialogueTurn] = Field(
        default_factory=list, description="对话历史召回细节（按需 RAG 检索）"
    )
    priority_order: list[str] = Field(
        default_factory=lambda: [
            "key_facts",
            "user_profile",
            "dialogue_history",
        ],
        description="三层优先级（高 -> 低）",
    )
    degraded_layers: list[str] = Field(
        default_factory=list,
        description="降级层列表（如 ['user_profile'] 表示画像缺失）",
    )
    assembled_at: datetime = Field(
        default_factory=datetime.now, description="组装时间"
    )


# ---------------------------------------------------------------------------
# ContextIntegrator: three-layer context integrator
# ---------------------------------------------------------------------------


class ContextIntegrator:
    """Three-layer context integrator (corresponds to D14 / task 7.1-7.6).

    Args:
        profile_store: User profile store
        key_fact_store: Key fact store
        dialogue_store: Dialogue history store
        config: Global configuration (includes ContextConfig.recall_top_k and other parameters)
    """

    # Three-layer priority constants (high -> low)
    PRIORITY_KEY_FACTS = "key_facts"
    PRIORITY_USER_PROFILE = "user_profile"
    PRIORITY_DIALOGUE_HISTORY = "dialogue_history"

    # Identifier field for "points acknowledged by the user" in key facts
    ACKNOWLEDGED_FLAG_KEY = "acknowledged"  # content["acknowledged"] = True
    ACKNOWLEDGED_FACT_TAG = "acknowledged_point"

    def __init__(
        self,
        profile_store: UserProfileStore,
        key_fact_store: KeyFactStore,
        dialogue_store: DialogueHistoryStore,
        config: Config | None = None,
    ) -> None:
        self.profile_store = profile_store
        self.key_fact_store = key_fact_store
        self.dialogue_store = dialogue_store
        self.config = config or get_config()

    # ------------------------------------------------------------------
    # 7.1-7.3 Three-layer context assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        session_id: str,
        user_id: str,
        current_input: str,
    ) -> ContextBundle:
        """Assemble three-layer context (corresponds to task 7.1-7.3).

        Args:
            session_id: Current session ID
            user_id: User ID
            current_input: Current user input (used for RAG recall)

        Returns:
            ContextBundle: three-layer context integration bundle
        """
        degraded: list[str] = []

        # Layer 1: User profile (task 7.1)
        profile = self.profile_store.get(user_id)
        if profile is None:
            degraded.append(self.PRIORITY_USER_PROFILE)

        # Layer 2: Key facts (task 7.2)
        # Get all key facts for this session, filter out expired and revoked ones
        all_facts = self.key_fact_store.get_by_session(session_id)
        active_facts = [
            f
            for f in all_facts
            if f.status == "active" and not self.is_expired(f)
        ]
        if not active_facts:
            degraded.append(self.PRIORITY_KEY_FACTS)

        # Layer 3: Dialogue history recall (task 7.3)
        # Summary is always resident
        summary = self.dialogue_store.get_summary(session_id)
        # Details via on-demand RAG retrieval (using current input as query)
        recall_top_k = self.config.context.recall_top_k
        recalled = self.dialogue_store.search_semantic(
            session_id, current_input, top_k=recall_top_k
        )
        # Supplement the most recent N turns of short-term memory (ensure short-term context is not lost)
        short_window = self.config.context.short_term_window
        recent = self.dialogue_store.get_recent(session_id, n=short_window)
        recalled = self._merge_unique_turns(recalled, recent)
        if summary is None and not recalled:
            degraded.append(self.PRIORITY_DIALOGUE_HISTORY)

        return ContextBundle(
            session_id=session_id,
            user_id=user_id,
            current_input=current_input,
            user_profile=profile,
            key_facts=active_facts,
            dialogue_summary=summary,
            recalled_details=recalled,
            priority_order=[
                self.PRIORITY_KEY_FACTS,
                self.PRIORITY_USER_PROFILE,
                self.PRIORITY_DIALOGUE_HISTORY,
            ],
            degraded_layers=degraded,
        )

    # ------------------------------------------------------------------
    # 7.4 Conflict resolution
    # ------------------------------------------------------------------

    def resolve_conflict(
        self,
        facts: list[KeyFact],
        profile: UserProfile | None,
    ) -> dict[str, Any]:
        """Three-layer context conflict resolution (corresponds to task 7.4).

        Priority: Key facts ("acknowledged points" highest) > User profile > Dialogue history
        This method performs conflict detection on preferences/tendencies within facts and profile,
        and returns the resolution result and conflict list.

        Args:
            facts: List of key facts
            profile: User profile (may be None)

        Returns:
            {
                "winner": "key_facts" | "user_profile" | "none",
                "resolved_values": {key: value},
                "conflicts": [{key, key_fact_value, profile_value, winner, reason}],
                "acknowledged_points": [...],
            }
        """
        # 1. Extract preferences/acknowledged points from key facts
        acknowledged: list[dict[str, Any]] = []
        fact_values: dict[str, Any] = {}
        for f in facts:
            # "Acknowledged point": content has acknowledged=True or fact_type=key_fact with tag
            if self._is_acknowledged(f):
                ack_key = (
                    f.content.get("topic")
                    or f.content.get("key")
                    or f.content.get("subject")
                    or f.fact_id
                )
                acknowledged.append(
                    {
                        "key": ack_key,
                        "value": f.content.get("value")
                        or f.content.get("resolution")
                        or f.content.get("content"),
                        "fact_id": f.fact_id,
                        "turn": f.turn,
                    }
                )
                fact_values[str(ack_key)] = f.content
            # Preference values in general key facts
            if "preference" in f.content:
                key = str(f.content.get("topic") or f.content.get("key") or "pref")
                fact_values[key] = f.content.get("preference")

        # 2. Extract preferences from profile
        profile_values: dict[str, Any] = {}
        if profile is not None:
            for k, v in profile.preferences.items():
                profile_values[str(k)] = v
            for topic, tendency in profile.topic_tendencies.items():
                profile_values[f"tendency:{topic}"] = tendency

        # 3. Conflict detection: same key exists in both facts and profile with different values
        conflicts: list[dict[str, Any]] = []
        resolved_values: dict[str, Any] = {}
        winner = "none"

        all_keys = set(fact_values.keys()) | set(profile_values.keys())
        for key in all_keys:
            in_facts = key in fact_values
            in_profile = key in profile_values
            if in_facts and in_profile:
                fv = fact_values[key]
                pv = profile_values[key]
                if not self._values_equal(fv, pv):
                    # Conflict: key facts take priority (spec D14)
                    conflicts.append(
                        {
                            "key": key,
                            "key_fact_value": fv,
                            "profile_value": pv,
                            "winner": self.PRIORITY_KEY_FACTS,
                            "reason": "关键事实优先于用户画像（D14）",
                        }
                    )
                    resolved_values[key] = fv
                    winner = self.PRIORITY_KEY_FACTS
                else:
                    resolved_values[key] = fv
            elif in_facts:
                resolved_values[key] = fact_values[key]
                if winner == "none":
                    winner = self.PRIORITY_KEY_FACTS
            else:
                resolved_values[key] = profile_values[key]
                if winner == "none":
                    winner = self.PRIORITY_USER_PROFILE

        return {
            "winner": winner,
            "resolved_values": resolved_values,
            "conflicts": conflicts,
            "acknowledged_points": acknowledged,
            "acknowledged_count": len(acknowledged),
        }

    # ------------------------------------------------------------------
    # 7.5 Timeliness management
    # ------------------------------------------------------------------

    def is_expired(self, fact: KeyFact) -> bool:
        """Determine whether a key fact has expired (corresponds to task 7.5).

        Rules:
        - expires_at is None -> never expires
        - expires_at < now -> expired
        - status == "expired" -> expired
        - status == "revoked" -> revoked (treated as expired)
        """
        if fact.status in ("expired", "revoked"):
            return True
        if fact.expires_at is None:
            return False
        return fact.expires_at < datetime.now()

    def filter_expired_facts(self, facts: list[KeyFact]) -> list[KeyFact]:
        """Filter out expired facts and return the list of still-valid facts."""
        return [f for f in facts if not self.is_expired(f)]

    def mark_expired_facts(self, session_id: str) -> list[str]:
        """Scan all key facts in the session and mark expired facts as expired.

        Returns:
            List of fact_ids marked as expired
        """
        marked: list[str] = []
        all_facts = self.key_fact_store.get_by_session(session_id)
        for f in all_facts:
            if f.status == "active" and self.is_expired(f):
                f.status = "expired"
                # Re-save (in-memory storage can modify the object directly; persistent implementations need to re-save)
                self.key_fact_store.save(f)
                marked.append(f.fact_id)
        return marked

    # ------------------------------------------------------------------
    # 7.6 Observability
    # ------------------------------------------------------------------

    def explain_influence(self, bundle: ContextBundle) -> dict[str, Any]:
        """Return a context influence report (corresponds to task 7.6 observability).

        The report includes:
        - Number of injected fragments per layer
        - Source identifiers per layer
        - Matched evidence fragments (key fact content fragments, dialogue history fragments)
        - Degradation reasons
        - Conflict resolution summary

        Returns:
            {
                "layers": {
                    "user_profile": {"injected": bool, "source": "...", "fields": [...]},
                    "key_facts": {"injected": bool, "count": N, "evidence": [...]},
                    "dialogue_history": {"injected": bool, "summary_len": N, "details_count": N, "top_turns": [...]},
                },
                "degraded_layers": [...],
                "conflict_resolution": {...},
                "priority_order": [...],
            }
        """
        # Key fact evidence fragments
        fact_evidence: list[dict[str, Any]] = []
        acknowledged_count = 0
        for f in bundle.key_facts:
            is_ack = self._is_acknowledged(f)
            if is_ack:
                acknowledged_count += 1
            fact_evidence.append(
                {
                    "fact_id": f.fact_id,
                    "fact_type": f.fact_type.value,
                    "turn": f.turn,
                    "content_preview": self._preview_content(f.content),
                    "acknowledged": is_ack,
                    "expired": self.is_expired(f),
                }
            )

        # Dialogue history evidence fragments
        detail_evidence: list[dict[str, Any]] = []
        for t in bundle.recalled_details:
            detail_evidence.append(
                {
                    "turn": t.turn,
                    "role": t.role,
                    "content_preview": t.content[:100],
                    "has_summary": t.summary is not None,
                }
            )

        # Conflict resolution summary
        conflict = self.resolve_conflict(
            bundle.key_facts, bundle.user_profile
        )

        # Profile fields
        profile_fields: list[str] = []
        profile_source = ""
        if bundle.user_profile is not None:
            profile_fields = list(bundle.user_profile.preferences.keys()) + [
                f"tendency:{k}" for k in bundle.user_profile.topic_tendencies
            ]
            profile_source = f"industry={bundle.user_profile.industry}"

        return {
            "layers": {
                "user_profile": {
                    "injected": bundle.user_profile is not None,
                    "source": profile_source,
                    "fields": profile_fields,
                },
                "key_facts": {
                    "injected": len(bundle.key_facts) > 0,
                    "count": len(bundle.key_facts),
                    "acknowledged_count": acknowledged_count,
                    "evidence": fact_evidence,
                },
                "dialogue_history": {
                    "injected": bundle.dialogue_summary is not None
                    or len(bundle.recalled_details) > 0,
                    "summary_len": len(bundle.dialogue_summary or ""),
                    "details_count": len(bundle.recalled_details),
                    "top_turns": detail_evidence,
                },
            },
            "degraded_layers": bundle.degraded_layers,
            "conflict_resolution": {
                "winner": conflict["winner"],
                "conflicts_count": len(conflict["conflicts"]),
                "conflicts": conflict["conflicts"],
                "acknowledged_points": conflict["acknowledged_points"],
                "acknowledged_count": conflict["acknowledged_count"],
            },
            "priority_order": bundle.priority_order,
            "assembled_at": bundle.assembled_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal utility methods
    # ------------------------------------------------------------------

    def _is_acknowledged(self, fact: KeyFact) -> bool:
        """Determine whether a key fact is a "point acknowledged by the user"."""
        if fact.content.get(self.ACKNOWLEDGED_FLAG_KEY) is True:
            return True
        if fact.content.get("tag") == self.ACKNOWLEDGED_FACT_TAG:
            return True
        if fact.content.get("status") == "acknowledged":
            return True
        return False

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """Loose equality check (avoids false positives when comparing dict vs str)."""
        if a == b:
            return True
        # Tolerance: dict vs str
        if isinstance(a, dict) and isinstance(b, str):
            return a.get("value") == b or str(a) == b
        if isinstance(b, dict) and isinstance(a, str):
            return b.get("value") == a or str(b) == a
        return False

    @staticmethod
    def _preview_content(content: dict[str, Any]) -> str:
        """Take the first 100 characters of key fact content as a preview."""
        s = str(content)
        return s[:100] + ("..." if len(s) > 100 else "")

    @staticmethod
    def _merge_unique_turns(
        primary: list[DialogueTurn],
        secondary: list[DialogueTurn],
    ) -> list[DialogueTurn]:
        """Merge two dialogue history lists, deduplicate by turn, keep ascending by turn."""
        seen: set[int] = set()
        merged: list[DialogueTurn] = []
        for t in primary + secondary:
            if t.turn not in seen:
                seen.add(t.turn)
                merged.append(t)
        merged.sort(key=lambda t: t.turn)
        return merged
