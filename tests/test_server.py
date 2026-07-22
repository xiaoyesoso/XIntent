"""Tests for the FastAPI server endpoints, focused on the main /agent/intent endpoint.

The server's `load_dotenv()` runs at import time and may load a real API_KEY
from .env, which would make tests hit real LLM endpoints (slow, nondeterministic).
We patch `get_pipeline` and `get_intent_pipeline` directly to force MockLLMClient.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from user_input_normalization.llm.mock import MockLLMClient
from user_input_normalization.pipeline import NormalizationPipeline
from user_input_normalization.storage import (
    MemoryDialogueHistoryStore,
    MemoryFewShotStore,
    MemoryKeyFactStore,
    MemoryUserProfileStore,
    MemoryVocabStore,
)

# Import server AFTER stubbing load_dotenv so no real API_KEY leaks in.
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]

from user_input_normalization.server import app, _build_default_registry  # noqa: E402
from intent_recognition import (  # noqa: E402
    IntentRecognitionConfig,
    IntentRecognitionPipeline,
)
from intent_recognition.storage import (  # noqa: E402
    MemoryIntentHistoryStore,
    MemorySlotStateStore,
)


@pytest.fixture(autouse=True)
def _force_mock_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace server pipelines with MockLLMClient-backed instances per test.

    This guarantees fast, deterministic tests regardless of any .env file.
    The mock LLM returns a complete normalization result so the completeness
    checker does not pause the pipeline for clarification on simple inputs.
    """
    import json
    import re

    import user_input_normalization.server as srv
    from user_input_normalization.config import get_config

    mock_llm = MockLLMClient()

    def _default_norm_handler(sys_prompt: str, user_prompt: str) -> str:
        """Return a minimal but complete normalization result.

        Extracts the raw input from the user prompt (embedded as
        `# 用户输入\n{raw_input}`) so normalized_input reflects the original.
        """
        m = re.search(r"#\s*用户输入\n(.+?)(?:\n#|\Z)", user_prompt, re.DOTALL)
        raw = m.group(1).strip() if m else user_prompt[:200]
        return json.dumps(
            {
                "normalized_input": raw,
                "spo": {"subject": "用户", "predicate": "请求", "obj": ""},
                "pronoun_resolutions": [],
                "quantifiable_adjectives": [],
                "term_mappings": [],
                "completions": [],
            },
            ensure_ascii=False,
        )

    mock_llm.set_default_handler(_default_norm_handler)

    # Use domain agent_type to avoid clarification pauses on incomplete SPO
    # (the mock returns a placeholder SPO; we don't want it to block the pipeline)
    cfg = get_config()
    cfg.pipeline.agent_type = "domain"

    norm_pipe = NormalizationPipeline(
        llm_client=mock_llm,
        key_fact_store=MemoryKeyFactStore(),
        fewshot_store=MemoryFewShotStore(),
        vocab_store=MemoryVocabStore(),
        profile_store=MemoryUserProfileStore(),
        dialogue_store=MemoryDialogueHistoryStore(),
        config=cfg,
    )
    registry = _build_default_registry()
    intent_pipe = IntentRecognitionPipeline(
        llm_client=mock_llm,
        registry=registry,
        config=IntentRecognitionConfig(),
        slot_state_store=MemorySlotStateStore(),
        intent_history_store=MemoryIntentHistoryStore(),
        deep_llm_client=mock_llm,
    )

    monkeypatch.setattr(srv, "get_pipeline", lambda: norm_pipe)
    monkeypatch.setattr(srv, "get_intent_pipeline", lambda: (intent_pipe, registry))


client = TestClient(app)


class TestRootAndHealth:
    """Smoke tests for info endpoints."""

    def test_root_lists_main_endpoint_first(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "XIntent API"
        assert body["main_endpoint"] == "POST /agent/intent"
        assert "agent_intent" in body["endpoints"]
        assert "/agent/intent" in body["endpoints"]["agent_intent"]

    def test_health_returns_version_and_backend(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "llm_backend" in body


class TestAgentIntentMain:
    """Tests for the main POST /agent/intent endpoint."""

    def test_code_layer_hit_with_normalization(self):
        """End-to-end: raw input -> normalization -> code-layer intent hit."""
        resp = client.post("/agent/intent", json={
            "raw_input": "我要退款",
            "session_id": "s-test-1",
            "turn": 1,
        })
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()

        # Intent recognition result (top-level)
        assert body["intent"] == "refund"
        assert body["source"] == "code-layer"
        assert body["confidence"] == 1.0
        assert body["layer_reached"] == 1

        # Pipeline meta
        assert body["raw_input"] == "我要退款"
        assert body["normalized_input"]  # non-empty
        assert body["pipeline_path"] == ["normalization", "intent-recognition"]
        assert body["skipped_normalization"] is False
        assert body["paused_at_normalization"] is False

        # Normalization detail is attached
        assert body["normalization"] is not None
        assert "normalized_input" in body["normalization"]
        assert "stage_reached" in body["normalization"]
        assert isinstance(body["normalization"]["classification_tags"], list)

    def test_skip_normalization(self):
        """When skip_normalization=True, raw_input feeds intent recognition directly."""
        resp = client.post("/agent/intent", json={
            "raw_input": "我要退款",
            "session_id": "s-test-2",
            "turn": 1,
            "skip_normalization": True,
        })
        assert resp.status_code == 200
        body = resp.json()

        assert body["intent"] == "refund"
        assert body["source"] == "code-layer"
        # Pipeline skipped normalization
        assert body["pipeline_path"] == ["intent-recognition"]
        assert body["skipped_normalization"] is True
        assert body["paused_at_normalization"] is False
        # normalization detail is None because skipped
        assert body["normalization"] is None
        # normalized_input equals raw_input when skipped
        assert body["normalized_input"] == body["raw_input"]

    def test_ui_event_page_guidance_skipping_norm(self):
        """UI event triggers page guidance (code layer) when normalization is skipped.

        We skip normalization here to avoid the MockLLM's default normalization
        of the terse input "继续" affecting the result. The page-guidance matcher
        keys off the `event` field, not the text.
        """
        resp = client.post("/agent/intent", json={
            "raw_input": "继续",
            "session_id": "s-test-3",
            "turn": 1,
            "event": "click:next",
            "skip_normalization": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "code-layer"
        assert body["layer_reached"] == 1
        assert body["intent"] is not None

    def test_unsupported_intent_rejected(self):
        """Unsupported input is rejected with a reason; normalization still runs."""
        resp = client.post("/agent/intent", json={
            "raw_input": "今天天气怎么样",
            "session_id": "s-test-4",
            "turn": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] is None
        assert body["rejection_reason"] is not None and len(body["rejection_reason"]) > 0
        assert body["pipeline_path"] == ["normalization", "intent-recognition"]
        assert body["normalization"] is not None

    def test_missing_required_field(self):
        """Missing required field `raw_input` returns 422."""
        resp = client.post("/agent/intent", json={
            "session_id": "s-test-5",
        })
        assert resp.status_code == 422

    def test_missing_session_id(self):
        """Missing required field `session_id` returns 422."""
        resp = client.post("/agent/intent", json={
            "raw_input": "我要退款",
        })
        assert resp.status_code == 422

    def test_observation_passes_through_to_normalization(self):
        """observation reaches the normalization stage without error.

        Whether stage_reached is 'deep' depends on the normalizer detecting
        an unquantified adjective (internal behavior). Here we only verify
        the main endpoint passes observation through and the pipeline runs
        both stages.
        """
        resp = client.post("/agent/intent", json={
            "raw_input": "帮我推荐更有性价比的牛仔裤",
            "session_id": "s-test-6",
            "turn": 1,
            "observation": {"current_price": 200, "available_prices": [100, 150, 200, 300]},
        })
        assert resp.status_code == 200
        body = resp.json()
        # Both stages ran (normalization did not pause for clarification)
        assert body["pipeline_path"] == ["normalization", "intent-recognition"]
        assert body["paused_at_normalization"] is False
        assert body["normalization"] is not None
        # stage_reached is either "pre" or "deep" depending on whether the
        # mock detected an unquantified adjective; both are acceptable here
        assert body["normalization"]["stage_reached"] in ("pre", "deep")


class TestSubEndpointsStillAvailable:
    """Sub-endpoints /normalize and /recognize remain accessible for direct calls."""

    def test_normalize_still_works(self):
        resp = client.post("/normalize", json={
            "raw_input": "我要退款",
            "session_id": "s-sub-1",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "normalized_input" in body
        assert "stage_reached" in body

    def test_recognize_still_works(self):
        resp = client.post("/recognize", json={
            "text": "我要退款",
            "session_id": "s-sub-2",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "refund"
        assert body["source"] == "code-layer"

    def test_intents_list(self):
        resp = client.get("/intents")
        assert resp.status_code == 200
        body = resp.json()
        assert "intents" in body
        names = [i["name"] for i in body["intents"]]
        assert "refund" in names
        assert "product_recommendation" in names


class TestExtensionOverridesAndBugs:
    """Regression tests for D17-D24 per-request overrides and related bugs.

    Covers:
    - Per-request override config leak fix (overrides must not persist across requests)
    - `source` field when paused at normalization (should not be "code-layer")
    - /recognize endpoint accepts the same override flags as /agent/intent
    - D22 reuse works end-to-end via the main endpoint
    """

    def test_override_does_not_leak_across_requests(self):
        """Per-request D22 override must not leak into a subsequent request.

        Regression for a bug where `_apply_extension_overrides` mutated the
        singleton pipeline's config in-place without restoring it, so a
        request with `reuse_previous_intent=True` left D22 enabled for all
        following requests.
        """
        import user_input_normalization.server as srv

        # Request A: enable D22
        client.post("/agent/intent", json={
            "raw_input": "查询订单 12345",
            "session_id": "leak-a",
            "turn": 1,
            "skip_normalization": True,
            "reuse_previous_intent": True,
        })
        # Request B: no override flag (None) - D22 should be OFF again
        client.post("/agent/intent", json={
            "raw_input": "查询订单 67890",
            "session_id": "leak-b",
            "turn": 1,
            "skip_normalization": True,
        })
        pipe, _ = srv.get_intent_pipeline()
        assert pipe._config.reuse_strategy.enable is False
        assert pipe._reuse_strategy is None

    def test_override_does_not_leak_retrieval_or_vector(self):
        """Same leak check for D17 (retrieval) and D21 (vector_fallback)."""
        import user_input_normalization.server as srv

        client.post("/agent/intent", json={
            "raw_input": "test",
            "session_id": "leak-d17",
            "turn": 1,
            "skip_normalization": True,
            "enable_retrieval": True,
            "enable_vector_fallback": True,
        })
        pipe, _ = srv.get_intent_pipeline()
        # After the request completes, config should be restored to defaults
        assert pipe._config.retrieval.enable is False
        assert pipe._config.vector_fallback.enable is False

    def test_source_empty_when_paused_at_normalization(self):
        """When normalization pauses, `source` must not claim "code-layer".

        Regression for a cosmetic bug where the early-exit response used the
        `source` field's default value "code-layer" even though no intent
        recognition layer actually ran.
        """
        # Force a clarification pause by using a general agent_type (not domain)
        # and a terse input. The MockLLM returns a placeholder SPO that the
        # completeness checker will flag as incomplete.
        import user_input_normalization.server as srv
        from user_input_normalization.config import get_config

        # Build a separate normalization pipeline with general agent_type
        # to trigger the clarification pause path.
        from user_input_normalization.llm.mock import MockLLMClient
        from user_input_normalization.pipeline import NormalizationPipeline
        from user_input_normalization.storage import (
            MemoryDialogueHistoryStore,
            MemoryFewShotStore,
            MemoryKeyFactStore,
            MemoryUserProfileStore,
            MemoryVocabStore,
        )

        mock_llm = MockLLMClient()
        # Default handler returns minimal SPO -> triggers clarification pause
        import json
        mock_llm.set_default_handler(lambda sys_p, user_p: json.dumps({
            "normalized_input": "test",
            "spo": {"subject": "", "predicate": "", "obj": ""},
            "pronoun_resolutions": [],
            "quantifiable_adjectives": [],
            "term_mappings": [],
            "completions": [],
        }, ensure_ascii=False))

        cfg = get_config()
        cfg.pipeline.agent_type = "general"
        norm_pipe = NormalizationPipeline(
            llm_client=mock_llm,
            key_fact_store=MemoryKeyFactStore(),
            fewshot_store=MemoryFewShotStore(),
            vocab_store=MemoryVocabStore(),
            profile_store=MemoryUserProfileStore(),
            dialogue_store=MemoryDialogueHistoryStore(),
            config=cfg,
        )
        original = srv.get_pipeline
        srv.get_pipeline = lambda: norm_pipe
        try:
            resp = client.post("/agent/intent", json={
                "raw_input": "它",
                "session_id": "pause-test",
                "turn": 1,
            })
            assert resp.status_code == 200
            body = resp.json()
            if body.get("paused_at_normalization"):
                # source must NOT be "code-layer" when no recognition ran
                assert body["source"] != "code-layer"
                assert body["source"] == ""
        finally:
            srv.get_pipeline = original

    def test_recognize_accepts_override_flags(self):
        """/recognize endpoint accepts the same D17/D21/D22 flags as /agent/intent."""
        resp = client.post("/recognize", json={
            "text": "我要退款",
            "session_id": "recognize-override",
            "turn": 1,
            "enable_retrieval": True,
            "enable_vector_fallback": False,
            "reuse_previous_intent": False,
        })
        assert resp.status_code == 200
        # Config should be restored after the request
        import user_input_normalization.server as srv
        pipe, _ = srv.get_intent_pipeline()
        assert pipe._config.retrieval.enable is False

    def test_d22_reuse_via_main_endpoint(self):
        """D22 reuse works end-to-end when enabled per-request.

        Turn 1 stores an intent via code-layer; turn 2 with reuse enabled
        should return source='reused' (unless an intent-switch marker appears).
        """
        # Turn 1: code-layer hit stores 'order_query' in history
        r1 = client.post("/agent/intent", json={
            "raw_input": "查询订单 12345",
            "session_id": "reuse-e2e",
            "turn": 1,
            "skip_normalization": True,
            "reuse_previous_intent": True,
        })
        assert r1.status_code == 200
        assert r1.json()["intent"] == "order_query"

        # Turn 2: reuse should kick in (no intent-switch marker)
        r2 = client.post("/agent/intent", json={
            "raw_input": "再看一下",
            "session_id": "reuse-e2e",
            "turn": 2,
            "skip_normalization": True,
            "reuse_previous_intent": True,
        })
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["source"] == "reused"
        assert body2["intent"] == "order_query"
        assert body2["confidence"] == 1.0

    def test_d22_reuse_skipped_on_intent_switch_marker(self):
        """D22 reuse is skipped when input contains an intent-switch marker."""
        # Turn 1: store intent
        client.post("/agent/intent", json={
            "raw_input": "查询订单 12345",
            "session_id": "reuse-switch",
            "turn": 1,
            "skip_normalization": True,
            "reuse_previous_intent": True,
        })
        # Turn 2: intent-switch marker -> reuse must NOT trigger
        r2 = client.post("/agent/intent", json={
            "raw_input": "换个话题,我要退款",
            "session_id": "reuse-switch",
            "turn": 2,
            "skip_normalization": True,
            "reuse_previous_intent": True,
        })
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["source"] != "reused"

    def test_d20_fields_present_in_main_response(self):
        """D20 multi-intent fields (relations, pending_intents) are always present."""
        resp = client.post("/agent/intent", json={
            "raw_input": "我要退款",
            "session_id": "d20-fields",
            "turn": 1,
            "skip_normalization": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "relations" in body
        assert "pending_intents" in body
        assert isinstance(body["relations"], list)
        assert isinstance(body["pending_intents"], list)


class TestD31ProtocolFields:
    """Tests for D31 structured output protocol fields (tasks 10.13-10.15).

    The 10-field protocol is the handoff contract between normalization and
    intent recognition. The main endpoint surfaces these fields at the top
    level of AgentIntentResponse; /recognize surfaces them with
    normalized_query="" (no normalization runs).
    """

    PROTOCOL_FIELDS = (
        "normalized_query",
        "sub_tasks",
        "independent_intents",
        "verified_evidence",
        "provisional_evidence",
        "assumptions_disclosed",
        "assumptions",
        "arbitration_breakdown",
    )

    def test_main_endpoint_returns_all_protocol_fields(self):
        """Task 10.13: POST /agent/intent surfaces all D31 protocol fields."""
        resp = client.post("/agent/intent", json={
            "raw_input": "我要退款",
            "session_id": "d31-protocol",
            "turn": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        # All protocol fields must be present
        for field in self.PROTOCOL_FIELDS:
            assert field in body, f"missing protocol field: {field}"
        # normalized_query is non-empty when normalization ran
        assert body["normalized_query"]
        # Default values are correct types
        assert isinstance(body["sub_tasks"], list)
        assert isinstance(body["independent_intents"], list)
        assert isinstance(body["verified_evidence"], list)
        assert isinstance(body["provisional_evidence"], list)
        assert isinstance(body["assumptions_disclosed"], bool)
        assert isinstance(body["assumptions"], list)
        assert isinstance(body["arbitration_breakdown"], dict)

    def test_main_endpoint_normalized_query_reflects_normalization(self):
        """Task 10.13: normalized_query comes from normalization output when present."""
        resp = client.post("/agent/intent", json={
            "raw_input": "帮我推荐一款3000元以内的手机",
            "session_id": "d31-nq",
            "turn": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        # normalized_query should be populated (from normalization or raw input)
        assert body["normalized_query"]
        # pipeline_path confirms normalization ran
        assert body["pipeline_path"] == ["normalization", "intent-recognition"]

    def test_main_endpoint_evidence_collected_for_verified_slots(self):
        """Task 10.13: verified_evidence is populated for code-layer hits.

        Code-layer hits (keyword/regex match) tag the current input as
        verified evidence, so verified_evidence should be non-empty.
        """
        resp = client.post("/agent/intent", json={
            "raw_input": "我要退款",
            "session_id": "d31-evidence",
            "turn": 1,
            "skip_normalization": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        # Code-layer hit -> verified_evidence captures the current input
        assert len(body["verified_evidence"]) >= 1
        # All entries have the expected shape
        for ev in body["verified_evidence"]:
            assert "content" in ev
            assert "grade" in ev
            assert ev["grade"] == "verified"

    def test_recognize_endpoint_normalized_query_empty(self):
        """Task 10.14: POST /recognize returns normalized_query as empty string.

        /recognize does not run normalization, so normalized_query must be ""
        (not None, not the raw text).
        """
        resp = client.post("/recognize", json={
            "text": "我要退款",
            "session_id": "d31-recognize",
        })
        assert resp.status_code == 200
        body = resp.json()
        # normalized_query is explicitly "" on /recognize
        assert body["normalized_query"] == ""
        # Other protocol fields are still present
        for field in self.PROTOCOL_FIELDS:
            assert field in body, f"missing protocol field on /recognize: {field}"

    def test_recognize_endpoint_protocol_field_types(self):
        """Task 10.14: /recognize protocol fields have correct default types."""
        resp = client.post("/recognize", json={
            "text": "我要退款",
            "session_id": "d31-recognize-types",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["sub_tasks"], list)
        assert isinstance(body["independent_intents"], list)
        assert isinstance(body["verified_evidence"], list)
        assert isinstance(body["provisional_evidence"], list)
        assert isinstance(body["assumptions_disclosed"], bool)
        assert isinstance(body["assumptions"], list)
        assert isinstance(body["arbitration_breakdown"], dict)

    def test_sub_intents_alias_syncs_with_independent_intents(self):
        """Task 10.15: sub_intents is a deprecated alias of independent_intents.

        When independent_intents is populated (e.g. multi-intent decomposition
        in L3), the legacy sub_intents field mirrors it so old clients keep
        working. The two fields must stay in sync.
        """
        # Use a multi-intent scenario via L3 deep LLM
        import json
        import user_input_normalization.server as srv

        mock_llm = MockLLMClient()

        # L2 returns low confidence to force escalation to L3
        def handler(sys_p, _user_p):
            if "深度推理" in sys_p or "复杂场景" in sys_p:
                # L3 returns multi-intent decomposition
                return json.dumps({
                    "intent": "product_recommendation",
                    "confidence": 0.85,
                    "slots": {"category": "手机"},
                    "missing_slots": [],
                    "hard_constraints": [],
                    "soft_constraints": [],
                    "independent_intents": ["product_recommendation", "order_query"],
                    "sub_tasks": ["比价"],
                    "intent_switched": False,
                })
            # L2 low confidence -> escalate
            return json.dumps({
                "intent": "product_recommendation",
                "confidence": 0.3,
                "slots": {},
                "missing_slots": [],
                "hard_constraints": [],
                "soft_constraints": [],
            })
        mock_llm.set_default_handler(handler)

        # Build a pipeline that will hit L3 (input must miss code layer + L2)
        registry = _build_default_registry()
        intent_pipe = IntentRecognitionPipeline(
            llm_client=mock_llm,
            registry=registry,
            config=IntentRecognitionConfig(),
            slot_state_store=MemorySlotStateStore(),
            intent_history_store=MemoryIntentHistoryStore(),
            deep_llm_client=mock_llm,
        )
        original = srv.get_intent_pipeline
        srv.get_intent_pipeline = lambda: (intent_pipe, registry)
        try:
            resp = client.post("/agent/intent", json={
                "raw_input": "那个，就是，帮我看看有没有什么手机可以推荐的，顺便查下订单",
                "session_id": "d31-alias",
                "turn": 1,
                "skip_normalization": True,
            })
            assert resp.status_code == 200
            body = resp.json()
            # independent_intents should be populated by L3
            assert len(body["independent_intents"]) >= 1
            # sub_intents (legacy alias) must mirror independent_intents
            assert "sub_intents" in body
            assert body["sub_intents"] == body["independent_intents"]
        finally:
            srv.get_intent_pipeline = original

    def test_sub_intents_alias_empty_when_no_multi_intent(self):
        """Task 10.15: sub_intents alias is empty list when no multi-intent.

        For a simple code-layer hit (no multi-intent decomposition), both
        independent_intents and sub_intents should be empty lists.
        """
        resp = client.post("/agent/intent", json={
            "raw_input": "我要退款",
            "session_id": "d31-alias-empty",
            "turn": 1,
            "skip_normalization": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        # No multi-intent -> both fields empty
        assert body["independent_intents"] == []
        assert body.get("sub_intents", []) == []
