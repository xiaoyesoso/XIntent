"""FastAPI server for user input normalization.

Run locally:
    python -m user_input_normalization.server

Run with uvicorn:
    uvicorn user_input_normalization.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from .llm.base import LLMClient
from .llm.mock import MockLLMClient
from .llm.openai_client import OpenAICompatibleClient
from .pipeline import NormalizationPipeline, PipelineResult
from .storage import (
    MemoryDialogueHistoryStore,
    MemoryFewShotStore,
    MemoryKeyFactStore,
    MemoryUserProfileStore,
    MemoryVocabStore,
)

# Load .env
load_dotenv()

# Single source of truth for the API version
APP_VERSION = "0.2.0"

app = FastAPI(
    title="XIntent API",
    description=(
        "XIntent - agent intent framework. Main entry point: POST /agent/intent - takes raw user input, "
        "runs two-stage normalization internally, then three-layer intent recognition waterfall "
        "(Code Layer / Lightweight LLM / Deep LLM) + slot filling + rejection/clarification. "
        "Sub-endpoints /normalize and /recognize remain available for internal use or direct calls."
    ),
    version=APP_VERSION,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class NormalizeRequest(BaseModel):
    """Request body for normalization."""

    raw_input: str = Field(..., description="User raw input text", examples=["第二个适合生产吗？"])
    session_id: str = Field(..., description="Session ID", examples=["s1"])
    user_id: str | None = Field(default=None, description="User ID", examples=["u1"])
    turn: int = Field(default=0, description="Dialogue turn number", examples=[1])
    observation: dict | None = Field(
        default=None,
        description="ReAct observation (for deep normalization)",
        examples=[{"current_price": 200}],
    )


class NormalizeResponse(BaseModel):
    """Response body for normalization."""

    normalized_input: str = Field(..., description="Normalized input")
    pronoun_resolutions: list[dict] = Field(default_factory=list)
    quantifiable_adjectives: list[dict] = Field(default_factory=list)
    term_mappings: list[dict] = Field(default_factory=list)
    completions: list[dict] = Field(default_factory=list)
    classification_tags: list[str] = Field(default_factory=list)
    stage_reached: str = Field(..., description="Stage reached: pre / deep")
    paused_for_clarification: bool = Field(default=False)
    clarification: dict | None = Field(default=None)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    llm_backend: str


# ---------------------------------------------------------------------------
# Global pipeline instance (lazy init)
# ---------------------------------------------------------------------------

_pipeline: NormalizationPipeline | None = None
_llm_backend: str = "unknown"


def _get_llm_backend_label() -> str:
    """Return a human-readable label for the LLM backend in use.

    Determines the label lazily from environment variables so that /health
    returns an accurate backend even before any pipeline has been built.
    """
    if _llm_backend != "unknown":
        return _llm_backend
    api_key = os.getenv("API_KEY", "")
    if api_key:
        flash_model = os.getenv("FLASH_LLM_MODEL") or "unknown"
        pro_model = os.getenv("PRO_LLM_MODEL") or "unknown"
        return f"OpenAI-compatible (flash={flash_model}, pro={pro_model})"
    return "MockLLM (no API_KEY set)"


def get_pipeline() -> NormalizationPipeline:
    """Get or create the pipeline singleton."""
    global _pipeline, _llm_backend

    if _pipeline is not None:
        return _pipeline

    # Choose LLM client based on API_KEY availability
    api_key = os.getenv("API_KEY", "")
    if api_key:
        llm: LLMClient = OpenAICompatibleClient()
        _llm_backend = _get_llm_backend_label()
    else:
        llm = MockLLMClient()
        _llm_backend = "MockLLM (no API_KEY set)"

    _pipeline = NormalizationPipeline(
        llm_client=llm,
        key_fact_store=MemoryKeyFactStore(),
        fewshot_store=MemoryFewShotStore(),
        vocab_store=MemoryVocabStore(),
        profile_store=MemoryUserProfileStore(),
        dialogue_store=MemoryDialogueHistoryStore(),
    )
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "name": "XIntent API",
        "version": APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "main_endpoint": "POST /agent/intent",
        "endpoints": {
            "agent_intent": "POST /agent/intent (main - normalization + intent recognition)",
            "normalize": "POST /normalize (sub - normalization only)",
            "recognize": "POST /recognize (sub - intent recognition only)",
            "intents": "GET /intents",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        llm_backend=_get_llm_backend_label(),
    )


@app.post("/normalize", response_model=NormalizeResponse, tags=["normalization"])
async def normalize(request: NormalizeRequest):
    """Normalize user input through the two-stage pipeline.

    - Classifies input into 6 categories (anaphora, missing, expression, semantic, subjective, external fact)
    - Pre-normalization: pronoun resolution, ellipsis completion, sentence correction, term standardization
    - Deep normalization: adjective quantification, external fact resolution (if observation provided)
    - Returns structured result with completeness check
    """
    pipe = get_pipeline()
    try:
        result: PipelineResult = pipe.process(
            raw_input=request.raw_input,
            session_id=request.session_id,
            user_id=request.user_id,
            turn=request.turn,
            observation=request.observation,
        )
    except Exception as e:
        logger.error("Normalization failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Normalization failed: {e}") from e


    return NormalizeResponse(
        normalized_input=result.result.normalized_input,
        pronoun_resolutions=[pr.model_dump() for pr in result.result.pronoun_resolutions],
        quantifiable_adjectives=[a.model_dump() for a in result.result.quantifiable_adjectives],
        term_mappings=[t.model_dump() for t in result.result.term_mappings],
        completions=[c.model_dump() for c in result.result.completions],
        classification_tags=[t.value for t in result.result.classification_tags],
        stage_reached=result.stage_reached,
        paused_for_clarification=result.paused_for_clarification,
        clarification=result.clarification.model_dump() if result.clarification else None,
    )


# ---------------------------------------------------------------------------
# Intent Recognition endpoints
# ---------------------------------------------------------------------------

from intent_recognition import (  # noqa: E402
    IntentDefinition,
    IntentRecognitionConfig,
    IntentRecognitionPipeline,
    IntentRegistry,
    SlotDefinition,
)
from intent_recognition.storage import (  # noqa: E402
    MemoryIntentHistoryStore,
    MemorySlotStateStore,
)

_intent_pipeline: IntentRecognitionPipeline | None = None
_intent_registry: IntentRegistry | None = None


def _build_default_registry() -> IntentRegistry:
    """Build a default intent registry with common e-commerce intents."""
    reg = IntentRegistry()
    reg.register(IntentDefinition(
        name="product_recommendation",
        description="用户希望系统推荐商品、帮忙挑选商品，或根据预算、用途、偏好给出购买建议",
        positive_examples=["3000以内有什么手机推荐", "帮我推荐个耳机"],
        negative_examples=["iPhone 16的电池容量是多少 -> product_query"],
        slots=[
            SlotDefinition(name="category", required=True, description="商品类目，如手机、耳机"),
            SlotDefinition(name="budget_max", required=False, description="最高预算"),
            SlotDefinition(name="usage_scenario", required=False, description="使用场景"),
        ],
    ))
    reg.register(IntentDefinition(
        name="product_query",
        description="用户希望查询某个商品、品牌、型号、参数、价格或库存信息",
        positive_examples=["iPhone 16的电池容量是多少"],
        negative_examples=["3000以内有什么手机推荐 -> product_recommendation"],
    ))
    reg.register(IntentDefinition(
        name="product_comparison",
        description="用户希望比较两个或多个商品、品牌或型号",
        positive_examples=["iPhone 16和小米15哪个好"],
        negative_examples=["iPhone 16的电池容量是多少 -> product_query"],
    ))
    reg.register(IntentDefinition(
        name="order_query",
        description="用户希望查询订单状态、物流信息",
        positive_examples=["我的订单到哪了", "查询订单 12345"],
    ))
    reg.register(IntentDefinition(
        name="refund",
        description="用户希望申请退款或售后",
        positive_examples=["我要退款", "这个商品我想退货"],
        parent_intent="order_query",
    ))
    return reg


def get_intent_pipeline() -> tuple[IntentRecognitionPipeline, IntentRegistry]:
    """Get or create the intent recognition pipeline singleton."""
    global _intent_pipeline, _intent_registry

    if _intent_pipeline is not None and _intent_registry is not None:
        return _intent_pipeline, _intent_registry

    api_key = os.getenv("API_KEY", "")
    if api_key:
        llm = OpenAICompatibleClient()
        deep_llm = OpenAICompatibleClient(model_tier="pro")
    else:
        llm = MockLLMClient()
        deep_llm = llm

    _intent_registry = _build_default_registry()
    _intent_pipeline = IntentRecognitionPipeline(
        llm_client=llm,
        registry=_intent_registry,
        config=IntentRecognitionConfig(),
        slot_state_store=MemorySlotStateStore(),
        intent_history_store=MemoryIntentHistoryStore(),
        deep_llm_client=deep_llm,
    )
    return _intent_pipeline, _intent_registry


class RecognizeRequest(BaseModel):
    """Request body for intent recognition."""

    text: str = Field(..., description="Normalized user input text", examples=["帮我推荐一个手机"])
    session_id: str = Field(default="default", description="Session ID", examples=["s1"])
    user_id: str | None = Field(default=None, description="User ID", examples=["u1"])
    turn: int = Field(default=1, description="Dialogue turn number", examples=[1])
    context: dict | None = Field(default=None, description="Additional context")
    event: str | None = Field(default=None, description="UI event for page guidance", examples=["click:next"])
    dialogue_history: list[dict] | None = Field(default=None, description="Previous dialogue turns")
    # D17-D24 extension flags (mirror /agent/intent; None = use pipeline config)
    enable_retrieval: bool | None = Field(
        default=None,
        description="Override D17 retrieval-based candidate narrowing (None=use config, True/False=force)",
    )
    enable_vector_fallback: bool | None = Field(
        default=None,
        description="Override D21 vector matching fallback (None=use config, True/False=force)",
    )
    reuse_previous_intent: bool | None = Field(
        default=None,
        description="Override D22 intent reuse with rollback (None=use config, True/False=force)",
    )


class RecognizeResponse(BaseModel):
    """Response body for intent recognition.

    D31 protocol fields are exposed here too. ``normalized_query`` is always
    an empty string for ``/recognize`` because this endpoint does not run
    normalization (callers feed pre-normalized text). Callers that need the
    normalized query should use ``POST /agent/intent`` instead.
    """

    intent: str | None = Field(default=None, description="Recognized intent")
    confidence: float = Field(default=0.0, description="Confidence score")
    source: str = Field(default="code-layer", description="Which layer produced this")
    slots: dict = Field(default_factory=dict, description="Extracted slot values")
    missing_slots: list[str] = Field(default_factory=list, description="Missing required slots")
    need_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_question: str | None = Field(default=None, description="Question to ask user")
    hard_constraints: list[dict] = Field(default_factory=list)
    soft_constraints: list[dict] = Field(default_factory=list)
    rejection_reason: str | None = Field(default=None, description="Rejection reason if unsupported")
    layer_reached: int = Field(default=1, description="Highest layer reached")
    sub_intents: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-intents (deprecated alias of independent_intents, D27)",
    )
    intent_switched: bool = Field(default=False, description="Whether intent switched")
    previous_intent: str | None = Field(default=None, description="Previous turn's intent")
    relations: list[dict] = Field(default_factory=list, description="D20: Dependency relations between sub-intents")
    pending_intents: list[str] = Field(default_factory=list, description="D20: Pending intents in topological order")
    # D26/D27/D31 protocol fields (mirrored from AgentIntentResponse)
    normalized_query: str = Field(
        default="",
        description="D31: Always empty for /recognize (no normalization ran). Use /agent/intent for the full protocol.",
    )
    sub_tasks: list[str] = Field(
        default_factory=list,
        description="D27: Intra-flow sub-tasks",
    )
    independent_intents: list[str] = Field(
        default_factory=list,
        description="D27: Independent intents (primary field; sub_intents is deprecated alias)",
    )
    verified_evidence: list[dict] = Field(
        default_factory=list,
        description="D26: Verified evidence from current input / context",
    )
    provisional_evidence: list[dict] = Field(
        default_factory=list,
        description="D26: Provisional evidence from history / speculation / defaults",
    )
    assumptions_disclosed: bool = Field(default=False, description="D26: Whether assumptions were disclosed")
    assumptions: list[str] = Field(default_factory=list, description="D26: Disclosed assumptions")
    arbitration_breakdown: dict = Field(
        default_factory=dict,
        description="D28: Five-factor arbitration breakdown; empty when arbitration did not run",
    )


@app.get("/intents", tags=["intent-recognition"])
async def list_intents():
    """List all registered intent definitions."""
    _, registry = get_intent_pipeline()
    return {
        "intents": [
            {
                "name": d.name,
                "description": d.description,
                "parent_intent": d.parent_intent,
                "slots": [s.model_dump() for s in d.slots],
                "positive_examples": d.positive_examples,
                "negative_examples": d.negative_examples,
            }
            for d in registry.list_all()
        ]
    }


@app.post("/recognize", response_model=RecognizeResponse, tags=["intent-recognition"])
async def recognize(request: RecognizeRequest):
    """Recognize user intent through the three-layer waterfall pipeline.

    - Layer 1: Code-based (keyword, regex, rule engine) - fast, no LLM
    - Layer 2: Lightweight LLM (Flash/Mini) with confidence routing
    - Layer 3: Deep reasoning LLM for complex scenarios
    - Slot Filling: parameter extraction, constraint identification
    - Rejection/Clarification: unsupported and unclear dual exits
    """
    pipe, _ = get_intent_pipeline()
    # Apply D17/D21/D22 per-request overrides (same pattern as /agent/intent).
    cfg_snapshot = _apply_extension_overrides(
        pipe,
        request.enable_retrieval,
        request.enable_vector_fallback,
        request.reuse_previous_intent,
    )
    try:
        result = pipe.recognize(
            text=request.text,
            session_id=request.session_id,
            user_id=request.user_id or "default",
            turn=request.turn,
            context=request.context,
            event=request.event,
            dialogue_history=request.dialogue_history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent recognition failed: {e}") from e
    finally:
        _restore_extension_config(pipe, cfg_snapshot)

    return RecognizeResponse(
        intent=result.intent,
        confidence=result.confidence,
        source=result.source.value,
        slots=result.slots,
        missing_slots=result.missing_slots,
        need_clarification=result.need_clarification,
        clarification_question=result.clarification_question,
        hard_constraints=[c.model_dump() for c in result.hard_constraints],
        soft_constraints=[c.model_dump() for c in result.soft_constraints],
        rejection_reason=result.rejection_reason,
        layer_reached=result.layer_reached,
        sub_intents=result.sub_intents,
        intent_switched=result.intent_switched,
        previous_intent=result.previous_intent,
        relations=[r.model_dump() for r in result.relations],
        pending_intents=result.pending_intents,
        # D31 protocol fields (normalized_query is always "" for /recognize)
        normalized_query="",
        sub_tasks=result.sub_tasks,
        independent_intents=result.independent_intents,
        verified_evidence=[e.model_dump() for e in result.verified_evidence],
        provisional_evidence=[e.model_dump() for e in result.provisional_evidence],
        assumptions_disclosed=result.assumptions_disclosed,
        assumptions=result.assumptions,
        arbitration_breakdown=result.arbitration_breakdown,
    )


# ---------------------------------------------------------------------------
# Main endpoint: POST /agent/intent
# ---------------------------------------------------------------------------


class AgentIntentRequest(BaseModel):
    """Request body for the main agent intent endpoint.

    This endpoint orchestrates the full pipeline: normalization (internal)
    followed by three-layer intent recognition. Most callers should use this
    endpoint instead of calling /normalize and /recognize separately.
    """

    raw_input: str = Field(..., description="User raw input text", examples=["第二个适合生产吗？"])
    session_id: str = Field(..., description="Session ID (shared by both stages)", examples=["s1"])
    user_id: str | None = Field(default=None, description="User ID", examples=["u1"])
    turn: int = Field(default=1, description="Dialogue turn number", examples=[1])
    context: dict | None = Field(
        default=None,
        description="Additional context for intent recognition, e.g. {'page': 'product_detail'}",
    )
    event: str | None = Field(
        default=None,
        description="UI event for page guidance (code layer), e.g. 'click:next'",
    )
    observation: dict | None = Field(
        default=None,
        description="ReAct observation for deep normalization (triggers deep stage)",
    )
    dialogue_history: list[dict] | None = Field(
        default=None,
        description="Previous dialogue turns, e.g. [{'role': 'user', 'content': '...'}]",
    )
    skip_normalization: bool = Field(
        default=False,
        description="Skip normalization stage (use when input is already normalized)",
    )
    # D17-D24 extension flags (all default None = use pipeline config, True/False = override)
    enable_retrieval: bool | None = Field(
        default=None,
        description="Override D17 retrieval-based candidate narrowing (None=use config, True/False=force)",
    )
    enable_vector_fallback: bool | None = Field(
        default=None,
        description="Override D21 vector matching fallback (None=use config, True/False=force)",
    )
    reuse_previous_intent: bool | None = Field(
        default=None,
        description="Override D22 intent reuse with rollback (None=use config, True/False=force)",
    )


class AgentIntentNormalizationDetail(BaseModel):
    """Normalization sub-result attached to the agent intent response."""

    normalized_input: str
    pronoun_resolutions: list[dict] = Field(default_factory=list)
    quantifiable_adjectives: list[dict] = Field(default_factory=list)
    term_mappings: list[dict] = Field(default_factory=list)
    completions: list[dict] = Field(default_factory=list)
    classification_tags: list[str] = Field(default_factory=list)
    stage_reached: str
    paused_for_clarification: bool = False
    clarification: dict | None = None
    # D31: Normalization-stage evidence sources (verified/provisional) that
    # feed into the structured output protocol. Empty when normalization is
    # skipped or evidence grading is disabled.
    normalized_query: str = Field(
        default="",
        description="D31: Normalized query string handed off to intent recognition",
    )
    verified_evidence: list[dict] = Field(
        default_factory=list,
        description="D26/D31: Evidence grounded in current input / context / confirmed KeyFact",
    )
    provisional_evidence: list[dict] = Field(
        default_factory=list,
        description="D26/D31: Evidence inferred from history / speculation / defaults",
    )


class AgentIntentResponse(BaseModel):
    """Unified response of the main agent intent endpoint.

    Top-level fields are the intent recognition result (the primary answer).
    The `normalization` field carries the detailed normalization sub-result
    when the normalization stage ran. Meta fields describe the pipeline path.

    D31 structured-output protocol fields (``normalized_query``,
    ``sub_tasks``, ``independent_intents``, ``verified_evidence``,
    ``provisional_evidence``) are exposed at the top level so callers can
    consume a single unified contract. ``sub_intents`` is kept as a
    deprecated alias of ``independent_intents``.
    """

    # Intent recognition result (top-level - the primary answer)
    intent: str | None = Field(default=None, description="Recognized intent, or None if rejected/paused")
    confidence: float = Field(default=0.0)
    source: str = Field(default="code-layer", description="Which layer produced the intent: code-layer / lightweight-llm / deep-llm")
    slots: dict = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    need_clarification: bool = Field(default=False)
    clarification_question: str | None = Field(default=None)
    hard_constraints: list[dict] = Field(default_factory=list)
    soft_constraints: list[dict] = Field(default_factory=list)
    rejection_reason: str | None = Field(default=None)
    layer_reached: int = Field(default=0, description="Highest intent-recognition layer reached (0 if paused at normalization)")
    sub_intents: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-intents (deprecated alias of independent_intents, D27)",
    )
    intent_switched: bool = Field(default=False)
    previous_intent: str | None = Field(default=None)
    # D20: Multi-intent extension fields
    relations: list[dict] = Field(
        default_factory=list,
        description="Dependency relations between sub-intents: [{src, dst, constraints}]",
    )
    pending_intents: list[str] = Field(
        default_factory=list,
        description="Pending intents in topological order (first already executed)",
    )

    # D26/D27/D31: Structured output protocol fields
    normalized_query: str = Field(
        default="",
        description="D31: Normalized query string handed off from normalization to intent recognition (empty if normalization skipped)",
    )
    sub_tasks: list[str] = Field(
        default_factory=list,
        description="D27: Intra-flow sub-tasks (e.g. compare_with_reference, verify_delivery). Do NOT enter multi-intent arbitration.",
    )
    independent_intents: list[str] = Field(
        default_factory=list,
        description="D27: Independent intents that trigger different business flows (primary field; sub_intents is deprecated alias)",
    )
    verified_evidence: list[dict] = Field(
        default_factory=list,
        description="D26: Evidence grounded in current input / context / confirmed KeyFact",
    )
    provisional_evidence: list[dict] = Field(
        default_factory=list,
        description="D26: Evidence inferred from history / speculation / defaults",
    )
    assumptions_disclosed: bool = Field(
        default=False,
        description="D26: True when provisional evidence was disclosed as assumptions instead of triggering clarification",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="D26: List of disclosed assumptions (e.g. '假设尺码为 M（来自历史画像）')",
    )
    arbitration_breakdown: dict = Field(
        default_factory=dict,
        description="D28: Five-factor arbitration breakdown (factor -> pass/fail/skipped) and reason; empty when arbitration did not run",
    )

    # Normalization detail (nested - supporting info)
    normalization: AgentIntentNormalizationDetail | None = Field(
        default=None, description="Normalization sub-result (None if skipped)"
    )

    # Pipeline meta
    raw_input: str = Field(description="Original user input (echoed back)")
    normalized_input: str = Field(description="Text actually fed into intent recognition")
    pipeline_path: list[str] = Field(
        default_factory=list,
        description="Stages executed, e.g. ['normalization', 'intent-recognition']",
    )
    skipped_normalization: bool = Field(default=False)
    paused_at_normalization: bool = Field(
        default=False,
        description="True if pipeline paused at normalization stage awaiting user clarification",
    )


def _apply_extension_overrides(
    intent_pipe: IntentRecognitionPipeline,
    enable_retrieval: bool | None,
    enable_vector_fallback: bool | None,
    reuse_previous_intent: bool | None,
) -> tuple[bool, bool, bool]:
    """Apply per-request D17/D21/D22 config overrides on the pipeline.

    Mutates the pipeline config in-place for the duration of one request.
    Returns a snapshot of the original config values so the caller can
    restore them via ``_restore_extension_config`` after recognition.

    Note: the pipeline is a singleton, so overrides MUST be restored to
    prevent leaking across subsequent requests (a previous version of this
    function had exactly that leak).
    """
    cfg = intent_pipe._config
    orig_retrieval = cfg.retrieval.enable
    orig_vector = cfg.vector_fallback.enable
    orig_reuse = cfg.reuse_strategy.enable

    changed = False
    if enable_retrieval is not None and cfg.retrieval.enable != enable_retrieval:
        cfg.retrieval.enable = enable_retrieval
        changed = True
    if enable_vector_fallback is not None and cfg.vector_fallback.enable != enable_vector_fallback:
        cfg.vector_fallback.enable = enable_vector_fallback
        changed = True
    if reuse_previous_intent is not None and cfg.reuse_strategy.enable != reuse_previous_intent:
        cfg.reuse_strategy.enable = reuse_previous_intent
        changed = True
    if changed:
        intent_pipe._init_extensions()
    return orig_retrieval, orig_vector, orig_reuse


def _restore_extension_config(
    intent_pipe: IntentRecognitionPipeline,
    snapshot: tuple[bool, bool, bool],
) -> None:
    """Restore pipeline config to its pre-request state.

    Pairs with ``_apply_extension_overrides`` to ensure per-request overrides
    do not leak into subsequent requests sharing the singleton pipeline.
    """
    orig_retrieval, orig_vector, orig_reuse = snapshot
    cfg = intent_pipe._config
    restore_needed = False
    if cfg.retrieval.enable != orig_retrieval:
        cfg.retrieval.enable = orig_retrieval
        restore_needed = True
    if cfg.vector_fallback.enable != orig_vector:
        cfg.vector_fallback.enable = orig_vector
        restore_needed = True
    if cfg.reuse_strategy.enable != orig_reuse:
        cfg.reuse_strategy.enable = orig_reuse
        restore_needed = True
    if restore_needed:
        intent_pipe._init_extensions()


@app.post("/agent/intent", response_model=AgentIntentResponse, tags=["agent-intent"])
async def agent_intent(request: AgentIntentRequest):
    """Main entry point for Agent intent recognition.

    Orchestrates the full pipeline end-to-end:
    1. Normalize raw user input via two-stage normalization pipeline
       (pre-normalization + optional deep-normalization when `observation` is provided).
    2. If normalization pauses for clarification (low-confidence pronoun etc.),
       return early with `intent=None` and `paused_at_normalization=True` —
       the client should resolve the clarification before retrying.
    3. Otherwise feed the normalized text into the three-layer intent recognition
       waterfall (Code Layer → Lightweight LLM → Deep LLM) with slot filling
       and rejection/clarification dual exits.
    4. Return a unified response: intent + slots + constraints at the top level,
       plus the full normalization detail nested under `normalization`.

    Production callers should prefer this endpoint. The `/normalize` and
    `/recognize` sub-endpoints remain available for debugging or when only
    one stage is needed.
    """
    norm_pipe = get_pipeline()
    intent_pipe, _ = get_intent_pipeline()

    raw_input = request.raw_input
    pipeline_path: list[str] = []
    normalization_detail: AgentIntentNormalizationDetail | None = None
    normalized_text = raw_input
    paused_at_normalization = False
    skipped_normalization = request.skip_normalization

    # Stage 1: Normalization (skippable)
    if not skipped_normalization:
        pipeline_path.append("normalization")
        try:
            norm_result: PipelineResult = norm_pipe.process(
                raw_input=raw_input,
                session_id=request.session_id,
                user_id=request.user_id,
                turn=request.turn,
                observation=request.observation,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Normalization failed: {e}"
            ) from e

        normalization_detail = AgentIntentNormalizationDetail(
            normalized_input=norm_result.result.normalized_input,
            pronoun_resolutions=[pr.model_dump() for pr in norm_result.result.pronoun_resolutions],
            quantifiable_adjectives=[a.model_dump() for a in norm_result.result.quantifiable_adjectives],
            term_mappings=[t.model_dump() for t in norm_result.result.term_mappings],
            completions=[c.model_dump() for c in norm_result.result.completions],
            classification_tags=[t.value for t in norm_result.result.classification_tags],
            stage_reached=norm_result.stage_reached,
            paused_for_clarification=norm_result.paused_for_clarification,
            clarification=norm_result.clarification.model_dump() if norm_result.clarification else None,
            # D31: surface the normalized query string handed off to
            # intent recognition. verified/provisional evidence lists are
            # populated by the intent pipeline post-processing; they are
            # empty at the normalization stage and filled later when the
            # intent result is built.
            normalized_query=norm_result.result.normalized_input or raw_input,
            verified_evidence=[],
            provisional_evidence=[],
        )
        # Use normalized text (fall back to raw if normalization returned empty)
        normalized_text = norm_result.result.normalized_input or raw_input

        # If normalization paused for clarification, do NOT proceed to intent recognition.
        # The client must resolve the clarification first, then retry.
        if norm_result.paused_for_clarification:
            paused_at_normalization = True
            clar_question = None
            if norm_result.clarification is not None:
                clar_question = norm_result.clarification.question
            return AgentIntentResponse(
                intent=None,
                confidence=0.0,
                source="",  # No intent-recognition layer ran (paused at normalization)
                need_clarification=True,
                clarification_question=clar_question,
                # D31: protocol fields default to empty / False here because
                # intent recognition never ran.
                normalized_query=normalized_text,
                normalization=normalization_detail,
                raw_input=raw_input,
                normalized_input=normalized_text,
                pipeline_path=pipeline_path,
                skipped_normalization=skipped_normalization,
                paused_at_normalization=paused_at_normalization,
            )
    else:
        # Skipped normalization - use raw input directly as the recognition text
        normalized_text = raw_input

    # Stage 2: Intent recognition
    pipeline_path.append("intent-recognition")

    # Apply D17/D21/D22 per-request overrides on the pipeline config.
    # Snapshot the original config so it can be restored after recognition,
    # preventing per-request overrides from leaking into subsequent requests
    # that share the singleton pipeline.
    cfg_snapshot = _apply_extension_overrides(
        intent_pipe,
        request.enable_retrieval,
        request.enable_vector_fallback,
        request.reuse_previous_intent,
    )

    try:
        intent_result = intent_pipe.recognize(
            text=normalized_text,
            session_id=request.session_id,
            user_id=request.user_id or "default",
            turn=request.turn,
            context=request.context,
            event=request.event,
            dialogue_history=request.dialogue_history,
        )
    except Exception as e:
        logger.error("Intent recognition failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Intent recognition failed: {e}"
        ) from e
    finally:
        # Always restore config so overrides do not leak across requests.
        _restore_extension_config(intent_pipe, cfg_snapshot)

    return AgentIntentResponse(
        intent=intent_result.intent,
        confidence=intent_result.confidence,
        source=intent_result.source.value,
        slots=intent_result.slots,
        missing_slots=intent_result.missing_slots,
        need_clarification=intent_result.need_clarification,
        clarification_question=intent_result.clarification_question,
        hard_constraints=[c.model_dump() for c in intent_result.hard_constraints],
        soft_constraints=[c.model_dump() for c in intent_result.soft_constraints],
        rejection_reason=intent_result.rejection_reason,
        layer_reached=intent_result.layer_reached,
        sub_intents=intent_result.sub_intents,
        intent_switched=intent_result.intent_switched,
        previous_intent=intent_result.previous_intent,
        relations=[r.model_dump() for r in intent_result.relations],
        pending_intents=intent_result.pending_intents,
        # D26/D27/D31 structured output protocol fields
        normalized_query=intent_result.normalized_query or normalized_text,
        sub_tasks=intent_result.sub_tasks,
        independent_intents=intent_result.independent_intents,
        verified_evidence=[e.model_dump() for e in intent_result.verified_evidence],
        provisional_evidence=[e.model_dump() for e in intent_result.provisional_evidence],
        assumptions_disclosed=intent_result.assumptions_disclosed,
        assumptions=intent_result.assumptions,
        arbitration_breakdown=intent_result.arbitration_breakdown,
        normalization=normalization_detail,
        raw_input=raw_input,
        normalized_input=normalized_text,
        pipeline_path=pipeline_path,
        skipped_normalization=skipped_normalization,
        paused_at_normalization=paused_at_normalization,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run the server with uvicorn."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "user_input_normalization.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
