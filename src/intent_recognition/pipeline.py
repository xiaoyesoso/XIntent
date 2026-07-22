"""Intent recognition pipeline - orchestrates L1 -> L2 -> L3 -> Slot Filling -> Rejection/Clarification (D1, D2)."""

from __future__ import annotations

import logging
from typing import Any

from user_input_normalization.llm.base import LLMClient

from .code_layer import CodeLayerClassifier
from .config import IntentRecognitionConfig
from .deep_llm import DeepLLMClassifier
from .intent_registry import IntentRegistry
from .lightweight_llm import LightweightLLMClassifier
from .lightweight_llm.confidence_router import ConfidenceRouter
from .models import (
    Evidence,
    EvidenceGrade,
    IntentRecognitionResult,
    RecognitionSource,
    SlotState,
    SlotValue,
)
from .rejection_clarification import RejectionClarificationHandler
from .slot_filling import ConstraintExtractor, CrossTurnSlotMerger, SlotExtractor
from .storage import (
    MemoryFewShotStore,
    MemoryIntentHistoryStore,
    MemorySlotStateStore,
    MemoryVectorMatchStore,
    FewShotStore,
    IntentHistoryStore,
    SlotStateStore,
    VectorMatchStore,
)

logger = logging.getLogger(__name__)


class IntentRecognitionPipeline:
    """Three-layer waterfall intent recognition pipeline.

    Flow:
        [D22 Reuse?] -> L1 (Code + D21 Vector) -> [D17 Retrieval?] -> L2 (Light LLM + D18 FewShot + D23 Arbiter) -> L3 (Deep LLM + D20 MultiIntent) -> Slot Filling -> Rejection/Clarification

    Designed to run BEFORE ReAct/TAO loop, consuming normalized input
    from user-input-normalization pipeline.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: IntentRegistry,
        config: IntentRecognitionConfig | None = None,
        *,
        slot_state_store: SlotStateStore | None = None,
        intent_history_store: IntentHistoryStore | None = None,
        deep_llm_client: LLMClient | None = None,
        fewshot_store: FewShotStore | None = None,
        vector_match_store: VectorMatchStore | None = None,
    ) -> None:
        self._config = config or IntentRecognitionConfig()
        self._registry = registry
        self._llm_client = llm_client
        self._slot_store = slot_state_store or MemorySlotStateStore()
        self._history_store = intent_history_store or MemoryIntentHistoryStore()
        self._fewshot_store = fewshot_store or MemoryFewShotStore()
        self._vector_match_store = vector_match_store or MemoryVectorMatchStore()

        # Layer 1: Code-based
        self._code_layer = CodeLayerClassifier()

        # Layer 2: Lightweight LLM
        self._lightweight_llm = LightweightLLMClassifier(
            llm_client=llm_client,
            registry=registry,
            config=self._config,
        )

        # Layer 3: Deep LLM (may use different model)
        self._deep_llm = DeepLLMClassifier(
            llm_client=deep_llm_client or llm_client,
            registry=registry,
            config=self._config,
        )

        # Slot filling
        self._slot_extractor = SlotExtractor(llm_client=llm_client, registry=registry)
        self._constraint_extractor = ConstraintExtractor(llm_client=llm_client)
        self._slot_merger = CrossTurnSlotMerger(self._slot_store)

        # Rejection / Clarification
        self._reject_clarify = RejectionClarificationHandler(
            registry=registry,
            history_store=self._history_store,
            config=self._config,
        )

        # Confidence router for L2
        self._confidence_router = ConfidenceRouter(self._config.confidence)

        # D17-D24 extension modules (lazily initialized, None when disabled)
        self._candidate_retriever = None
        self._dynamic_fewshot_injector = None
        self._vector_matcher = None
        self._reuse_strategy = None
        self._arbiter = None
        self._init_extensions()

    def _init_extensions(self) -> None:
        """Initialize D17-D24 extension modules based on config.

        All extension modules are reset to ``None`` first so that toggling
        a config flag from enabled back to disabled actually releases the
        previously created module. Without this reset, per-request overrides
        that re-init extensions would leave stale modules attached to the
        singleton pipeline.
        """
        # Reset all extension modules so disabled flags truly disable them.
        self._candidate_retriever = None
        self._dynamic_fewshot_injector = None
        self._vector_matcher = None
        self._reuse_strategy = None
        self._arbiter = None

        # D17: Candidate retriever
        if self._config.retrieval.enable:
            try:
                from .lightweight_llm.candidate_retriever import create_retriever
                self._candidate_retriever = create_retriever(
                    self._config.retrieval, self._llm_client
                )
            except Exception as e:
                logger.warning(f"D17 candidate retriever init failed: {e}")

        # D18: Dynamic few-shot injector
        if self._config.dynamic_fewshot.dynamic_enabled:
            try:
                from .lightweight_llm.dynamic_fewshot import DynamicFewShotInjector
                self._dynamic_fewshot_injector = DynamicFewShotInjector(
                    self._fewshot_store, self._config.dynamic_fewshot
                )
            except Exception as e:
                logger.warning(f"D18 dynamic fewshot init failed: {e}")

        # D21: Vector matching fallback
        if self._config.vector_fallback.enable:
            try:
                from .code_layer.vector_matcher import VectorMatcher
                self._vector_matcher = VectorMatcher(
                    self._vector_match_store, self._config.vector_fallback
                )
            except Exception as e:
                logger.warning(f"D21 vector matcher init failed: {e}")

        # D22: Intent reuse strategy
        if self._config.reuse_strategy.enable:
            try:
                from .intent_reuse_strategy import IntentReuseStrategy
                self._reuse_strategy = IntentReuseStrategy(
                    self._history_store,
                    self._config.reuse_strategy,
                    self._config.failure_signals,
                )
            except Exception as e:
                logger.warning(f"D22 reuse strategy init failed: {e}")

        # D23: Multi-recognizer arbiter
        if self._config.arbiter.enable:
            try:
                from .multi_recognizer_arbiter import MultiRecognizerArbiter
                self._arbiter = MultiRecognizerArbiter(self._config.arbiter)
                # Register built-in recognizers
                if self._vector_matcher is not None:
                    self._arbiter.register("vector", self._vector_matcher.match)
                self._arbiter.register("rule", lambda text, ctx=None: self._code_layer.rule_engine.match(text, ctx or {}))
                self._arbiter.register("lightweight_llm", lambda text, ctx=None: self._lightweight_llm.classify(text, ctx))
            except Exception as e:
                logger.warning(f"D23 arbiter init failed: {e}")

    def recognize(
        self,
        text: str,
        session_id: str = "default",
        user_id: str = "default",
        turn: int = 1,
        context: dict[str, Any] | None = None,
        event: str | None = None,
        dialogue_history: list[dict] | None = None,
    ) -> IntentRecognitionResult:
        """Run the three-layer waterfall recognition pipeline.

        Args:
            text: Normalized user input (from user-input-normalization)
            session_id: Session identifier for cross-turn state
            user_id: User identifier
            turn: Turn number in the conversation
            context: Additional context (page state, user profile, etc.)
            event: UI event (e.g., "click:next") for page guidance
            dialogue_history: Previous dialogue turns for deep LLM

        Returns:
            IntentRecognitionResult with intent, slots, constraints, etc.
        """
        ctx = context or {}

        # D31: The normalized query string is stamped onto each result in
        # ``_post_process`` via the ``text`` argument; no extra bookkeeping
        # is needed here.

        # Check for implicit failure signal from previous turn (D11)
        if self._config.enable_implicit_eval and turn > 1:
            if self._reject_clarify.check_implicit_failure(text):
                logger.warning(
                    f"Implicit failure signal detected in turn {turn} for session {session_id}"
                )

        # Get previous intent for switch detection
        previous_intent = self._history_store.get_previous_intent(session_id)

        # D22: Intent reuse early exit (before L1)
        if self._reuse_strategy is not None:
            reused = self._reuse_strategy.try_reuse(session_id, text, turn)
            if reused is not None:
                logger.info(f"D22 reused previous intent: {reused.intent}")
                self._history_store.add(session_id, reused, turn)
                return reused

        # Layer 1: Code-based recognition (+ D21 vector fallback)
        result: IntentRecognitionResult | None = None
        if self._config.enable_code_layer:
            result = self._code_layer.classify(text, ctx, event)

            # D21: Vector matching fallback (if L1 code layer missed)
            if result is None and self._vector_matcher is not None:
                result = self._vector_matcher.match(text)

        if result is not None:
            # L1 hit - fast path
            result = self._post_process(result, session_id, turn, text)
            self._history_store.add(session_id, result, turn)
            return result

        # D17: Candidate narrowing (before L2)
        candidate_intents: list | None = None
        if self._candidate_retriever is not None:
            try:
                all_intents = self._registry.list_all()
                candidate_intents = self._candidate_retriever.retrieve(text, all_intents, ctx)
                logger.info(f"D17 narrowed to {len(candidate_intents)} candidates")
            except Exception as e:
                logger.warning(f"D17 retrieval failed: {e}")

        # Layer 2: Lightweight LLM (+ D23 arbiter, D18 dynamic fewshot)
        l2_result: IntentRecognitionResult | None = None
        if self._config.enable_lightweight_llm:
            # D23: Multi-recognizer arbitration
            if self._arbiter is not None:
                try:
                    arbiter_result = self._arbiter.arbitrate(text, ctx)
                    if arbiter_result is not None:
                        result = self._post_process(arbiter_result, session_id, turn, text)
                        self._history_store.add(session_id, result, turn)
                        return result
                except Exception as e:
                    logger.warning(f"D23 arbiter failed: {e}, falling back to L2")

            # D18: Dynamic few-shot injection
            fewshot_str = ""
            if self._dynamic_fewshot_injector is not None:
                try:
                    fewshot_str = self._dynamic_fewshot_injector.inject(text)
                except Exception as e:
                    logger.warning(f"D18 fewshot injection failed: {e}")

            result = self._lightweight_llm.classify(text, ctx)

            # Apply confidence routing
            route = self._confidence_router.route(result.confidence)

            if route == "accept":
                result = self._post_process(result, session_id, turn, text)
                self._history_store.add(session_id, result, turn)
                return result

            if route == "clarify":
                # Medium confidence - try slot filling, may need clarification
                result = self._post_process(result, session_id, turn, text)
                if result.need_clarification:
                    self._history_store.add(session_id, result, turn)
                    return result
                # If no clarification needed, accept
                self._history_store.add(session_id, result, turn)
                return result

            # route == "escalate" -> fall through to L3.
            # D28: Keep L2 result for cross-layer arbitration below.
            l2_result = result
            logger.info(f"L2 confidence {result.confidence:.2f} < threshold, escalating to L3")

        # Layer 3: Deep reasoning LLM
        if self._config.enable_deep_llm:
            try:
                l3_result = self._deep_llm.classify(
                    text,
                    context=ctx,
                    dialogue_history=dialogue_history,
                    previous_intent=previous_intent,
                )
            except Exception as e:
                logger.error(f"L3 failed: {e}")
                # Graceful degradation: fallback to L2 result if available
                if result is not None:
                    logger.warning("Degrading to L2 result after L3 failure")
                    result = self._post_process(result, session_id, turn, text)
                    self._history_store.add(session_id, result, turn)
                    return result
                # If no L2 result either, return unsupported
                return self._reject_clarify.handle_unsupported(text)

            # D28 task 4.7: Three-layer result-disagreement arbitration.
            # L1 already early-returned on hit; by here L1 has missed.
            # If L2 and L3 disagree on intent, arbitrate via five factors.
            if (
                self._config.arbitration.enable_five_factor
                and l2_result is not None
                and l2_result.intent is not None
                and l3_result.intent is not None
                and l2_result.intent != l3_result.intent
            ):
                arb = self._confidence_router.arbitrate_l2_l3(l2_result, l3_result)
                logger.info(
                    f"D28 L2/L3 disagreement arbitrated to '{arb.decision}': "
                    f"{arb.reason}"
                )
                if arb.decision == "clarify":
                    # Force clarification instead of accepting L3
                    l3_result.need_clarification = True
                    if not l3_result.clarification_question:
                        l3_result.clarification_question = (
                            "请补充更多信息以确认您的意图。"
                        )
                # If "accept", L3 result stands. If "escalate", we still
                # return L3 since there's no L4; pipeline degrades gracefully.

            result = self._post_process(l3_result, session_id, turn, text)
            self._history_store.add(session_id, result, turn)
            return result

        # All layers disabled or failed
        if result is not None:
            result = self._post_process(result, session_id, turn, text)
            self._history_store.add(session_id, result, turn)
            return result

        return self._reject_clarify.handle_unsupported(text)

    def _post_process(
        self,
        result: IntentRecognitionResult,
        session_id: str,
        turn: int,
        text: str,
    ) -> IntentRecognitionResult:
        """Post-process: slot filling, constraint extraction, cross-turn merge."""
        # D31: Stamp the normalized query string onto every result so the
        # server can surface it in the unified response.
        if not result.normalized_query:
            result.normalized_query = text

        if result.intent is None:
            # Unsupported - check if we should reject
            if result.rejection_reason is None:
                return self._reject_clarify.handle_unsupported(text)
            return result

        # Slot filling
        if result.intent and self._registry.has_intent(result.intent):
            # Extract additional slots if not already done
            if not result.slots:
                try:
                    slots, missing = self._slot_extractor.extract(text, result.intent)
                    result.slots = slots
                    result.missing_slots = missing
                except Exception as e:
                    logger.warning(f"Slot extraction failed: {e}")

            # Extract constraints if not already done
            if not result.hard_constraints and not result.soft_constraints:
                hard, soft = self._constraint_extractor.extract_separated(text)
                result.hard_constraints = hard
                result.soft_constraints = soft

            # Cross-turn merge
            state = self._slot_merger.merge(
                session_id=session_id,
                new_slots=result.slots,
                turn=turn,
                intent=result.intent,
                hard_constraints=result.hard_constraints,
                soft_constraints=result.soft_constraints,
            )

            # Update missing slots from accumulated state
            state = self._slot_merger.update_missing_slots(
                state, result.intent, self._registry
            )
            result.missing_slots = state.missing_slots
            result.slots = state.slots
            result.hard_constraints = state.hard_constraints
            result.soft_constraints = state.soft_constraints

            # D26: Collect evidence for the recognized intent
            if self._config.evidence.enable_grading:
                self._collect_evidence(result, text, session_id, turn)
                # D26: Hard-operation pre-check
                self._check_hard_op_evidence(result, session_id)

            # Check if clarification needed
            if result.missing_slots:
                # Check max clarifications
                if not self._reject_clarify.check_max_clarifications(session_id):
                    result.need_clarification = True
                    result.clarification_question = self._generate_clarification_question(
                        result.missing_slots
                    )
                else:
                    # Max clarifications exceeded - degrade to best guess
                    logger.warning(
                        f"Max clarifications exceeded for session {session_id}, "
                        "accepting best guess"
                    )

            # Check intent switch
            previous = self._history_store.get_previous_intent(session_id)
            if previous and previous != result.intent:
                result.intent_switched = True
                result.previous_intent = previous

        return result

    # ------------------------------------------------------------------
    # D26: Evidence grading (verified / provisional)
    # ------------------------------------------------------------------
    def _collect_evidence(
        self,
        result: IntentRecognitionResult,
        text: str,
        session_id: str,
        turn: int,
    ) -> None:
        """Collect verified and provisional evidence for the result.

        Verified evidence comes from the current input and current context.
        Provisional evidence comes from historical profile, speculation, or
        default values. Slot values extracted from the current input are
        tagged ``verified`` by default (via ``get_slot_evidence``); slots
        filled from cross-turn state that originated from profile/default
        are tagged ``provisional``.
        """
        verified: list[Evidence] = []
        provisional: list[Evidence] = []

        # Current input is always verified evidence
        if text:
            verified.append(Evidence(
                content=f"当前输入：{text}",
                grade=EvidenceGrade.VERIFIED,
                source="current_input",
            ))

        # Slots extracted from current input default to verified.
        # Slots that came from cross-turn state and are NOT in the current
        # text are treated as provisional (they may have originated from
        # profile or earlier speculation).
        if result.slots:
            for name, value in result.slots.items():
                if value is None:
                    continue
                # If value is already a SlotValue, respect its grade
                if isinstance(value, SlotValue):
                    if value.evidence_grade == EvidenceGrade.PROVISIONAL:
                        provisional.append(Evidence(
                            content=f"槽位 {name}={value.value}（来自历史/推测）",
                            grade=EvidenceGrade.PROVISIONAL,
                            source="user_profile",
                        ))
                    else:
                        verified.append(Evidence(
                            content=f"槽位 {name}={value.value}",
                            grade=EvidenceGrade.VERIFIED,
                            source="current_input",
                        ))
                elif isinstance(value, dict) and value.get("evidence_grade") == "provisional":
                    provisional.append(Evidence(
                        content=f"槽位 {name}={value.get('value')}（来自历史/推测）",
                        grade=EvidenceGrade.PROVISIONAL,
                        source="user_profile",
                    ))
                else:
                    # Plain value from current input: verified by default
                    verified.append(Evidence(
                        content=f"槽位 {name}={value}",
                        grade=EvidenceGrade.VERIFIED,
                        source="current_input",
                    ))

        result.verified_evidence = verified
        result.provisional_evidence = provisional

    def _upgrade_provisional_to_verified(
        self,
        result: IntentRecognitionResult,
        slot_name: str,
        confirmed_value: str,
        source: str = "user_confirmation",
    ) -> None:
        """D26: Upgrade a provisional slot to verified.

        Supports three upgrade paths:
        - ``user_confirmation``: user explicitly confirmed ("对，我要 M 码")
        - ``clarification_reply``: user answered a clarification question
        - ``observation_check``: Observation verified the value

        Verified evidence CANNOT be downgraded to provisional.
        """
        # Update the slot value to verified
        value = result.slots.get(slot_name)
        if value is None:
            return
        if isinstance(value, SlotValue):
            value.evidence_grade = EvidenceGrade.VERIFIED
            value.evidence_ref = f"用户确认：{confirmed_value}"
        elif isinstance(value, dict):
            value["evidence_grade"] = EvidenceGrade.VERIFIED.value
            value["evidence_ref"] = f"用户确认：{confirmed_value}"
        else:
            # Wrap plain value into SlotValue with verified grade
            result.slots[slot_name] = SlotValue(
                value=confirmed_value or str(value),
                evidence_grade=EvidenceGrade.VERIFIED,
                evidence_ref=f"用户确认：{confirmed_value}",
            )

        # Move the evidence from provisional to verified
        new_verified: list[Evidence] = list(result.verified_evidence)
        new_provisional: list[Evidence] = []
        for ev in result.provisional_evidence:
            if slot_name in ev.content:
                new_verified.append(Evidence(
                    content=f"用户确认：{slot_name}={confirmed_value}",
                    grade=EvidenceGrade.VERIFIED,
                    source=source,
                ))
            else:
                new_provisional.append(ev)
        result.verified_evidence = new_verified
        result.provisional_evidence = new_provisional

    def _check_hard_op_evidence(
        self,
        result: IntentRecognitionResult,
        session_id: str,
    ) -> None:
        """D26: Check evidence grades before hard operations.

        When ``require_verified_for_hard_ops=True`` and the intent is in
        ``high_risk_intents``, all required slots must be ``verified``.
        If any required slot is ``provisional``, either trigger clarification
        or disclose assumptions.
        """
        cfg = self._config.evidence
        if not cfg.require_verified_for_hard_ops:
            return
        if not result.intent:
            return
        if result.intent not in cfg.high_risk_intents:
            return

        # Find provisional slots among required slots
        required = self._registry.get_required_slots(result.intent)
        provisional_slots: list[str] = []
        for slot_name in required:
            sv = result.get_slot_evidence(slot_name)
            if sv is not None and sv.evidence_grade == EvidenceGrade.PROVISIONAL:
                provisional_slots.append(slot_name)

        if not provisional_slots:
            return

        # Trigger clarification for the first provisional slot
        if not self._reject_clarify.check_max_clarifications(session_id):
            result.need_clarification = True
            result.clarification_question = self._generate_clarification_question(
                provisional_slots
            )
        else:
            # Max clarifications exceeded: disclose assumptions instead
            result.assumptions_disclosed = True
            result.assumptions = [
                f"假设 {name}={result.slots.get(name)}（来自历史画像）"
                for name in provisional_slots
            ]

    @staticmethod
    def _generate_clarification_question(missing_slots: list[str]) -> str:
        """Generate a clarification question for missing slots."""
        if not missing_slots:
            return ""
        if len(missing_slots) == 1:
            slot = missing_slots[0]
            if "budget" in slot or "price" in slot:
                return f"您的{slot}大概是多少？"
            if "category" in slot:
                return f"您希望提供哪类{slot}？"
            return f"请提供{slot}信息。"
        return f"请补充以下信息：{'、'.join(missing_slots)}"

    def get_slot_state(self, session_id: str) -> SlotState | None:
        """Get current slot state for a session."""
        return self._slot_store.get(session_id)

    def get_intent_history(self, session_id: str) -> list[IntentRecognitionResult]:
        """Get intent recognition history for a session."""
        return self._history_store.get_history(session_id)

    def reset_session(self, session_id: str) -> None:
        """Reset all state for a session."""
        self._slot_store.delete(session_id)
