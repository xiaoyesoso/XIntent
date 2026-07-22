"""pre-normalization main normalizer (corresponds to D1-D8, D16).

Preliminary normalization processing before intent recognition, including:
- Pronoun resolution, omission completion, broken sentence correction, term standardization, jargon explanation, polysemous word disambiguation
- Structured output, completeness check
- Pronoun resolution table cross-turn reuse, indexing by name
- few-shot example retrieval injection
- Responsibility boundary constraints
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

from ..classification import InputClassifier
from ..clarification import ClarificationHandler
from ..config import Config, get_config
from ..context import ContextIntegrator
from ..llm.base import LLMClient
from ..models import (
    CompletionField,
    CompletenessStatus,
    FactType,
    FewShotExample,
    InputProblemType,
    KeyFact,
    ModifierExplanation,
    NormalizationResult,
    NormalizationStage,
    PronounResolution,
    QuantifiableAdjective,
    SubjectPredicateObject,
    TermMapping,
    UserProfile,
    VocabEntry,
)
from ..storage.base import (
    DialogueHistoryStore,
    FewShotStore,
    KeyFactStore,
    UserProfileStore,
    VocabStore,
)
from ..vocabulary import VocabularyTable
from .completeness_checker import CompletenessChecker
from .prompts import CANNOT_DO, CAN_DO, SYSTEM_PROMPT, format_context, format_fewshot, format_vocab


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class PreNormalizer:
    """pre-normalization main normalizer.

    Executes independently before intent recognition, serving as a general-purpose,
    C-end Agent-oriented independent normalization stage.
    Strictly adheres to responsibility boundaries: does not answer questions, does not execute tools,
    does not judge intent, does not fabricate facts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        key_fact_store: KeyFactStore,
        fewshot_store: FewShotStore,
        vocab_store: VocabStore | None = None,
        profile_store: UserProfileStore | None = None,
        dialogue_store: DialogueHistoryStore | None = None,
        config: Config | None = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or get_config()
        self._key_facts = key_fact_store
        self._fewshots = fewshot_store
        self._vocab_store = vocab_store
        self._profile_store = profile_store
        self._dialogue_store = dialogue_store

        # Assemble dependency modules
        self._classifier = InputClassifier(config=self._config)
        self._completeness = CompletenessChecker()
        self._clarification = ClarificationHandler(self._config)

        # Vocabulary table service (optional)
        self._vocab_table: VocabularyTable | None = None
        if vocab_store:
            self._vocab_table = VocabularyTable(vocab_store, self._config)

        # Context integrator (optional, requires three stores)
        self._context_integrator: ContextIntegrator | None = None
        if profile_store and key_fact_store and dialogue_store:
            self._context_integrator = ContextIntegrator(
                profile_store, key_fact_store, dialogue_store, self._config
            )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def normalize(
        self,
        raw_input: str,
        session_id: str,
        user_id: str | None = None,
        turn: int = 0,
    ) -> NormalizationResult:
        """Execute pre-normalization (main entry).

        Flow:
        1. Classify input problems
        2. Assemble context (three layers)
        3. Retrieve vocabulary table for injection
        4. Retrieve few-shot examples for injection
        5. Build responsibility boundary prompt
        6. Call the large model
        7. Parse structured output
        8. Completeness check
        9. Write pronoun resolution table to key facts
        10. Return result
        """
        # 1. Classify
        tags = self._classifier.classify(raw_input)

        # 2. Assemble context (task 3.1)
        context_bundle = None
        if self._context_integrator and user_id:
            context_bundle = self._context_integrator.assemble(
                session_id, user_id, raw_input
            )

        # 3. Vocabulary table retrieval injection (task 3.2)
        vocab_entries: list[VocabEntry] = []
        if self._vocab_table:
            industry = "通用"
            if context_bundle and context_bundle.user_profile:
                industry = context_bundle.user_profile.industry
            vocab_entries = self._vocab_table.inject_context(
                raw_input, industry, user_id
            )

        # 4. few-shot example retrieval injection (task 3.3)
        fewshot_examples = []
        if self._config.fewshot.enabled:
            fewshot_examples = self._fewshots.search(
                raw_input, top_k=self._config.fewshot.top_k
            )

        # 5. Build prompt (task 3.4 responsibility boundary)
        user_prompt = self._build_user_prompt(
            raw_input, context_bundle, vocab_entries, fewshot_examples
        )

        # 6. Call the large model
        llm_response = self._llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        # 7. Parse structured output (task 3.10)
        result = self._parse_llm_output(llm_response, raw_input, tags)

        # 8. Completeness check (task 3.11)
        check, route_to, route_reason = self._completeness.check_and_route(result)
        result.completeness = check
        if route_to:
            result.route_to = route_to

        # 9. Write pronoun resolution table to key facts (task 3.12)
        if result.pronoun_resolutions:
            self._write_pronoun_facts(session_id, turn, result.pronoun_resolutions)

        # 10. Naming convention check (task 3.13)
        self._ensure_named_entities(result)

        # 11. Sink strange inputs as few-shot (task 3.14)
        if self._is_strange_input(raw_input, tags):
            self._sink_to_fewshot(raw_input, result, tags)

        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        raw_input: str,
        context_bundle: Any,
        vocab_entries: list[VocabEntry],
        fewshot_examples: list[FewShotExample],
    ) -> str:
        """Build user prompt (inject context + vocabulary + few-shot + responsibility boundary)."""
        parts: list[str] = []

        # Context
        if context_bundle:
            parts.append(format_context(context_bundle))

        # Vocabulary
        if vocab_entries:
            parts.append(format_vocab(vocab_entries))

        # few-shot examples
        if fewshot_examples:
            parts.append(format_fewshot(fewshot_examples))

        # User input
        parts.append(f"# 用户输入\n{raw_input}")

        # Responsibility boundary reinforcement
        parts.append(f"\n# 再次强调职责边界\n{CANNOT_DO}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM output parsing (task 3.10 structured output)
    # ------------------------------------------------------------------

    def _parse_llm_output(
        self,
        llm_response: str,
        raw_input: str,
        tags: list[InputProblemType],
    ) -> NormalizationResult:
        """Parse large model output into NormalizationResult."""
        # Try to extract JSON
        data = self._extract_json(llm_response)

        if not data:
            # Parsing failed, return minimal result
            return NormalizationResult(
                raw_input=raw_input,
                normalized_input=raw_input,
                classification_tags=tags,
            )

        # Parse each field
        spo_data = data.get("spo", {})
        modifiers_data = data.get("modifiers", {})
        pronoun_data = data.get("pronoun_resolutions", [])
        adj_data = data.get("quantifiable_adjectives", [])
        term_data = data.get("term_mappings", [])
        completion_data = data.get("completions", [])

        pronoun_resolutions = [
            PronounResolution(
                pronoun=p.get("pronoun") or "",
                resolved_to=p.get("resolved_to") or "",
                confidence=float(p.get("confidence") or 0.5),
                evidence_source=p.get("evidence_source") or "",
                named_entity=p.get("named_entity"),
            )
            for p in pronoun_data
            # Skip entries where pronoun is empty (LLM returned null/empty)
            if p.get("pronoun")
        ]

        quantifiable_adjectives = [
            QuantifiableAdjective(
                adjective=a.get("adjective") or "",
                quantified=a.get("quantified", False),
                quantified_value=a.get("quantified_value"),
                route_to=NormalizationStage.DEEP
                if not a.get("quantified", False)
                else None,
            )
            for a in adj_data
            if a.get("adjective")
        ]

        term_mappings = [
            TermMapping(
                original=t.get("original") or "",
                standard=t.get("standard") or "",
                source=t.get("source") or "vocabulary-table",
            )
            for t in term_data
        ]

        completions = [
            CompletionField(
                field=c.get("field") or "",
                content=c.get("content") or "",
                source=c.get("source") or "",
            )
            for c in completion_data
            if c.get("field") or c.get("content")
        ]

        return NormalizationResult(
            raw_input=raw_input,
            normalized_input=data.get("normalized_input") or raw_input,
            spo=SubjectPredicateObject(
                subject=spo_data.get("subject"),
                subject_source=spo_data.get("subject_source"),
                predicate=spo_data.get("predicate"),
                obj=spo_data.get("obj"),
                obj_source=spo_data.get("obj_source"),
            ),
            modifiers=ModifierExplanation(
                attributive=modifiers_data.get("attributive"),
                adverbial=modifiers_data.get("adverbial"),
                complement=modifiers_data.get("complement"),
            ),
            pronoun_resolutions=pronoun_resolutions,
            quantifiable_adjectives=quantifiable_adjectives,
            term_mappings=term_mappings,
            completions=completions,
            classification_tags=tags,
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract JSON object from text."""
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract ```json ... ``` block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to extract the first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # Write pronoun resolution table to key facts (task 3.12 cross-turn reuse)
    # ------------------------------------------------------------------

    def _write_pronoun_facts(
        self,
        session_id: str,
        turn: int,
        resolutions: list[PronounResolution],
    ) -> None:
        """Write the pronoun resolution table to key fact storage for cross-turn reuse."""
        for pr in resolutions:
            # Check if it already exists (cross-turn reuse)
            existing = self._key_facts.find_pronoun_resolution(
                session_id, pr.pronoun
            )
            if existing:
                # Update confidence (take the higher one)
                if pr.confidence > existing.confidence:
                    existing.confidence = pr.confidence
                    existing.content = {
                        "pronoun": pr.pronoun,
                        "resolved_to": pr.resolved_to,
                        "confidence": pr.confidence,
                        "evidence_source": pr.evidence_source,
                        "named_entity": pr.named_entity,
                    }
                    existing.created_at = datetime.now()
                continue

            # Create new key fact
            fact = KeyFact(
                fact_id=_gen_id("fact"),
                session_id=session_id,
                turn=turn,
                fact_type=FactType.PRONOUN_RESOLUTION,
                content={
                    "pronoun": pr.pronoun,
                    "resolved_to": pr.resolved_to,
                    "confidence": pr.confidence,
                    "evidence_source": pr.evidence_source,
                    "named_entity": pr.named_entity,
                },
                confidence=pr.confidence,
            )
            self._key_facts.save(fact)

    # ------------------------------------------------------------------
    # Naming convention (task 3.13 indexing by name)
    # ------------------------------------------------------------------

    def _ensure_named_entities(self, result: NormalizationResult) -> None:
        """Assign semantic names to resolved entities (indexing by name D7).

        If the resolution result has no named_entity, attempt to generate one from resolved_to.
        """
        for pr in result.pronoun_resolutions:
            if not pr.named_entity and pr.resolved_to:
                # Use resolved_to as the named entity
                pr.named_entity = pr.resolved_to

    # ------------------------------------------------------------------
    # Sink strange inputs as few-shot (task 3.14)
    # ------------------------------------------------------------------

    def _is_strange_input(
        self, raw_input: str, tags: list[InputProblemType]
    ) -> bool:
        """Determine whether the input is strange (worth sinking as a few-shot example)."""
        # Inputs with anaphora or semantic problems are usually "stranger"
        strange_types = {InputProblemType.ANAPHORA, InputProblemType.SEMANTIC}
        return bool(strange_types & set(tags))

    def _sink_to_fewshot(
        self,
        raw_input: str,
        result: NormalizationResult,
        tags: list[InputProblemType],
    ) -> None:
        """Sink strange inputs as few-shot examples into the store."""
        # Check if similar examples already exist
        existing = self._fewshots.search(raw_input, top_k=1)
        if existing:
            # Similar example already exists, do not duplicate
            return

        example = FewShotExample(
            example_id=_gen_id("ex"),
            input_type=tags,
            input=raw_input,
            context_summary="",
            normalized_output=result.model_dump(),
        )
        self._fewshots.save(example)

    # ------------------------------------------------------------------
    # Clarification mechanism integration
    # ------------------------------------------------------------------

    def check_clarification(
        self,
        session_id: str,
        result: NormalizationResult,
    ) -> list:
        """Check whether the normalization result needs to trigger clarification."""
        return self._clarification.check_result(session_id, result)

    @property
    def clarification_handler(self) -> ClarificationHandler:
        """Get the clarification handler."""
        return self._clarification

    # ------------------------------------------------------------------
    # Responsibility boundary validation (for regression testing)
    # ------------------------------------------------------------------

    def validate_boundary(self, result: NormalizationResult) -> dict[str, bool]:
        """Verify whether the normalization result violates responsibility boundaries.

        Used for regression testing: ensures pre-normalization does not overstep its authority.
        """
        violations: dict[str, bool] = {
            "answered_question": False,  # Should not directly answer questions
            "executed_tool": False,  # Should not execute tools
            "made_recommendation": False,  # Should not make recommendations
            "fabricated_facts": False,  # Should not fabricate facts
        }

        normalized = result.normalized_input.lower()

        # Check for signs of direct answering (e.g. specific values, recommendation language)
        # Note: this is a heuristic check, not fully accurate
        recommendation_patterns = [
            r"推荐.*给.*你",
            r"建议.*选择",
            r"答案.*是",
        ]
        for pattern in recommendation_patterns:
            if re.search(pattern, normalized):
                violations["answered_question"] = True

        return violations
