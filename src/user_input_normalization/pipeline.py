"""User input normalization pipeline orchestrator (corresponds to D1 two-stage pipeline).

Chains classification -> pre-normalization -> deep-normalization into a complete pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .classification import InputClassifier
from .clarification import ClarificationHandler
from .config import Config, get_config
from .context import ContextIntegrator
from .deep_normalization import DeepNormalizer
from .llm.base import LLMClient
from .models import (
    ClarificationRequest,
    InputProblemType,
    NormalizationResult,
    NormalizationStage,
)
from .pre_normalization import PreNormalizer
from .quantification import QuantificationEngine
from .storage.base import (
    DialogueHistoryStore,
    FewShotStore,
    KeyFactStore,
    UserProfileStore,
    VocabStore,
)


class PipelineResult(BaseModel):
    """Pipeline processing result."""

    result: NormalizationResult = Field(description="规范化结果")
    clarification: ClarificationRequest | None = Field(
        default=None, description="澄清请求（如有）"
    )
    stage_reached: str = Field(description="达到的阶段：classification / pre / deep")
    paused_for_clarification: bool = Field(
        default=False, description="是否暂停等待澄清"
    )


class NormalizationPipeline:
    """User input normalization pipeline (corresponds to D1 two-stage pipeline).

    Flow:
    1. Classification (input-classification)
    2. pre-normalization (before intent recognition)
    3. Completeness check -> routing
    4. If clarification needed -> pause and return clarification request
    5. If routed to deep -> deep-normalization (within ReAct loop)
    6. Return final result
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
        self._config = config or get_config()
        self._llm = llm_client

        # Initialize stage modules
        self._classifier = InputClassifier(config=self._config)
        self._pre_normalizer = PreNormalizer(
            llm_client=llm_client,
            key_fact_store=key_fact_store,
            fewshot_store=fewshot_store,
            vocab_store=vocab_store,
            profile_store=profile_store,
            dialogue_store=dialogue_store,
            config=self._config,
        )
        self._clarification = self._pre_normalizer.clarification_handler

        # deep-normalization requires quantification engine
        self._quant_engine = QuantificationEngine(
            llm_client=llm_client, vocab_store=vocab_store, config=self._config
        )
        self._deep_normalizer = DeepNormalizer(
            llm_client=llm_client,
            quantification_engine=self._quant_engine,
            key_fact_store=key_fact_store,
            dialogue_store=dialogue_store,
            config=self._config,
        )

        # Attribute retrieval pronoun resolution
        from .attribute_resolution import AttributeResolver

        self._attribute_resolver = AttributeResolver(
            llm_client=llm_client,
            dialogue_store=dialogue_store,
            key_fact_store=key_fact_store,
            config=self._config,
        )

        self._dialogue_store = dialogue_store

    def process(
        self,
        raw_input: str,
        session_id: str,
        user_id: str | None = None,
        turn: int = 0,
        observation: dict | None = None,
    ) -> PipelineResult:
        """Execute the complete normalization pipeline.

        Args:
            raw_input: user raw input
            session_id: session ID
            user_id: user ID (optional)
            turn: dialogue turn
            observation: Observation in the ReAct loop (optional, used for deep stage)

        Returns:
            PipelineResult: contains normalization result, clarification request (if any), and stage reached
        """
        # 1. Classify
        tags = self._classifier.classify(raw_input)

        # 2. pre-normalization
        pre_result = self._pre_normalizer.normalize(
            raw_input=raw_input,
            session_id=session_id,
            user_id=user_id,
            turn=turn,
        )

        stage_reached = "pre"

        # 3. Check whether clarification is needed
        clarifications = self._clarification.check_result(session_id, pre_result)
        if clarifications and self._config.pipeline.agent_type != "domain":
            return PipelineResult(
                result=pre_result,
                clarification=clarifications[0],
                stage_reached=stage_reached,
                paused_for_clarification=True,
            )

        # 4. Route to deep-normalization
        if (
            pre_result.route_to == NormalizationStage.DEEP
            and self._config.pipeline.enable_deep_normalization
        ):
            deep_result = self._deep_normalizer.process(
                session_id=session_id,
                turn=turn,
                pre_result=pre_result,
                user_id=user_id,
                observation=observation,
            )
            stage_reached = "deep"
            return PipelineResult(
                result=deep_result,
                clarification=None,
                stage_reached=stage_reached,
                paused_for_clarification=False,
            )

        # 5. Check whether attribute retrieval pronoun resolution is needed (long-span attribute anaphora)
        if (
            self._config.pipeline.enable_attribute_resolution
            and InputProblemType.ANAPHORA in tags
            and self._is_attribute_anaphora(raw_input)
        ):
            attr_result = self._attribute_resolver.resolve(
                session_id=session_id,
                user_input=raw_input,
                user_id=user_id,
            )
            if attr_result.confidence > self._config.clarify.theta_clarify:
                # Append attribute resolution result to pre_result
                pre_result.pronoun_resolutions.append(
                    __import__(
                        "user_input_normalization.models",
                        fromlist=["PronounResolution"],
                    ).PronounResolution(
                        pronoun=attr_result.pronoun,
                        resolved_to=attr_result.resolved_to,
                        confidence=attr_result.confidence,
                        evidence_source="属性检索指代消解",
                    )
                )

        return PipelineResult(
            result=pre_result,
            clarification=None,
            stage_reached=stage_reached,
            paused_for_clarification=False,
        )

    def resume_after_clarification(
        self,
        session_id: str,
        user_response: str,
        original_input: str,
        user_id: str | None = None,
        turn: int = 0,
    ) -> PipelineResult:
        """Resume processing after clarification.

        After the user answers the clarification question, merge the original input with the user response and reprocess.
        """
        self._clarification.receive_response(session_id, user_response)
        # Merge user response as supplementary context and reprocess
        merged_input = f"{original_input}（用户澄清：{user_response}）"
        return self.process(
            raw_input=merged_input,
            session_id=session_id,
            user_id=user_id,
            turn=turn,
        )

    def react_step(
        self,
        session_id: str,
        turn: int,
        pre_result: NormalizationResult,
        thought: str,
        action: str,
        observation: Any,
    ) -> NormalizationResult:
        """ReAct loop single step: embeds deep normalization within Thought -> Action -> Observation."""
        return self._deep_normalizer.react_step(
            session_id=session_id,
            turn=turn,
            pre_result=pre_result,
            thought=thought,
            action=action,
            observation=observation,
        )

    @staticmethod
    def _is_attribute_anaphora(input_text: str) -> bool:
        """Determine whether it is an attribute anaphora (e.g. "看樱花的那个地方")."""
        import re

        # Attribute anaphora patterns: ...的(那个|那个地方|那个东西|那个项目)
        patterns = [
            r".+的(那个|那个地方|那个东西|那个项目|那个人|那个方案)",
            r"(上次|之前|刚才).+的.+",
        ]
        for pattern in patterns:
            if re.search(pattern, input_text):
                return True
        return False
