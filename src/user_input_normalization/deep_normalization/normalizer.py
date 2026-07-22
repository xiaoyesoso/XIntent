"""Deep normalizer (corresponding to tasks 5.1-5.6 / D1 / D13).

Executes within the ReAct loop, handling content marked as unquantified by pre-normalization:
1. Judgment word quantification (task 5.2): transforms "性价比", "再高级一点", etc. into tool parameters
2. External fact resolution (task 5.3): resolves inputs like "现在最便宜的" by combining tool returns
3. Observation-dependent backtracking resolution (task 5.4): re-infers anaphora after tool returns
4. Context window management (task 5.5): dynamically assembles Observation + historical details
5. Result writeback to key fact storage (task 5.6)
6. ReAct loop single-step integration (task 5.1)
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..config import Config, get_config
from ..llm.base import LLMClient
from ..models import (
    CompletenessCheck,
    CompletenessStatus,
    FactType,
    KeyFact,
    NormalizationResult,
    NormalizationStage,
    PronounResolution,
    QuantifiableAdjective,
)
from ..quantification.engine import QuantificationEngine
from ..storage.base import DialogueHistoryStore, KeyFactStore


def _gen_fact_id(prefix: str = "fact") -> str:
    """Generate a fact ID."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class DeepNormalizer:
    """Deep normalizer (corresponds to D1 deep processing stage / D13 quantification).

    Receives the structured result of pre-normalization, and within the ReAct loop completes:
    - Quantification of unquantified adjectives
    - External fact resolution dependent on tool returns
    - Anaphora backtracking correction based on Observation
    - Result writeback to key fact storage

    Args:
        llm_client: LLM client (for external fact resolution, backtracking inference)
        quantification_engine: Quantification engine (implemented in task 6)
        key_fact_store: Key fact storage (writeback of resolution results)
        dialogue_store: Dialogue history storage (context window management)
        config: Global configuration
    """

    # Context window priority (high -> low), corresponds to spec "window overflow priority retention"
    _WINDOW_PRIORITY: list[str] = [
        "key_facts",
        "observation",
        "recent_dialogue",
        "remote_summary",
    ]

    # Default context window token limit (rough estimate, approximated by character count)
    _DEFAULT_WINDOW_LIMIT: int = 4000

    def __init__(
        self,
        llm_client: LLMClient,
        quantification_engine: QuantificationEngine,
        key_fact_store: KeyFactStore,
        dialogue_store: DialogueHistoryStore,
        config: Config | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.quantification_engine = quantification_engine
        self.key_fact_store = key_fact_store
        self.dialogue_store = dialogue_store
        self.config = config or get_config()
        # Observability: record processing statistics
        self._stats: dict[str, Any] = {
            "total_processed": 0,
            "adjectives_quantified": 0,
            "external_facts_resolved": 0,
            "pronouns_re_resolved": 0,
            "writebacks": 0,
        }

    # ------------------------------------------------------------------
    # Main entry (task 5.1 integration)
    # ------------------------------------------------------------------

    def process(
        self,
        session_id: str,
        turn: int,
        pre_result: NormalizationResult,
        user_id: str,
        observation: dict[str, Any] | str | None = None,
    ) -> NormalizationResult:
        """Main entry: handle unquantified content marked by pre-normalization.

        Steps:
        1. Iterate over items with quantified=False in pre_result.quantifiable_adjectives
        2. Call quantification_engine.quantify() to quantify
        3. Handle external facts (if observation is present, resolve by combining tool returns)
        4. If observation is present, re-infer anaphora (task 5.4)
        5. Update pre_result's quantifiable_adjectives
        6. Writeback results to key fact storage (task 5.6)
        7. Return the updated NormalizationResult

        Args:
            session_id: Session ID
            turn: Current dialogue turn
            pre_result: Structured result from the pre-normalization stage
            user_id: User ID
            observation: Observation returned by the tool in the ReAct loop (may be dict / str / None)

        Returns:
            Updated NormalizationResult (based on a copy of pre_result)
        """
        self._stats["total_processed"] += 1

        # Copy to avoid modifying the input
        result = pre_result.model_copy(deep=True)

        # 1+2+5. Quantify unquantified adjectives (task 5.2)
        context = self._build_quantify_context(session_id, user_id, observation)
        result.quantifiable_adjectives = self.quantify_adjectives(
            result.quantifiable_adjectives, context
        )

        # 3. External fact resolution (task 5.3)
        external_resolved: dict[str, Any] = {}
        if observation is not None:
            external_resolved = self.resolve_external_fact(
                session_id, result.normalized_input or result.raw_input, observation
            )

        # 4. Backtracking resolution based on Observation (task 5.4)
        re_resolved: list[PronounResolution] = []
        if observation is not None:
            re_resolved = self.re_resolve_with_observation(
                session_id, result, observation
            )
            if re_resolved:
                result.pronoun_resolutions = re_resolved

        # Update completeness check: if all adjectives are quantified, mark as complete
        result.completeness = self._update_completeness(result.completeness, result)

        # Routing suggestion: if fully processed, do not route to deep
        if self._is_fully_processed(result):
            result.route_to = None
        else:
            result.route_to = NormalizationStage.DEEP

        # 6. Result writeback (task 5.6)
        self.writeback(
            session_id,
            turn,
            result,
            external_resolved=external_resolved,
            re_resolved=re_resolved,
        )

        return result

    # ------------------------------------------------------------------
    # task 5.2: Judgment word quantification
    # ------------------------------------------------------------------

    def quantify_adjectives(
        self,
        adjectives: list[QuantifiableAdjective],
        context: dict[str, Any],
    ) -> list[QuantifiableAdjective]:
        """Quantify all unquantified adjectives (task 5.2).

        Iterates over items with quantified=False in the list, calls QuantificationEngine to quantify;
        already-quantified items remain unchanged.

        Args:
            adjectives: List of quantifiable adjectives marked by pre-normalization
            context: Quantification context (includes current_price / current_tier / user_profile, etc.)

        Returns:
            Updated list of QuantifiableAdjective (order preserved)
        """
        updated: list[QuantifiableAdjective] = []
        for adj in adjectives:
            if adj.quantified:
                # Already quantified, keep as-is
                updated.append(adj)
                continue
            # Call quantification engine to quantify
            quantified = self.quantification_engine.quantify(adj.adjective, context)
            # Preserve the original route_to field for audit (set to None if already quantified)
            quantified.route_to = None
            updated.append(quantified)
            self._stats["adjectives_quantified"] += 1
        return updated

    # ------------------------------------------------------------------
    # task 5.3: External fact resolution
    # ------------------------------------------------------------------

    def resolve_external_fact(
        self,
        session_id: str,
        input_text: str,
        observation: dict[str, Any] | str,
    ) -> dict[str, Any]:
        """External fact resolution (task 5.3): resolve inputs like "现在最便宜的" by combining observation.

        Strategy:
        - Hand the user input and Observation to the LLM, letting it determine the referent based on real tool returns
        - Strictly forbid fabricating real-time information: when observation is empty, return an unresolved marker
        - On failure, return needs_clarification=True for the upper layer to trigger clarification

        Args:
            session_id: Session ID
            input_text: User's original input (or normalized input)
            observation: Real data returned by the tool (dict or string)

        Returns:
            Resolution result dict, containing:
            - resolved: bool whether resolution succeeded
            - resolved_entity: str | None the specific object resolved
            - evidence: str evidence source
            - needs_clarification: bool whether to trigger clarification
            - raw_observation: the original observation
        """
        self._stats["external_facts_resolved"] += 1

        # observation is empty: forbid fabrication, mark for clarification
        if observation is None:
            return {
                "resolved": False,
                "resolved_entity": None,
                "evidence": "observation 缺失，禁止伪造实时信息",
                "needs_clarification": True,
                "raw_observation": None,
            }

        # Unify to string for LLM processing
        if isinstance(observation, dict):
            obs_text = json.dumps(observation, ensure_ascii=False, default=str)
        else:
            obs_text = str(observation)

        # observation is an empty string: also forbid fabrication
        if not obs_text.strip():
            return {
                "resolved": False,
                "resolved_entity": None,
                "evidence": "observation 为空，禁止伪造实时信息",
                "needs_clarification": True,
                "raw_observation": obs_text,
            }

        system_prompt = (
            "你是外部事实消解助手。基于工具返回的真实数据（Observation），"
            "消解用户输入中的外部事实表达（如'现在最便宜的''最近哪个更火'）。\n"
            "严格规则：\n"
            "1. 必须基于 Observation 中的真实数据，禁止伪造实时信息\n"
            "2. 若 Observation 不足以消解，返回 needs_clarification=true\n"
            "输出 JSON 格式：\n"
            '{"resolved": true/false, "resolved_entity": "具体对象或null", '
            '"evidence": "证据说明", "needs_clarification": true/false}'
        )
        user_prompt = (
            f"用户输入：{input_text}\n"
            f"工具返回（Observation）：{obs_text}\n"
            "请基于 Observation 消解外部事实，输出 JSON。"
        )

        try:
            response = self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError, Exception):
            # LLM response parsing failed: conservative degradation, no fabrication
            return {
                "resolved": False,
                "resolved_entity": None,
                "evidence": f"LLM 返回解析失败，原始响应: {response[:200]}",
                "needs_clarification": True,
                "raw_observation": obs_text,
            }

        return {
            "resolved": bool(data.get("resolved", False)),
            "resolved_entity": data.get("resolved_entity"),
            "evidence": data.get("evidence", "基于 Observation 消解"),
            "needs_clarification": bool(data.get("needs_clarification", False)),
            "raw_observation": obs_text,
        }

    # ------------------------------------------------------------------
    # task 5.4: Observation-dependent backtracking resolution
    # ------------------------------------------------------------------

    def re_resolve_with_observation(
        self,
        session_id: str,
        pre_result: NormalizationResult,
        observation: dict[str, Any] | str,
    ) -> list[PronounResolution]:
        """Observation-dependent backtracking resolution (task 5.4).

        When the Observation in the ReAct loop returns new facts, re-evaluate the
        anaphora resolution result from the pre-normalization stage, and backtrack-correct if necessary.

        Example (spec scenario):
        - pre-normalization once placeholder-resolved "那个" to "商品 C"
        - Observation returns "商品 C 已下架"
        - This method re-resolves "那个" to another candidate and records the correction reason

        Args:
            session_id: Session ID
            pre_result: pre-normalization structured result
            observation: Observation returned by the tool

        Returns:
            Updated list of PronounResolution (if no correction, consistent with the original list)
        """
        if not pre_result.pronoun_resolutions:
            return list(pre_result.pronoun_resolutions)

        if isinstance(observation, dict):
            obs_text = json.dumps(observation, ensure_ascii=False, default=str)
        else:
            obs_text = str(observation)

        if not obs_text.strip():
            return list(pre_result.pronoun_resolutions)

        # Build a summary of the existing resolution table for LLM reference
        existing_table = [
            {
                "pronoun": pr.pronoun,
                "resolved_to": pr.resolved_to,
                "confidence": pr.confidence,
                "evidence_source": pr.evidence_source,
            }
            for pr in pre_result.pronoun_resolutions
        ]

        system_prompt = (
            "你是指代消解回溯助手。ReAct 循环中工具返回了新事实（Observation），"
            "请基于新事实重新评估现有的指代消解表格，必要时回溯修正。\n"
            "规则：\n"
            "1. 若 Observation 表明原消解对象已失效（如下架/删除/不存在），需修正为新候选\n"
            "2. 修正时需提高置信度（有新证据支撑）\n"
            "3. 若 Observation 与原消解无冲突，保持原结果\n"
            "4. 禁止伪造候选，若无合理新候选则保持原结果并标注 needs_review=true\n"
            "输出 JSON 格式：\n"
            '{"resolutions": [{"pronoun": "...", "resolved_to": "...", '
            '"confidence": 0.0, "evidence_source": "...", '
            '"corrected": true/false, "correction_reason": "..."}]}'
        )
        user_prompt = (
            f"现有消解表格：{json.dumps(existing_table, ensure_ascii=False)}\n"
            f"Observation：{obs_text}\n"
            "请评估是否需要回溯修正，输出 JSON。"
        )

        try:
            response = self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError, Exception):
            # Parsing failed: keep the original resolution result, no correction
            return list(pre_result.pronoun_resolutions)

        raw_list = data.get("resolutions", [])
        if not isinstance(raw_list, list) or not raw_list:
            return list(pre_result.pronoun_resolutions)

        # Convert LLM output to a list of PronounResolution
        updated: list[PronounResolution] = []
        corrected_count = 0
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            pronoun = item.get("pronoun") or ""
            resolved_to = item.get("resolved_to") or ""
            if not pronoun or not resolved_to:
                continue
            confidence = float(item.get("confidence", 0.8))
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            evidence_source = item.get("evidence_source") or "Observation 回溯"
            corrected = bool(item.get("corrected", False))
            if corrected:
                correction_reason = item.get("correction_reason", "")
                evidence_source = (
                    f"{evidence_source}（回溯修正: {correction_reason}）"
                )
                corrected_count += 1
            updated.append(
                PronounResolution(
                    pronoun=pronoun,
                    resolved_to=resolved_to,
                    confidence=confidence,
                    evidence_source=evidence_source,
                )
            )

        if corrected_count > 0:
            self._stats["pronouns_re_resolved"] += corrected_count

        # If LLM output is incomplete, keep unprocessed items from the original table
        if len(updated) < len(pre_result.pronoun_resolutions):
            existing_pronouns = {u.pronoun for u in updated}
            for pr in pre_result.pronoun_resolutions:
                if pr.pronoun not in existing_pronouns:
                    updated.append(pr)

        return updated

    # ------------------------------------------------------------------
    # task 5.5: Context window management
    # ------------------------------------------------------------------

    def manage_context_window(
        self,
        session_id: str,
        observation: dict[str, Any] | str | None,
        pre_result: NormalizationResult,
    ) -> str:
        """Context window management (task 5.5): dynamically assemble Observation + historical details.

        Retain by priority (spec "window overflow priority retention"):
        Key facts > Current-turn tool return > Recent N turns of dialogue > Remote dialogue summary

        Args:
            session_id: Session ID
            observation: Current-turn tool return
            pre_result: pre-normalization result

        Returns:
            Assembled context window string
        """
        sections: list[str] = []
        total_chars = 0
        limit = self._DEFAULT_WINDOW_LIMIT

        # Priority 1: Key facts (persisted within this session)
        facts = self.key_fact_store.get_by_session(session_id)
        if facts:
            fact_lines: list[str] = []
            for f in facts:
                if f.status != "active":
                    continue
                content_str = json.dumps(f.content, ensure_ascii=False, default=str)
                fact_lines.append(
                    f"  - [{f.fact_type.value}] turn={f.turn}: {content_str}"
                )
            if fact_lines:
                section = "【关键事实】\n" + "\n".join(fact_lines)
                sections.append(section)
                total_chars += len(section)

        # Priority 2: Current-turn Observation
        if observation is not None:
            if isinstance(observation, dict):
                obs_text = json.dumps(observation, ensure_ascii=False, default=str)
            else:
                obs_text = str(observation)
            section = f"【当前轮 Observation】\n  {obs_text}"
            sections.append(section)
            total_chars += len(section)

        # Priority 3: Recent N turns of dialogue
        recent_n = self.config.context.short_term_window
        recent_turns = self.dialogue_store.get_recent(session_id, n=recent_n)
        if recent_turns:
            turn_lines: list[str] = []
            for t in recent_turns:
                line = f"  - turn={t.turn} role={t.role}: {t.content}"
                if total_chars + len(line) > limit:
                    break
                turn_lines.append(line)
                total_chars += len(line)
            if turn_lines:
                section = "【近期对话】\n" + "\n".join(turn_lines)
                sections.append(section)

        # Priority 4: Remote dialogue summary (can be trimmed)
        summary = self.dialogue_store.get_summary(session_id)
        if summary and total_chars < limit:
            remaining = limit - total_chars
            trimmed = summary[:remaining] if len(summary) > remaining else summary
            section = f"【对话摘要】\n  {trimmed}"
            sections.append(section)

        # Attach pre-normalization's normalized input as reference
        if pre_result.normalized_input:
            section = f"【规范化后输入】\n  {pre_result.normalized_input}"
            sections.append(section)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # task 5.6: Result writeback
    # ------------------------------------------------------------------

    def writeback(
        self,
        session_id: str,
        turn: int,
        result: NormalizationResult,
        external_resolved: dict[str, Any] | None = None,
        re_resolved: list[PronounResolution] | None = None,
    ) -> None:
        """Result writeback to key fact storage (task 5.6).

        Writeback content includes:
        - Quantified tool parameters (each adjective with quantified=True)
        - Resolved external facts (external_resolved)
        - Backtracking correction records (items with corrected=True in re_resolved)

        Atomicity guarantee: each KeyFact is saved independently; failure does not affect others.

        Args:
            session_id: Session ID
            turn: Current turn
            result: NormalizationResult after deep normalization
            external_resolved: External fact resolution result (optional)
            re_resolved: Backtracking-corrected anaphora list (optional)
        """
        # 1. Quantification parameter writeback
        for adj in result.quantifiable_adjectives:
            if not adj.quantified or adj.quantified_value is None:
                continue
            fact = KeyFact(
                fact_id=_gen_fact_id("fact"),
                session_id=session_id,
                turn=turn,
                fact_type=FactType.QUANTIFICATION,
                content={
                    "adjective": adj.adjective,
                    "quantified_value": adj.quantified_value,
                    "stage": NormalizationStage.DEEP.value,
                },
                confidence=0.9,
            )
            self.key_fact_store.save(fact)
            self._stats["writebacks"] += 1

        # 2. External fact resolution result writeback
        if external_resolved and external_resolved.get("resolved"):
            fact = KeyFact(
                fact_id=_gen_fact_id("fact"),
                session_id=session_id,
                turn=turn,
                fact_type=FactType.KEY_FACT,
                content={
                    "type": "external_fact_resolution",
                    "resolved_entity": external_resolved.get("resolved_entity"),
                    "evidence": external_resolved.get("evidence"),
                    "raw_observation": external_resolved.get("raw_observation"),
                    "stage": NormalizationStage.DEEP.value,
                },
                confidence=0.85,
            )
            self.key_fact_store.save(fact)
            self._stats["writebacks"] += 1

        # 3. Backtracking correction record writeback
        if re_resolved:
            for pr in re_resolved:
                if "回溯" not in pr.evidence_source:
                    continue
                fact = KeyFact(
                    fact_id=_gen_fact_id("fact"),
                    session_id=session_id,
                    turn=turn,
                    fact_type=FactType.PRONOUN_RESOLUTION,
                    content={
                        "pronoun": pr.pronoun,
                        "resolved_to": pr.resolved_to,
                        "confidence": pr.confidence,
                        "evidence_source": pr.evidence_source,
                        "correction_recorded": True,
                        "stage": NormalizationStage.DEEP.value,
                    },
                    confidence=pr.confidence,
                )
                self.key_fact_store.save(fact)
                self._stats["writebacks"] += 1

    # ------------------------------------------------------------------
    # task 5.1: ReAct loop single-step integration
    # ------------------------------------------------------------------

    def react_step(
        self,
        session_id: str,
        turn: int,
        pre_result: NormalizationResult,
        thought: str,
        action: str,
        observation: dict[str, Any] | str | None,
        user_id: str = "",
    ) -> NormalizationResult:
        """ReAct loop single-step integration (task 5.1).

        Embeds deep normalization within Thought -> Action -> Observation:
        - Receives Thought (reasoning process), Action (tool call), Observation (tool return)
        - Calls process() to complete deep normalization
        - Does not break the ReAct loop's termination condition (does not force a new turn when termination is already satisfied)

        Args:
            session_id: Session ID
            turn: Current turn
            pre_result: pre-normalization result
            thought: Reasoning text of the ReAct Thought step
            action: Tool call description of the ReAct Action step
            observation: Tool return of the ReAct Observation step
            user_id: User ID

        Returns:
            Updated NormalizationResult
        """
        # Treat Thought / Action as part of the context, Observation as the main input
        # If observation is empty (e.g. Action failed), still attempt to quantify adjectives
        return self.process(
            session_id=session_id,
            turn=turn,
            pre_result=pre_result,
            user_id=user_id,
            observation=observation,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get deep normalization statistics (observability)."""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _build_quantify_context(
        self,
        session_id: str,
        user_id: str,
        observation: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        """Build the quantification context.

        Combines real-time data in Observation (e.g. current_price) with historical key facts.
        """
        context: dict[str, Any] = {}

        # Extract quantification baselines from Observation (e.g. current_price / current_tier)
        if isinstance(observation, dict):
            for key in ("current_price", "current_tier", "current_quality_rank",
                        "current_brand_tier"):
                if key in observation:
                    context[key] = observation[key]
        elif isinstance(observation, str) and observation:
            # String-form Observation cannot directly extract structured fields; keep as-is for LLM reference
            context["observation_text"] = observation

        # Supplement quantification baselines from key facts
        facts = self.key_fact_store.get_by_session(session_id)
        for f in facts:
            if f.fact_type == FactType.QUANTIFICATION and f.status == "active":
                content = f.content
                # If historical quantification record contains current_price, it can serve as a baseline
                if "current_price" not in context and "current_price" in content:
                    context["current_price"] = content["current_price"]

        return context

    @staticmethod
    def _update_completeness(
        original: CompletenessCheck | None,
        result: NormalizationResult,
    ) -> CompletenessCheck:
        """Update the completeness check based on deep normalization results."""
        adjectives_quantified = all(
            adj.quantified for adj in result.quantifiable_adjectives
        ) if result.quantifiable_adjectives else True

        pronouns_resolved = all(
            pr.confidence >= 0.6 for pr in result.pronoun_resolutions
        ) if result.pronoun_resolutions else True

        spo_complete = original.spo_complete if original else True

        # Determine final status
        if spo_complete and pronouns_resolved and adjectives_quantified:
            status = CompletenessStatus.COMPLETE
        elif not adjectives_quantified:
            status = CompletenessStatus.INCOMPLETE_ADJECTIVE_UNQUANTIFIED
        elif not pronouns_resolved:
            status = CompletenessStatus.INCOMPLETE_PRONOUN_UNRESOLVED
        else:
            status = CompletenessStatus.INCOMPLETE_MISSING_ARGUMENT

        return CompletenessCheck(
            spo_complete=spo_complete,
            pronouns_resolved=pronouns_resolved,
            adjectives_quantified=adjectives_quantified,
            result=status,
        )

    @staticmethod
    def _is_fully_processed(result: NormalizationResult) -> bool:
        """Determine whether deep normalization is fully complete."""
        if not result.quantifiable_adjectives:
            return True
        return all(adj.quantified for adj in result.quantifiable_adjectives)
