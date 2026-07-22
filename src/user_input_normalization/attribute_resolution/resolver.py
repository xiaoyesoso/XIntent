"""Attribute + retrieval anaphora resolver (corresponding to tasks 9.1-9.7 / D9).

For long time-span, attribute anaphora (e.g. "上次推荐的看樱花的那个地方"),
uses two-step inference + compensation mechanism to complete anaphora resolution.

Core scenario (D9):
    The user was recommended 鸡鸣寺 for cherry blossom viewing before a holiday;
    after the holiday, the user inputs "你上次推荐的看樱花的那个地方……"
    - Step 1: extract attributes [樱花, 旅游, 推荐], vector-search the dialogue history, recall details
    - Step 2: the LLM infers "看樱花的那个地方" => "鸡鸣寺" based on the window content
    - Compensation mechanism: on retrieval failure, extract attributes to trigger a tool call, then re-infer

Difficulties:
1. Long time-span (short-term memory fails, RAG long-term memory needed)
2. Attribute anaphora (one more layer of attribute matching than "这个"/"那个")
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from ..config import Config, get_config
from ..llm.base import LLMClient
from ..models import (
    AttributeResolutionResult,
    FactType,
    KeyFact,
)
from ..storage.base import DialogueHistoryStore, KeyFactStore


def _gen_fact_id(prefix: str = "fact") -> str:
    """Generate a fact ID."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# Common verb patterns for attribute extraction (corresponds to spec "attribute anaphora recognition")
_VERB_PATTERNS: list[str] = [
    "看", "穿", "戴", "吃", "去", "买", "玩", "读", "听", "用",
    "推荐", "讨论", "提到", "说过", "聊过",
]

# Time attribute keywords
_TIME_KEYWORDS: list[str] = [
    "上次", "上次推荐", "之前", "之前推荐", "刚才", "最近", "昨天",
]


class AttributeResolver:
    """Attribute + retrieval anaphora resolver (corresponds to D9).

    Two-step inference + compensation mechanism:
    1. extract_attributes: extract attribute keywords from user input
    2. recall_details: vector-search the dialogue history store, recall top-k details
    3. infer: the LLM infers the referent based on window content
    4. compensation: on retrieval failure, extract attributes to trigger a tool call, then re-infer

    Args:
        llm_client: LLM client
        dialogue_store: Dialogue history storage (supports semantic search)
        key_fact_store: Key fact storage (writeback of resolution results)
        config: Global configuration
    """

    # Default confidence threshold (below this triggers compensation or clarification)
    _DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7

    # Maximum retry count for the compensation mechanism
    _MAX_COMPENSATION_RETRIES: int = 2

    # Simulated tool name for the compensation mechanism
    _COMPENSATION_TOOL_NAME: str = "search_history_by_attributes"

    def __init__(
        self,
        llm_client: LLMClient,
        dialogue_store: DialogueHistoryStore,
        key_fact_store: KeyFactStore,
        config: Config | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.dialogue_store = dialogue_store
        self.key_fact_store = key_fact_store
        self.config = config or get_config()
        # Recall quality monitoring (task 9.7)
        self._recall_stats: dict[str, Any] = {
            "total_resolutions": 0,
            "successful_recalls": 0,
            "failed_recalls": 0,
            "compensation_triggered": 0,
            "compensation_succeeded": 0,
            "clarification_triggered": 0,
            "avg_confidence": 0.0,
            "confidence_samples": [],  # list[float]
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def resolve(
        self,
        session_id: str,
        user_input: str,
        user_id: str | None = None,
    ) -> AttributeResolutionResult:
        """Main entry: two-step inference + compensation mechanism.

        Flow:
        1. extract_attributes(user_input) -> list[str]
        2. recall_details(session_id, attributes) -> list[str] (vector search)
        3. if recalled: infer(session_id, user_input, recalled) -> (result, confidence)
        4. if not recalled or confidence < threshold:
           compensation(session_id, user_input, attributes)
        5. writeback(session_id, result)
        6. return AttributeResolutionResult

        Args:
            session_id: Session ID
            user_input: User's original input
            user_id: User ID (optional, for personal vocabulary, etc.)

        Returns:
            AttributeResolutionResult
        """
        self._recall_stats["total_resolutions"] += 1

        # 1. Attribute extraction (task 9.1)
        attributes = self.extract_attributes(user_input)

        # 2. Vector retrieval recall (task 9.2)
        top_k = self.config.context.recall_top_k
        recalled_details = self.recall_details(session_id, attributes, top_k=top_k)

        # Extract the attribute anaphora source text (e.g. "看樱花的那个地方")
        pronoun = self._extract_pronoun_phrase(user_input)

        # 3. Two-step inference (task 9.3)
        resolved_to = ""
        confidence = 0.0
        compensation_used = False
        tool_called: str | None = None

        if recalled_details:
            self._recall_stats["successful_recalls"] += 1
            resolved_to, confidence = self.infer(
                session_id, user_input, recalled_details
            )
        else:
            self._recall_stats["failed_recalls"] += 1

        # 4. Start compensation mechanism on retrieval failure or low confidence (task 9.4)
        threshold = self._DEFAULT_CONFIDENCE_THRESHOLD
        if not recalled_details or confidence < threshold:
            (
                resolved_to_comp,
                confidence_comp,
                compensation_used,
                tool_called,
            ) = self.compensation(session_id, user_input, attributes)
            # Compensation result takes priority (if compensation succeeded and confidence is higher)
            if confidence_comp > confidence:
                resolved_to = resolved_to_comp
                confidence = confidence_comp

        # 5. Build the result
        result = AttributeResolutionResult(
            pronoun=pronoun,
            extracted_attributes=attributes,
            recalled_details=recalled_details,
            resolved_to=resolved_to,
            confidence=confidence,
            compensation_used=compensation_used,
            tool_called=tool_called,
        )

        # 6. Writeback (task 9.6)
        # Infer turn: use dialogue history length + 1 as an approximation
        recent = self.dialogue_store.get_recent(session_id, n=1)
        turn = (recent[-1].turn + 1) if recent else 1
        self.writeback(session_id, turn, result)

        # Update statistics
        self._recall_stats["confidence_samples"].append(confidence)
        if compensation_used:
            self._recall_stats["compensation_triggered"] += 1
            if confidence >= threshold:
                self._recall_stats["compensation_succeeded"] += 1
        if confidence < self.config.clarify.theta_clarify:
            self._recall_stats["clarification_triggered"] += 1

        return result

    # ------------------------------------------------------------------
    # task 9.1: Attribute extraction
    # ------------------------------------------------------------------

    def extract_attributes(self, user_input: str) -> list[str]:
        """Attribute extraction (task 9.1): extract attribute keywords from user input.

        Combines rules + LLM extraction:
        - Rules: match common patterns like "看XX的", "穿XX的", "上次推荐的"
        - LLM: semantic extraction for parts not covered by rules

        Examples:
            "看樱花的那个地方" -> ["樱花", "看樱花", "地方"]
            "上次推荐的那个" -> ["上次", "推荐", "上次推荐"]
            "穿红色衣服、戴眼镜的同事" -> ["红色衣服", "戴眼镜", "穿红色衣服"]

        Args:
            user_input: User input

        Returns:
            List of attribute keywords (deduplicated)
        """
        if not user_input:
            return []

        attributes: list[str] = []

        # Rule 1: verb + object pattern (看樱花, 穿红色衣服, 戴眼镜)
        for verb in _VERB_PATTERNS:
            # Match "看XX的" or "看XX、" forms
            pattern = rf"{verb}([\u4e00-\u9fff]{{1,6}})(?:的|，|,|、)"
            matches = re.findall(pattern, user_input)
            for m in matches:
                attributes.append(m)
                attributes.append(verb + m)

        # Rule 2: time keywords
        for time_kw in _TIME_KEYWORDS:
            if time_kw in user_input:
                attributes.append(time_kw)

        # Rule 3: recommendation-related
        if "推荐" in user_input:
            attributes.append("推荐")
        if "旅游" in user_input or "玩" in user_input:
            attributes.append("旅游")

        # Rule 4: LLM fallback extraction (supplement attributes not covered by rules)
        llm_attrs = self._extract_attributes_via_llm(user_input)
        attributes.extend(llm_attrs)

        # Deduplicate and preserve order
        seen: set[str] = set()
        unique: list[str] = []
        for attr in attributes:
            if attr and attr not in seen and len(attr) <= 10:
                seen.add(attr)
                unique.append(attr)

        return unique

    def _extract_attributes_via_llm(self, user_input: str) -> list[str]:
        """Extract attribute keywords via LLM (fallback)."""
        system_prompt = (
            "你是属性关键词提取助手。从用户输入中提取用于检索历史对话的属性关键词。\n"
            "规则：\n"
            "1. 提取名词、动宾短语作为属性\n"
            "2. 忽略'那个''这个''地方''东西'等无指代意义的词\n"
            "3. 输出 JSON: {\"attributes\": [\"关键词1\", \"关键词2\"]}\n"
            "示例：\n"
            "输入: '你上次推荐的看樱花的那个地方'\n"
            "输出: {\"attributes\": [\"樱花\", \"推荐\", \"看樱花\"]}"
        )
        user_prompt = f"用户输入：{user_input}\n请提取属性关键词，输出 JSON。"
        try:
            response = self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            data = json.loads(response)
            attrs = data.get("attributes", [])
            if isinstance(attrs, list):
                return [str(a) for a in attrs if isinstance(a, str)]
        except (json.JSONDecodeError, TypeError, Exception):
            pass
        return []

    # ------------------------------------------------------------------
    # task 9.2: Vector retrieval recall
    # ------------------------------------------------------------------

    def recall_details(
        self,
        session_id: str,
        attributes: list[str],
        top_k: int = 5,
    ) -> list[str]:
        """Vector retrieval recall (task 9.2): use attribute keywords to search the dialogue history store.

        Args:
            session_id: Session ID
            attributes: List of attribute keywords
            top_k: Recall top-k entries

        Returns:
            List of recalled dialogue history content (sorted by relevance, deduplicated)
        """
        if not attributes:
            return []

        # Concatenate attribute keywords into a query
        query = " ".join(attributes)

        # Call the dialogue history storage's semantic search
        turns = self.dialogue_store.search_semantic(
            session_id=session_id,
            query=query,
            top_k=top_k,
        )

        # Extract content and deduplicate
        seen: set[str] = set()
        details: list[str] = []
        for t in turns:
            if t.content and t.content not in seen:
                seen.add(t.content)
                details.append(t.content)
        return details

    # ------------------------------------------------------------------
    # task 9.3: Two-step inference
    # ------------------------------------------------------------------

    def infer(
        self,
        session_id: str,
        user_input: str,
        recalled_details: list[str],
    ) -> tuple[str, float]:
        """Two-step inference (task 9.3): after recall, the LLM infers the referent based on window content.

        The LLM MUST infer based on the recalled window content; fabrication is forbidden.
        If the recalled content is insufficient for inference, return low confidence.

        Args:
            session_id: Session ID
            user_input: User's original input
            recalled_details: Recalled dialogue history details

        Returns:
            (resolution result, confidence) tuple
        """
        if not recalled_details:
            return "", 0.0

        details_text = "\n".join(
            f"  {i + 1}. {d}" for i, d in enumerate(recalled_details)
        )

        system_prompt = (
            "你是指代消解助手。基于召回的历史对话细节，推断用户输入中属性指代的"
            "具体对象。\n"
            "规则：\n"
            "1. 必须基于召回的对话细节进行推断，禁止伪造\n"
            "2. 若召回内容不足以确定指代对象，返回低置信度\n"
            "3. 输出 JSON: {\"resolved_to\": \"消解结果\", \"confidence\": 0.0-1.0, "
            "\"reasoning\": \"推断依据\"}\n"
            "示例：\n"
            "用户输入: '你上次推荐的看樱花的那个地方'\n"
            "召回细节: ['我推荐鸡鸣寺看樱花', '鸡鸣寺的樱花很有名']\n"
            "输出: {\"resolved_to\": \"鸡鸣寺\", \"confidence\": 0.92, "
            "\"reasoning\": \"召回细节中明确提到推荐鸡鸣寺看樱花\"}"
        )
        user_prompt = (
            f"用户输入：{user_input}\n"
            f"召回的历史对话细节：\n{details_text}\n"
            "请基于召回细节推断属性指代对象，输出 JSON。"
        )

        try:
            response = self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            data = json.loads(response)
            resolved_to = str(data.get("resolved_to", ""))
            confidence = float(data.get("confidence", 0.0))
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            return resolved_to, confidence
        except (json.JSONDecodeError, TypeError, ValueError, Exception):
            return "", 0.0

    # ------------------------------------------------------------------
    # task 9.4: Retrieval-failure compensation mechanism
    # ------------------------------------------------------------------

    def compensation(
        self,
        session_id: str,
        user_input: str,
        attributes: list[str],
    ) -> tuple[str, float, bool, str]:
        """Retrieval-failure compensation mechanism (task 9.4).

        When step-1 recall does not go well:
        1. Extract attributes as parameters to trigger a tool call (simulating the search_history_by_attributes tool)
        2. The LLM again infers based on the returned dialogue history details

        The compensation mechanism limits the retry count (default 2); if still failing, return low confidence.

        Args:
            session_id: Session ID
            user_input: User's original input
            attributes: List of attribute keywords

        Returns:
            (resolution result, confidence, whether compensation was used, tool name) tuple
        """
        tool_name = self._COMPENSATION_TOOL_NAME
        threshold = self._DEFAULT_CONFIDENCE_THRESHOLD

        best_resolved = ""
        best_confidence = 0.0

        for retry in range(self._MAX_COMPENSATION_RETRIES):
            # Simulate tool call: use a broader query to search dialogue history
            # In production, a real tool would be called here; we use dialogue_store to simulate
            tool_details = self._simulate_tool_call(session_id, attributes, retry)

            if not tool_details:
                continue

            # The LLM again infers based on the details returned by the tool
            resolved, confidence = self.infer(session_id, user_input, tool_details)

            if confidence > best_confidence:
                best_confidence = confidence
                best_resolved = resolved

            # Exit early if confidence is sufficient
            if best_confidence >= threshold:
                break

        return best_resolved, best_confidence, True, tool_name

    def _simulate_tool_call(
        self,
        session_id: str,
        attributes: list[str],
        retry: int,
    ) -> list[str]:
        """Simulate the search_history_by_attributes tool call.

        In production, this should be replaced with a real tool call. Here we use dialogue_store's
        semantic search to simulate, expanding top_k on each retry to simulate "broader retrieval".
        """
        # Expand retrieval range on retry
        top_k = self.config.context.recall_top_k + (retry + 1) * 3
        # Use attributes + original input as a combined query
        query = " ".join(attributes)
        if not query:
            return []

        turns = self.dialogue_store.search_semantic(
            session_id=session_id,
            query=query,
            top_k=top_k,
        )
        seen: set[str] = set()
        details: list[str] = []
        for t in turns:
            if t.content and t.content not in seen:
                seen.add(t.content)
                details.append(t.content)
        return details

    # ------------------------------------------------------------------
    # task 9.5: Confidence assessment
    # ------------------------------------------------------------------

    def assess_confidence(
        self,
        result: str,
        recalled_details: list[str],
    ) -> float:
        """Confidence assessment (task 9.5).

        Comprehensively considers:
        - Relevance of recalled fragments (quantity and quality)
        - Attribute match degree
        - LLM inference consistency

        Args:
            result: Resolution result
            recalled_details: Recalled details

        Returns:
            Confidence [0.0, 1.0]
        """
        if not result or not recalled_details:
            return 0.0

        # Factor 1: number of recalled fragments (more means higher confidence, but with an upper bound)
        recall_factor = min(1.0, len(recalled_details) / 3.0)

        # Factor 2: frequency of the result in recalled details
        match_count = sum(
            1 for d in recalled_details if result in d or d in result
        )
        match_factor = min(1.0, match_count / 2.0)

        # Factor 3: result length reasonableness (too short or too long is penalized)
        length = len(result)
        if 2 <= length <= 10:
            length_factor = 1.0
        elif 1 <= length <= 20:
            length_factor = 0.7
        else:
            length_factor = 0.3

        # Weighted combination
        confidence = (
            recall_factor * 0.3
            + match_factor * 0.5
            + length_factor * 0.2
        )
        return round(confidence, 3)

    # ------------------------------------------------------------------
    # task 9.6: Resolution result writeback
    # ------------------------------------------------------------------

    def writeback(
        self,
        session_id: str,
        turn: int,
        result: AttributeResolutionResult,
    ) -> None:
        """Resolution result writeback to key fact storage (task 9.6).

        Writeback content includes:
        - Attribute anaphora source text
        - Resolution result
        - Attribute keywords
        - Recall source
        - Confidence

        Args:
            session_id: Session ID
            turn: Current turn
            result: Resolution result
        """
        # Only writeback on successful resolution
        if not result.resolved_to:
            return

        fact = KeyFact(
            fact_id=_gen_fact_id("fact"),
            session_id=session_id,
            turn=turn,
            fact_type=FactType.PRONOUN_RESOLUTION,
            content={
                "pronoun": result.pronoun,
                "resolved_to": result.resolved_to,
                "extracted_attributes": result.extracted_attributes,
                "recalled_details_count": len(result.recalled_details),
                "confidence": result.confidence,
                "compensation_used": result.compensation_used,
                "tool_called": result.tool_called,
                "source": "attribute_resolution",
            },
            confidence=result.confidence,
        )
        self.key_fact_store.save(fact)

    # ------------------------------------------------------------------
    # task 9.7: Recall quality monitoring
    # ------------------------------------------------------------------

    def get_recall_stats(self) -> dict[str, Any]:
        """Recall quality monitoring (task 9.7).

        Returns statistical metrics:
        - total_resolutions: total resolution count
        - successful_recalls: successful recall count
        - failed_recalls: failed recall count
        - compensation_triggered: compensation trigger count
        - compensation_succeeded: compensation success count
        - clarification_triggered: clarification trigger count
        - avg_confidence: average confidence
        - recall_success_rate: recall success rate
        - compensation_trigger_rate: compensation trigger rate
        """
        stats = dict(self._recall_stats)
        total = stats["total_resolutions"]
        if total > 0:
            stats["recall_success_rate"] = stats["successful_recalls"] / total
            stats["compensation_trigger_rate"] = (
                stats["compensation_triggered"] / total
            )
            samples = stats.get("confidence_samples", [])
            if samples:
                stats["avg_confidence"] = sum(samples) / len(samples)
        else:
            stats["recall_success_rate"] = 0.0
            stats["compensation_trigger_rate"] = 0.0
        # Remove raw sample data
        stats.pop("confidence_samples", None)
        return stats

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pronoun_phrase(user_input: str) -> str:
        """Extract the attribute anaphora phrase from user input.

        Examples:
            "你上次推荐的看樱花的那个地方……" -> "看樱花的那个地方"
            "上次那个穿红色衣服的同事" -> "穿红色衣服的同事"
        """
        # Match "...的(那个|那个)(地方|东西|人|同事|...)"
        # Simplified: extract the attribute part before "的" + "的那个XX"
        patterns = [
            r"(?:看|穿|戴|吃|去|买|玩|读|听|用)([\u4e00-\u9fff]{1,8})的(那个|那个)?[\u4e00-\u9fff]{0,4}",
            r"(上次|之前|刚才)([^\u4e00-\u9fff]*)([\u4e00-\u9fff]{1,8})的(那个|那个)?[\u4e00-\u9fff]{0,4}",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                return match.group(0)

        # Fallback: return the substring containing "那个"
        if "那个" in user_input:
            idx = user_input.find("那个")
            # Look backward for "的"
            start = user_input.rfind("的", 0, idx)
            if start >= 0:
                # Look further backward for the attribute start
                start2 = user_input.rfind("的", 0, start)
                if start2 >= 0:
                    return user_input[start2 + 1: idx + 4]
                return user_input[start + 1: idx + 4]
            return user_input[idx: idx + 4]

        # Final fallback: return the entire input
        return user_input
