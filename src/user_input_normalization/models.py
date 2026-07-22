"""User input normalization framework - data models.

Corresponds to SDD decisions D4 (structured output format), D6 (pronoun resolution table cross-turn reuse),
D10 (vocabulary entry), D15 (storage selection).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------


class InputProblemType(str, Enum):
    """Six major categories of input problems (corresponds to input-classification spec)."""

    ANAPHORA = "指代问题"  # Anaphora: this, that, the first one
    MISSING = "缺失问题"  # Missing: missing subject/object/constraint
    EXPRESSION = "表达问题"  # Expression: colloquial, broken sentences, disordered word order
    SEMANTIC = "词义问题"  # Semantic: non-standard terms, jargon, polysemous words
    SUBJECTIVE = "主观判断问题"  # Subjective judgment: best value for money, a bit more advanced
    EXTERNAL_FACT = "外部事实问题"  # External fact: which one is more popular recently


class VocabLevel(str, Enum):
    """Vocabulary table levels (corresponds to D10)."""

    PUBLIC = "public"  # Public vocabulary
    PERSONAL = "personal"  # User personal vocabulary


class VocabStatus(str, Enum):
    """Vocabulary entry status (corresponds to D11 self-iteration)."""

    CANDIDATE = "candidate"  # Candidate (offline analysis marked, below threshold)
    PROMOTED = "promoted"  # Promoted (reached threshold, manually approved)
    DRAFT = "draft"  # Draft (pending manual review)


class FactType(str, Enum):
    """Key fact types."""

    PRONOUN_RESOLUTION = "pronoun_resolution"  # Pronoun resolution
    KEY_FACT = "key_fact"  # General key fact
    NAMED_ENTITY = "named_entity"  # Named entity (indexed by name D7)
    QUANTIFICATION = "quantification"  # Quantification result


class NormalizationStage(str, Enum):
    """Normalization stages (corresponds to D1 two-stage pipeline)."""

    PRE = "pre"  # Before intent recognition
    DEEP = "deep"  # Within ReAct loop


class CompletenessStatus(str, Enum):
    """Completeness check status (corresponds to D16)."""

    COMPLETE = "complete"
    INCOMPLETE_MISSING_ARGUMENT = "未完成-缺宾语"
    INCOMPLETE_PRONOUN_UNRESOLVED = "未完成-代词未消解"
    INCOMPLETE_ADJECTIVE_UNQUANTIFIED = "未完成-形容词未量化"


class ClarificationReason(str, Enum):
    """Clarification trigger reasons (corresponds to D8)."""

    PRONOUN_RESOLUTION_FAILED = "代词消解失败"
    WORD_SENSE_LOW_CONFIDENCE = "词义消歧低置信"
    SUBJECTIVE_NO_BASELINE = "主观判断无基准"
    COMPLETION_NO_EVIDENCE = "补全无证据"
    UNKNOWN_VOCAB = "未知词汇"


# ---------------------------------------------------------------------------
# Pronoun resolution table (D4 / D6)
# ---------------------------------------------------------------------------


class PronounResolution(BaseModel):
    """Pronoun resolution table entry (corresponds to D4 structured output, D6 cross-turn reuse)."""

    pronoun: str = Field(description="代词，如 '第二个'、'那个'")
    resolved_to: str = Field(description="指代对象，如 'TCC方案'")
    confidence: float = Field(ge=0.0, le=1.0, description="消解置信度")
    evidence_source: str = Field(description="证据来源，如 '对话历史第3轮'、'用户画像'")
    named_entity: str | None = Field(default=None, description="语义化名称（D7 按名索引），如 'TCC方案'")


# ---------------------------------------------------------------------------
# Structured output fields (D4)
# ---------------------------------------------------------------------------


class SubjectPredicateObject(BaseModel):
    """Subject-predicate-object structure."""

    subject: str | None = Field(default=None, description="主语")
    subject_source: str | None = Field(default=None, description="主语来源：原文/补全/对话历史第N轮")
    predicate: str | None = Field(default=None, description="谓语")
    obj: str | None = Field(default=None, description="宾语")
    obj_source: str | None = Field(default=None, description="宾语来源")


class ModifierExplanation(BaseModel):
    """Attributive/adverbial/complement explanation fields."""

    attributive: str | None = Field(default=None, description="定语解释")
    adverbial: str | None = Field(default=None, description="状语解释")
    complement: str | None = Field(default=None, description="补语解释")


class QuantifiableAdjective(BaseModel):
    """Quantifiable adjective fields."""

    adjective: str = Field(description="形容词，如 '最有性价比'")
    quantified: bool = Field(default=False, description="是否已量化")
    quantified_value: dict[str, Any] | None = Field(default=None, description="量化值，如 {'price_range': [150, 200]}")
    route_to: NormalizationStage | None = Field(default=None, description="路由目标：未量化时路由至 deep")


class TermMapping(BaseModel):
    """Term mapping table entry."""

    original: str = Field(description="原始术语，如 'RAG'")
    standard: str = Field(description="标准术语，如 '检索增强生成'")
    source: str = Field(default="vocabulary-table", description="来源")


class CompletionField(BaseModel):
    """Completion field and source."""

    field: str = Field(description="补全字段，如 '主语'")
    content: str = Field(description="补全内容")
    source: str = Field(description="补全来源，如 '对话历史第2轮'")


class CompletenessCheck(BaseModel):
    """Completeness check result (corresponds to D16)."""

    spo_complete: bool = Field(description="主谓宾是否完整")
    pronouns_resolved: bool = Field(description="代词是否完全消解")
    adjectives_quantified: bool = Field(description="形容词是否已量化")
    result: CompletenessStatus = Field(description="校验结果")


class NormalizationResult(BaseModel):
    """pre-normalization structured output (corresponds to D4 complete structure).

    This is the core data structure of the entire normalization pipeline, serving as a key fact throughout the dialogue lifecycle.
    """

    normalized_input: str = Field(description="规范化后的完整句子")
    spo: SubjectPredicateObject = Field(default_factory=SubjectPredicateObject, description="主谓宾")
    modifiers: ModifierExplanation = Field(default_factory=ModifierExplanation, description="定状补解释")
    pronoun_resolutions: list[PronounResolution] = Field(
        default_factory=list, description="指代消解表格（D6 跨轮复用）"
    )
    quantifiable_adjectives: list[QuantifiableAdjective] = Field(
        default_factory=list, description="可量化形容词字段"
    )
    term_mappings: list[TermMapping] = Field(default_factory=list, description="术语映射表")
    completions: list[CompletionField] = Field(default_factory=list, description="补全字段及来源")
    completeness: CompletenessCheck | None = Field(default=None, description="完整性校验结果")
    classification_tags: list[InputProblemType] = Field(default_factory=list, description="分类标签")
    route_to: NormalizationStage | None = Field(default=None, description="路由建议")
    raw_input: str = Field(default="", description="原始用户输入")


# ---------------------------------------------------------------------------
# Key fact storage models (D6 / D15)
# ---------------------------------------------------------------------------


class KeyFact(BaseModel):
    """Key fact (corresponds to D6 pronoun resolution table cross-turn reuse, D15 storage)."""

    fact_id: str = Field(description="事实唯一 ID")
    session_id: str = Field(description="会话 ID")
    turn: int = Field(description="对话轮次")
    fact_type: FactType = Field(description="事实类型")
    content: dict[str, Any] = Field(description="事实内容")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime | None = Field(default=None, description="过期时间，None 表示不过期")
    status: str = Field(default="active", description="active / expired / revoked")


# ---------------------------------------------------------------------------
# Vocabulary table models (D10 / D11 / D12)
# ---------------------------------------------------------------------------


class AlternativeMeaning(BaseModel):
    """Polysemous word alternative meanings."""

    industry: str = Field(description="行业，如 'IT'、'医疗'")
    meaning: str = Field(description="含义")


class VocabEntry(BaseModel):
    """Vocabulary entry (corresponds to D10 two-level vocabulary, D11 self-iteration, D12 industry profile)."""

    vocab_id: str = Field(description="词汇唯一 ID")
    term: str = Field(description="词汇/缩写/黑话，如 'CRM'、'抓手'")
    level: VocabLevel = Field(description="层级：public / personal")
    industry: str = Field(default="通用", description="所属行业")
    standard_meaning: str = Field(description="标准含义")
    alternative_meanings: list[AlternativeMeaning] = Field(
        default_factory=list, description="多义词备选含义"
    )
    source: str = Field(default="offline_analysis", description="来源")
    occurrence_count: int = Field(default=0, description="出现次数")
    discussant_count: int = Field(default=0, description="讨论人数")
    consecutive_count: int = Field(default=0, description="连续讨论次数")
    status: VocabStatus = Field(default=VocabStatus.CANDIDATE)
    user_id: str | None = Field(default=None, description="个人词汇表所属用户")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# few-shot example models (D5)
# ---------------------------------------------------------------------------


class FewShotExample(BaseModel):
    """few-shot example (corresponds to D5 retrieval injection)."""

    example_id: str = Field(description="例子唯一 ID")
    input_type: list[InputProblemType] = Field(default_factory=list, description="输入类型标签")
    input: str = Field(description="用户原始输入")
    context_summary: str = Field(default="", description="上下文摘要")
    normalized_output: dict[str, Any] = Field(description="规范化输出（JSON）")
    embedding: list[float] | None = Field(default=None, description="向量嵌入")
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# User profile models (D14)
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """User profile (corresponds to one of D14 three-layer contexts)."""

    user_id: str = Field(description="用户 ID")
    industry: str = Field(default="通用", description="所属行业")
    preferences: dict[str, Any] = Field(default_factory=dict, description="偏好")
    behavior_history: list[dict[str, Any]] = Field(default_factory=list, description="历史行为")
    topic_tendencies: dict[str, str] = Field(
        default_factory=dict,
        description="话题倾向，如 {'三国演义': '电视剧'}",
    )


# ---------------------------------------------------------------------------
# Dialogue history models (D14)
# ---------------------------------------------------------------------------


class DialogueTurn(BaseModel):
    """Single turn of dialogue history."""

    turn: int = Field(description="轮次号")
    role: str = Field(description="user / assistant / system")
    content: str = Field(description="内容")
    summary: str | None = Field(default=None, description="摘要")
    embedding: list[float] | None = Field(default=None, description="向量嵌入（用于 RAG 检索）")
    timestamp: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Clarification request models (D8)
# ---------------------------------------------------------------------------


class ClarificationRequest(BaseModel):
    """Clarification request (corresponds to D8 clarification mechanism)."""

    reason: ClarificationReason = Field(description="触发原因")
    item: str = Field(description="待澄清项，如 '那个帅气的同事'")
    candidates: list[str] = Field(default_factory=list, description="候选列表（可能为空）")
    question: str = Field(description="向用户展示的澄清问题")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="当前推断置信度")


# ---------------------------------------------------------------------------
# Quantification rule models (D13)
# ---------------------------------------------------------------------------


class QuantificationRule(BaseModel):
    """Adjective quantification rule (corresponds to D13 Spec judgment rules)."""

    adjective: str = Field(description="形容词，如 '性价比'")
    strategy: str = Field(description="量化策略描述")
    tool_params_template: dict[str, Any] = Field(description="工具参数模板")
    explanation: str = Field(default="", description="可解释性说明")


# ---------------------------------------------------------------------------
# Attribute retrieval pronoun resolution models (D9)
# ---------------------------------------------------------------------------


class AttributeResolutionResult(BaseModel):
    """Attribute + retrieval pronoun resolution result (corresponds to D9)."""

    pronoun: str = Field(description="属性指代，如 '看樱花的那个地方'")
    extracted_attributes: list[str] = Field(description="提取的属性关键词，如 ['樱花', '旅游']")
    recalled_details: list[str] = Field(default_factory=list, description="召回的历史对话细节")
    resolved_to: str = Field(description="消解结果，如 '鸡鸣寺'")
    confidence: float = Field(ge=0.0, le=1.0)
    compensation_used: bool = Field(default=False, description="是否使用了补偿机制")
    tool_called: str | None = Field(default=None, description="补偿机制调用的工具名")
