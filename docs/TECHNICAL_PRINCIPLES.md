# Technical Principles

> [中文技术原理文档](TECHNICAL_PRINCIPLES.zh-CN.md)
>
> Related: [HTTP API Reference](API.md)

This document details the technical principles, design philosophy, and theoretical foundations of the XIntent framework's two modules — **user input normalization** (sections 1-13) and **three-layer intent recognition** (sections 14-22). For each principle, the **service interfaces** and **code implementations** are marked.

---

## Table of Contents

### Module 1: User Input Normalization

- [1. Problem Background: Why Agents Can't Understand User Input](#1-problem-background-why-agents-cant-understand-user-input)
- [2. Linguistic Foundations of the Six Input Problem Categories](#2-linguistic-foundations-of-the-six-input-problem-categories)
- [3. Two-Stage Pipeline Design Rationale](#3-two-stage-pipeline-design-rationale)
- [4. Responsibility Boundary: Why LLMs "Over-Step"](#4-responsibility-boundary-why-llms-over-step)
- [5. Pronoun Resolution and Cross-Turn Memory](#5-pronoun-resolution-and-cross-turn-memory)
- [6. Named Indexing: Cognitive Load and Reference Accuracy](#6-named-indexing-cognitive-load-and-reference-accuracy)
- [7. Clarification Mechanism: Handling Uncertainty](#7-clarification-mechanism-handling-uncertainty)
- [8. Attribute Retrieval Resolution: RAG Long-Term Memory](#8-attribute-retrieval-resolution-rag-long-term-memory)
- [9. Vocabulary Table Self-Evolution: Emergent Knowledge](#9-vocabulary-table-self-evolution-emergent-knowledge)
- [10. Adjective Quantification: From Subjective to Objective](#10-adjective-quantification-from-subjective-to-objective)
- [11. Three-Layer Context Integration](#11-three-layer-context-integration)
- [12. Completeness Check: The Gatekeeper](#12-completeness-check-the-gatekeeper)
- [13. Few-Shot Retrieval Injection: Economics and Feedback Loop](#13-few-shot-retrieval-injection-economics-and-feedback-loop)

### Module 2: Three-Layer Intent Recognition

- [14. Why "Just Call a Big Model" Isn't Enough](#14-why-just-call-a-big-model-isnt-enough)
- [15. Three-Layer Waterfall Architecture](#15-three-layer-waterfall-architecture)
- [16. Confidence Routing and Multi-Signal Fusion](#16-confidence-routing-and-multi-signal-fusion)
- [17. Slot Filling: Hard/Soft Constraints and Cross-Turn Accumulation](#17-slot-filling-hardsoft-constraints-and-cross-turn-accumulation)
- [18. Rejection and Clarification: Dual Exits](#18-rejection-and-clarification-dual-exits)
- [19. Next-Turn Implicit Evaluation](#19-next-turn-implicit-evaluation)
- [20. Intent Boundary Definition and Few-Shot Injection](#20-intent-boundary-definition-and-few-shot-injection)
- [21. Hierarchical Intents and Decision Tree](#21-hierarchical-intents-and-decision-tree)
- [22. Evaluation Metrics: 4-Tier System and 9-Scenario Test Set](#22-evaluation-metrics-4-tier-system-and-9-scenario-test-set)

---

## 1. Problem Background: Why Agents Can't Understand User Input

### The Real-World Challenge

In real-world scenarios, user input is highly casual-just like everyday speech. Users produce broken sentences, inversions, pronouns, abbreviations, and even "slang." This is not a user problem; it's the fundamental nature of natural language: **natural language is redundant, ambiguous, and context-dependent**.

Traditional software constrains user input through UI (forms, dropdowns, validation rules). Agents, however, directly receive natural language text, losing the "input normalization" protection that the UI layer provides. This leads to:

- **Intent recognition failure**: Missing subjects cause intent drift ("市场占有率多少？" - whose market share?)
- **Context breakdown**: Cross-turn anaphora cannot be resolved (what does "第二个" in turn 3 refer to?)
- **Parameter drift**: Unquantified adjectives cause uncontrolled tool call parameters ("性价比" interpreted as "cheap")
- **Fact fabrication**: LLM "hallucinations" invent non-existent information

### Analogy: Requirements Analysis

User input normalization is the Agent equivalent of "requirements analysis" in traditional software engineering. In traditional SE, requirements analysis is the foundation for all subsequent design, development, and testing; in Agents, input normalization is the foundation for intent recognition, tool calling, and final output. **If the input isn't normalized, everything downstream is wasted effort**.

### Implementation

The entire framework is exposed as a service through `POST /normalize`. The user passes raw text, and the service returns a structured normalization result:

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "user_id": "u1", "turn": 2}'
```

- **Service entry**: `POST /normalize` (see [API Reference](API.md#3-post-normalize--user-input-normalization))
- **Orchestrator**: `NormalizationPipeline` (`src/user_input_normalization/pipeline.py`)
- **Server startup**: `python -m user_input_normalization.server` (`src/user_input_normalization/server.py`)

---

## 2. Linguistic Foundations of the Six Input Problem Categories

The framework categorizes user input problems into six types, each with corresponding linguistic/cognitive science foundations:

### 2.1 Anaphora

**Linguistic basis**: "Anaphora Resolution" in pragmatics. Pronouns have no fixed meaning; their referents depend on context.

**Typical manifestations**: Ordinal anaphora ("第二个适合生产吗？"), temporal anaphora ("刚才那个项目怎么包装？"), attribute anaphora ("看樱花的那个地方")

**Challenge**: Resolution requires recalling historical dialogue details; short-term memory fails across multi-turn, long time-span conversations.

### 2.2 Missing (Ellipsis)

**Linguistic basis**: Ellipsis is a common economy principle in natural language-speakers omit parts that can be inferred from shared context.

**Typical manifestations**: Missing subject ("市场占有率多少？"), missing object ("帮我优化一下。"), missing constraints ("推荐一条裤子。")

### 2.3 Expression

**Linguistic basis**: The fundamental difference between spoken and written language. Speech is linear, jumping, and self-correcting; writing is structured and complete.

**Typical manifestations**: Disordered word order ("这个不太行，换个更像面试能讲的。"), mid-stream correction ("不是，我说的是另一个。")

### 2.4 Semantic

**Linguistic basis**: "Polysemy" and "Jargon" in semantics. The same word has different meanings in different domains.

**Typical manifestations**: Abbreviations (RAG, CRM), slang (抓手、赋能、闭环、不够 P8), synonyms (知识库 / 检索增强 / RAG)

### 2.5 Subjective

**Linguistic basis**: Subjectivity and Appraisal Theory. Judgment word meanings depend on personal preferences, scenarios, and standards.

**Typical manifestations**: "哪个最有性价比？", "再高级一点。"

**Challenge**: Subjective judgment words must be converted to quantifiable parameters, otherwise tool calls are uncontrollable. Many Agents handle this crudely: "性价比高" directly equals "low price," ignoring quality, performance, brand, and after-sales.

### 2.6 External Fact

**Linguistic basis**: Expressions referring to external world states, whose truth values depend on real-time data.

**Typical manifestations**: "最近哪个框架更火？", "现在最便宜的是哪个？"

**Challenge**: Cannot be completed in the pre-processing stage; must rely on tool calls returning real-time data. Cannot be fabricated.

### Implementation

Classification results are returned via the `classification_tags` field of the API response (array, multi-label supported):

```json
{
  "classification_tags": ["指代问题", "主观判断问题"]
}
```

- **API response field**: `classification_tags` (see [API Reference - classification_tags possible values](API.md#classification_tags-possible-values))
- **Classifier**: `InputClassifier` (`src/user_input_normalization/classification/classifier.py`)
- **Classification rules**: `CLASSIFICATION_RULES` (`src/user_input_normalization/classification/rules.py`, with 82 semantic rules + 80 known terms)
- **Routing logic**: `SUBJECTIVE`/`EXTERNAL_FACT` -> `deep` stage; others -> `pre` stage

---

## 3. Two-Stage Pipeline Design Rationale

### The Core Contradiction

User input normalization has a fundamental contradiction:
- **Some problems** (like "第二个" anaphora) can be solved immediately before intent recognition
- **Some problems** (like "现在最便宜的" external facts) can only be solved after tool calls

If everything is done in pre-processing: cross-time anaphora and external facts cannot be handled.
If everything is deferred to the ReAct loop: simple anaphora also waits for tool calls, adding unnecessary latency.

### Solution: Two-Stage Division of Labor

```
User Input -> [pre-normalization] -> Intent Recognition -> [deep-normalization] -> Output
                ↑                                           ↑
           Immediately solvable                      Needs context/tools
```

| Stage | Timing | Handles | Does NOT handle |
|-------|--------|---------|-----------------|
| pre | Before intent recognition | Anaphora, ellipsis, sentence correction, term standardization | Quantification needing tool returns, external facts |
| deep | Inside ReAct loop | Adjective quantification, external fact resolution, Observation re-resolution | What's already done in pre |

### Implementation

The `observation` parameter of the API request controls whether to enter the deep stage, and the `stage_reached` field of the response indicates the actual stage reached:

```bash
# pre stage only (no observation)
curl -X POST http://localhost:8000/normalize \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "turn": 2}'
# -> "stage_reached": "pre"

# Trigger deep stage (with observation)
curl -X POST http://localhost:8000/normalize \
  -d '{"raw_input": "帮我推荐更有性价比的牛仔裤", "session_id": "s2", "turn": 1,
       "observation": {"current_price": 200}}'
# -> "stage_reached": "deep"
```

- **API request field**: `observation` (optional, passing tool return data triggers the deep stage)
- **API response field**: `stage_reached` (`"pre"` or `"deep"`)
- **Orchestrator**: `NormalizationPipeline.process()` (`src/user_input_normalization/pipeline.py`)
- **pre stage**: `PreNormalizer` (`src/user_input_normalization/pre_normalization/normalizer.py`)
- **deep stage**: `DeepNormalizer` (`src/user_input_normalization/deep_normalization/normalizer.py`)

---

## 4. Responsibility Boundary: Why LLMs "Over-Step"

### The LLM "Over-Autonomy" Problem

LLMs are trained to be "helpful assistants," tending to complete user requests as much as possible. Without constraints, it will do input normalization, intent recognition, and answering all at once:

- User asks "市场占有率多少？", LLM directly fabricates a number to answer
- User asks "帮我推荐裤子", LLM directly recommends products
- User asks "现在最便宜的", LLM fabricates real-time prices

This is disastrous in the normalization stage: the responsibility of normalization is to **understand the input**, not to **answer the question**.

### Solution: Explicit "Can Do / Cannot Do" List

| Can Do (8 items) | Cannot Do (8 prohibitions) |
|------------------|---------------------------|
| Pronoun resolution | Do not directly answer user questions |
| Ellipsis completion | Do not execute tools |
| Sentence correction | Do not make recommendations |
| Term standardization | Do not judge business intent |
| Slang explanation | Do not generate complete solutions |
| Polysemy disambiguation | Do not fabricate facts |
| Determine if clarification needed | Do not guess low-confidence content |
| Determine if search needed | Do not fabricate real-time information |

**Principle**: Explicit lists are the simplest and most effective constraint method. Vague role descriptions (like "you are an assistant") are insufficient to constrain LLM behavior. Lists convert implicit expectations into explicit rules, enabling auditing and regression testing.

### Implementation

The responsibility boundary is injected as a hard constraint via the System Prompt, and the normalization result will **not contain** direct answers to user questions:

- **Prompt constants**: `SYSTEM_PROMPT`, `CAN_DO`, `CANNOT_DO` (`src/user_input_normalization/pre_normalization/prompts.py`)
- **Boundary validation**: `PreNormalizer.validate_boundary()` checks for over-stepping (`normalizer.py`)
- **Regression tests**: `tests/test_pre_normalization.py::TestResponsibilityBoundary` verifies no answering / no tool execution / no fact fabrication

---

## 5. Pronoun Resolution and Cross-Turn Memory

### Pronoun Resolution Table as "Key Facts"

Pronoun resolution is not a one-time operation-results need **cross-turn reuse**:

```
Turn 3: "第一个方案适合生产吗？" -> Resolve: 第一个方案 -> TCC方案
Turn 5: "那个 TCC 方案成本多少？" -> Direct hit by name indexing
Turn 7: "第一个呢？" -> Reuse turn 3 resolution
```

If re-resolved every turn: high cost, potential inconsistency, poor user experience.

### Implementation

Resolution results are returned via the `pronoun_resolutions` field of the API response, and are automatically reused across turns under the same `session_id`:

```json
{
  "pronoun_resolutions": [
    {
      "pronoun": "第二个",
      "resolved_to": "TCC方案",
      "confidence": 0.95,
      "evidence_source": "对话历史第3轮",
      "named_entity": "TCC方案"
    }
  ]
}
```

- **API response field**: `pronoun_resolutions` (see [API Reference - pronoun_resolutions item structure](API.md#pronoun_resolutions-item-structure))
- **Data model**: `PronounResolution` (`src/user_input_normalization/models.py`)
- **Cross-turn storage**: `KeyFactStore.find_pronoun_resolution()` (`src/user_input_normalization/storage/base.py`)
- **Write logic**: `PreNormalizer._write_pronoun_facts()` (`normalizer.py`)
- **Cross-turn reuse tests**: `tests/test_pre_normalization.py::TestCrossTurnReuse`

---

## 6. Named Indexing: Cognitive Load and Reference Accuracy

### The Problem with Ordinal References

If the Agent outputs "方案一、方案二、方案三" and the user later references "第二个", the LLM is **prone to reference errors** during inference: ordinals have no semantics, and LLM attention may misalign.

### Advantages of Semantic Naming

If the Agent outputs "TCC方案、轻量方案、企业方案" and the user references "TCC方案": name-based indexing has higher match accuracy, lower cognitive load, and avoids inference errors.

**Principle**: This is the "rectification of names" concept-if names are not correct, language cannot flow; if language cannot flow, things cannot be done. Ordinals are weak identifiers; semantic names are strong identifiers.

### Implementation

- **API response field**: `pronoun_resolutions[].named_entity` (semantic name)
- **Naming logic**: `PreNormalizer._ensure_named_entities()` (`normalizer.py`)
- **Named indexing tests**: `tests/test_pre_normalization.py::TestPronounResolution::test_named_entity_set`

---

## 7. Clarification Mechanism: Handling Uncertainty

### Why Can't "Guess"

Error accumulation is a real risk: Turn 1 guesses wrong -> Turn 2 continues reasoning based on the wrong result -> final answer completely deviates.

### Confidence Threshold Mechanism

```
Confidence >= 0.6 -> Resolve directly
Confidence < 0.6  -> Trigger clarification, ask user
```

**Principle**: This is "better to not answer than to answer wrong." The threshold θ_clarify is configurable (default 0.6), balancing efficiency and accuracy.

### Implementation

When `paused_for_clarification` is `true`, the response contains a `clarification` object, and the client should pause processing and present the clarification question to the user:

```json
{
  "paused_for_clarification": true,
  "clarification": {
    "reason": "代词消解失败",
    "item": "那个帅气的同事",
    "candidates": [],
    "question": "您说的'那个帅气的同事'具体是指什么？请提供更多信息。",
    "confidence": 0.2
  }
}
```

- **API response fields**: `paused_for_clarification`, `clarification` (see [API Reference - clarification structure](API.md#clarification-structure-when-paused_for_clarification-is-true))
- **Handler**: `ClarificationHandler` (`src/user_input_normalization/clarification/handler.py`)
- **Config items**: `theta_clarify=0.6`, `max_consecutive_clarifications=3` (`src/user_input_normalization/config.py`)
- **Resume entry**: `NormalizationPipeline.resume_after_clarification()` (`pipeline.py`)

---

## 8. Attribute Retrieval Resolution: RAG Long-Term Memory

### Problem Scenario

Before a holiday, the user asks the Agent to recommend a travel destination. The Agent recommends Jiming Temple (鸡鸣寺) for cherry blossoms. After the holiday, the user inputs: "你上次推荐的看樱花的那个地方……"

Challenges: long time span (short-term memory has expired) + attribute anaphora ("看樱花的" is an attribute modifier).

### Two-Step Inference + Compensation Mechanism

1. **Attribute extraction + vector search**: Extract ["樱花", "旅游"], search historical dialogue store to recall details
2. **LLM inference**: Infer based on window content => "鸡鸣寺"
3. **Compensation mechanism** (when search fails): Extract attributes to issue tool calls, then re-reason

### Implementation

Attribute retrieval resolution is automatically triggered in the pipeline (when an attribute anaphora pattern is detected), and resolution results are written back to `pronoun_resolutions`:

- **Resolver**: `AttributeResolver` (`src/user_input_normalization/attribute_resolution/resolver.py`)
- **Two-step inference**: `AttributeResolver.resolve()` -> `extract_attributes()` -> `recall_details()` -> `infer()`
- **Compensation mechanism**: `AttributeResolver.compensation()` (simulated tool call + re-reasoning)
- **Trigger condition**: `NormalizationPipeline._is_attribute_anaphora()` detects attribute anaphora patterns
- **Tests**: `tests/test_attribute_resolution.py::TestJimingTempleCase` (Jiming Temple end-to-end case)

---

## 9. Vocabulary Table Self-Evolution: Emergent Knowledge

### Problem

The same word/abbreviation has different meanings in different industries, domains, and contexts. Manually maintaining vocabulary tables is unrealistic-new words constantly emerge.

### Self-Iteration Mechanism

```
Online dialogue -> Offline analysis -> Mark candidate terms -> Threshold check -> Promote to vocab table
```

**Threshold rules**: Total count > 100 / Discussant count > 3 (public vocabulary) / Consecutive discussion > 10

**Principle**: This is "emergent knowledge"-the vocabulary table emerges bottom-up from actual dialogue, threshold filtering ensures stability, and human review serves as a backstop.

### Implementation

Term standardization results are returned via the `term_mappings` field of the API response:

```json
{
  "term_mappings": [
    {"original": "RAG", "standard": "检索增强生成", "source": "vocabulary-table"}
  ]
}
```

- **API response field**: `term_mappings` (see [API Reference - term_mappings item structure](API.md#term_mappings-item-structure))
- **Vocabulary service**: `VocabularyTable` (`src/user_input_normalization/vocabulary/table.py`)
- **Offline analyzer**: `OfflineAnalyzer` (`src/user_input_normalization/vocabulary/offline_analyzer.py`)
- **Storage interface**: `VocabStore` (`src/user_input_normalization/storage/base.py`)
- **Config items**: `min_total_count=100`, `min_discussant_count=3`, `min_consecutive_count=10` (`config.py`)

---

## 10. Adjective Quantification: From Subjective to Objective

### Problem

Any unquantified adjective causes Agent output drift. Many Agents handle this crudely: "性价比高" directly equals "low price," ignoring quality, performance, brand, and after-sales.

### Quantification Strategies

"性价比" has two reasonable interpretations:

| Strategy | Meaning | Tool Parameters |
|----------|---------|-----------------|
| Same price, better quality | At the same price point, better quality/config/service | `price_range: [150, 200], quality_rank: top 30%` |
| Same quality, lower price | At the same quality level, lower price | `price_range: [100, 150], quality_rank: top 50%` |

### Implementation

Quantification results are returned via the `quantifiable_adjectives` field of the API response, and the request must include `observation` (e.g. current price) to complete quantification:

```bash
curl -X POST http://localhost:8000/normalize \
  -d '{"raw_input": "帮我推荐更有性价比的牛仔裤", "session_id": "s2", "turn": 1,
       "observation": {"current_price": 200}}'
```

```json
{
  "quantifiable_adjectives": [
    {
      "adjective": "性价比",
      "quantified": true,
      "quantified_value": {"price_range": [150, 200], "quality_rank": "top 30%"},
      "route_to": null
    }
  ],
  "stage_reached": "deep"
}
```

- **API request field**: `observation` (passes current price and other context to trigger deep-stage quantification)
- **API response field**: `quantifiable_adjectives` (see [API Reference - quantifiable_adjectives item structure](API.md#quantifiable_adjectives-item-structure))
- **Quantification engine**: `QuantificationEngine` (`src/user_input_normalization/quantification/engine.py`)
- **Built-in rules**: `DEFAULT_RULES` (6 rules: 性价比, 划算, 再高级一点, etc.) (`quantification/rules.py`)
- **Deep normalization**: `DeepNormalizer.quantify_adjectives()` (`deep_normalization/normalizer.py`)

---

## 11. Three-Layer Context Integration

### Core Idea

> All normalization measures can be summarized in one sentence: deeply integrate context.

| Layer | Content | Source | Purpose |
|-------|---------|--------|---------|
| User Profile | Industry, preferences, behavior history | User profile store | Disambiguation tendency |
| Key Facts | Preferences, opinions, attitudes, acknowledged points | Key fact store | Cross-turn reuse, completion basis |
| Dialogue History | Summaries + details | Short-term memory + RAG long-term memory | Anaphora resolution evidence, ellipsis completion source |

### Implementation

The three-layer context is automatically assembled inside `PreNormalizer.normalize()`, associated via `user_id` and `session_id`:

- **Integrator**: `ContextIntegrator.assemble()` (`src/user_input_normalization/context/integrator.py`)
- **Context model**: `ContextBundle` (user_profile + key_facts + dialogue_summary + recalled_details)
- **Priority conflict**: `ContextIntegrator.resolve_conflict()` (Key Facts > User Profile > Dialogue History)
- **Storage interfaces**: `UserProfileStore`, `KeyFactStore`, `DialogueHistoryStore` (`storage/base.py`)
- **Observability**: `ContextIntegrator.explain_influence()` returns an influence report for each layer

---

## 12. Completeness Check: The Gatekeeper

### Why Check Is Needed

Normalization is not "call the LLM once and done"-the LLM may miss things: incomplete subject-predicate-object, unresolved pronouns, unquantified adjectives.

### Three Checks + Routing

```
Check passes -> Flow to intent recognition
Missing SPO / unresolved pronouns -> Trigger clarification (paused_for_clarification: true)
Unquantified adjectives -> Route to deep-normalization (stage_reached: "deep")
```

### Implementation

Check results affect the `stage_reached` and `paused_for_clarification` fields of the API response:

- **Checker**: `CompletenessChecker` (`src/user_input_normalization/pre_normalization/completeness_checker.py`)
- **Check method**: `check_and_route()` returns `(check_result, route_target, route_reason)`
- **Check items**: `spo_complete` (SPO), `pronouns_resolved` (pronouns), `adjectives_quantified` (adjectives)
- **API impact**: Unquantified adjectives -> `stage_reached: "deep"`; unresolved pronouns -> `paused_for_clarification: true`

---

## 13. Few-Shot Retrieval Injection: Economics and Feedback Loop

### Problem

Few-shot examples significantly improve normalization quality, but injecting all examples would overflow the context window.

### Retrieval Injection + Online Sinking

```
User input -> Vectorize -> Search top-k similar examples -> Inject into Prompt
Online encounters strange input -> Successfully resolved -> Organize as few-shot example -> Save to library -> Next similar input can reference
```

**Principle**: This is a "data flywheel"-the longer the system runs, the richer the example library, and the higher the normalization quality.

### Implementation

Few-shot injection is automatically executed inside `PreNormalizer.normalize()`, transparent to API callers:

- **Example storage**: `FewShotStore` (`src/user_input_normalization/storage/base.py`)
- **Retrieval injection**: `PreNormalizer._build_user_prompt()` calls `FewShotStore.search(top_k=3)`
- **Online sinking**: `PreNormalizer._sink_to_fewshot()` automatically saves strange inputs to the library
- **Formatting**: `format_fewshot()` (`pre_normalization/prompts.py`)
- **Config items**: `top_k=3`, `enabled=True` (`config.py`)

---

## Summary

The core ideas of user input normalization (Module 1) can be summarized as:

1. **Deeply integrate context** - all measures revolve around this one sentence
2. **Two-stage division of labor** - what can be solved immediately goes to pre; what needs tools goes to deep
3. **Explicit constraints** - responsibility boundaries, completeness checks, clarification thresholds prevent LLM over-stepping and guessing
4. **Structured output** - resolution tables, SPO, quantification fields enable downstream programmatic consumption
5. **Self-evolution** - vocabulary table iteration, few-shot sinking make the system smarter with use

The core ideas of three-layer intent recognition (Module 2) can be summarized as:

1. **Waterfall escalation** - start cheap and fast, escalate only when needed (Code → Flash → Pro)
2. **Multi-signal confidence** - never trust LLM self-reported confidence alone; fuse with rule match, vector similarity, historical accuracy
3. **Slot filling as part of recognition** - intent + parameters + constraints are one structured output, not separate stages
4. **Dual exits for uncertainty** - unsupported intent is rejected with reason; unclear intent triggers a bounded clarification loop
5. **Closed-loop evaluation** - test set + next-turn implicit signals feed back into metrics for continuous improvement

### Principle-to-Interface Mapping Table

#### Module 1: Normalization

| Technical Principle | API Request Field | API Response Field | Core Class |
|---------------------|-------------------|--------------------|------------|
| Six-category classification | - | `classification_tags` | `InputClassifier` |
| Two-stage pipeline | `observation` | `stage_reached` | `NormalizationPipeline` |
| Responsibility boundary | - | (no answer content appears) | `SYSTEM_PROMPT` / `CAN_DO` / `CANNOT_DO` |
| Pronoun resolution | `session_id` | `pronoun_resolutions` | `PreNormalizer` / `KeyFactStore` |
| Named indexing | - | `pronoun_resolutions[].named_entity` | `PreNormalizer` |
| Clarification mechanism | - | `paused_for_clarification` / `clarification` | `ClarificationHandler` |
| Attribute retrieval resolution | `session_id` | `pronoun_resolutions` (written back) | `AttributeResolver` |
| Vocabulary table | - | `term_mappings` | `VocabularyTable` / `OfflineAnalyzer` |
| Adjective quantification | `observation` | `quantifiable_adjectives` | `QuantificationEngine` / `DeepNormalizer` |
| Ellipsis completion | - | `completions` | `PreNormalizer` |
| Completeness check | - | `stage_reached` / `paused_for_clarification` | `CompletenessChecker` |
| Few-shot injection | - | (automatic internally) | `FewShotStore` / `PreNormalizer` |

#### Module 2: Intent Recognition

| Technical Principle | API Request Field | API Response Field | Core Class |
|---------------------|-------------------|--------------------|------------|
| Three-layer waterfall | `text` | `source` / `layer_reached` | `IntentRecognitionPipeline` |
| Confidence routing | - | `confidence` / `need_clarification` | `ConfidenceRouter` |
| Multi-signal fusion | - | `confidence` | `MultiSignalFuser` |
| Slot filling | - | `slots` / `missing_slots` | `SlotExtractor` |
| Hard/soft constraints | - | `hard_constraints` / `soft_constraints` | `ConstraintExtractor` |
| Cross-turn slot accumulation | `session_id` / `turn` | `slots` (accumulated) | `CrossTurnSlotMerger` / `SlotStateStore` |
| Rejection (unsupported) | - | `rejection_reason` | `RejectionClarificationHandler` |
| Clarification (unclear) | - | `need_clarification` / `clarification_question` | `RejectionClarificationHandler` |
| Intent switching | `session_id` | `intent_switched` / `previous_intent` | `IntentHistoryStore` |
| Page guidance (code layer) | `event` | `intent` / `source: "code-layer"` | `PageGuidanceMatcher` |
| Hierarchical intents | - | (intent with `parent_intent`) | `IntentRegistry` |
| Evaluation metrics | - | (via `TestRunner`, not HTTP) | `MetricsCalculator` / `TestSet` |

---

## 14. Why "Just Call a Big Model" Isn't Enough

### The Interview Question

When asked "how do you do intent recognition in your Agent?" many candidates answer "call a big model to classify." In engineering this is wrong for three reasons:

1. **Cost and latency**: A single deep reasoning call can take 5-15 seconds. Most user inputs ("continue", "next", "refund") are unambiguous and don't need it.
2. **Confidence is not a probability**: LLM self-reported `confidence` is a "self-judgment score," not a calibrated probability. Treating it as ground truth leads to over-acceptance of wrong answers.
3. **No dual exits**: A single LLM call produces one answer. Engineering needs explicit handling for "unsupported" (intent not in candidate list) and "unclear" (low confidence or missing required slots) — two distinct failure modes.

### Implementation

The framework introduces a three-layer waterfall that addresses all three concerns:

- **Service entry**: `POST /recognize` (see [API Reference](API.md#4-post-recognize--intent-recognition))
- **Orchestrator**: `IntentRecognitionPipeline` (`src/intent_recognition/pipeline.py`)
- **Design decisions**: D1 (waterfall), D2 (independent stage), D10 (dual exits)

---

## 15. Three-Layer Waterfall Architecture

### The Cascade

```
Normalized Input -> [L1 Code Layer] --miss--> [L2 Lightweight LLM] --low conf--> [L3 Deep LLM]
                        |                              |                              |
                        +------- hit (conf=1.0) ------+---------- high conf --------+
                                                       |
                                              [Slot Filling] -> [Rejection / Clarification]
                                                       |
                                          IntentRecognitionResult
```

**Layer 1 - Code Layer** (D3): Three recognition methods — page guidance (UI event → intent), keyword/regex matching, rule engine (context state + normalized input). Returns first match or "unmatched" signal. **Zero LLM calls, < 10ms.**

**Layer 2 - Lightweight LLM** (D4, D5): Flash/Mini model with structured prompt (candidate intents + slots + few-shot). Confidence routing: `>=0.85` accept, `0.6-0.85` clarify, `<0.6` escalate to L3.

**Layer 3 - Deep LLM** (D6): Deep reasoning model for 5 complex scenarios — complex expressions (病句/倒装), cross-turn context dependency, intent switching, multi-intent decomposition, implicit info completion.

### Why Waterfall, Not Parallel

Parallel execution of all three layers would 3× API cost, and Flash model confidence vs. Pro model confidence are not directly comparable. The waterfall escalates only on miss or low-confidence, so most requests stop at L1 or L2.

### Graceful Degradation

If L3 fails (timeout, API error), the pipeline falls back to the L2 result with a warning log, rather than failing the whole request.

### Implementation

- **L1 Code Layer**: `src/intent_recognition/code_layer/` (`classifier.py`, `page_guidance.py`, `keyword_matcher.py`, `rule_engine.py`)
- **L2 Lightweight LLM**: `src/intent_recognition/lightweight_llm/` (`classifier.py`, `confidence_router.py`, `multi_signal.py`, `prompts.py`)
- **L3 Deep LLM**: `src/intent_recognition/deep_llm/` (`classifier.py`, `prompts.py`)
- **Orchestrator**: `IntentRecognitionPipeline.recognize()` (`src/intent_recognition/pipeline.py`)
- **Config flags**: `enable_code_layer`, `enable_lightweight_llm`, `enable_deep_llm` (`src/intent_recognition/config.py`)

---

## 16. Confidence Routing and Multi-Signal Fusion

### The Problem with LLM Confidence

LLM self-reported confidence is a "self-judgment score" — it reflects the model's internal certainty, not a calibrated probability. Two failure modes:

- **Over-confident on wrong answers**: Model says 0.9 confidence on a misclassified intent
- **Under-confident on right answers**: Model says 0.5 on a correct intent, triggering unnecessary escalation

### Multi-Signal Fusion (D5)

The framework fuses four signals:

| Signal | Weight | Source |
|--------|--------|--------|
| LLM confidence | 0.5 | Model output |
| Rule match | 0.2 | Code layer keyword/regex hit on same intent |
| Vector similarity | 0.2 | Embedding similarity to positive examples |
| Historical accuracy | 0.1 | Per-intent accuracy from evaluation store |

The fused confidence is then routed:

- `>= 0.85` → accept
- `>= 0.6 and < 0.85` → clarify (ask user)
- `< 0.6` → escalate to L3

### Implementation

- **Confidence router**: `ConfidenceRouter` (`src/intent_recognition/lightweight_llm/confidence_router.py`)
- **Multi-signal fusion**: `MultiSignalFuser` (`src/intent_recognition/lightweight_llm/multi_signal.py`)
- **Config**: `ConfidenceConfig` (`src/intent_recognition/config.py`) — `accept_threshold=0.85`, `clarify_threshold=0.6`, weight fields

---

## 17. Slot Filling: Hard/Soft Constraints and Cross-Turn Accumulation

### Slot Filling as Part of Recognition (D7)

Intent recognition is not just "what intent" — it's "what intent + what parameters + what constraints." Slot filling is integrated into the recognition pipeline, not a separate stage.

### Hard vs Soft Constraints (D8)

| Constraint Type | Examples | Semantics |
|----------------|----------|-----------|
| Hard | "不超过3000", "必须华为", "不要二手" | Must satisfy; filter results |
| Soft | "最好轻薄", "优先蓝色", "续航越长越好" | Best effort; sort or boost |

The framework uses regex patterns for common Chinese constraint expressions, falling back to LLM extraction for complex cases. Constraints are returned as normalized expressions (e.g., `price<=3000`) plus raw text for traceability.

### Cross-Turn Accumulation (D9)

In a ReAct/TAO loop, users iteratively refine their request across turns:

```
Turn 1: "帮我推荐一款3000元以内的手机"  -> {category: 手机, budget_max: 3000}
Turn 2: "预算可以到4000"                -> {category: 手机, budget_max: 4000}  (latest-wins)
Turn 3: "最好是华为"                    -> {category: 手机, budget_max: 4000, brand: 华为}  (accumulate)
```

**Latest-wins strategy**: When the same slot appears in a new turn, the new value overwrites the old. Conflict is recorded for debugging.

**Constraint accumulation**: Constraints don't overwrite — unique expressions are appended. `has_constraint_conflict()` detects contradictory hard constraints (e.g., `price<100` vs `price>200`).

### Implementation

- **Slot extractor**: `SlotExtractor` (`src/intent_recognition/slot_filling/extractor.py`)
- **Constraint extractor**: `ConstraintExtractor` (`src/intent_recognition/slot_filling/constraints.py`)
- **Cross-turn merger**: `CrossTurnSlotMerger` (`src/intent_recognition/slot_filling/cross_turn.py`)
- **Slot state store**: `SlotStateStore` (`src/intent_recognition/storage/base.py`, memory impl in `memory.py`)
- **API fields**: `slots`, `missing_slots`, `hard_constraints`, `soft_constraints`

---

## 18. Rejection and Clarification: Dual Exits

### Two Distinct Failure Modes

| Mode | Trigger | Response |
|------|---------|----------|
| Unsupported (拒识) | Intent not in candidate list | `rejection_reason` with supported intents listed |
| Unclear (澄清) | Low confidence OR missing required slots | `clarification_question` to ask user |

A single "I don't know" answer conflates these two — the user can't tell whether to rephrase (unsupported) or provide more info (unclear).

### Rejection: Tell User What IS Supported

When the intent is not in the candidate list, the rejection reason includes:

1. The unsupported content (echoed back)
2. The list of supported intents with descriptions and positive examples
3. The redirect hints from negative examples (e.g., `"iPhone 16的电池容量是多少 -> product_query"`)

### Clarification: Bounded Loop with Convergence Tracking

Clarification is not infinite. The framework tracks consecutive clarifications per session:

- After `max_consecutive_clarifications` (default 3), the system degrades to "best guess" mode — accepts the highest-confidence candidate rather than asking again.
- If the user's next turn resolves the clarification (e.g., provides the missing slot), the consecutive counter resets.

### Implementation

- **Handler**: `RejectionClarificationHandler` (`src/intent_recognition/rejection_clarification/handler.py`)
- **Config**: `ClarificationConfig.max_consecutive_clarifications=3` (`src/intent_recognition/config.py`)
- **API fields**: `rejection_reason`, `need_clarification`, `clarification_question`

---

## 19. Next-Turn Implicit Evaluation

### The Problem with Explicit Feedback

Asking users "did I get that right?" after every intent recognition is terrible UX. Most users won't answer, and those who do are biased toward reporting errors, not successes.

### Implicit Signal Detection (D11)

The framework detects failure signals in the **next turn's input**:

```
Turn N: "帮我推荐手机"  -> intent: product_recommendation
Turn N+1: "你理解错了，我要查订单"  -> failure signal detected for Turn N
```

Default failure signals: `"你理解错了"`, `"不是这个意思"`, `"不是，我不是"`, `"不对，我要的是"`.

When a failure signal is detected:

1. The previous turn's intent is marked as incorrect in the evaluation store
2. The current turn's intent is recognized fresh (the failure signal often contains the correct intent)

### Convergence Tracking

For clarification loops, the framework tracks whether the user's task progressed after clarification:

- **Resolved**: Next turn's input is a normal request (not a failure signal, not another clarification) → reset consecutive counter
- **Unresolved**: Next turn's input is another clarification response → increment consecutive counter

### Implementation

- **Failure signal detection**: `RejectionClarificationHandler.detect_implicit_failure()` (`src/intent_recognition/rejection_clarification/handler.py`)
- **Config**: `IntentRecognitionConfig.failure_signals` (`src/intent_recognition/config.py`)
- **Evaluation store**: `EvaluationStore` (`src/intent_recognition/storage/base.py`)

---

## 20. Intent Boundary Definition and Few-Shot Injection

### The Boundary Problem

Intents overlap. "iPhone 16的电池容量是多少" could be `product_query` (query specs) or `product_recommendation` (suggest based on battery). Without explicit boundaries, the LLM will guess inconsistently.

### Positive + Negative Examples (D12)

Each `IntentDefinition` includes:

- **Positive examples**: Inputs that definitely belong to this intent
- **Negative examples**: Inputs that look similar but belong to a different intent, with redirect target (e.g., `"3000以内有什么手机推荐 -> product_recommendation"`)

These are injected into the L2 prompt to define boundaries.

### Few-Shot Retrieval Injection

For confusing intent pairs, the framework retrieves the top-k most similar historical examples (from `FewShotStore`) and injects them into the prompt. This is the same mechanism as Module 1's few-shot injection (section 13), reused across modules.

### Implementation

- **Intent registry**: `IntentRegistry` (`src/intent_recognition/intent_registry.py`) — stores `IntentDefinition` with positive/negative examples
- **Prompt builder**: `build_prompt()` in `lightweight_llm/prompts.py` — injects candidate intents, slots, boundaries, few-shot
- **Few-shot store**: `FewShotStore` (shared with Module 1, `src/user_input_normalization/storage/base.py`)
- **Config**: `fewshot_top_k=3`, `fewshot_enabled=True` (`src/intent_recognition/config.py`)

---

## 21. Hierarchical Intents and Decision Tree

### The Context Window Problem

When the intent space grows beyond ~20 intents, listing all of them in the LLM prompt overflows the context window and degrades accuracy.

### Hierarchical Dispatch (D13)

The framework supports parent/child intent relationships:

```python
IntentDefinition(name="refund", parent_intent="order_query", ...)
```

When hierarchical dispatch is enabled:

1. L1/L2 first recognizes the parent intent (e.g., `order_query`)
2. The candidate list is narrowed to children of `order_query` (e.g., `refund`, `exchange`, `cancel`)
3. L1/L2 runs again on the narrowed list

This turns a flat 100-intent problem into two 10-intent problems.

### When to Enable

Hierarchical dispatch is **off by default** (`enable_hierarchical=False`) because it adds latency (two recognition passes). Enable it only when the intent space is large enough to hurt L2 accuracy.

### Implementation

- **Intent registry**: `IntentRegistry` (`src/intent_recognition/intent_registry.py`) — `parent_intent` field, `get_children()` method
- **Config flag**: `enable_hierarchical` (`src/intent_recognition/config.py`)
- **Default intents**: The server's `_build_default_registry()` in `server.py` registers `refund` as a child of `order_query`

---

## 22. Evaluation Metrics: 4-Tier System and 9-Scenario Test Set

### 4-Tier Metrics (D15)

| Tier | Metric | What It Measures |
|------|--------|------------------|
| 1. Top-K Accuracy | Top-1, Top-3 | Is the correct intent in the top-K? |
| 2. Per-Intent | Per-intent accuracy | Which intents are confused with which? |
| 3. Rejection/Clarification | 拒识准确率, 误拒率, 漏拒率, 澄清触发准确率, 澄清后收敛率 | Are the dual exits working correctly? |
| 4. Slot Filling | 槽位准确率, 槽位召回率, 必填槽位完整率, 槽位更新准确率, 约束识别准确率 | Are parameters extracted correctly? |

### Benchmark Targets

- **Domain-specific Agent**: 0.99 (the framework's target market)
- **General Agent**: 0.85 (baseline)

### 9-Scenario Test Set (D16)

The framework includes a built-in test set covering 9 scenario types:

1. Clear/simple intent (L1/L2 hit)
2. Complex expression (L3 needed)
3. Multi-intent input (decomposition)
4. Cross-turn context dependency
5. Intent switching
6. Slot filling (missing required slots)
7. Rejection (unsupported intent)
8. Clarification (unclear intent)
9. Hierarchical dispatch

### Implementation

- **Metrics calculator**: `MetricsCalculator` (`src/intent_recognition/evaluation/metrics.py`)
- **Test set**: `TestSet` (`src/intent_recognition/evaluation/test_set.py`) — 9 scenario types, sample cases
- **Test runner**: `TestRunner` (`src/intent_recognition/evaluation/runner.py`) — runs test set against pipeline, generates report
- **Config**: `EvaluationConfig` (`src/intent_recognition/config.py`) — `top_k=3`, benchmark targets

---

## Accuracy Improvements (D17-D25)

The following optional capabilities (all default OFF) address five types of intent recognition accuracy problems identified in production. Each can be enabled via config or per-request override flags on `POST /agent/intent`.

### Problem-Solution Mapping (D25)

| Problem | Solution | Config |
|---------|----------|--------|
| Large intent space → L2 confusion | D17: Retrieval-based candidate narrowing | `retrieval.enable` |
| Static few-shot insufficient | D18: Dynamic few-shot injection | `dynamic_fewshot.dynamic_enabled` |
| Intent boundary overlap | D19: Intent orthogonality check | `orthogonality.enable_check` |
| Pseudo multi-intent (process descriptions) | D20: True multi-intent decomposition | `multi_intent.enable` |
| Code layer misses (no keyword match) | D21: Vector matching fallback | `vector_fallback.enable` |
| Unnecessary re-recognition on same task | D22: Intent reuse with rollback | `reuse_strategy.enable` |
| Single recognizer uncertainty | D23: Multi-recognizer arbiter | `arbiter.enable` |
| Need for domain-specific model | D24: Fine-tuning integration point | `fine_tuning.enable` |

### D17: Retrieval-based Candidate Narrowing

Before Layer 2 LLM classification, retrieve Top-N candidate intents using vector similarity, LLM coarse classification, or hybrid retrieval. Dynamic N adjusts based on context (large intent space → more candidates; small → fewer).

- **Config**: `RetrievalConfig(enable=False, method="vector"|"llm_coarse"|"hybrid", top_n=10, dynamic_n=True)`
- **Per-request override**: `enable_retrieval: true|false` on `POST /agent/intent` and `POST /recognize`
- **Implementation**: `src/intent_recognition/lightweight_llm/candidate_retriever.py`

### D18: Dynamic Few-Shot Injection

Injects dynamically retrieved historical examples (similar to current input) alongside static few-shot examples. Uses Jaccard token overlap to find similar past inputs from `FewShotStore`.

- **Config**: `DynamicFewShotConfig(dynamic_enabled=False, dynamic_top_k=3, static_kind_tag="static")`
- **Implementation**: `src/intent_recognition/lightweight_llm/dynamic_fewshot.py`

### D19: Intent Orthogonality Check

Detects overlap between intent definitions (similarity > threshold) and provides `split_intent()` / `merge_with_param()` operations to maintain a clean, orthogonal intent space.

- **Config**: `OrthogonalityConfig(enable_check=False, overlap_threshold=0.7)`
- **Implementation**: `IntentRegistry.detect_overlap()`, `split_intent()`, `merge_with_param()` in `src/intent_recognition/intent_registry.py`

### D20: True Multi-Intent Decomposition

Distinguishes true multi-intent requests from process descriptions (e.g., "recommend a phone and then check out"). Returns dependency relations between sub-intents and a topologically sorted pending list.

- **Config**: `MultiIntentConfig(enable=True, sequential_execution=True, filter_process_description=True)`
- **Response fields**: `relations: [{src, dst, constraints}]`, `pending_intents: ["intent_a", "intent_b"]`
- **Implementation**: `src/intent_recognition/deep_llm/classifier.py` (parsing + topological sort)

### D21: Vector Matching Fallback

When Layer 1 code-layer misses (no keyword/regex/rule match), falls back to vector similarity matching against a `VectorMatchStore`. Uses a 128-dim hashing vectorizer (pure Python, no numpy dependency) with cosine similarity.

- **Config**: `VectorFallbackConfig(enable=False, similarity_threshold=0.92, top_k=1)`
- **Per-request override**: `enable_vector_fallback: true|false`
- **Implementation**: `src/intent_recognition/code_layer/vector_matcher.py`
- **Source value**: `"vector-fallback"` in the `source` response field

### D22: Intent Reuse with Rollback

When the user continues the same task across turns, reuses the previous turn's intent directly (confidence=1.0, source="reused"), skipping the entire waterfall. Rolls back on implicit failure signals (D11) or intent-switch markers.

- **Config**: `ReuseStrategyConfig(enable=False, rollback_on_failure_signal=True, rollback_on_tool_failure_count=3)`
- **Per-request override**: `reuse_previous_intent: true|false`
- **Intent switch markers**: `"换个话题"`, `"不是这个"`, `"我要问别的"`, `"切换"`, `"换一个"`, `"不是说这个"`, `"我想问的是"`
- **Implementation**: `src/intent_recognition/intent_reuse_strategy.py`
- **Source value**: `"reused"` in the `source` response field

### D23: Multi-Recognizer Arbiter

Runs multiple recognizers (vector, rule, lightweight LLM) in parallel and arbitrates via vote (majority) or weighted_score (configurable weights). Useful when a single recognizer is uncertain.

- **Config**: `ArbiterConfig(enable=False, mode="vote"|"weighted_score", recognizers=["vector","rule","lightweight_llm"], weights={"vector":0.8,"rule":0.6,"lightweight_llm":1.0})`
- **Implementation**: `src/intent_recognition/multi_recognizer_arbiter.py`
- **Source values**: `"arbiter-vote"` or `"arbiter-weighted"`

### D24: Fine-Tuning Integration Point

Provides a training data export pipeline (JSONL format) and a `model_tier="fine_tuned"` LLM client tier that reads the `FINE_TUNED_MODEL` env var. Design-only — no actual training is performed; the exported data can be used to fine-tune a domain-specific model externally.

- **Config**: `FineTuningConfig(enable=False, model=None)`
- **Implementation**: `src/intent_recognition/training_data_exporter.py`, `evaluate_fine_tuned()` in evaluation runner
- **Env var**: `FINE_TUNED_MODEL` for the fine-tuned model name

### Per-Request Override Pattern

All D17/D21/D22 capabilities support per-request override via the `POST /agent/intent` and `POST /recognize` request body:

| Field | Type | Default | Effect |
|-------|------|---------|--------|
| `enable_retrieval` | `bool \| null` | `null` | Override D17 (`null`=use config, `true`/`false`=force) |
| `enable_vector_fallback` | `bool \| null` | `null` | Override D21 |
| `reuse_previous_intent` | `bool \| null` | `null` | Override D22 |

Overrides are applied to a snapshot of the pipeline config and restored after the request completes, so they do not leak across requests sharing the singleton pipeline.

### Accuracy Presets

The `accuracy_preset` config field (`"balanced"` by default) provides a shorthand for enabling multiple capabilities at once. Available presets:
- `"balanced"` (default): all D17-D24 OFF, rely on the three-layer waterfall
- `"high_accuracy"`: enable D17 + D21 + D22 for maximum accuracy (higher cost)
- `"low_cost"`: all OFF (same as balanced, explicit)

## Interview Insights (D26-D31)

Six design decisions distilled from intent-recognition interview materials
(resume framing, post-class exercises, reuse templates). They complement
D1-D25 with evidence grading, a strict sub-task boundary, five-factor
routing arbitration, extended evaluation metrics, a solution-evolution
methodology, and a unified structured-output protocol. All six default
ON because they are additive (they enrich output without changing
existing behavior).

### D26: Evidence Grading

Every slot value and fact is tagged with an evidence grade:

- **`verified`** — sourced from the current input, the active context, or
  a confirmed `KeyFact`. Hard operations (refund, payment) may proceed.
- **`provisional`** — sourced from history, user profile, speculation, or
  a default value. Hard operations must either upgrade the evidence
  (user confirmation / observation check) or disclose the assumption.

The pipeline collects evidence via `_collect_evidence()`, upgrades
provisional evidence via `_upgrade_provisional_to_verified()`, and
enforces the hard-op check via `_check_hard_op_evidence()` when
`evidence.require_verified_for_hard_ops=True` and the intent is in
`evidence.high_risk_intents`.

- **Config**: `EvidenceConfig(enable_grading=True, require_verified_for_hard_ops=True, high_risk_intents=[])`
- **Implementation**: `IntentRecognitionPipeline._collect_evidence()` / `_upgrade_provisional_to_verified()` / `_check_hard_op_evidence()` in `src/intent_recognition/pipeline.py`
- **Models**: `Evidence`, `EvidenceGrade`, `SlotValue` in `src/intent_recognition/models.py`

### D27: sub_tasks vs independent_intents Boundary

**The job of an intent is to select a business flow.** Only goals that
independently trigger *different* business flows are `independent_intents`;
intra-flow steps (price comparison, delivery check, parameter lookup) are
`sub_tasks`, slots, or constraints.

- `independent_intents` enter D20 multi-intent governance (process
  filtering, `relations`, `pending_intents`, sequential execution).
- `sub_tasks` are recorded as execution steps within the main flow and
  do **NOT** enter `relations`, `pending_intents`, or process filtering.

The deep LLM prompt (rules 6-10) and the lightweight LLM
(`detect_boundary_simple()` with `_SUB_TASK_PATTERNS` /
`_INDEPENDENT_MARKERS`) both enforce this boundary.

- **Config**: `BoundaryConfig(enable_sub_tasks=True, strict_mode=False)`
- **Implementation**: `src/intent_recognition/deep_llm/prompts.py` (rules 6-10), `src/intent_recognition/deep_llm/classifier.py`, `src/intent_recognition/lightweight_llm/classifier.py`

### D28: Five-Factor Routing Arbitration

Confidence is a **signal**, not a probability. When L2 confidence falls
in the ambiguous zone `[clarify_threshold, accept_threshold)`, the
five-factor arbitration runs in priority order:

| # | Factor | Fail action |
|---|--------|-------------|
| 1 | Rule validation | adjust confidence (±0.05) |
| 2 | Slot completeness | Clarify |
| 3 | Hard-constraint risk (high-risk intent + provisional required slot) | Clarify |
| 4 | Candidate gap (Top1-Top2 < threshold) | Escalate to L3 |
| 5 | Confidence + historical accuracy | Clarify if history poor and adjusted conf < accept |

The first failing factor that mandates Clarify/Escalate wins. Factor 5
reuses the D5 multi-signal fusion score; it does not replace it.

For L2/L3 disagreement, `arbitrate_l2_l3()` applies: L3 high-confidence +
complete slots → accept L3; both below clarify threshold → force Clarify;
otherwise run five-factor on L3.

- **Config**: `ArbitrationConfig(enable_five_factor=True, candidate_gap_threshold=0.1, risk_aware_clarify=True, high_risk_intents=[])`
- **Implementation**: `src/intent_recognition/lightweight_llm/confidence_router.py` (`ArbitrationInput`, `ArbitrationDecision`, `arbitrate()`, `arbitrate_l2_l3()`)
- **Output**: `arbitration_breakdown` field on `IntentRecognitionResult` (separate from `signals` which is `dict[str, float]`)

### D29: Extended Evaluation Metrics

Beyond the D15 4-tier metrics, D29 adds 9 fine-grained metrics plus an
online-feedback loop:

| Metric | Method | What it measures |
|--------|--------|------------------|
| Confusion matrix | `compute_confusion_matrix()` | N×N expected-vs-predicted |
| Hard-sample accuracy | `compute_hard_sample_accuracy()` | Accuracy on `is_hard=True` cases |
| Rejection accuracy | `compute_rejection_accuracy()` | `false_reject_rate`, `missed_reject_rate`, `rejection_precision` (English keys) |
| Clarification convergence | `compute_clarification_convergence_rate()` | First-round convergence rate |
| Slot recall | `compute_slot_recall()` | `correct_slots / expected_slots` |
| Slot completeness | `compute_slot_completeness()` | Sessions with all required slots filled |
| Constraint identification | `compute_constraint_identification_rate()` | `correct_constraints / total_constraints` |
| State update accuracy | `compute_state_update_accuracy()` | Multi-turn slot update correctness |

**Online feedback loop**: `TestSet.import_online_samples()` feeds
production failures (collected via `TestRunner.collect_online_failures()`
using D11 failure signals) back into the offline test set.

- **Config**: `ExtendedEvaluationConfig` toggles each metric
- **Implementation**: `src/intent_recognition/evaluation/metrics.py`, `src/intent_recognition/evaluation/test_set.py`, `src/intent_recognition/evaluation/runner.py`

### D30: Solution Evolution Methodology

The framework evolves through three stages, each addressing the
accuracy bottlenecks of the previous one:

| Stage | Architecture | Key mechanisms |
|-------|-------------|----------------|
| **v0** | Single deep LLM, no normalization | Brute-force intent recognition |
| **v1-stage1** | + L2 lightweight LLM | Cost reduction, confidence routing |
| **v1-stage2** | + L1 code layer + D5 multi-signal fusion | Three-layer waterfall, <10ms fast path |
| **v1-stage3** | + D17-D24 advanced mechanisms | Retrieval, dynamic fewshot, vector fallback, reuse, arbiter, fine-tuning |
| **v2** | + D26-D31 interview insights | Evidence grading, boundary, arbitration, extended eval, protocol |

`TestRunner.assess_evolution_stage(config)` inspects the config and
returns the current stage string. Use it to communicate maturity in
interviews and to gate feature expectations.

**60-second interview skeleton** (fill in the blanks):

> "We model intent recognition as a ___-layer waterfall: a code layer
> for <10ms keyword/regex hits, a lightweight LLM for the ambiguous
> middle, and a deep LLM for ___ complex scenarios. Confidence is
> treated as a ___ (signal), so we run a five-factor ___ (arbitration)
> before accepting. Every slot value carries an evidence ___ (grade):
> verified or provisional; hard operations require ___ (verified)
> evidence or must disclose assumptions. We evaluate with ___ (9)
> metrics including a confusion matrix and online-failure feedback loop,
> and the system evolves through v0 → v1 → v2 stages."

- **Implementation**: `src/intent_recognition/evaluation/runner.py::assess_evolution_stage()`

### D31: Structured Output Protocol

A 10-field protocol serves as the handoff contract between normalization
and intent recognition, and between intent recognition and downstream
ReAct/TAO loops:

| # | Field | Source | Default |
|---|-------|--------|---------|
| 1 | `normalized_query` | Normalization output (or raw input if skipped) | `""` on `/recognize` |
| 2 | `intent` | Recognition result | `null` |
| 3 | `sub_tasks` | D27 intra-flow steps | `[]` |
| 4 | `independent_intents` | D27 multi-flow goals | `[]` |
| 5 | `slots` | Slot extractor | `{}` |
| 6 | `missing_slots` | Slot extractor | `[]` |
| 7 | `hard_constraints` | Constraint extractor | `[]` |
| 8 | `soft_constraints` | Constraint extractor | `[]` |
| 9 | `verified_evidence` | D26 evidence collector | `[]` |
| 10 | `provisional_evidence` | D26 evidence collector | `[]` |

`sub_intents` is kept as a **deprecated alias** of `independent_intents`
(synced via `model_validator(mode="after")`) so old clients keep working.

- **Config**: `ProtocolConfig(enable_structured_output=True, deprecate_sub_intents=False)`
- **Implementation**: `AgentIntentResponse` / `AgentIntentNormalizationDetail` / `RecognizeResponse` in `src/user_input_normalization/server.py`
