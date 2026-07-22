# Agent Can't Understand Humans? I Split This Problem Into Two Modules

> [中文技术原理文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/TECHNICAL_PRINCIPLES.zh-CN.md)
>
> Related: [HTTP API Reference](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md) | [README](https://github.com/xiaoyesoso/XIntent/blob/main/README.md)

You tell your Agent, 「show me the second one, something cheaper.」 The Agent freezes.

Which one? The second what? How much cheaper?

This conversation happens every day. The user thinks they're being perfectly clear. The Agent feels like it's listening to a foreign language. And honestly, it's not the Agent's fault. Natural language is packed with pronouns, omissions, slang, and subjective judgments. Feeding raw user input directly into an LLM is like dropping a tourist who just arrived in Beijing into a conversation full of 「那个」「这玩意儿」「性价比高的」. Every word is recognizable, but the meaning is completely lost.

I spent a long time on this problem, and eventually realized it's actually two problems.

The first problem, the Agent doesn't understand what the user is saying. The user says 「the second one,」 and the Agent has no idea what that refers to. The user says 「something cheaper,」 and the Agent doesn't know how much cheaper. This is an input normalization problem. You need to clean up the user's input before intent recognition even runs.

The second problem, the Agent doesn't know what the user wants to do. Even if the input is clean, is 「help me pick a phone」 a recommendation intent or a search intent? Should it call a tool? What parameters does it need? This is an intent recognition problem, and a single LLM call is neither fast enough nor reliable enough to handle it.

XIntent is designed around these two problems. Two modules, each handling one. Input normalization runs before intent recognition, transforming casual, ambiguous, context-dependent user input into structured, machine-consumable text. The three-layer waterfall intent recognition then takes that clean input and starts from the cheapest code layer, escalating to a lightweight LLM, and only reaching the deep LLM when truly needed, balancing cost, speed, and accuracy.

Later, production usage revealed more edge cases, so we added a set of optional accuracy improvement mechanisms, each targeting a specific failure mode. Then we took the system to job interviews, and the interviewers' questions helped us discover blind spots we'd never noticed, leading to six enhancements born from real interview feedback.

This article walks through the entire design process. Not a reference manual, but a conversation with someone who built this system from scratch, sitting across from you, talking about why each decision was made, what pitfalls were hit, and which parts they're genuinely proud of.

For each principle, the **service interfaces** and **code implementations** are marked.

---

## Problem Background: Why Agents Can't Understand User Input

### The Real-World Challenge

Here's something I've noticed building Agents. In real-world scenarios, user input is highly casual, just like everyday speech. Users produce broken sentences, inversions, pronouns, abbreviations, and even slang. This is not a user problem. It's the fundamental nature of natural language. **Natural language is redundant, ambiguous, and context-dependent**.

Traditional software constrains user input through UI, forms, dropdowns, validation rules. Agents, however, directly receive natural language text. They lose the input normalization protection that the UI layer provides. This leads to several problems that anyone who has built an Agent will recognize.

- **Intent recognition failure** happens when missing subjects cause intent drift. 「市场占有率多少？」, whose market share?
- **Context breakdown** happens when cross-turn anaphora cannot be resolved. What does 「第二个」 in turn 3 refer to?
- **Parameter drift** happens when unquantified adjectives cause uncontrolled tool call parameters. 「性价比」 interpreted as 「cheap」.
- **Fact fabrication** happens when LLM hallucinations invent non-existent information.

### The Requirements Analysis Analogy

Think of user input normalization as the Agent equivalent of requirements analysis in traditional software engineering. In traditional SE, requirements analysis is the foundation for all subsequent design, development, and testing. In Agents, input normalization is the foundation for intent recognition, tool calling, and final output. **If the input isn't normalized, everything downstream is wasted effort**.

### Implementation

The entire framework is exposed as a service through `POST /normalize`. The user passes raw text, and the service returns a structured normalization result.

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "user_id": "u1", "turn": 2}'
```

- The service entry is `POST /normalize` (see [API Reference](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md#3-post-normalize--user-input-normalization))
- The orchestrator is `NormalizationPipeline` in `src/user_input_normalization/pipeline.py`
- To start the server, run `python -m user_input_normalization.server` from `src/user_input_normalization/server.py`

---

## Linguistic Foundations of the Six Input Problem Categories

The framework categorizes user input problems into six types, each with corresponding linguistic and cognitive science foundations. Let me walk you through them.

### Anaphora

The linguistic basis here is Anaphora Resolution in pragmatics. Pronouns have no fixed meaning, their referents depend entirely on context.

Typical manifestations include ordinal anaphora (「第二个适合生产吗？」), temporal anaphora (「刚才那个项目怎么包装？」), and attribute anaphora (「看樱花的那个地方」).

The challenge is that resolution requires recalling historical dialogue details. Short-term memory fails across multi-turn, long time-span conversations.

### Missing (Ellipsis)

The linguistic basis is that ellipsis is a common economy principle in natural language. Speakers omit parts that can be inferred from shared context.

Typical manifestations include missing subject (「市场占有率多少？」), missing object (「帮我优化一下。」), and missing constraints (「推荐一条裤子。」).

### Expression

The linguistic basis is the fundamental difference between spoken and written language. Speech is linear, jumping, and self-correcting. Writing is structured and complete.

Typical manifestations include disordered word order (「这个不太行，换个更像面试能讲的。」) and mid-stream correction (「不是，我说的是另一个。」).

### Semantic

The linguistic basis is Polysemy and Jargon in semantics. The same word has different meanings in different domains.

Typical manifestations include abbreviations (RAG, CRM), slang (抓手、赋能、闭环、不够 P8), and synonyms (知识库 / 检索增强 / RAG).

### Subjective

The linguistic basis is Subjectivity and Appraisal Theory. Judgment word meanings depend on personal preferences, scenarios, and standards.

Typical manifestations include 「哪个最有性价比？」 and 「再高级一点。」

The challenge here is real. Subjective judgment words must be converted to quantifiable parameters, otherwise tool calls are uncontrollable. Many Agents handle this crudely. 「性价比高」 directly equals 「low price」, ignoring quality, performance, brand, and after-sales. That's a big problem.

### External Fact

The linguistic basis is expressions referring to external world states, whose truth values depend on real-time data.

Typical manifestations include 「最近哪个框架更火？」 and 「现在最便宜的是哪个？」

The challenge is that this cannot be completed in the pre-processing stage. It must rely on tool calls returning real-time data. Cannot be fabricated.

### Implementation

Classification results are returned via the `classification_tags` field of the API response, as an array with multi-label support.

```json
{
  "classification_tags": ["指代问题", "主观判断问题"]
}
```

- The API response field is `classification_tags` (see [API Reference - classification_tags possible values](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md#classification_tags-possible-values))
- The classifier is `InputClassifier` in `src/user_input_normalization/classification/classifier.py`
- Classification rules live in `CLASSIFICATION_RULES` in `src/user_input_normalization/classification/rules.py`, with 82 semantic rules and 80 known terms
- Routing logic sends `SUBJECTIVE`/`EXTERNAL_FACT` to the `deep` stage, others to the `pre` stage

---

## Two-Stage Pipeline Design Rationale

### The Core Contradiction

Here's the tension at the heart of input normalization. Some problems, like 「第二个」 anaphora, can be solved immediately before intent recognition. Other problems, like 「现在最便宜的」 external facts, can only be solved after tool calls return real-time data.

If you do everything in pre-processing, cross-time anaphora and external facts cannot be handled. If you defer everything to the ReAct loop, simple anaphora also waits for tool calls, adding unnecessary latency.

Neither extreme works. You need a middle ground.

### Two-Stage Division of Labor

The solution is to split normalization into two stages with a clear division of labor.

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

The `observation` parameter of the API request controls whether to enter the deep stage, and the `stage_reached` field of the response indicates the actual stage reached.

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

- The API request field is `observation`, optional, passing tool return data triggers the deep stage
- The API response field is `stage_reached`, either `"pre"` or `"deep"`
- The orchestrator is `NormalizationPipeline.process()` in `src/user_input_normalization/pipeline.py`
- The pre stage lives in `PreNormalizer` in `src/user_input_normalization/pre_normalization/normalizer.py`
- The deep stage lives in `DeepNormalizer` in `src/user_input_normalization/deep_normalization/normalizer.py`

---

## Responsibility Boundary: Why LLMs "Over-Step"

### The LLM Over-Autonomy Problem

LLMs are trained to be helpful assistants. They want to complete every user request as thoroughly as possible. That sounds great until you realize what happens in practice. Without constraints, the LLM will happily do input normalization, intent recognition, AND answer the question all at once.

- User asks 「市场占有率多少？」, LLM directly fabricates a number to answer
- User asks 「帮我推荐裤子」, LLM directly recommends products
- User asks 「现在最便宜的」, LLM fabricates real-time prices

This is disastrous in the normalization stage. The responsibility of normalization is to **understand the input**, not to **answer the question**.

### The Explicit Can-Do / Cannot-Do List

The solution is surprisingly simple. Give the LLM an explicit list of what it can and cannot do.

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

The principle here is that explicit lists are the simplest and most effective constraint method. Vague role descriptions like 「you are an assistant」 are insufficient to constrain LLM behavior. Lists convert implicit expectations into explicit rules, enabling auditing and regression testing.

### Implementation

The responsibility boundary is injected as a hard constraint via the System Prompt, and the normalization result will **not contain** direct answers to user questions.

- The prompt constants are `SYSTEM_PROMPT`, `CAN_DO`, `CANNOT_DO` in `src/user_input_normalization/pre_normalization/prompts.py`
- The boundary validation is `PreNormalizer.validate_boundary()` which checks for over-stepping in `normalizer.py`
- Regression tests in `tests/test_pre_normalization.py::TestResponsibilityBoundary` verify no answering, no tool execution, no fact fabrication

---

## Pronoun Resolution and Cross-Turn Memory

### Pronoun Resolution Table as Key Facts

Pronoun resolution is not a one-time operation. The results need cross-turn reuse.

```
Turn 3: "第一个方案适合生产吗？" -> Resolve: 第一个方案 -> TCC方案
Turn 5: "那个 TCC 方案成本多少？" -> Direct hit by name indexing
Turn 7: "第一个呢？" -> Reuse turn 3 resolution
```

If you re-resolve every turn, you get high cost, potential inconsistency, and poor user experience. Nobody wants that.

### Implementation

Resolution results are returned via the `pronoun_resolutions` field of the API response, and are automatically reused across turns under the same `session_id`.

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

- The API response field is `pronoun_resolutions` (see [API Reference - pronoun_resolutions item structure](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md#pronoun_resolutions-item-structure))
- The data model is `PronounResolution` in `src/user_input_normalization/models.py`
- Cross-turn storage uses `KeyFactStore.find_pronoun_resolution()` in `src/user_input_normalization/storage/base.py`
- The write logic is `PreNormalizer._write_pronoun_facts()` in `normalizer.py`
- Cross-turn reuse tests are in `tests/test_pre_normalization.py::TestCrossTurnReuse`

---

## Named Indexing: Cognitive Load and Reference Accuracy

### The Problem with Ordinal References

If the Agent outputs 「方案一、方案二、方案三」 and the user later references 「第二个」, the LLM is prone to reference errors during inference. Ordinals have no semantics, and LLM attention may misalign.

### The Advantages of Semantic Naming

Now consider this. If the Agent outputs 「TCC方案、轻量方案、企业方案」 and the user references 「TCC方案」, name-based indexing has higher match accuracy, lower cognitive load, and avoids inference errors.

This connects to the ancient concept of the rectification of names. If names are not correct, language cannot flow. If language cannot flow, things cannot be done. Ordinals are weak identifiers. Semantic names are strong identifiers.

### Implementation

- The API response field is `pronoun_resolutions[].named_entity`, the semantic name
- The naming logic is `PreNormalizer._ensure_named_entities()` in `normalizer.py`
- Named indexing tests are in `tests/test_pre_normalization.py::TestPronounResolution::test_named_entity_set`

---

## Clarification Mechanism: Handling Uncertainty

### Why Guessing Doesn't Work

Error accumulation is a real risk. Turn 1 guesses wrong. Turn 2 continues reasoning based on the wrong result. The final answer completely deviates. By the time you notice, the damage is done.

### Confidence Threshold Mechanism

```
Confidence >= 0.6 -> Resolve directly
Confidence < 0.6  -> Trigger clarification, ask user
```

The principle is simple, better to not answer than to answer wrong. The threshold θ_clarify is configurable, defaulting to 0.6, balancing efficiency and accuracy.

### Implementation

When `paused_for_clarification` is `true`, the response contains a `clarification` object, and the client should pause processing and present the clarification question to the user.

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

- The API response fields are `paused_for_clarification` and `clarification` (see [API Reference - clarification structure](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md#clarification-structure-when-paused_for_clarification-is-true))
- The handler is `ClarificationHandler` in `src/user_input_normalization/clarification/handler.py`
- The config items are `theta_clarify=0.6` and `max_consecutive_clarifications=3` in `src/user_input_normalization/config.py`
- The resume entry is `NormalizationPipeline.resume_after_clarification()` in `pipeline.py`

---

## Attribute Retrieval Resolution: RAG Long-Term Memory

### Problem Scenario

Here's a scenario that sounds simple but is genuinely hard. Before a holiday, the user asks the Agent to recommend a travel destination. The Agent recommends Jiming Temple (鸡鸣寺) for cherry blossoms. After the holiday, the user inputs, 「你上次推荐的看樱花的那个地方……」

The challenges stack up. Long time span means short-term memory has expired. Attribute anaphora means 「看樱花的」 is an attribute modifier, not a direct reference.

### Two-Step Inference + Compensation Mechanism

1. **Attribute extraction + vector search**. Extract ["樱花", "旅游"], search historical dialogue store to recall details
2. **LLM inference**. Infer based on window content, which gives us 「鸡鸣寺」
3. **Compensation mechanism** when search fails. Extract attributes to issue tool calls, then re-reason

### Implementation

Attribute retrieval resolution is automatically triggered in the pipeline when an attribute anaphora pattern is detected, and resolution results are written back to `pronoun_resolutions`.

- The resolver is `AttributeResolver` in `src/user_input_normalization/attribute_resolution/resolver.py`
- The two-step inference flow is `AttributeResolver.resolve()` then `extract_attributes()` then `recall_details()` then `infer()`
- The compensation mechanism is `AttributeResolver.compensation()` which simulates a tool call and re-reasoning
- The trigger condition is `NormalizationPipeline._is_attribute_anaphora()` which detects attribute anaphora patterns
- Tests are in `tests/test_attribute_resolution.py::TestJimingTempleCase`, the Jiming Temple end-to-end case

---

## Vocabulary Table Self-Evolution: Emergent Knowledge

### The Problem

The same word or abbreviation has different meanings in different industries, domains, and contexts. Manually maintaining vocabulary tables is unrealistic. New words constantly emerge, and you can't keep up.

### The Self-Iteration Mechanism

```
Online dialogue -> Offline analysis -> Mark candidate terms -> Threshold check -> Promote to vocab table
```

The threshold rules are designed to ensure stability. Total count > 100. Discussant count > 3 for public vocabulary. Consecutive discussion > 10.

This is emergent knowledge. The vocabulary table emerges bottom-up from actual dialogue. Threshold filtering ensures stability, and human review serves as a backstop.

### Implementation

Term standardization results are returned via the `term_mappings` field of the API response.

```json
{
  "term_mappings": [
    {"original": "RAG", "standard": "检索增强生成", "source": "vocabulary-table"}
  ]
}
```

- The API response field is `term_mappings` (see [API Reference - term_mappings item structure](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md#term_mappings-item-structure))
- The vocabulary service is `VocabularyTable` in `src/user_input_normalization/vocabulary/table.py`
- The offline analyzer is `OfflineAnalyzer` in `src/user_input_normalization/vocabulary/offline_analyzer.py`
- The storage interface is `VocabStore` in `src/user_input_normalization/storage/base.py`
- The config items are `min_total_count=100`, `min_discussant_count=3`, `min_consecutive_count=10` in `config.py`

---

## Adjective Quantification: From Subjective to Objective

### The Problem

Any unquantified adjective causes Agent output drift. Many Agents handle this crudely. 「性价比高」 directly equals 「low price」, ignoring quality, performance, brand, and after-sales. That kind of simplification leads to terrible recommendations.

### Quantification Strategies

「性价比」 has two reasonable interpretations, and both are valid.

| Strategy | Meaning | Tool Parameters |
|----------|---------|-----------------|
| Same price, better quality | At the same price point, better quality/config/service | `price_range: [150, 200], quality_rank: top 30%` |
| Same quality, lower price | At the same quality level, lower price | `price_range: [100, 150], quality_rank: top 50%` |

### Implementation

Quantification results are returned via the `quantifiable_adjectives` field of the API response, and the request must include `observation` (like current price) to complete quantification.

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

- The API request field is `observation`, which passes current price and other context to trigger deep-stage quantification
- The API response field is `quantifiable_adjectives` (see [API Reference - quantifiable_adjectives item structure](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md#quantifiable_adjectives-item-structure))
- The quantification engine is `QuantificationEngine` in `src/user_input_normalization/quantification/engine.py`
- Built-in rules are in `DEFAULT_RULES` with 6 rules covering 性价比, 划算, 再高级一点, and more, in `quantification/rules.py`
- Deep normalization uses `DeepNormalizer.quantify_adjectives()` in `deep_normalization/normalizer.py`

---

## Three-Layer Context Integration

### The Core Idea

> All normalization measures can be summarized in one sentence, deeply integrate context.

| Layer | Content | Source | Purpose |
|-------|---------|--------|---------|
| User Profile | Industry, preferences, behavior history | User profile store | Disambiguation tendency |
| Key Facts | Preferences, opinions, attitudes, acknowledged points | Key fact store | Cross-turn reuse, completion basis |
| Dialogue History | Summaries + details | Short-term memory + RAG long-term memory | Anaphora resolution evidence, ellipsis completion source |

### Implementation

The three-layer context is automatically assembled inside `PreNormalizer.normalize()`, associated via `user_id` and `session_id`.

- The integrator is `ContextIntegrator.assemble()` in `src/user_input_normalization/context/integrator.py`
- The context model is `ContextBundle` with user_profile, key_facts, dialogue_summary, and recalled_details
- Priority conflict resolution uses `ContextIntegrator.resolve_conflict()` with the order Key Facts > User Profile > Dialogue History
- The storage interfaces are `UserProfileStore`, `KeyFactStore`, `DialogueHistoryStore` in `storage/base.py`
- For observability, `ContextIntegrator.explain_influence()` returns an influence report for each layer

---

## Completeness Check: The Gatekeeper

### Why a Check Is Needed

Normalization is not 「call the LLM once and done.」 The LLM may miss things. Incomplete subject-predicate-object. Unresolved pronouns. Unquantified adjectives. You need a gatekeeper.

### Three Checks + Routing

```
Check passes -> Flow to intent recognition
Missing SPO / unresolved pronouns -> Trigger clarification (paused_for_clarification: true)
Unquantified adjectives -> Route to deep-normalization (stage_reached: "deep")
```

### Implementation

Check results affect the `stage_reached` and `paused_for_clarification` fields of the API response.

- The checker is `CompletenessChecker` in `src/user_input_normalization/pre_normalization/completeness_checker.py`
- The check method is `check_and_route()` which returns `(check_result, route_target, route_reason)`
- The check items are `spo_complete` for SPO, `pronouns_resolved` for pronouns, `adjectives_quantified` for adjectives
- The API impact is that unquantified adjectives route to `stage_reached: "deep"`, and unresolved pronouns trigger `paused_for_clarification: true`

---

## Few-Shot Retrieval Injection: Economics and Feedback Loop

### The Problem

Few-shot examples significantly improve normalization quality. But injecting all examples would overflow the context window. You need to be selective.

### Retrieval Injection + Online Sinking

```
User input -> Vectorize -> Search top-k similar examples -> Inject into Prompt
Online encounters strange input -> Successfully resolved -> Organize as few-shot example -> Save to library -> Next similar input can reference
```

This is a data flywheel. The longer the system runs, the richer the example library, and the higher the normalization quality. It gets smarter with use.

### Implementation

Few-shot injection is automatically executed inside `PreNormalizer.normalize()`, transparent to API callers.

- The example storage is `FewShotStore` in `src/user_input_normalization/storage/base.py`
- The retrieval injection is `PreNormalizer._build_user_prompt()` which calls `FewShotStore.search(top_k=3)`
- Online sinking uses `PreNormalizer._sink_to_fewshot()` which automatically saves strange inputs to the library
- The formatting function is `format_fewshot()` in `pre_normalization/prompts.py`
- The config items are `top_k=3` and `enabled=True` in `config.py`

---

## Summary

The core ideas of user input normalization, Module 1, can be boiled down to a few key insights.

1. **Deeply integrate context**. All measures revolve around this one sentence.
2. **Two-stage division of labor**. What can be solved immediately goes to pre. What needs tools goes to deep.
3. **Explicit constraints**. Responsibility boundaries, completeness checks, clarification thresholds prevent LLM over-stepping and guessing.
4. **Structured output**. Resolution tables, SPO, quantification fields enable downstream programmatic consumption.
5. **Self-evolution**. Vocabulary table iteration, few-shot sinking make the system smarter with use.

Coming back to Module 2, the core ideas of three-layer intent recognition are equally clear.

1. **Waterfall escalation**. Start cheap and fast, escalate only when needed, from Code to Flash to Pro.
2. **Multi-signal confidence**. Never trust LLM self-reported confidence alone. Fuse with rule match, vector similarity, historical accuracy.
3. **Slot filling as part of recognition**. Intent plus parameters plus constraints are one structured output, not separate stages.
4. **Dual exits for uncertainty**. Unsupported intent is rejected with reason. Unclear intent triggers a bounded clarification loop.
5. **Closed-loop evaluation**. Test set plus next-turn implicit signals feed back into metrics for continuous improvement.

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

## Why "Just Call a Big Model" Isn't Enough

### The Interview Question

When asked 「how do you do intent recognition in your Agent?」 many candidates answer 「call a big model to classify.」 In engineering, this is wrong for three reasons.

1. **Cost and latency**. A single deep reasoning call can take 5-15 seconds. Most user inputs like 「continue」, 「next」, 「refund」 are unambiguous and don't need it.
2. **Confidence is not a probability**. LLM self-reported `confidence` is a self-judgment score, not a calibrated probability. Treating it as ground truth leads to over-acceptance of wrong answers.
3. **No dual exits**. A single LLM call produces one answer. Engineering needs explicit handling for 「unsupported」 (intent not in candidate list) and 「unclear」 (low confidence or missing required slots). These are two distinct failure modes.

### Implementation

The framework introduces a three-layer waterfall that addresses all three concerns.

- The service entry is `POST /recognize` (see [API Reference](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.md#4-post-recognize--intent-recognition))
- The orchestrator is `IntentRecognitionPipeline` in `src/intent_recognition/pipeline.py`
- The design decisions are the waterfall architecture, the independent stage, and the dual exits

---

## Three-Layer Waterfall Architecture

### The Cascade

This is where the architecture gets elegant. Instead of one big call, you cascade through three layers, escalating only when the previous layer can't handle it.

```
Normalized Input -> [L1 Code Layer] --miss--> [L2 Lightweight LLM] --low conf--> [L3 Deep LLM]
                        |                              |                              |
                        +------- hit (conf=1.0) ------+---------- high conf --------+
                                                       |
                                              [Slot Filling] -> [Rejection / Clarification]
                                                       |
                                          IntentRecognitionResult
```

**Layer 1, Code Layer** has three recognition methods. Page guidance maps UI events to intents. Keyword and regex matching catches explicit patterns. The rule engine combines context state with normalized input. It returns the first match or an 「unmatched」 signal. **Zero LLM calls, under 10ms.**

**Layer 2, Lightweight LLM** uses a Flash or Mini model with a structured prompt containing candidate intents, slots, and few-shot examples. Confidence routing sends `>=0.85` to accept, `0.6-0.85` to clarify, and `<0.6` to escalate to L3.

**Layer 3, Deep LLM** is the deep reasoning model for 5 complex scenarios. Complex expressions like 病句 and 倒装. Cross-turn context dependency. Intent switching. Multi-intent decomposition. Implicit info completion.

### Why Waterfall, Not Parallel

Parallel execution of all three layers would cost 3× the API calls, and Flash model confidence versus Pro model confidence are not directly comparable. The waterfall escalates only on miss or low-confidence, so most requests stop at L1 or L2. It's economics and accuracy working together.

### Graceful Degradation

If L3 fails, maybe a timeout or API error, the pipeline falls back to the L2 result with a warning log rather than failing the whole request. The user still gets an answer, just not the deepest one.

### Implementation

- L1 Code Layer lives in `src/intent_recognition/code_layer/` with `classifier.py`, `page_guidance.py`, `keyword_matcher.py`, `rule_engine.py`
- L2 Lightweight LLM lives in `src/intent_recognition/lightweight_llm/` with `classifier.py`, `confidence_router.py`, `multi_signal.py`, `prompts.py`
- L3 Deep LLM lives in `src/intent_recognition/deep_llm/` with `classifier.py`, `prompts.py`
- The orchestrator is `IntentRecognitionPipeline.recognize()` in `src/intent_recognition/pipeline.py`
- The config flags are `enable_code_layer`, `enable_lightweight_llm`, `enable_deep_llm` in `src/intent_recognition/config.py`

---

## Confidence Routing and Multi-Signal Fusion

### The Problem with LLM Confidence

Here's something that took me a while to appreciate. LLM self-reported confidence is a self-judgment score. It reflects the model's internal certainty, not a calibrated probability. Two failure modes result from this.

- **Over-confident on wrong answers**. The model says 0.9 confidence on a misclassified intent.
- **Under-confident on right answers**. The model says 0.5 on a correct intent, triggering unnecessary escalation.

You can't just trust the number. You need more signals.

### Multi-Signal Fusion

The framework fuses four signals to produce a more reliable confidence score.

| Signal | Weight | Source |
|--------|--------|--------|
| LLM confidence | 0.5 | Model output |
| Rule match | 0.2 | Code layer keyword/regex hit on same intent |
| Vector similarity | 0.2 | Embedding similarity to positive examples |
| Historical accuracy | 0.1 | Per-intent accuracy from evaluation store |

The fused confidence is then routed through three thresholds.

- `>= 0.85` means accept
- `>= 0.6 and < 0.85` means clarify, ask the user
- `< 0.6` means escalate to L3

### Implementation

- The confidence router is `ConfidenceRouter` in `src/intent_recognition/lightweight_llm/confidence_router.py`
- The multi-signal fusion is `MultiSignalFuser` in `src/intent_recognition/lightweight_llm/multi_signal.py`
- The config is `ConfidenceConfig` in `src/intent_recognition/config.py` with `accept_threshold=0.85`, `clarify_threshold=0.6`, and the weight fields

---

## Slot Filling: Hard/Soft Constraints and Cross-Turn Accumulation

### Slot Filling as Part of Recognition

Intent recognition is not just 「what intent.」 It's 「what intent plus what parameters plus what constraints.」 Slot filling is integrated into the recognition pipeline, not a separate stage. This was a deliberate design choice.

### Hard vs Soft Constraints

| Constraint Type | Examples | Semantics |
|----------------|----------|-----------|
| Hard | 「不超过3000」, 「必须华为」, 「不要二手」 | Must satisfy, filter results |
| Soft | 「最好轻薄」, 「优先蓝色」, 「续航越长越好」 | Best effort, sort or boost |

The framework uses regex patterns for common Chinese constraint expressions, falling back to LLM extraction for complex cases. Constraints are returned as normalized expressions like `price<=3000` plus raw text for traceability.

### Cross-Turn Accumulation

In a ReAct/TAO loop, users iteratively refine their request across turns. This is natural conversation behavior.

```
Turn 1: "帮我推荐一款3000元以内的手机"  -> {category: 手机, budget_max: 3000}
Turn 2: "预算可以到4000"                -> {category: 手机, budget_max: 4000}  (latest-wins)
Turn 3: "最好是华为"                    -> {category: 手机, budget_max: 4000, brand: 华为}  (accumulate)
```

The **latest-wins strategy** means when the same slot appears in a new turn, the new value overwrites the old. Conflicts are recorded for debugging.

**Constraint accumulation** works differently. Constraints don't overwrite. Unique expressions are appended. The `has_constraint_conflict()` function detects contradictory hard constraints, like `price<100` versus `price>200`.

### Implementation

- The slot extractor is `SlotExtractor` in `src/intent_recognition/slot_filling/extractor.py`
- The constraint extractor is `ConstraintExtractor` in `src/intent_recognition/slot_filling/constraints.py`
- The cross-turn merger is `CrossTurnSlotMerger` in `src/intent_recognition/slot_filling/cross_turn.py`
- The slot state store is `SlotStateStore` in `src/intent_recognition/storage/base.py`, with memory impl in `memory.py`
- The API fields are `slots`, `missing_slots`, `hard_constraints`, `soft_constraints`

---

## Rejection and Clarification: Dual Exits

### Two Distinct Failure Modes

| Mode | Trigger | Response |
|------|---------|----------|
| Unsupported (拒识) | Intent not in candidate list | `rejection_reason` with supported intents listed |
| Unclear (澄清) | Low confidence OR missing required slots | `clarification_question` to ask user |

A single 「I don't know」 answer conflates these two. The user can't tell whether to rephrase (unsupported) or provide more info (unclear). That's terrible UX.

### Rejection Tells the User What IS Supported

When the intent is not in the candidate list, the rejection reason includes three things.

1. The unsupported content, echoed back
2. The list of supported intents with descriptions and positive examples
3. The redirect hints from negative examples, like `"iPhone 16的电池容量是多少 -> product_query"`

### Clarification Has a Bounded Loop with Convergence Tracking

Clarification is not infinite. The framework tracks consecutive clarifications per session.

- After `max_consecutive_clarifications` (default 3), the system degrades to 「best guess」 mode. It accepts the highest-confidence candidate rather than asking again.
- If the user's next turn resolves the clarification, like providing the missing slot, the consecutive counter resets.

### Implementation

- The handler is `RejectionClarificationHandler` in `src/intent_recognition/rejection_clarification/handler.py`
- The config is `ClarificationConfig.max_consecutive_clarifications=3` in `src/intent_recognition/config.py`
- The API fields are `rejection_reason`, `need_clarification`, `clarification_question`

---

## Next-Turn Implicit Evaluation

### The Problem with Explicit Feedback

Asking users 「did I get that right?」 after every intent recognition is terrible UX. Most users won't answer. Those who do are biased toward reporting errors, not successes. You get skewed data and annoyed users.

### Implicit Signal Detection

The framework detects failure signals in the **next turn's input**. No extra UI needed.

```
Turn N: "帮我推荐手机"  -> intent: product_recommendation
Turn N+1: "你理解错了，我要查订单"  -> failure signal detected for Turn N
```

Default failure signals are 「你理解错了」, 「不是这个意思」, 「不是，我不是」, and 「不对，我要的是」.

When a failure signal is detected, two things happen.

1. The previous turn's intent is marked as incorrect in the evaluation store
2. The current turn's intent is recognized fresh, since the failure signal often contains the correct intent

### Convergence Tracking

For clarification loops, the framework tracks whether the user's task progressed after clarification.

- **Resolved** means the next turn's input is a normal request, not a failure signal or another clarification. The consecutive counter resets.
- **Unresolved** means the next turn's input is another clarification response. The consecutive counter increments.

### Implementation

- The failure signal detection is `RejectionClarificationHandler.detect_implicit_failure()` in `src/intent_recognition/rejection_clarification/handler.py`
- The config is `IntentRecognitionConfig.failure_signals` in `src/intent_recognition/config.py`
- The evaluation store is `EvaluationStore` in `src/intent_recognition/storage/base.py`

---

## Intent Boundary Definition and Few-Shot Injection

### The Boundary Problem

Intents overlap. 「iPhone 16的电池容量是多少」 could be `product_query` (query specs) or `product_recommendation` (suggest based on battery). Without explicit boundaries, the LLM will guess inconsistently. One day it picks one, the next day the other.

### Positive + Negative Examples

Each `IntentDefinition` includes two types of examples.

- **Positive examples** are inputs that definitely belong to this intent
- **Negative examples** are inputs that look similar but belong to a different intent, with a redirect target, like `"3000以内有什么手机推荐 -> product_recommendation"`

These are injected into the L2 prompt to define boundaries. It's like teaching a child the difference between a cat and a dog by showing examples of each, including the tricky cases.

### Few-Shot Retrieval Injection

For confusing intent pairs, the framework retrieves the top-k most similar historical examples from `FewShotStore` and injects them into the prompt. This is the same mechanism as Module 1's few-shot injection (section 13), reused across modules.

### Implementation

- The intent registry is `IntentRegistry` in `src/intent_recognition/intent_registry.py`, storing `IntentDefinition` with positive and negative examples
- The prompt builder is `build_prompt()` in `lightweight_llm/prompts.py`, injecting candidate intents, slots, boundaries, and few-shot
- The few-shot store is shared with Module 1 in `src/user_input_normalization/storage/base.py`
- The config is `fewshot_top_k=3` and `fewshot_enabled=True` in `src/intent_recognition/config.py`

---

## Hierarchical Intents and Decision Tree

### The Context Window Problem

When the intent space grows beyond roughly 20 intents, listing all of them in the LLM prompt overflows the context window and degrades accuracy. The LLM gets confused by too many options.

### Hierarchical Dispatch

The framework supports parent and child intent relationships.

```python
IntentDefinition(name="refund", parent_intent="order_query", ...)
```

When hierarchical dispatch is enabled, the flow changes.

1. L1/L2 first recognizes the parent intent, like `order_query`
2. The candidate list narrows to children of `order_query`, like `refund`, `exchange`, `cancel`
3. L1/L2 runs again on the narrowed list

This turns a flat 100-intent problem into two 10-intent problems. Much more manageable.

### When to Enable

Hierarchical dispatch is **off by default** (`enable_hierarchical=False`) because it adds latency from two recognition passes. Enable it only when the intent space is large enough to hurt L2 accuracy.

### Implementation

- The intent registry is `IntentRegistry` in `src/intent_recognition/intent_registry.py` with the `parent_intent` field and `get_children()` method
- The config flag is `enable_hierarchical` in `src/intent_recognition/config.py`
- The default intents are registered in the server's `_build_default_registry()` in `server.py`, where `refund` is a child of `order_query`

---

## Evaluation Metrics: 4-Tier System and 9-Scenario Test Set

### 4-Tier Metrics

| Tier | Metric | What It Measures |
|------|--------|------------------|
| 1. Top-K Accuracy | Top-1, Top-3 | Is the correct intent in the top-K? |
| 2. Per-Intent | Per-intent accuracy | Which intents are confused with which? |
| 3. Rejection/Clarification | 拒识准确率, 误拒率, 漏拒率, 澄清触发准确率, 澄清后收敛率 | Are the dual exits working correctly? |
| 4. Slot Filling | 槽位准确率, 槽位召回率, 必填槽位完整率, 槽位更新准确率, 约束识别准确率 | Are parameters extracted correctly? |

### Benchmark Targets

- **Domain-specific Agent** targets 0.99, which is the framework's target market
- **General Agent** targets 0.85 as a baseline

### 9-Scenario Test Set

The framework includes a built-in test set covering 9 scenario types.

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

- The metrics calculator is `MetricsCalculator` in `src/intent_recognition/evaluation/metrics.py`
- The test set is `TestSet` in `src/intent_recognition/evaluation/test_set.py` with 9 scenario types and sample cases
- The test runner is `TestRunner` in `src/intent_recognition/evaluation/runner.py`, which runs the test set against the pipeline and generates a report
- The config is `EvaluationConfig` in `src/intent_recognition/config.py` with `top_k=3` and benchmark targets

---

## Accuracy Improvements

The following optional capabilities, all default OFF except multi-intent detection which defaults ON, address eight types of intent recognition accuracy problems identified in production. Each can be enabled via config or per-request override flags on `POST /agent/intent`.

These are the extensions we built after shipping the three-layer waterfall and watching it hit real-world edge cases. Each one solves a specific problem we encountered, and each one was designed to be independently toggleable so you only pay for what you need.

### Problem and Solution Mapping

| Problem | Solution | Config |
|---------|----------|--------|
| Large intent space causes L2 confusion | Retrieval-based candidate narrowing | `retrieval.enable` |
| Static few-shot insufficient | Dynamic few-shot injection | `dynamic_fewshot.dynamic_enabled` |
| Intent boundary overlap | Intent orthogonality check | `orthogonality.enable_check` |
| Pseudo multi-intent (process descriptions) | True multi-intent decomposition | `multi_intent.enable` |
| Code layer misses (no keyword match) | Vector matching fallback | `vector_fallback.enable` |
| Unnecessary re-recognition on same task | Intent reuse with rollback | `reuse_strategy.enable` |
| Single recognizer uncertainty | Multi-recognizer arbiter | `arbiter.enable` |
| Need for domain-specific model | Fine-tuning integration point | `fine_tuning.enable` |

### Retrieval-Based Candidate Narrowing

Imagine walking into a restaurant with a 200-item menu. You'd be paralyzed. But if the waiter says, 「based on what you've said, here are 5 dishes you might like,」 suddenly choosing is easy. This mechanism does exactly this for the LLM.

Before Layer 2 sends the candidate list to the LLM, it narrows the field to Top-N using vector similarity, LLM coarse classification, or a hybrid approach. The N is dynamic too, more candidates when the intent space is large, fewer when it's small. This prevents the LLM from getting confused by too many options and improves accuracy significantly.

- The config is `RetrievalConfig(enable=False, method="vector"|"llm_coarse"|"hybrid", top_n=10, dynamic_n=True)`
- The per-request override is `enable_retrieval: true|false` on `POST /agent/intent` and `POST /recognize`
- The implementation is in `src/intent_recognition/lightweight_llm/candidate_retriever.py`

### Dynamic Few-Shot Injection

Static few-shot examples are like a teacher who always uses the same examples regardless of what the student is struggling with. It works, but it's not optimal. This mechanism changes this by dynamically retrieving historical examples that are most similar to the current input.

The mechanism uses Jaccard token overlap to find relevant past inputs from the `FewShotStore`. So if the user asks about refund policies, the few-shot examples will be about refunds, not about product recommendations. The right example at the right time makes a big difference.

- The config is `DynamicFewShotConfig(dynamic_enabled=False, dynamic_top_k=3, static_kind_tag="static")`
- The implementation is in `src/intent_recognition/lightweight_llm/dynamic_fewshot.py`

### Intent Orthogonality Check

When two intent definitions overlap too much, the LLM will flip-flop between them. It's like having two buttons that do almost the same thing. Users can never remember which one to press, and neither can the LLM.

This detects the overlap using a similarity threshold and provides `split_intent()` and `merge_with_param()` operations to maintain a clean, orthogonal intent space. Keep your intents orthogonal, and classification accuracy goes up automatically.

- The config is `OrthogonalityConfig(enable_check=False, overlap_threshold=0.7)`
- The implementation uses `IntentRegistry.detect_overlap()`, `split_intent()`, and `merge_with_param()` in `src/intent_recognition/intent_registry.py`

### True Multi-Intent Decomposition

There's a huge difference between 「recommend a phone and then check out」 and 「recommend a phone and book a flight.」 The first is a process description, a single intent with multiple steps. The second is a true multi-intent request, two independent goals that each trigger a different business flow.

This mechanism knows the difference. It filters out process descriptions, identifies true multi-intent requests, and returns dependency relations between sub-intents along with a topologically sorted pending list. This prevents the system from treating every 「and」 sentence as multi-intent, which would create unnecessary complexity.

- The config is `MultiIntentConfig(enable=True, sequential_execution=True, filter_process_description=True)`
- The response fields are `relations: [{src, dst, constraints}]` and `pending_intents: ["intent_a", "intent_b"]`
- The implementation is in `src/intent_recognition/deep_llm/classifier.py` with parsing and topological sort

### Vector Matching Fallback

Sometimes the user says something that means the same thing but uses completely different words. The keyword matcher misses it. The regex doesn't catch it. But the meaning is the same.

This adds a vector matching fallback to the code layer. It uses a 128-dim hashing vectorizer, pure Python with no numpy dependency, and computes cosine similarity against a pre-built `VectorMatchStore`. When the similarity exceeds the threshold, it catches what the keyword matcher missed. This is one of my favorite mechanisms because it adds semantic understanding to the fastest layer without any LLM calls.

- The config is `VectorFallbackConfig(enable=False, similarity_threshold=0.92, top_k=1)`
- The per-request override is `enable_vector_fallback: true|false`
- The implementation is in `src/intent_recognition/code_layer/vector_matcher.py`
- The source value is `"vector-fallback"` in the `source` response field

### Intent Reuse with Rollback

If the user said 「recommend a phone」 in turn 1 and 「make it cheaper」 in turn 2, they're still talking about phone recommendation. There's no need to re-run the entire waterfall.

This reuses the previous turn's intent directly, with confidence 1.0 and source marked as 「reused.」 But here's the clever part. It rolls back when it detects failure signals or intent-switch markers like 「换个话题」 or 「不是这个.」 So it's fast when the user is continuing the same task, and smart enough to know when they've switched.

- The config is `ReuseStrategyConfig(enable=False, rollback_on_failure_signal=True, rollback_on_tool_failure_count=3)`
- The per-request override is `reuse_previous_intent: true|false`
- The intent switch markers are 「换个话题」, 「不是这个」, 「我要问别的」, 「切换」, 「换一个」, 「不是说这个」, and 「我想问的是」
- The implementation is in `src/intent_recognition/intent_reuse_strategy.py`
- The source value is `"reused"` in the `source` response field

### Multi-Recognizer Arbiter

When one doctor isn't sure, you get a second opinion. This does the same thing for intent recognition. It runs multiple recognizers in parallel, vector, rule, lightweight LLM, and arbitrates via vote (majority wins) or weighted_score (configurable weights per recognizer).

This is useful when a single recognizer is uncertain and you want cross-validation before committing to an intent. The extra cost is worth it for high-stakes classifications.

- The config is `ArbiterConfig(enable=False, mode="vote"|"weighted_score", recognizers=["vector","rule","lightweight_llm"], weights={"vector":0.8,"rule":0.6,"lightweight_llm":1.0})`
- The implementation is in `src/intent_recognition/multi_recognizer_arbiter.py`
- The source values are `"arbiter-vote"` or `"arbiter-weighted"`

### Fine-Tuning Integration Point

This one is about leaving the door open. This mechanism provides a training data export pipeline in JSONL format and a `model_tier="fine_tuned"` LLM client tier. It doesn't actually train a model. It collects the data, exports it, and when you're ready to fine-tune a domain-specific model externally, the infrastructure is already there.

The exported data can be used with any fine-tuning pipeline. Think of it as building the runway before you build the plane.

- The config is `FineTuningConfig(enable=False, model=None)`
- The implementation is in `src/intent_recognition/training_data_exporter.py` and `evaluate_fine_tuned()` in the evaluation runner
- The env var is `FINE_TUNED_MODEL` for the fine-tuned model name

### Per-Request Override Pattern

All retrieval, vector fallback, and reuse capabilities support per-request override via the `POST /agent/intent` and `POST /recognize` request body.

| Field | Type | Default | Effect |
|-------|------|---------|--------|
| `enable_retrieval` | `bool \| null` | `null` | Override retrieval (`null`=use config, `true`/`false`=force) |
| `enable_vector_fallback` | `bool \| null` | `null` | Override vector fallback |
| `reuse_previous_intent` | `bool \| null` | `null` | Override reuse |

Overrides are applied to a snapshot of the pipeline config and restored after the request completes, so they do not leak across requests sharing the singleton pipeline.

### Accuracy Presets

The `accuracy_preset` config field (`"balanced"` by default) provides a shorthand for enabling multiple capabilities at once. Available presets are as follows.

- `"balanced"` (default) means all extensions are OFF, rely on the three-layer waterfall
- `"high_accuracy"` means enable retrieval + vector fallback + reuse for maximum accuracy, at higher cost
- `"low_cost"` means all OFF, same as balanced but explicit

## Interview Insights

Six design decisions distilled from intent-recognition interview materials, resume framing, post-class exercises, and reuse templates. They complement the core architecture with evidence grading, a strict sub-task boundary, five-factor routing arbitration, extended evaluation metrics, a solution-evolution methodology, and a unified structured-output protocol.

All six default ON because they are additive. They enrich output without changing existing behavior. Think of them as the polish layer that takes the system from functional to interview-ready.

### Evidence Grading

Here's something I find genuinely elegant. Not all slot values are equally trustworthy. 「The user just said their budget is 3000」 is a fact. 「The user's profile says they usually spend 3000」 is a guess. This tags every slot value with an evidence grade.

- **`verified`** means the value is sourced from the current input, the active context, or a confirmed `KeyFact`. Hard operations like refunds and payments may proceed.
- **`provisional`** means the value is sourced from history, user profile, speculation, or a default value. Hard operations must either upgrade the evidence through user confirmation or observation check, or disclose the assumption.

And here's the key design decision. Once a value is tagged `verified`, it stays `verified`. You cannot downgrade it. This prevents provisional noise from overwriting confirmed facts. The upgrade path is one-way, and that's intentional.

The pipeline collects evidence via `_collect_evidence()`, upgrades provisional evidence via `_upgrade_provisional_to_verified()`, and enforces the hard-op check via `_check_hard_op_evidence()` when `evidence.require_verified_for_hard_ops=True` and the intent is in `evidence.high_risk_intents`.

- The config is `EvidenceConfig(enable_grading=True, require_verified_for_hard_ops=True, high_risk_intents=[])`
- The implementation is in `IntentRecognitionPipeline._collect_evidence()` / `_upgrade_provisional_to_verified()` / `_check_hard_op_evidence()` in `src/intent_recognition/pipeline.py`
- The models are `Evidence`, `EvidenceGrade`, and `SlotValue` in `src/intent_recognition/models.py`

### sub_tasks vs independent_intents Boundary

The job of an intent is to select a business flow. This is a subtle but important distinction that took me a while to fully appreciate.

Comparing prices is a step within shopping, not a separate intent. It doesn't trigger a different business flow. Booking a flight IS a separate intent because it triggers a different business flow. This boundary draws this line clearly.

- `independent_intents` enter multi-intent governance, including process filtering, `relations`, `pending_intents`, and sequential execution.
- `sub_tasks` are recorded as execution steps within the main flow and do **NOT** enter `relations`, `pending_intents`, or process filtering.

The deep LLM prompt (rules 6-10) and the lightweight LLM (`detect_boundary_simple()` with `_SUB_TASK_PATTERNS` / `_INDEPENDENT_MARKERS`) both enforce this boundary. Two layers of defense, making sure the distinction holds.

- The config is `BoundaryConfig(enable_sub_tasks=True, strict_mode=False)`
- The implementation is in `src/intent_recognition/deep_llm/prompts.py` (rules 6-10), `src/intent_recognition/deep_llm/classifier.py`, and `src/intent_recognition/lightweight_llm/classifier.py`

### Five-Factor Routing Arbitration

Confidence is a **signal**, not a probability. I can't stress this enough. When L2 confidence falls in the ambiguous zone between `clarify_threshold` and `accept_threshold`, you need more information before making a routing decision. The arbitration checks five factors in strict priority order.

| # | Factor | Fail action |
|---|--------|-------------|
| 1 | Rule validation | adjust confidence (±0.05) |
| 2 | Slot completeness | Clarify |
| 3 | Hard-constraint risk (high-risk intent + provisional required slot) | Clarify |
| 4 | Candidate gap (Top1-Top2 < threshold) | Escalate to L3 |
| 5 | Confidence + historical accuracy | Clarify if history poor and adjusted conf < accept |

The first failing factor that mandates Clarify or Escalate wins. Factor 5 reuses the multi-signal fusion score, it does not replace it.

The priority order matters. Slot completeness must be checked before risk, otherwise a high-risk intent with missing slots would produce a confusing risk-related reason instead of the actionable 「missing required slots」 reason. Getting the order right is the difference between a helpful clarification and a confusing one.

For L2/L3 disagreement, `arbitrate_l2_l3()` applies the following logic. L3 high-confidence plus complete slots means accept L3. Both below clarify threshold means force Clarify. Otherwise, run five-factor on L3.

- The config is `ArbitrationConfig(enable_five_factor=True, candidate_gap_threshold=0.1, risk_aware_clarify=True, high_risk_intents=[])`
- The implementation is in `src/intent_recognition/lightweight_llm/confidence_router.py` with `ArbitrationInput`, `ArbitrationDecision`, `arbitrate()`, and `arbitrate_l2_l3()`
- The output lives in the `arbitration_breakdown` field on `IntentRecognitionResult`, separate from `signals` which is typed as `dict[str, float]`

### Extended Evaluation Metrics

You can't improve what you can't measure. This goes beyond the 4-tier metrics with 8 fine-grained metrics plus an online feedback loop.

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

But the really cool part is the online feedback loop. Production failures collected via failure signals get fed back into the offline test set through `TestSet.import_online_samples()`. The system literally learns from its mistakes in production and gets better over time.

- The config uses `ExtendedEvaluationConfig` which toggles each metric
- The implementation is in `src/intent_recognition/evaluation/metrics.py`, `src/intent_recognition/evaluation/test_set.py`, and `src/intent_recognition/evaluation/runner.py`

### Solution Evolution Methodology

This isn't just about features. It's about telling the story of how the system evolved. Each stage solved the bottlenecks of the previous one, and understanding this progression is key to communicating system maturity.

| Stage | Architecture | Key mechanisms |
|-------|-------------|----------------|
| **v0** | Single deep LLM, no normalization | Brute-force intent recognition |
| **v1-stage1** | + L2 lightweight LLM | Cost reduction, confidence routing |
| **v1-stage2** | + L1 code layer + multi-signal fusion | Three-layer waterfall, <10ms fast path |
| **v1-stage3** | + advanced mechanisms | Retrieval, dynamic fewshot, vector fallback, reuse, arbiter, fine-tuning |
| **v2** | + interview insights | Evidence grading, boundary, arbitration, extended eval, protocol |

`TestRunner.assess_evolution_stage(config)` inspects the config and returns the current stage string. Use it to communicate maturity in interviews and to gate feature expectations.

Here's a 60-second interview skeleton you can fill in as you go.

> 「We model intent recognition as a ___-layer waterfall, a code layer for <10ms keyword/regex hits, a lightweight LLM for the ambiguous middle, and a deep LLM for ___ complex scenarios. Confidence is treated as a ___ (signal), so we run a five-factor ___ (arbitration) before accepting. Every slot value carries an evidence ___ (grade), verified or provisional. Hard operations require ___ (verified) evidence or must disclose assumptions. We evaluate with ___ (9) metrics including a confusion matrix and online-failure feedback loop, and the system evolves through v0 -> v1 -> v2 stages.」

- The implementation is in `src/intent_recognition/evaluation/runner.py::assess_evolution_stage()`

### Structured Output Protocol

When modules talk to each other, they need a shared language. This defines that language with a 10-field protocol that serves as the handoff contract between normalization and intent recognition, and between intent recognition and downstream ReAct/TAO loops.

| # | Field | Source | Default |
|---|-------|--------|---------|
| 1 | `normalized_query` | Normalization output (or raw input if skipped) | `""` on `/recognize` |
| 2 | `intent` | Recognition result | `null` |
| 3 | `sub_tasks` | Intra-flow steps | `[]` |
| 4 | `independent_intents` | Multi-flow goals | `[]` |
| 5 | `slots` | Slot extractor | `{}` |
| 6 | `missing_slots` | Slot extractor | `[]` |
| 7 | `hard_constraints` | Constraint extractor | `[]` |
| 8 | `soft_constraints` | Constraint extractor | `[]` |
| 9 | `verified_evidence` | Evidence collector | `[]` |
| 10 | `provisional_evidence` | Evidence collector | `[]` |

For backward compatibility, `sub_intents` is kept as a **deprecated alias** of `independent_intents`, synced via `model_validator(mode="after")` so old clients keep working. When you're ready to fully migrate, set `protocol.deprecate_sub_intents=True` to omit `sub_intents` from responses entirely.

- The config is `ProtocolConfig(enable_structured_output=True, deprecate_sub_intents=False)`
- The implementation is in `AgentIntentResponse` / `AgentIntentNormalizationDetail` / `RecognizeResponse` in `src/user_input_normalization/server.py`
