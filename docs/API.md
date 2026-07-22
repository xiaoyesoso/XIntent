# HTTP API Reference

> [中文 API 文档](API.zh-CN.md)

REST API documentation for the XIntent service. The framework's primary goal is **Agent intent recognition** — turn raw user input into a structured intent + slots + constraints. The main endpoint `POST /agent/intent` orchestrates the full pipeline end-to-end (normalization internally, then three-layer intent recognition). Sub-endpoints `/normalize` and `/recognize` remain available for internal use or direct calls when only one stage is needed.

## Base URL

```
http://localhost:8000
```

When deployed via Docker, the default port is `8000`. Change via `PORT` environment variable.

## Authentication

No API key required for HTTP requests. The backend LLM API key is configured via `.env` file on the server side.

## Interactive Docs

| URL | Description |
|-----|-------------|
| `/docs` | Swagger UI - interactive testing |
| `/redoc` | ReDoc - readable documentation |

## Endpoint Overview

| # | Method | Path | Role | Description |
|---|--------|------|------|-------------|
| 1 | `GET` | `/` | info | API info + version + endpoint list |
| 2 | `GET` | `/health` | info | Health check + LLM backend label |
| **3** | **`POST`** | **`/agent/intent`** | **main** | **Main entry point: raw input → normalization (internal) → intent recognition → unified result** |
| 4 | `POST` | `/normalize` | sub | Normalization only (internal component, exposed for debugging) |
| 5 | `POST` | `/recognize` | sub | Intent recognition only (expects pre-normalized text) |
| 6 | `GET` | `/intents` | sub | List registered intent definitions |

> **Production callers should prefer `POST /agent/intent`.** The sub-endpoints exist for debugging, partial use, or when the caller already has normalized text.

---

## Endpoints

### 1. GET `/` - API Info

Returns basic API information.

#### Request

```http
GET / HTTP/1.1
Host: localhost:8000
```

#### Response `200 OK`

```json
{
  "name": "XIntent API",
  "version": "0.2.0",
  "docs": "/docs",
  "redoc": "/redoc",
  "main_endpoint": "POST /agent/intent",
  "endpoints": {
    "agent_intent": "POST /agent/intent (main - normalization + intent recognition)",
    "normalize": "POST /normalize (sub - normalization only)",
    "recognize": "POST /recognize (sub - intent recognition only)",
    "intents": "GET /intents",
    "health": "GET /health"
  }
}
```

---

### 2. GET `/health` - Health Check

Returns service health status and LLM backend info.

#### Request

```http
GET /health HTTP/1.1
Host: localhost:8000
```

#### Response `200 OK`

```json
{
  "status": "ok",
  "version": "0.2.0",
  "llm_backend": "OpenAI-compatible (flash=deepseek-ai/DeepSeek-V4-Flash, pro=deepseek-ai/DeepSeek-V4-Pro)"
}
```

> If `API_KEY` is not configured, `llm_backend` will be `"MockLLM (no API_KEY set)"`.

---

### 3. POST `/agent/intent` - Agent Intent Recognition (Main Endpoint)

**This is the main entry point of the service.** It orchestrates the full pipeline end-to-end:

1. **Normalization (internal)**: runs the two-stage normalization pipeline on the raw input (pre-normalization + optional deep-normalization when `observation` is provided).
2. **Early exit on clarification**: if normalization pauses for clarification (e.g. low-confidence pronoun), the endpoint returns early with `intent=null` and `paused_at_normalization=true` — the client must resolve the clarification before retrying.
3. **Intent recognition**: feeds the normalized text into the three-layer waterfall (Code Layer → Lightweight LLM → Deep LLM) with slot filling and rejection/clarification dual exits.
4. **Unified response**: returns intent + slots + constraints at the top level, plus the full normalization detail nested under `normalization`, plus pipeline meta (`pipeline_path`, `raw_input`, `normalized_input`).

Production callers should prefer this endpoint over calling `/normalize` and `/recognize` separately.

#### Request

```http
POST /agent/intent HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

##### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `raw_input` | `string` | Yes | - | User raw input text |
| `session_id` | `string` | Yes | - | Session ID (shared by normalization and intent recognition stages) |
| `user_id` | `string` | No | `null` | User ID |
| `turn` | `integer` | No | `1` | Dialogue turn number |
| `context` | `object` | No | `null` | Additional context for intent recognition, e.g. `{"page": "product_detail"}` |
| `event` | `string` | No | `null` | UI event for page guidance (code layer), e.g. `"click:next"` |
| `observation` | `object` | No | `null` | ReAct observation for deep normalization (triggers deep stage) |
| `dialogue_history` | `array[object]` | No | `null` | Previous dialogue turns, e.g. `[{"role": "user", "content": "..."}]` |
| `skip_normalization` | `boolean` | No | `false` | Skip normalization stage (use when input is already normalized) |
| `enable_retrieval` | `boolean \| null` | No | `null` | Override D17 retrieval-based candidate narrowing (`null`=use config, `true`/`false`=force) |
| `enable_vector_fallback` | `boolean \| null` | No | `null` | Override D21 vector matching fallback (`null`=use config, `true`/`false`=force) |
| `reuse_previous_intent` | `boolean \| null` | No | `null` | Override D22 intent reuse with rollback (`null`=use config, `true`/`false`=force) |

> **Per-request overrides**: D17/D21/D22 override flags mutate the pipeline config only for the duration of one request and are restored afterward, so they do not leak across requests sharing the singleton pipeline.

##### Example 1: Code-layer hit (keyword match, no LLM)

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

##### Example 2: L2 LLM with slot filling

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "帮我推荐一款3000元以内的手机",
    "session_id": "s2",
    "turn": 1
  }'
```

##### Example 3: Cross-turn slot accumulation

```bash
# Turn 1: establish context
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# Turn 2: update budget (same session_id) - slot accumulates
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

##### Example 4: UI event (page guidance)

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "继续",
    "session_id": "s3",
    "turn": 1,
    "event": "click:next"
  }'
```

##### Example 5: Skip normalization (input already normalized)

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "我要退款",
    "session_id": "s4",
    "turn": 1,
    "skip_normalization": true
  }'
```

##### Example 6: With observation (triggers deep normalization)

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "帮我推荐更有性价比的牛仔裤",
    "session_id": "s5",
    "turn": 1,
    "observation": {"current_price": 200, "available_prices": [100, 150, 200, 300]}
  }'
```

#### Response `200 OK`

##### Response Body

Top-level fields are the intent recognition result (the primary answer). The `normalization` field carries the detailed normalization sub-result. Meta fields describe the pipeline path.

| Field | Type | Description |
|-------|------|-------------|
| `intent` | `string \| null` | Recognized intent (null if rejected or paused at normalization) |
| `confidence` | `number` | Confidence score [0.0, 1.0] |
| `source` | `string` | Which layer produced the intent: `"code-layer"`, `"lightweight-llm"`, `"deep-llm"`, `"vector-fallback"`, `"reused"`, `"arbiter-vote"`, `"arbiter-weighted"`, or `""` if paused at normalization |
| `slots` | `object` | Extracted slot values, e.g. `{"category": "手机", "budget_max": "3000"}` |
| `missing_slots` | `array[string]` | Required slots not yet provided |
| `need_clarification` | `boolean` | Whether clarification is needed (from normalization or intent recognition) |
| `clarification_question` | `string \| null` | Question to ask user (if needed) |
| `hard_constraints` | `array[object]` | Hard constraints (must satisfy) |
| `soft_constraints` | `array[object]` | Soft constraints (best effort) |
| `rejection_reason` | `string \| null` | Rejection reason (if unsupported) |
| `layer_reached` | `integer` | Highest intent-recognition layer reached (0 if paused at normalization) |
| `sub_intents` | `array[string]` | **Deprecated alias** of `independent_intents` (D27). Kept for backward compatibility; new callers should read `independent_intents` instead. Auto-synced via Pydantic `model_validator`. |
| `independent_intents` | `array[string]` | D27: Independent intents that each trigger a distinct business flow (multi-intent input). Intra-flow steps are NOT listed here — those are `sub_tasks`. |
| `sub_tasks` | `array[object]` | D27: Intra-flow steps (e.g. price comparison, delivery check). Each item: `{"name": string, "description": string}`. Do NOT enter D20 multi-intent governance. |
| `normalized_query` | `string` | D31: The normalized query string handed from normalization to intent recognition. Empty string when normalization is skipped. |
| `verified_evidence` | `array[object]` | D26: Evidence backed by current input, context, or confirmed KeyFact. Each item: `{"slot": string, "value": string, "source": string}`. Hard operations require verified evidence. |
| `provisional_evidence` | `array[object]` | D26: Evidence from history, profile, speculation, or default. Same item shape as `verified_evidence`. Must be upgraded or have assumptions disclosed. |
| `assumptions_disclosed` | `boolean` | D26: Whether provisional evidence assumptions have been disclosed in the response. |
| `assumptions` | `array[string]` | D26: Human-readable list of assumptions made when provisional evidence was used. |
| `arbitration_breakdown` | `object` | D28: Five-factor arbitration breakdown (rule_validation, slot_completeness, hard_constraint_risk, candidate_gap, confidence_history). Present when five-factor arbitration ran. |
| `intent_switched` | `boolean` | Whether intent switched from previous turn |
| `previous_intent` | `string \| null` | Previous turn's intent (if any) |
| `relations` | `array[object]` | D20: Dependency relations between independent_intents, e.g. `[{"src": "intent_b", "dst": "intent_a", "constraints": []}]` |
| `pending_intents` | `array[string]` | D20: Pending intents in topological order (first already executed) |
| `normalization` | `object \| null` | Normalization sub-result (null if `skip_normalization=true`) |
| `raw_input` | `string` | Original user input (echoed back) |
| `normalized_input` | `string` | Text actually fed into intent recognition |
| `pipeline_path` | `array[string]` | Stages executed, e.g. `["normalization", "intent-recognition"]` |
| `skipped_normalization` | `boolean` | Whether normalization was skipped |
| `paused_at_normalization` | `boolean` | True if pipeline paused at normalization awaiting user clarification |

##### `normalization` sub-object structure (when not skipped)

| Field | Type | Description |
|-------|------|-------------|
| `normalized_input` | `string` | Normalized input text |
| `pronoun_resolutions` | `array[object]` | Pronoun resolution table (see section 4) |
| `quantifiable_adjectives` | `array[object]` | Quantified adjective results (see section 4) |
| `term_mappings` | `array[object]` | Term standardization mappings (see section 4) |
| `completions` | `array[object]` | Ellipsis completion fields (see section 4) |
| `classification_tags` | `array[string]` | Input problem categories (see section 4) |
| `stage_reached` | `string` | Normalization stage reached: `"pre"` or `"deep"` |
| `paused_for_clarification` | `boolean` | Whether normalization paused for clarification |
| `clarification` | `object \| null` | Normalization clarification request (if any) |

##### `hard_constraints` / `soft_constraints` item structure

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | `"hard"` or `"soft"` |
| `expression` | `string` | Normalized constraint expression, e.g. `"price<=3000"` |
| `raw_text` | `string` | Original constraint text, e.g. `"3000元以内"` |

##### `source` possible values

| Value | Description |
|-------|-------------|
| `""` | No intent-recognition layer ran (paused at normalization) |
| `"code-layer"` | Layer 1 (keyword/regex/rule) - no LLM, < 10ms |
| `"lightweight-llm"` | Layer 2 (Flash/Mini model) + confidence routing |
| `"deep-llm"` | Layer 3 (deep reasoning model) for complex scenarios |
| `"vector-fallback"` | D21: Vector matching fallback (when L1 code-layer missed) |
| `"reused"` | D22: Reused previous turn's intent (confidence=1.0) |
| `"arbiter-vote"` | D23: Multi-recognizer arbiter (vote mode) |
| `"arbiter-weighted"` | D23: Multi-recognizer arbiter (weighted_score mode) |

##### `pipeline_path` possible values

| Value | Description |
|-------|-------------|
| `["normalization", "intent-recognition"]` | Full pipeline ran (default) |
| `["normalization"]` | Paused at normalization (clarification needed) |
| `["intent-recognition"]` | Normalization skipped (`skip_normalization=true`) |

#### Response Example 1: Code-layer hit (full pipeline)

```json
{
  "intent": "refund",
  "confidence": 1.0,
  "source": "code-layer",
  "slots": {},
  "missing_slots": [],
  "need_clarification": false,
  "clarification_question": null,
  "hard_constraints": [],
  "soft_constraints": [],
  "rejection_reason": null,
  "layer_reached": 1,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": null,
  "normalization": {
    "normalized_input": "我要退款",
    "pronoun_resolutions": [],
    "quantifiable_adjectives": [],
    "term_mappings": [],
    "completions": [],
    "classification_tags": [],
    "stage_reached": "pre",
    "paused_for_clarification": false,
    "clarification": null
  },
  "raw_input": "我要退款",
  "normalized_input": "我要退款",
  "pipeline_path": ["normalization", "intent-recognition"],
  "skipped_normalization": false,
  "paused_at_normalization": false
}
```

#### Response Example 2: L2 LLM with slot filling

```json
{
  "intent": "product_recommendation",
  "confidence": 0.95,
  "source": "lightweight-llm",
  "slots": {"category": "手机", "budget_max": "3000"},
  "missing_slots": [],
  "need_clarification": false,
  "clarification_question": null,
  "hard_constraints": [
    {"type": "hard", "expression": "price<=3000", "raw_text": "3000元以内"}
  ],
  "soft_constraints": [],
  "rejection_reason": null,
  "layer_reached": 2,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": null,
  "normalization": {
    "normalized_input": "帮我推荐一款3000元以内的手机",
    "pronoun_resolutions": [],
    "quantifiable_adjectives": [],
    "term_mappings": [],
    "completions": [],
    "classification_tags": ["主观判断问题"],
    "stage_reached": "pre",
    "paused_for_clarification": false,
    "clarification": null
  },
  "raw_input": "帮我推荐一款3000元以内的手机",
  "normalized_input": "帮我推荐一款3000元以内的手机",
  "pipeline_path": ["normalization", "intent-recognition"],
  "skipped_normalization": false,
  "paused_at_normalization": false
}
```

#### Response Example 3: Paused at normalization (clarification needed)

```json
{
  "intent": null,
  "confidence": 0.0,
  "source": "code-layer",
  "slots": {},
  "missing_slots": [],
  "need_clarification": true,
  "clarification_question": "您说的'那个帅气的同事'具体是指什么？请提供更多信息。",
  "hard_constraints": [],
  "soft_constraints": [],
  "rejection_reason": null,
  "layer_reached": 0,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": null,
  "normalization": {
    "normalized_input": "那个帅气的同事",
    "pronoun_resolutions": [],
    "quantifiable_adjectives": [],
    "term_mappings": [],
    "completions": [],
    "classification_tags": ["指代问题"],
    "stage_reached": "pre",
    "paused_for_clarification": true,
    "clarification": {
      "reason": "代词消解失败",
      "item": "那个帅气的同事",
      "candidates": [],
      "question": "您说的'那个帅气的同事'具体是指什么？请提供更多信息。",
      "confidence": 0.2
    }
  },
  "raw_input": "那个帅气的同事",
  "normalized_input": "那个帅气的同事",
  "pipeline_path": ["normalization"],
  "skipped_normalization": false,
  "paused_at_normalization": true
}
```

#### Response Example 4: Rejection (unsupported intent)

```json
{
  "intent": null,
  "confidence": 0.0,
  "source": "code-layer",
  "slots": {},
  "missing_slots": [],
  "need_clarification": false,
  "clarification_question": null,
  "hard_constraints": [],
  "soft_constraints": [],
  "rejection_reason": "intent_not_in_candidate_list | unsupported_content='今天天气怎么样' | supported_intents=[product_recommendation, product_query, product_comparison, order_query, refund]\n- product_recommendation: ...\n- product_query: ...\n- ...",
  "layer_reached": 1,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": null,
  "normalization": {
    "normalized_input": "今天天气怎么样",
    "pronoun_resolutions": [],
    "quantifiable_adjectives": [],
    "term_mappings": [],
    "completions": [],
    "classification_tags": [],
    "stage_reached": "pre",
    "paused_for_clarification": false,
    "clarification": null
  },
  "raw_input": "今天天气怎么样",
  "normalized_input": "今天天气怎么样",
  "pipeline_path": ["normalization", "intent-recognition"],
  "skipped_normalization": false,
  "paused_at_normalization": false
}
```

#### Error Response `500`

```json
{
  "detail": "Normalization failed: <error message>"
}
```

or

```json
{
  "detail": "Intent recognition failed: <error message>"
}
```

---

### 4. POST `/normalize` - User Input Normalization (Sub-endpoint)

Normalizes user input through the two-stage pipeline (pre-normalization + deep-normalization).

#### Request

```http
POST /normalize HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

##### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `raw_input` | `string` | Yes | - | User raw input text |
| `session_id` | `string` | Yes | - | Session ID (for cross-turn context reuse) |
| `user_id` | `string` | No | `null` | User ID (for user profile & personal vocab) |
| `turn` | `integer` | No | `0` | Dialogue turn number |
| `observation` | `object` | No | `null` | ReAct observation for deep normalization (e.g. tool return data) |

##### Example 1: Basic Pronoun Resolution

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "第二个适合生产吗？",
    "session_id": "s1",
    "user_id": "u1",
    "turn": 2
  }'
```

##### Example 2: Adjective Quantification (with observation)

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "帮我推荐更有性价比的牛仔裤",
    "session_id": "s2",
    "user_id": "u1",
    "turn": 1,
    "observation": {
      "current_price": 200,
      "available_prices": [100, 150, 200, 300]
    }
  }'
```

##### Example 3: Minimal Request

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "市场占有率多少？",
    "session_id": "s3"
  }'
```

#### Response `200 OK`

##### Response Body

| Field | Type | Description |
|-------|------|-------------|
| `normalized_input` | `string` | Normalized input text |
| `pronoun_resolutions` | `array[object]` | Pronoun resolution table |
| `quantifiable_adjectives` | `array[object]` | Quantified adjective results |
| `term_mappings` | `array[object]` | Term standardization mappings |
| `completions` | `array[object]` | Ellipsis completion fields |
| `classification_tags` | `array[string]` | Input problem categories |
| `stage_reached` | `string` | Pipeline stage reached: `"pre"` or `"deep"` |
| `paused_for_clarification` | `boolean` | Whether paused for user clarification |
| `clarification` | `object \| null` | Clarification request (if triggered) |

##### `pronoun_resolutions` item structure

| Field | Type | Description |
|-------|------|-------------|
| `pronoun` | `string` | Original pronoun, e.g. `"第二个"` |
| `resolved_to` | `string` | Resolved entity, e.g. `"TCC方案"` |
| `confidence` | `number` | Resolution confidence [0.0, 1.0] |
| `evidence_source` | `string` | Evidence source, e.g. `"对话历史第3轮"` |
| `named_entity` | `string \| null` | Semantic name for indexing |

##### `quantifiable_adjectives` item structure

| Field | Type | Description |
|-------|------|-------------|
| `adjective` | `string` | Adjective, e.g. `"性价比"` |
| `quantified` | `boolean` | Whether quantified |
| `quantified_value` | `object \| null` | Quantified tool params, e.g. `{"price_range": [150, 200]}` |
| `route_to` | `string \| null` | Route target if unquantified: `"deep"` |

##### `term_mappings` item structure

| Field | Type | Description |
|-------|------|-------------|
| `original` | `string` | Original term, e.g. `"RAG"` |
| `standard` | `string` | Standard term, e.g. `"检索增强生成"` |
| `source` | `string` | Source, e.g. `"vocabulary-table"` |

##### `completions` item structure

| Field | Type | Description |
|-------|------|-------------|
| `field` | `string` | Completed field, e.g. `"主语"` |
| `content` | `string` | Completed content |
| `source` | `string` | Source, e.g. `"对话历史第2轮"` |

##### `clarification` structure (when `paused_for_clarification` is `true`)

| Field | Type | Description |
|-------|------|-------------|
| `reason` | `string` | Trigger reason |
| `item` | `string` | Item to clarify |
| `candidates` | `array[string]` | Candidate list |
| `question` | `string` | Question for user |
| `confidence` | `number` | Current confidence |

##### `classification_tags` possible values

| Value | Description |
|-------|-------------|
| `"指代问题"` | Anaphora - pronoun/reference |
| `"缺失问题"` | Missing - incomplete input |
| `"表达问题"` | Expression - colloquial/broken |
| `"词义问题"` | Semantic - slang/polysemy |
| `"主观判断问题"` | Subjective - judgment words |
| `"外部事实问题"` | External fact - requires real-time data |

#### Response Example 1: Pronoun Resolution

```json
{
  "normalized_input": "TCC方案适合生产环境吗？",
  "pronoun_resolutions": [
    {
      "pronoun": "第二个",
      "resolved_to": "TCC方案",
      "confidence": 0.95,
      "evidence_source": "对话历史第3轮",
      "named_entity": "TCC方案"
    }
  ],
  "quantifiable_adjectives": [],
  "term_mappings": [],
  "completions": [],
  "classification_tags": ["指代问题"],
  "stage_reached": "pre",
  "paused_for_clarification": false,
  "clarification": null
}
```

#### Response Example 2: Adjective Quantification (Deep Stage)

```json
{
  "normalized_input": "推荐更有性价比的牛仔裤",
  "pronoun_resolutions": [],
  "quantifiable_adjectives": [
    {
      "adjective": "性价比",
      "quantified": true,
      "quantified_value": {
        "price_range": [150, 200],
        "quality_rank": "top 30%",
        "sort_by": "quality_desc"
      },
      "route_to": null
    }
  ],
  "term_mappings": [],
  "completions": [],
  "classification_tags": ["主观判断问题"],
  "stage_reached": "deep",
  "paused_for_clarification": false,
  "clarification": null
}
```

#### Response Example 3: Clarification Triggered

```json
{
  "normalized_input": "那个帅气的同事",
  "pronoun_resolutions": [],
  "quantifiable_adjectives": [],
  "term_mappings": [],
  "completions": [],
  "classification_tags": ["指代问题"],
  "stage_reached": "pre",
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

#### Error Response `500`

```json
{
  "detail": "Normalization failed: <error message>"
}
```

---

### 5. POST `/recognize` - Intent Recognition (Sub-endpoint)

Recognizes user intent through the three-layer waterfall pipeline (Code Layer → Lightweight LLM → Deep LLM) with slot filling and rejection/clarification.

#### Request

```http
POST /recognize HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

##### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | Yes | - | Normalized user input text (typically the output of `/normalize`) |
| `session_id` | `string` | No | `"default"` | Session ID (for cross-turn slot accumulation & intent history) |
| `user_id` | `string` | No | `null` | User ID |
| `turn` | `integer` | No | `1` | Dialogue turn number |
| `context` | `object` | No | `null` | Additional context (e.g. `{"page": "product_detail"}`) |
| `event` | `string` | No | `null` | UI event for page guidance (e.g. `"click:next"`, `"click:buy"`) |
| `dialogue_history` | `array[object]` | No | `null` | Previous dialogue turns, e.g. `[{"role": "user", "content": "..."}]` |
| `enable_retrieval` | `boolean \| null` | No | `null` | Override D17 retrieval-based candidate narrowing (`null`=use config, `true`/`false`=force) |
| `enable_vector_fallback` | `boolean \| null` | No | `null` | Override D21 vector matching fallback (`null`=use config, `true`/`false`=force) |
| `reuse_previous_intent` | `boolean \| null` | No | `null` | Override D22 intent reuse with rollback (`null`=use config, `true`/`false`=force) |

##### Example 1: Code Layer Hit (keyword match, no LLM)

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

##### Example 2: L2 LLM with Slot Filling

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "帮我推荐一款3000元以内的手机",
    "session_id": "s2",
    "turn": 1
  }'
```

##### Example 3: Cross-Turn Slot Accumulation

```bash
# Turn 1: establish context
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# Turn 2: update budget (same session_id) - slots accumulate
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

##### Example 4: With UI Event (page guidance)

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "继续",
    "session_id": "s3",
    "turn": 1,
    "event": "click:next"
  }'
```

##### Example 5: Unsupported Intent (rejection)

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天天气怎么样",
    "session_id": "s4",
    "turn": 1
  }'
```

#### Response `200 OK`

##### Response Body

| Field | Type | Description |
|-------|------|-------------|
| `intent` | `string \| null` | Recognized intent name (null if rejected) |
| `confidence` | `number` | Confidence score [0.0, 1.0] |
| `source` | `string` | Which layer produced this: `"code-layer"`, `"lightweight-llm"`, `"deep-llm"`, `"vector-fallback"`, `"reused"`, `"arbiter-vote"`, `"arbiter-weighted"` |
| `slots` | `object` | Extracted slot values, e.g. `{"category": "手机", "budget_max": "3000"}` |
| `missing_slots` | `array[string]` | Required slots not yet filled |
| `need_clarification` | `boolean` | Whether clarification is needed |
| `clarification_question` | `string \| null` | Question to ask user (if clarification needed) |
| `hard_constraints` | `array[object]` | Hard constraints (must satisfy) |
| `soft_constraints` | `array[object]` | Soft constraints (best effort) |
| `rejection_reason` | `string \| null` | Rejection reason if intent is unsupported |
| `layer_reached` | `integer` | Highest layer reached (1, 2, or 3) |
| `sub_intents` | `array[string]` | **Deprecated alias** of `independent_intents` (D27). Kept for backward compatibility; new callers should read `independent_intents` instead. |
| `independent_intents` | `array[string]` | D27: Independent intents that each trigger a distinct business flow (multi-intent input). Intra-flow steps are NOT listed here — those are `sub_tasks`. |
| `sub_tasks` | `array[object]` | D27: Intra-flow steps (e.g. price comparison, delivery check). Each item: `{"name": string, "description": string}`. Do NOT enter D20 multi-intent governance. |
| `normalized_query` | `string` | D31: The normalized query string. Always `""` for `/recognize` because normalization was not run by this endpoint. |
| `verified_evidence` | `array[object]` | D26: Evidence backed by current input, context, or confirmed KeyFact. Each item: `{"slot": string, "value": string, "source": string}`. |
| `provisional_evidence` | `array[object]` | D26: Evidence from history, profile, speculation, or default. Same item shape as `verified_evidence`. |
| `assumptions_disclosed` | `boolean` | D26: Whether provisional evidence assumptions have been disclosed in the response. |
| `assumptions` | `array[string]` | D26: Human-readable list of assumptions made when provisional evidence was used. |
| `arbitration_breakdown` | `object` | D28: Five-factor arbitration breakdown. Present when five-factor arbitration ran. |
| `intent_switched` | `boolean` | Whether intent switched from previous turn |
| `previous_intent` | `string \| null` | Previous turn's intent (if any) |
| `relations` | `array[object]` | D20: Dependency relations between independent_intents, e.g. `[{"src": "intent_b", "dst": "intent_a", "constraints": []}]` |
| `pending_intents` | `array[string]` | D20: Pending intents in topological order (first already executed) |

##### `hard_constraints` / `soft_constraints` item structure

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | `"hard"` or `"soft"` |
| `expression` | `string` | Normalized constraint expression, e.g. `"price<=3000"` |
| `raw_text` | `string` | Original constraint text, e.g. `"3000元以内"` |

##### `source` possible values

| Value | Description |
|-------|-------------|
| `"code-layer"` | Layer 1 (keyword/regex/rule) - no LLM, < 10ms |
| `"lightweight-llm"` | Layer 2 (Flash/Mini model) with confidence routing |
| `"deep-llm"` | Layer 3 (deep reasoning model) for complex scenarios |
| `"vector-fallback"` | D21: Vector matching fallback (when L1 code-layer missed) |
| `"reused"` | D22: Reused previous turn's intent (confidence=1.0) |
| `"arbiter-vote"` | D23: Multi-recognizer arbiter (vote mode) |
| `"arbiter-weighted"` | D23: Multi-recognizer arbiter (weighted_score mode) |

#### Response Example 1: Code Layer Hit

```json
{
  "intent": "refund",
  "confidence": 1.0,
  "source": "code-layer",
  "slots": {},
  "missing_slots": [],
  "need_clarification": false,
  "clarification_question": null,
  "hard_constraints": [],
  "soft_constraints": [],
  "rejection_reason": null,
  "layer_reached": 1,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": null
}
```

#### Response Example 2: L2 LLM with Slot Filling

```json
{
  "intent": "product_recommendation",
  "confidence": 0.95,
  "source": "lightweight-llm",
  "slots": {
    "category": "手机",
    "budget_max": "3000"
  },
  "missing_slots": [],
  "need_clarification": false,
  "clarification_question": null,
  "hard_constraints": [
    {
      "type": "hard",
      "expression": "price<=3000",
      "raw_text": "3000元以内"
    }
  ],
  "soft_constraints": [],
  "rejection_reason": null,
  "layer_reached": 2,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": null
}
```

#### Response Example 3: Cross-Turn Slot Accumulation

```json
{
  "intent": "product_recommendation",
  "confidence": 0.65,
  "source": "lightweight-llm",
  "slots": {
    "category": "手机",
    "budget_max": "4000"
  },
  "missing_slots": [],
  "need_clarification": true,
  "clarification_question": "请问您想推荐什么类型的商品？",
  "hard_constraints": [
    {"type": "hard", "expression": "price<=3000", "raw_text": "3000元以内"},
    {"type": "hard", "expression": "price<=4000", "raw_text": "预算可以到4000"}
  ],
  "soft_constraints": [],
  "rejection_reason": null,
  "layer_reached": 2,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": "product_recommendation"
}
```

#### Response Example 4: Rejection (Unsupported Intent)

```json
{
  "intent": null,
  "confidence": 0.0,
  "source": "code-layer",
  "slots": {},
  "missing_slots": [],
  "need_clarification": false,
  "clarification_question": null,
  "hard_constraints": [],
  "soft_constraints": [],
  "rejection_reason": "intent_not_in_candidate_list | unsupported_content='今天天气怎么样' | supported_intents=[product_recommendation, product_query, product_comparison, order_query, refund]\n- product_recommendation: ...\n- product_query: ...\n- ...",
  "layer_reached": 1,
  "sub_intents": [],
  "intent_switched": false,
  "previous_intent": null
}
```

#### Error Response `500`

```json
{
  "detail": "Intent recognition failed: <error message>"
}
```

---

### 6. GET `/intents` - List Registered Intents

Returns all registered intent definitions (used by the recognition pipeline).

#### Request

```http
GET /intents HTTP/1.1
Host: localhost:8000
```

#### Response `200 OK`

```json
{
  "intents": [
    {
      "name": "product_recommendation",
      "description": "用户希望系统推荐商品、帮忙挑选商品，或根据预算、用途、偏好给出购买建议",
      "parent_intent": null,
      "slots": [
        {"name": "category", "required": true, "description": "商品类目，如手机、耳机"},
        {"name": "budget_max", "required": false, "description": "最高预算"},
        {"name": "usage_scenario", "required": false, "description": "使用场景"}
      ],
      "positive_examples": ["3000以内有什么手机推荐", "帮我推荐个耳机"],
      "negative_examples": ["iPhone 16的电池容量是多少 -> product_query"]
    },
    {
      "name": "product_query",
      "description": "用户希望查询某个商品、品牌、型号、参数、价格或库存信息",
      "parent_intent": null,
      "slots": [],
      "positive_examples": ["iPhone 16的电池容量是多少"],
      "negative_examples": ["3000以内有什么手机推荐 -> product_recommendation"]
    },
    {
      "name": "product_comparison",
      "description": "用户希望比较两个或多个商品、品牌或型号",
      "parent_intent": null,
      "slots": [],
      "positive_examples": ["iPhone 16和小米15哪个好"],
      "negative_examples": ["iPhone 16的电池容量是多少 -> product_query"]
    },
    {
      "name": "order_query",
      "description": "用户希望查询订单状态、物流信息",
      "parent_intent": null,
      "slots": [],
      "positive_examples": ["我的订单到哪了", "查询订单 12345"],
      "negative_examples": []
    },
    {
      "name": "refund",
      "description": "用户希望申请退款或售后",
      "parent_intent": "order_query",
      "slots": [],
      "positive_examples": ["我要退款", "这个商品我想退货"],
      "negative_examples": []
    }
  ]
}
```

##### Intent item structure

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Intent name (unique identifier) |
| `description` | `string` | Intent description (used in LLM prompts) |
| `parent_intent` | `string \| null` | Parent intent name (for hierarchical intents, D13) |
| `slots` | `array[object]` | Slot definitions for this intent |
| `positive_examples` | `array[string]` | Positive example inputs |
| `negative_examples` | `array[string]` | Negative examples (with redirect target) |

##### Slot item structure

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Slot name |
| `required` | `boolean` | Whether this slot is required |
| `description` | `string` | Slot description (used in LLM prompts) |

---

## Code Examples

### Python (requests)

```python
import requests

# Normalization
response = requests.post("http://localhost:8000/normalize", json={
    "raw_input": "第二个适合生产吗？",
    "session_id": "s1",
    "user_id": "u1",
    "turn": 2,
})
data = response.json()
print(data["normalized_input"])

# Intent recognition
response = requests.post("http://localhost:8000/recognize", json={
    "text": "帮我推荐一款3000元以内的手机",
    "session_id": "s2",
    "turn": 1,
})
data = response.json()
print(data["intent"], data["confidence"], data["slots"])
```

### Python (async with httpx)

```python
import httpx
import asyncio

async def recognize():
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/recognize", json={
            "text": "我要退款",
            "session_id": "s3",
            "turn": 1,
        })
        return response.json()

result = asyncio.run(recognize())
print(result["intent"], result["source"])  # "refund" "code-layer"
```

### JavaScript (fetch)

```javascript
// Intent recognition
const response = await fetch("http://localhost:8000/recognize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "帮我推荐一款3000元以内的手机",
    session_id: "s1",
    turn: 1,
  }),
});
const data = await response.json();
console.log(data.intent, data.confidence, data.slots);
```

### curl

```bash
# Normalize
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "turn": 1}'

# Recognize
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "我要退款", "session_id": "s2", "turn": 1}'

# List intents
curl http://localhost:8000/intents

# Health check
curl http://localhost:8000/health
```

---

## Cross-Turn Context Reuse

Both modules support cross-turn context reuse via `session_id`:

### Normalization: pronoun resolution reuse

```bash
# Turn 1: Resolve "第二个" -> "TCC方案"
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "session-abc", "user_id": "u1", "turn": 1}'

# Turn 2: "第二个" is reused from key facts (no re-resolution)
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个成本多少？", "session_id": "session-abc", "user_id": "u1", "turn": 2}'
```

### Intent recognition: slot accumulation

```bash
# Turn 1: establish slots (category=手机, budget_max=3000)
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# Turn 2: update budget (budget_max: 3000 -> 4000, latest-wins)
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

---

## Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `422` | Validation error (missing required fields, wrong types) |
| `500` | Internal server error (normalization/recognition failed) |
