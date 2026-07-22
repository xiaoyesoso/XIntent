# HTTP API 接口文档

> [English API Reference](API.md)

XIntent 服务的 REST API 接口调用文档。本框架的核心目标是 **Agent 意图识别** —— 将用户原始输入转化为结构化的意图 + 槽位 + 约束。主接口 `POST /agent/intent` 端到端编排整个流水线（内部先做规范化，再做三层意图识别）。子接口 `/normalize` 与 `/recognize` 保留供内部调用或仅需单阶段时直接使用。

## 基础地址

```
http://localhost:8000
```

Docker 部署时默认端口为 `8000`，可通过 `PORT` 环境变量修改。

## 认证

HTTP 请求无需 API 密钥。后端 LLM 的 API 密钥通过服务端的 `.env` 文件配置。

## 交互式文档

| 地址 | 说明 |
|------|------|
| `/docs` | Swagger UI - 可交互测试 |
| `/redoc` | ReDoc - 可读文档 |

## 接口总览

| # | 方法 | 路径 | 角色 | 说明 |
|---|------|------|------|------|
| 1 | `GET` | `/` | 信息 | API 信息 + 版本 + 接口列表 |
| 2 | `GET` | `/health` | 信息 | 健康检查 + LLM 后端标签 |
| **3** | **`POST`** | **`/agent/intent`** | **主接口** | **主入口：原始输入 → 规范化（内部）→ 意图识别 → 统一结果** |
| 4 | `POST` | `/normalize` | 子接口 | 仅规范化（内部组件，暴露用于调试） |
| 5 | `POST` | `/recognize` | 子接口 | 仅意图识别（要求传入已规范化的文本） |
| 6 | `GET` | `/intents` | 子接口 | 列出已注册的意图定义 |

> **生产调用方应优先使用 `POST /agent/intent`。** 子接口用于调试、局部使用，或调用方已有规范化文本时直接使用。

---

## 接口列表

### 1. GET `/` - API 信息

返回 API 基本信息。

#### 请求

```http
GET / HTTP/1.1
Host: localhost:8000
```

#### 响应 `200 OK`

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

### 2. GET `/health` - 健康检查

返回服务健康状态和 LLM 后端信息。

#### 请求

```http
GET /health HTTP/1.1
Host: localhost:8000
```

#### 响应 `200 OK`

```json
{
  "status": "ok",
  "version": "0.2.0",
  "llm_backend": "OpenAI-compatible (flash=deepseek-ai/DeepSeek-V4-Flash, pro=deepseek-ai/DeepSeek-V4-Pro)"
}
```

> 未配置 `API_KEY` 时，`llm_backend` 显示为 `"MockLLM (no API_KEY set)"`。

---

### 3. POST `/agent/intent` - Agent 意图识别（主接口）

**这是服务的主入口。** 端到端编排整个流水线：

1. **规范化（内部）**：对原始输入跑两阶段规范化流水线（pre-normalization + 传 `observation` 时可选触发 deep-normalization）。
2. **澄清早退**：如果规范化阶段暂停等待澄清（如低置信度代词），主接口直接返回 `intent=null` 且 `paused_at_normalization=true` —— 客户端必须先处理澄清再重试。
3. **意图识别**：将规范化后的文本送入三层瀑布（代码层 → 轻量 LLM → 深度 LLM），带槽位填充和拒识/澄清双出口。
4. **统一响应**：顶层返回意图 + 槽位 + 约束，`normalization` 字段嵌套完整规范化详情，并附流水线元信息（`pipeline_path`、`raw_input`、`normalized_input`）。

生产调用方应优先使用此接口，而非分别调用 `/normalize` 与 `/recognize`。

#### 请求

```http
POST /agent/intent HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

##### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `raw_input` | `string` | 是 | - | 用户原始输入文本 |
| `session_id` | `string` | 是 | - | 会话 ID（规范化与意图识别共用） |
| `user_id` | `string` | 否 | `null` | 用户 ID |
| `turn` | `integer` | 否 | `1` | 对话轮次 |
| `context` | `object` | 否 | `null` | 意图识别的额外上下文，如 `{"page": "product_detail"}` |
| `event` | `string` | 否 | `null` | UI 事件（页面引导，代码层用），如 `"click:next"` |
| `observation` | `object` | 否 | `null` | ReAct 观测值（触发深度规范化） |
| `dialogue_history` | `array[object]` | 否 | `null` | 之前的对话轮次，如 `[{"role": "user", "content": "..."}]` |
| `skip_normalization` | `boolean` | 否 | `false` | 跳过规范化阶段（输入已规范化时使用） |
| `enable_retrieval` | `boolean \| null` | 否 | `null` | 覆盖 D17 检索式候选意图收窄（`null`=用配置，`true`/`false`=强制） |
| `enable_vector_fallback` | `boolean \| null` | 否 | `null` | 覆盖 D21 向量匹配兜底（`null`=用配置，`true`/`false`=强制） |
| `reuse_previous_intent` | `boolean \| null` | 否 | `null` | 覆盖 D22 意图复用（含回滚）（`null`=用配置，`true`/`false`=强制） |

> **Per-request 覆盖**：D17/D21/D22 覆盖标志仅在单次请求期间修改流水线配置，请求结束后自动恢复，不会泄漏到共享同一单例流水线的后续请求。

##### 示例 1：代码层命中（关键字匹配，无 LLM）

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

##### 示例 2：L2 LLM 带槽位填充

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "帮我推荐一款3000元以内的手机",
    "session_id": "s2",
    "turn": 1
  }'
```

##### 示例 3：跨轮槽位累积

```bash
# 第 1 轮：建立上下文
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# 第 2 轮：更新预算（相同 session_id）- 槽位累积
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

##### 示例 4：UI 事件（页面引导）

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

##### 示例 5：跳过规范化（输入已规范化）

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

##### 示例 6：带 observation（触发深度规范化）

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

#### 响应 `200 OK`

##### 响应体

顶层字段是意图识别结果（主答案）。`normalization` 字段携带详细规范化子结果。元信息字段描述流水线路径。

| 字段 | 类型 | 说明 |
|------|------|------|
| `intent` | `string \| null` | 识别到的意图（拒识或在规范化阶段暂停时为 null） |
| `confidence` | `number` | 置信度 [0.0, 1.0] |
| `source` | `string` | 产生意图的层：`"code-layer"`、`"lightweight-llm"`、`"deep-llm"`、`"vector-fallback"`、`"reused"`、`"arbiter-vote"`、`"arbiter-weighted"`，规范化暂停时为 `""` |
| `slots` | `object` | 提取到的槽位值，如 `{"category": "手机", "budget_max": "3000"}` |
| `missing_slots` | `array[string]` | 未填充的必填槽位 |
| `need_clarification` | `boolean` | 是否需要澄清（来自规范化或意图识别） |
| `clarification_question` | `string \| null` | 给用户的澄清问题（若需要） |
| `hard_constraints` | `array[object]` | 硬约束（必须满足） |
| `soft_constraints` | `array[object]` | 软约束（尽量满足） |
| `rejection_reason` | `string \| null` | 拒识原因（不支持时） |
| `layer_reached` | `integer` | 最高到达的意图识别层（在规范化暂停时为 0） |
| `sub_intents` | `array[string]` | **已弃用别名**，等价于 `independent_intents`（D27）。为向后兼容保留，新调用方应使用 `independent_intents`。由 Pydantic `model_validator` 自动同步。 |
| `independent_intents` | `array[string]` | D27：独立意图——每个意图各自触发不同业务流程（多意图输入）。流程内步骤不在此列，它们归入 `sub_tasks`。 |
| `sub_tasks` | `array[object]` | D27：流程内步骤（如比价、查物流）。元素结构：`{"name": string, "description": string}`。不进入 D20 多意图治理。 |
| `normalized_query` | `string` | D31：从规范化交接到意图识别的规范化查询字符串。跳过规范化时为空字符串。 |
| `verified_evidence` | `array[object]` | D26：已验证证据——来自当前输入、上下文或已确认 KeyFact。元素结构：`{"slot": string, "value": string, "source": string}`。硬操作要求已验证证据。 |
| `provisional_evidence` | `array[object]` | D26：暂定证据——来自历史、画像、推测或默认值。元素结构同 `verified_evidence`。必须升级或显式披露假设。 |
| `assumptions_disclosed` | `boolean` | D26：响应中是否已披露暂定证据的假设。 |
| `assumptions` | `array[string]` | D26：使用暂定证据时所做假设的人类可读列表。 |
| `arbitration_breakdown` | `object` | D28：五因素仲裁明细（rule_validation、slot_completeness、hard_constraint_risk、candidate_gap、confidence_history）。运行过五因素仲裁时存在。 |
| `intent_switched` | `boolean` | 是否相对上一轮切换了意图 |
| `previous_intent` | `string \| null` | 上一轮的意图（若有） |
| `relations` | `array[object]` | D20：独立意图间的依赖关系，如 `[{"src": "intent_b", "dst": "intent_a", "constraints": []}]` |
| `pending_intents` | `array[string]` | D20：拓扑排序后的待执行意图列表（首个已执行） |
| `normalization` | `object \| null` | 规范化子结果（`skip_normalization=true` 时为 null） |
| `raw_input` | `string` | 原始用户输入（回显） |
| `normalized_input` | `string` | 实际送入意图识别的文本 |
| `pipeline_path` | `array[string]` | 执行的阶段，如 `["normalization", "intent-recognition"]` |
| `skipped_normalization` | `boolean` | 是否跳过了规范化 |
| `paused_at_normalization` | `boolean` | 是否在规范化阶段暂停等待用户澄清 |

##### `normalization` 子对象结构（未跳过时）

| 字段 | 类型 | 说明 |
|------|------|------|
| `normalized_input` | `string` | 规范化后的输入文本 |
| `pronoun_resolutions` | `array[object]` | 指代消解表（见第 4 节） |
| `quantifiable_adjectives` | `array[object]` | 可量化形容词结果（见第 4 节） |
| `term_mappings` | `array[object]` | 术语标准化映射（见第 4 节） |
| `completions` | `array[object]` | 省略补全字段（见第 4 节） |
| `classification_tags` | `array[string]` | 输入问题分类标签（见第 4 节） |
| `stage_reached` | `string` | 规范化阶段：`"pre"` 或 `"deep"` |
| `paused_for_clarification` | `boolean` | 规范化是否暂停等待澄清 |
| `clarification` | `object \| null` | 规范化澄清请求（若有） |

##### `hard_constraints` / `soft_constraints` 元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `string` | `"hard"` 或 `"soft"` |
| `expression` | `string` | 规范化后的约束表达式，如 `"price<=3000"` |
| `raw_text` | `string` | 原始约束文本，如 `"3000元以内"` |

##### `source` 可能值

| 值 | 说明 |
|------|------|
| `""` | 未运行任何意图识别层（在规范化阶段暂停） |
| `"code-layer"` | Layer 1（关键字/正则/规则）- 无 LLM，< 10ms |
| `"lightweight-llm"` | Layer 2（Flash/Mini 模型）+ 置信度路由 |
| `"deep-llm"` | Layer 3（深度推理模型）处理复杂场景 |
| `"vector-fallback"` | D21：向量匹配兜底（L1 代码层未命中时） |
| `"reused"` | D22：复用上一轮意图（置信度=1.0） |
| `"arbiter-vote"` | D23：多识别器仲裁（投票模式） |
| `"arbiter-weighted"` | D23：多识别器仲裁（加权评分模式） |

##### `pipeline_path` 可能值

| 值 | 说明 |
|------|------|
| `["normalization", "intent-recognition"]` | 完整流水线运行（默认） |
| `["normalization"]` | 在规范化阶段暂停（需要澄清） |
| `["intent-recognition"]` | 跳过规范化（`skip_normalization=true`） |

#### 响应示例 1：代码层命中（完整流水线）

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

#### 响应示例 2：L2 LLM 带槽位填充

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

#### 响应示例 3：在规范化阶段暂停（需要澄清）

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

#### 响应示例 4：拒识（不支持的意图）

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

#### 错误响应 `500`

```json
{
  "detail": "Normalization failed: <error message>"
}
```

或

```json
{
  "detail": "Intent recognition failed: <error message>"
}
```

---

### 4. POST `/normalize` - 用户输入规范化（子接口）

通过两阶段流水线（初步规范化 + 深度规范化）规范化用户输入。

#### 请求

```http
POST /normalize HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

##### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `raw_input` | `string` | 是 | - | 用户原始输入文本 |
| `session_id` | `string` | 是 | - | 会话 ID（用于跨轮上下文复用） |
| `user_id` | `string` | 否 | `null` | 用户 ID（用于用户画像和个性化词汇） |
| `turn` | `integer` | 否 | `0` | 对话轮次 |
| `observation` | `object` | 否 | `null` | ReAct 观测值（用于深度规范化，如工具返回数据） |

##### 示例 1：基本指代消解

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

##### 示例 2：形容词量化（带 observation）

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

##### 示例 3：最小请求

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "市场占有率多少？",
    "session_id": "s3"
  }'
```

#### 响应 `200 OK`

##### 响应体

| 字段 | 类型 | 说明 |
|------|------|------|
| `normalized_input` | `string` | 规范化后的输入文本 |
| `pronoun_resolutions` | `array[object]` | 指代消解表 |
| `quantifiable_adjectives` | `array[object]` | 可量化形容词结果 |
| `term_mappings` | `array[object]` | 术语标准化映射 |
| `completions` | `array[object]` | 省略补全字段 |
| `classification_tags` | `array[string]` | 输入问题分类标签 |
| `stage_reached` | `string` | 流水线阶段：`"pre"` 或 `"deep"` |
| `paused_for_clarification` | `boolean` | 是否暂停等待用户澄清 |
| `clarification` | `object \| null` | 澄清请求（若触发） |

##### `pronoun_resolutions` 元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `pronoun` | `string` | 原始代词，如 `"第二个"` |
| `resolved_to` | `string` | 消解到的实体，如 `"TCC方案"` |
| `confidence` | `number` | 消解置信度 [0.0, 1.0] |
| `evidence_source` | `string` | 证据来源，如 `"对话历史第3轮"` |
| `named_entity` | `string \| null` | 语义命名（用于索引） |

##### `quantifiable_adjectives` 元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `adjective` | `string` | 形容词，如 `"性价比"` |
| `quantified` | `boolean` | 是否已量化 |
| `quantified_value` | `object \| null` | 量化后的工具参数，如 `{"price_range": [150, 200]}` |
| `route_to` | `string \| null` | 未量化时路由到：`"deep"` |

##### `term_mappings` 元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `original` | `string` | 原始术语，如 `"RAG"` |
| `standard` | `string` | 标准术语，如 `"检索增强生成"` |
| `source` | `string` | 来源，如 `"vocabulary-table"` |

##### `completions` 元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `field` | `string` | 补全字段，如 `"主语"` |
| `content` | `string` | 补全内容 |
| `source` | `string` | 来源，如 `"对话历史第2轮"` |

##### `clarification` 结构（`paused_for_clarification` 为 `true` 时）

| 字段 | 类型 | 说明 |
|------|------|------|
| `reason` | `string` | 触发原因 |
| `item` | `string` | 待澄清项 |
| `candidates` | `array[string]` | 候选列表 |
| `question` | `string` | 给用户的问题 |
| `confidence` | `number` | 当前置信度 |

##### `classification_tags` 可能值

| 值 | 说明 |
|------|------|
| `"指代问题"` | 代词/引用 |
| `"缺失问题"` | 输入不完整 |
| `"表达问题"` | 口语化/病句 |
| `"词义问题"` | 黑话/多义词 |
| `"主观判断问题"` | 判断词 |
| `"外部事实问题"` | 需实时数据 |

#### 响应示例 1：指代消解

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

#### 响应示例 2：形容词量化（深度阶段）

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

#### 响应示例 3：触发澄清

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

#### 错误响应 `500`

```json
{
  "detail": "Normalization failed: <error message>"
}
```

---

### 5. POST `/recognize` - 意图识别（子接口）

通过三层瀑布式流水线（代码层 → 轻量 LLM → 深度 LLM）识别用户意图，带槽位填充和拒识/澄清。

#### 请求

```http
POST /recognize HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

##### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | `string` | 是 | - | 规范化后的用户输入文本（通常是 `/normalize` 的输出） |
| `session_id` | `string` | 否 | `"default"` | 会话 ID（用于跨轮槽位累积和意图历史） |
| `user_id` | `string` | 否 | `null` | 用户 ID |
| `turn` | `integer` | 否 | `1` | 对话轮次 |
| `context` | `object` | 否 | `null` | 额外上下文，如 `{"page": "product_detail"}` |
| `event` | `string` | 否 | `null` | UI 事件（页面引导），如 `"click:next"`、`"click:buy"` |
| `dialogue_history` | `array[object]` | 否 | `null` | 之前的对话轮次，如 `[{"role": "user", "content": "..."}]` |
| `enable_retrieval` | `boolean \| null` | 否 | `null` | 覆盖 D17 检索式候选意图收窄（`null`=用配置，`true`/`false`=强制） |
| `enable_vector_fallback` | `boolean \| null` | 否 | `null` | 覆盖 D21 向量匹配兜底（`null`=用配置，`true`/`false`=强制） |
| `reuse_previous_intent` | `boolean \| null` | 否 | `null` | 覆盖 D22 意图复用（含回滚）（`null`=用配置，`true`/`false`=强制） |

##### 示例 1：代码层命中（关键字匹配，无 LLM）

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

##### 示例 2：L2 LLM 带槽位填充

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "帮我推荐一款3000元以内的手机",
    "session_id": "s2",
    "turn": 1
  }'
```

##### 示例 3：跨轮槽位累积

```bash
# 第 1 轮：建立上下文
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# 第 2 轮：更新预算（相同 session_id）- 槽位累积
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

##### 示例 4：带 UI 事件（页面引导）

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

##### 示例 5：不支持的意图（拒识）

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天天气怎么样",
    "session_id": "s4",
    "turn": 1
  }'
```

#### 响应 `200 OK`

##### 响应体

| 字段 | 类型 | 说明 |
|------|------|------|
| `intent` | `string \| null` | 识别到的意图名称（拒识时为 null） |
| `confidence` | `number` | 置信度 [0.0, 1.0] |
| `source` | `string` | 产生此结果的层：`"code-layer"`、`"lightweight-llm"`、`"deep-llm"`、`"vector-fallback"`、`"reused"`、`"arbiter-vote"`、`"arbiter-weighted"` |
| `slots` | `object` | 提取到的槽位值，如 `{"category": "手机", "budget_max": "3000"}` |
| `missing_slots` | `array[string]` | 未填充的必填槽位 |
| `need_clarification` | `boolean` | 是否需要澄清 |
| `clarification_question` | `string \| null` | 给用户的澄清问题（若需要） |
| `hard_constraints` | `array[object]` | 硬约束（必须满足） |
| `soft_constraints` | `array[object]` | 软约束（尽量满足） |
| `rejection_reason` | `string \| null` | 拒识原因（不支持时） |
| `layer_reached` | `integer` | 最高到达的层（1、2 或 3） |
| `sub_intents` | `array[string]` | **已弃用别名**，等价于 `independent_intents`（D27）。为向后兼容保留，新调用方应使用 `independent_intents`。 |
| `independent_intents` | `array[string]` | D27：独立意图——每个意图各自触发不同业务流程（多意图输入）。流程内步骤归入 `sub_tasks`。 |
| `sub_tasks` | `array[object]` | D27：流程内步骤（如比价、查物流）。元素结构：`{"name": string, "description": string}`。不进入 D20 多意图治理。 |
| `normalized_query` | `string` | D31：规范化查询字符串。`/recognize` 不运行规范化，此字段恒为 `""`。 |
| `verified_evidence` | `array[object]` | D26：已验证证据——来自当前输入、上下文或已确认 KeyFact。元素结构：`{"slot": string, "value": string, "source": string}`。 |
| `provisional_evidence` | `array[object]` | D26：暂定证据——来自历史、画像、推测或默认值。元素结构同 `verified_evidence`。 |
| `assumptions_disclosed` | `boolean` | D26：响应中是否已披露暂定证据的假设。 |
| `assumptions` | `array[string]` | D26：使用暂定证据时所做假设的人类可读列表。 |
| `arbitration_breakdown` | `object` | D28：五因素仲裁明细。运行过五因素仲裁时存在。 |
| `intent_switched` | `boolean` | 是否相对上一轮切换了意图 |
| `previous_intent` | `string \| null` | 上一轮的意图（若有） |
| `relations` | `array[object]` | D20：独立意图间的依赖关系，如 `[{"src": "intent_b", "dst": "intent_a", "constraints": []}]` |
| `pending_intents` | `array[string]` | D20：拓扑排序后的待执行意图列表（首个已执行） |

##### `hard_constraints` / `soft_constraints` 元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `string` | `"hard"` 或 `"soft"` |
| `expression` | `string` | 规范化后的约束表达式，如 `"price<=3000"` |
| `raw_text` | `string` | 原始约束文本，如 `"3000元以内"` |

##### `source` 可能值

| 值 | 说明 |
|------|------|
| `"code-layer"` | Layer 1（关键字/正则/规则）- 无 LLM，< 10ms |
| `"lightweight-llm"` | Layer 2（Flash/Mini 模型）+ 置信度路由 |
| `"deep-llm"` | Layer 3（深度推理模型）处理复杂场景 |
| `"vector-fallback"` | D21：向量匹配兜底（L1 代码层未命中时） |
| `"reused"` | D22：复用上一轮意图（置信度=1.0） |
| `"arbiter-vote"` | D23：多识别器仲裁（投票模式） |
| `"arbiter-weighted"` | D23：多识别器仲裁（加权评分模式） |

#### 响应示例 1：代码层命中

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

#### 响应示例 2：L2 LLM 带槽位填充

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

#### 响应示例 3：跨轮槽位累积

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

#### 响应示例 4：拒识（不支持的意图）

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

#### 错误响应 `500`

```json
{
  "detail": "Intent recognition failed: <error message>"
}
```

---

### 6. GET `/intents` - 列出已注册意图

返回所有已注册的意图定义（识别流水线使用）。

#### 请求

```http
GET /intents HTTP/1.1
Host: localhost:8000
```

#### 响应 `200 OK`

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

##### 意图元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 意图名称（唯一标识） |
| `description` | `string` | 意图描述（用于 LLM 提示词） |
| `parent_intent` | `string \| null` | 父意图名称（用于层次化意图，D13） |
| `slots` | `array[object]` | 该意图的槽位定义 |
| `positive_examples` | `array[string]` | 正例输入 |
| `negative_examples` | `array[string]` | 反例（带重定向目标） |

##### 槽位元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 槽位名称 |
| `required` | `boolean` | 是否必填 |
| `description` | `string` | 槽位描述（用于 LLM 提示词） |

---

## 代码示例

### Python (requests)

```python
import requests

# 规范化
response = requests.post("http://localhost:8000/normalize", json={
    "raw_input": "第二个适合生产吗？",
    "session_id": "s1",
    "user_id": "u1",
    "turn": 2,
})
data = response.json()
print(data["normalized_input"])

# 意图识别
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
// 意图识别
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
# 规范化
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "turn": 1}'

# 意图识别
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "我要退款", "session_id": "s2", "turn": 1}'

# 列出意图
curl http://localhost:8000/intents

# 健康检查
curl http://localhost:8000/health
```

---

## 跨轮上下文复用

两个模块都通过 `session_id` 支持跨轮上下文复用：

### 规范化：指代消解复用

```bash
# 第 1 轮：消解 "第二个" -> "TCC方案"
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "session-abc", "user_id": "u1", "turn": 1}'

# 第 2 轮："第二个" 从 key facts 复用（无需重新消解）
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个成本多少？", "session_id": "session-abc", "user_id": "u1", "turn": 2}'
```

### 意图识别：槽位累积

```bash
# 第 1 轮：建立槽位（category=手机, budget_max=3000）
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# 第 2 轮：更新预算（budget_max: 3000 -> 4000，latest-wins）
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"text": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

---

## 状态码

| 状态码 | 说明 |
|------|------|
| `200` | 成功 |
| `422` | 参数校验错误（缺少必填字段、类型错误） |
| `500` | 服务器内部错误（规范化/识别失败） |
