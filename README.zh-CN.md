# XIntent 意图框架

> [English](README.md)

XIntent 是一个 Agent 意图框架，提供**统一主接口** `POST /agent/intent` 端到端将用户原始输入转化为结构化的意图 + 槽位 + 约束。主接口内部串起两个互补模块：**用户输入规范化**（将混乱输入转为结构化输出）与 **三层意图识别**（代码层 → 轻量 LLM → 深度 LLM 瀑布式架构，带槽位填充和拒识/澄清）。子接口 `/normalize` 与 `/recognize` 保留供调试或单阶段使用。

## 简介

解决 Agent 流水线中的两个核心问题：
1. **"听不懂输入"** — Agent 连"那个""第二个""最划算""再高级一点"都听不懂。规范化模块覆盖六大类输入问题：指代、缺失、表达、词义、主观判断、外部事实。
2. **"不知道用户要做什么"** — 单次大模型调用做意图识别既不便宜也不可靠。意图识别模块采用三层瀑布式架构，按需逐层升级，带槽位填充、硬/软约束分离、拒识/澄清双出口。

```
                                POST /agent/intent（主接口）
                                         │
                                         ▼
用户输入 ──► [输入规范化] ──► [意图识别] ──► { intent, slots, constraints }
              （内部）         （三层瀑布）        统一响应
                                         │
                                         ▼
                              Prompt/Spec 加载 -> ReAct/TAO 循环 -> 输出
```

## 架构

### 主接口：`POST /agent/intent`（编排器）

端到端编排两个模块的完整流水线：

1. **规范化（内部）**：对原始输入跑两阶段规范化流水线。若规范化阶段暂停等待澄清，主接口直接返回 `intent=null` 且 `paused_at_normalization=true`。
2. **意图识别**：将规范化后的文本送入三层瀑布，带槽位填充和拒识/澄清双出口。
3. **统一响应**：顶层返回意图 + 槽位 + 约束，`normalization` 字段嵌套完整规范化详情，并附流水线元信息（`pipeline_path`、`raw_input`、`normalized_input`）。

通过 `skip_normalization=true` 可跳过规范化阶段（输入已规范化时使用）。

### 模块一：用户输入规范化（两阶段流水线）

- **pre-normalization**（意图识别之前）：指代消解、省略补全、病句修正、术语标准化、结构化输出、完整性校验
- **deep-normalization**（ReAct 循环中）：判断词量化、外部事实消解、依赖 Observation 回溯的消解

### 模块二：意图识别（三层瀑布式）

- **Layer 1 - 代码层**：页面引导 + 关键字/正则 + 规则引擎。零 LLM 调用，< 10ms。返回首个匹配或升级。
- **Layer 2 - 轻量 LLM**：Flash/Mini 模型 + 置信度路由（>=0.85 接受，0.6-0.85 澄清，<0.6 升级）。多信号融合：LLM 置信度 + 规则匹配 + 向量相似度 + 历史准确率。
- **Layer 3 - 深度 LLM**：深度推理模型，处理五类复杂场景 — 复杂表达、跨轮上下文、意图切换、多意图分解、隐式信息补全。
- **Slot Filling**：参数抽取、硬/软约束分离、跨轮累积（latest-wins + 冲突检测）。
- **拒识 / 澄清**：不支持意图 → 拒识并说明原因；不清晰意图 → 澄清提问；下一轮隐式失败信号检测。
- **准确率提升（D17-D24，全部可选，默认关闭）**：检索式候选收窄（D17）、动态 few-shot 注入（D18）、意图正交性检查（D19）、真正的多意图分解（D20）、向量匹配兜底（D21）、意图复用与回滚（D22）、多识别器仲裁（D23）、微调集成点（D24）。详见[技术原理](docs/TECHNICAL_PRINCIPLES.zh-CN.md)。
- **面试洞察（D26-D31，全部默认开启 -- 叠加式增强）**：证据分级（verified/provisional，D26）、sub_tasks 与 independent_intents 边界（D27）、五因子路由仲裁（D28）、扩展评估指标与线上反馈闭环（D29）、v0/v1/v2 方案演进方法论（D30）、10 字段结构化输出协议作为规范化到意图识别的交接契约（D31）。详见[技术原理](docs/TECHNICAL_PRINCIPLES.zh-CN.md)。

## 安装

```bash
pip install -e .
```

## 配置

复制 `.env.example` 为 `.env` 并填写 API 密钥：

```bash
cp .env.example .env
# 编辑 .env 填入你的 API_KEY
```

`.env` 文件内容：

```env
API_KEY=sk-your-key-here
FLASH_LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash   # Layer 2 + 规范化
PRO_LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro        # Layer 3 深度推理
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B          # 可选：用于检索
BASE_URL=https://api.openai.com/v1  # 任意 OpenAI 兼容的 API 提供商

# 可选：LLM 客户端超时与重试（推理模型思考时间长时需要调整）
# LLM_TIMEOUT=120      # 请求超时秒数（默认：120）
# LLM_MAX_RETRIES=1    # 失败重试次数（默认：1，设 0 禁用）
```

> 注意：未配置 `API_KEY` 时，服务自动降级为 MockLLM 模式（用于开发测试）。

## 本地启动服务

**方式一：直接运行**

```bash
python -m user_input_normalization.server
```

服务启动后访问 `http://localhost:8000`

**方式二：uvicorn 热重载（开发模式）**

```bash
uvicorn user_input_normalization.server:app --host 0.0.0.0 --port 8000 --reload
```

**方式三：指定端口**

```bash
PORT=9000 python -m user_input_normalization.server
```

## Docker 启动

**方式一：docker-compose（推荐）**

```bash
docker compose up -d
```

服务启动后访问 `http://localhost:8000`，查看日志：

```bash
docker compose logs -f
```

停止服务：

```bash
docker compose down
```

**方式二：docker build + run**

```bash
docker build -t agent-intent .
docker run -d --name agent-intent -p 8000:8000 --env-file .env agent-intent
```

## API 接口

| # | 接口 | 方法 | 角色 | 说明 |
|---|------|------|------|------|
| 1 | `/` | GET | 信息 | API 信息 + 版本 + 接口列表 |
| 2 | `/health` | GET | 信息 | 健康检查 + LLM 后端标识 |
| **3** | **`/agent/intent`** | **POST** | **主接口** | **主入口：原始输入 → 规范化（内部）→ 意图识别 → 统一结果** |
| 4 | `/normalize` | POST | 子接口 | 仅两阶段规范化（调试 / 单阶段使用） |
| 5 | `/recognize` | POST | 子接口 | 仅三层意图识别（要求传入已规范化的文本） |
| 6 | `/intents` | GET | 子接口 | 列出已注册意图定义 |
| 7 | `/docs` | GET | 信息 | Swagger UI 交互式文档 |
| 8 | `/redoc` | GET | 信息 | ReDoc 文档 |

> **生产调用方应优先使用 `POST /agent/intent`。** 子接口用于调试、局部使用，或调用方已有规范化文本时直接使用。完整请求/响应格式见 [docs/API.zh-CN.md](docs/API.zh-CN.md)。

### 主接口示例（`POST /agent/intent`）

**代码层命中（关键字匹配，无 LLM）：**

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

**L2 LLM 带槽位填充：**

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "帮我推荐一款3000元以内的手机",
    "session_id": "s2",
    "turn": 1
  }'
```

**跨轮槽位累积（相同 session_id，下一轮）：**

```bash
# 第 1 轮：建立上下文
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# 第 2 轮：更新预算（槽位累积）
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

**带 UI 事件（页面引导）或 observation（深度规范化）：**

```bash
# UI 事件用于代码层页面引导
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "继续", "session_id": "s3", "turn": 1, "event": "click:next"}'

# observation 触发深度规范化
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "帮我推荐更有性价比的牛仔裤", "session_id": "s4", "turn": 1,
       "observation": {"current_price": 200, "available_prices": [100, 150, 200, 300]}}'
```

**跳过规范化（输入已规范化）：**

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "我要退款", "session_id": "s5", "turn": 1, "skip_normalization": true}'
```

### 子接口示例

**仅规范化（`POST /normalize`）：**

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

**仅意图识别（`POST /recognize`，要求传入已规范化文本）：**

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

**列出已注册意图：**

```bash
curl http://localhost:8000/intents
```

## 运行测试

```bash
pytest
```

共 650 个测试，覆盖 23 个测试文件，全部通过约 0.7 秒。测试使用 `MockLLMClient`（无真实 API 调用）。

## 文档

- [HTTP API 接口文档](docs/API.zh-CN.md) - REST API 接口、请求/响应格式、代码调用示例
- [技术原理文档](docs/TECHNICAL_PRINCIPLES.zh-CN.md) - 设计思想、语言学基础、三层瀑布式架构
- [AGENTS.md](AGENTS.md) - AI Agent 协作指南

## 项目结构

```
.
├── src/
│   ├── user_input_normalization/     # 模块一：输入规范化
│   │   ├── server.py                  # FastAPI 服务（两个模块的统一入口）
│   │   ├── pipeline.py                # NormalizationPipeline 编排器
│   │   ├── models.py                  # 数据模型与枚举
│   │   ├── config.py                  # 配置中心
│   │   ├── storage/                   # 存储层（抽象接口 + 内存实现）
│   │   ├── llm/                       # LLM 客户端（Mock + OpenAI 兼容）
│   │   ├── classification/            # 输入问题分类（六大类）
│   │   ├── pre_normalization/         # 初步规范化（核心模块）
│   │   ├── clarification/             # 澄清机制
│   │   ├── deep_normalization/        # 深度规范化
│   │   ├── quantification/            # 形容词量化
│   │   ├── context/                   # 上下文整合
│   │   ├── vocabulary/                # 词汇表
│   │   └── attribute_resolution/      # 属性检索指代消解
│   └── intent_recognition/            # 模块二：三层意图识别
│       ├── pipeline.py                # IntentRecognitionPipeline 编排器
│       ├── models.py                  # IntentDefinition, SlotDefinition, Constraint
│       ├── config.py                  # IntentRecognitionConfig
│       ├── intent_registry.py         # IntentRegistry（支持层次化）
│       ├── storage/                   # SlotStateStore, IntentHistoryStore, EvaluationStore
│       ├── code_layer/                # Layer 1：页面引导 + 关键字 + 规则引擎
│       ├── lightweight_llm/           # Layer 2：Flash LLM + 置信度路由 + 多信号融合
│       ├── deep_llm/                  # Layer 3：深度推理 LLM
│       ├── slot_filling/              # 槽位抽取 + 约束 + 跨轮合并
│       ├── rejection_clarification/   # 拒识 + 澄清双出口
│       └── evaluation/                # 指标 + 测试集 + 运行器
├── tests/                             # 650 个测试，23 个文件
├── docs/                              # API 文档 + 技术原理（中英双语）
├── AGENTS.md                          # AI Agent 协作指南
├── Dockerfile                         # Docker 镜像构建
├── docker-compose.yml                 # Docker Compose 编排
├── requirements.txt                   # Python 依赖清单
├── pyproject.toml                     # 项目配置
├── .env.example                       # 环境变量模板
└── .gitignore                         # Git 忽略规则
```
