# XIntent

> [中文文档](README.zh-CN.md)

XIntent is an agent intent framework with a single unified pipeline that turns raw user input into a structured intent + slots + constraints. The main endpoint `POST /agent/intent` orchestrates the full flow end-to-end: it internally runs **user input normalization** (transform messy input into structured output) and then **three-layer intent recognition** (Code Layer → Lightweight LLM → Deep LLM waterfall with slot filling and rejection/clarification). Sub-endpoints `/normalize` and `/recognize` are also exposed for debugging or single-stage use.

## Overview

Solving two problems in Agent pipelines:
1. **"Can't understand the input"** — Agents fail on inputs like "that one", "the second one", "best value", "a bit more advanced". The normalization module covers six categories of input problems: anaphora, missing, expression, semantic, subjective, external fact.
2. **"Don't know what the user wants"** — A single LLM call for intent recognition is neither cheap nor reliable. The intent recognition module uses a three-layer waterfall architecture that escalates only when needed, with slot filling, hard/soft constraint extraction, and rejection/clarification dual exits.

```
                                 POST /agent/intent (main endpoint)
                                         │
                                         ▼
User Input ──► [Normalization] ──► [Intent Recognition] ──► { intent, slots, constraints }
                (internal)          (3-layer waterfall)         unified response
                                         │
                                         ▼
                                Prompt/Spec -> ReAct/TAO Loop -> Output
```

## Architecture

### Main Endpoint: `POST /agent/intent` (orchestrator)

End-to-end pipeline that orchestrates both modules in one call:

1. **Normalization (internal)**: runs the two-stage normalization pipeline on the raw input. If normalization pauses for clarification, the endpoint returns early with `intent=null` and `paused_at_normalization=true`.
2. **Intent recognition**: feeds the normalized text into the three-layer waterfall with slot filling and rejection/clarification dual exits.
3. **Unified response**: returns intent + slots + constraints at the top level, plus the full normalization detail nested under `normalization`, plus pipeline meta (`pipeline_path`, `raw_input`, `normalized_input`).

A `skip_normalization` flag allows callers to bypass normalization when the input is already normalized.

### Module 1: User Input Normalization (two-stage pipeline)

- **pre-normalization** (before intent recognition): pronoun resolution, ellipsis completion, sentence correction, term standardization, structured output, completeness check
- **deep-normalization** (inside ReAct loop): adjective quantification, external fact resolution, observation-based re-resolution

### Module 2: Intent Recognition (three-layer waterfall)

- **Layer 1 - Code Layer**: page guidance + keyword/regex + rule engine. Zero LLM calls, < 10ms. Returns first match or escalates.
- **Layer 2 - Lightweight LLM**: Flash/Mini model with confidence routing (>=0.85 accept, 0.6-0.85 clarify, <0.6 escalate). Multi-signal fusion of LLM confidence + rule match + vector similarity + historical accuracy.
- **Layer 3 - Deep LLM**: deep reasoning model for 5 complex scenarios — complex expressions, cross-turn context, intent switching, multi-intent decomposition, implicit info completion.
- **Slot Filling**: parameter extraction, hard/soft constraint separation, cross-turn accumulation with latest-wins and conflict detection.
- **Rejection / Clarification**: unsupported intent → rejection with reason; unclear intent → clarification question; next-turn implicit failure signal detection.
- **Accuracy Improvements (D17-D24, all optional, default OFF)**: retrieval-based candidate narrowing (D17), dynamic few-shot injection (D18), intent orthogonality check (D19), true multi-intent decomposition (D20), vector matching fallback (D21), intent reuse with rollback (D22), multi-recognizer arbiter (D23), fine-tuning integration point (D24). See [Technical Principles](docs/TECHNICAL_PRINCIPLES.md) for details.
- **Interview Insights (D26-D31, all default ON - additive)**: evidence grading with verified/provisional levels (D26), sub_tasks vs independent_intents boundary (D27), five-factor routing arbitration (D28), extended evaluation metrics with online feedback loop (D29), v0/v1/v2 solution evolution methodology (D30), 10-field structured output protocol as normalization-to-recognition handoff contract (D31). See [Technical Principles](docs/TECHNICAL_PRINCIPLES.md) for details.

## Installation

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
# Edit .env to add your API_KEY
```

`.env` file contents:

```env
API_KEY=sk-your-key-here
FLASH_LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash   # Layer 2 + normalization
PRO_LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro        # Layer 3 deep reasoning
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B          # Optional: for retrieval
BASE_URL=https://api.openai.com/v1  # Any OpenAI-compatible provider

# Optional: LLM client timeout and retry (for reasoning models with long think time)
# LLM_TIMEOUT=120      # Request timeout in seconds (default: 120)
# LLM_MAX_RETRIES=1    # Max retries on failure (default: 1, set 0 to disable)
```

> Note: Without `API_KEY` configured, the service automatically falls back to MockLLM mode (for development/testing).

## Start the Service Locally

**Option 1: Direct run**

```bash
python -m user_input_normalization.server
```

Access the service at `http://localhost:8000`

**Option 2: uvicorn with hot reload (development)**

```bash
uvicorn user_input_normalization.server:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Custom port**

```bash
PORT=9000 python -m user_input_normalization.server
```

## Docker Deployment

**Option 1: docker-compose (recommended)**

```bash
docker compose up -d
```

Access the service at `http://localhost:8000`. View logs:

```bash
docker compose logs -f
```

Stop the service:

```bash
docker compose down
```

**Option 2: docker build + run**

```bash
docker build -t agent-intent .
docker run -d --name agent-intent -p 8000:8000 --env-file .env agent-intent
```

## API Endpoints

| # | Endpoint | Method | Role | Description |
|---|----------|--------|------|-------------|
| 1 | `/` | GET | info | API info + version + endpoint list |
| 2 | `/health` | GET | info | Health check + LLM backend label |
| **3** | **`/agent/intent`** | **POST** | **main** | **Main entry: raw input → normalization (internal) → intent recognition → unified result** |
| 4 | `/normalize` | POST | sub | Two-stage normalization only (debugging / single-stage) |
| 5 | `/recognize` | POST | sub | Three-layer intent recognition only (expects pre-normalized text) |
| 6 | `/intents` | GET | sub | List registered intent definitions |
| 7 | `/docs` | GET | info | Swagger UI interactive docs |
| 8 | `/redoc` | GET | info | ReDoc documentation |

> **Production callers should prefer `POST /agent/intent`.** Sub-endpoints exist for debugging, partial use, or when the caller already has normalized text. See [docs/API.md](docs/API.md) for full request/response schemas.

### Main endpoint examples (`POST /agent/intent`)

**Code-layer hit (keyword match, no LLM):**

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

**L2 LLM with slot filling:**

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "帮我推荐一款3000元以内的手机",
    "session_id": "s2",
    "turn": 1
  }'
```

**Cross-turn slot accumulation (same session_id, next turn):**

```bash
# Turn 1: establish context
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "帮我推荐一款3000元以内的手机", "session_id": "sess-1", "turn": 1}'

# Turn 2: update budget (slot accumulates)
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "预算可以到4000", "session_id": "sess-1", "turn": 2}'
```

**With UI event (page guidance) or observation (deep normalization):**

```bash
# UI event for code layer page guidance
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "继续", "session_id": "s3", "turn": 1, "event": "click:next"}'

# Observation triggers deep normalization
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "帮我推荐更有性价比的牛仔裤", "session_id": "s4", "turn": 1,
       "observation": {"current_price": 200, "available_prices": [100, 150, 200, 300]}}'
```

**Skip normalization (input already normalized):**

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "我要退款", "session_id": "s5", "turn": 1, "skip_normalization": true}'
```

### Sub-endpoint examples

**Normalization only (`POST /normalize`):**

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

**Intent recognition only (`POST /recognize`, expects normalized text):**

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "我要退款",
    "session_id": "s1",
    "turn": 1
  }'
```

**List registered intents:**

```bash
curl http://localhost:8000/intents
```

## Run Tests

```bash
pytest
```

650 tests across 23 test files, all passing in ~0.7s. Tests use `MockLLMClient` (no real API calls).

## Documentation

- [HTTP API Reference](docs/API.md) - REST API endpoints, request/response schemas, code examples
- [Technical Principles](docs/TECHNICAL_PRINCIPLES.md) - Design rationale, linguistic foundations, three-layer waterfall architecture
- [AGENTS.md](AGENTS.md) - AI agent guidance for working with this codebase

## Project Structure

```
.
├── src/
│   ├── user_input_normalization/     # Module 1: input normalization
│   │   ├── server.py                  # FastAPI server (single entry point for both modules)
│   │   ├── pipeline.py                # NormalizationPipeline orchestrator
│   │   ├── models.py                  # Data models & enums
│   │   ├── config.py                  # Configuration
│   │   ├── storage/                   # Storage layer (abstract + memory)
│   │   ├── llm/                       # LLM clients (Mock + OpenAI-compatible)
│   │   ├── classification/            # Input classification (6 categories)
│   │   ├── pre_normalization/         # Pre-normalization (core)
│   │   ├── clarification/             # Clarification mechanism
│   │   ├── deep_normalization/        # Deep normalization
│   │   ├── quantification/            # Adjective quantification
│   │   ├── context/                   # Context integration
│   │   ├── vocabulary/                # Vocabulary table
│   │   └── attribute_resolution/      # Attribute-based anaphora resolution
│   └── intent_recognition/            # Module 2: three-layer intent recognition
│       ├── pipeline.py                # IntentRecognitionPipeline orchestrator
│       ├── models.py                  # IntentDefinition, SlotDefinition, Constraint
│       ├── config.py                  # IntentRecognitionConfig
│       ├── intent_registry.py         # IntentRegistry (hierarchical)
│       ├── storage/                   # SlotStateStore, IntentHistoryStore, EvaluationStore
│       ├── code_layer/                # Layer 1: page guidance + keyword + rule engine
│       ├── lightweight_llm/           # Layer 2: Flash LLM + confidence router + multi-signal
│       ├── deep_llm/                  # Layer 3: deep reasoning LLM
│       ├── slot_filling/              # Slot extraction + constraints + cross-turn merger
│       ├── rejection_clarification/   # Rejection + clarification dual exits
│       └── evaluation/                # Metrics + test set + runner
├── tests/                             # 650 tests across 23 files
├── docs/                              # API docs + technical principles (bilingual)
├── AGENTS.md                          # AI agent guidance
├── Dockerfile                         # Docker build
├── docker-compose.yml                 # Docker compose
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project config
├── .env.example                       # Environment template
└── .gitignore                         # Git ignore rules
```
