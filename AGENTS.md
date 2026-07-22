# AGENTS.md

Guidance for AI agents working with this codebase.

## Project Overview

XIntent is an agent intent framework that turns raw user input into a structured intent + slots + constraints via a single unified main endpoint. The main endpoint `POST /agent/intent` orchestrates two complementary modules end-to-end:

1. **User Input Normalization** (`user_input_normalization`): Two-stage pipeline (pre-normalization + deep-normalization) that transforms messy user input (pronouns, ellipsis, slang, subjective adjectives) into structured, machine-consumable output. Called internally by the main endpoint; also exposed as `POST /normalize` for debugging or single-stage use.
2. **Intent Recognition** (`intent_recognition`): Three-layer waterfall architecture (Code Layer → Lightweight LLM → Deep LLM) with slot filling, rejection/clarification dual exits, and evaluation metrics. Consumes normalized input and produces structured intent + slots + constraints. Called internally by the main endpoint; also exposed as `POST /recognize` for callers that already have normalized text.

```
                                 POST /agent/intent  (main endpoint)
                                         │
                                         ▼
User Input ──► [Normalization] ──► [Intent Recognition] ──► { intent, slots, constraints }
                (internal)          (3-layer waterfall)         unified response
                                         │
                                         ▼
                                Prompt/Spec -> ReAct/TAO Loop -> Output
```

Sub-endpoints `/normalize` and `/recognize` remain available for internal use, debugging, or when only one stage is needed.

**Tech stack**: Python 3.10+, FastAPI, Pydantic 2.x, OpenAI-compatible LLM API

## Quick Commands

```bash
# Install (use miniconda3 python - browser-use-env lacks dependencies)
/Users/souljoy/miniconda3/bin/pip install -e ".[dev]"

# Run tests (650 tests, ~0.7s)
/Users/souljoy/miniconda3/bin/python -m pytest -v

# Start API server
/Users/souljoy/miniconda3/bin/python -m user_input_normalization.server

# Start with hot reload (dev)
/Users/souljoy/miniconda3/bin/python -m uvicorn user_input_normalization.server:app --reload

# Docker
docker compose up -d
```

> **Important**: Always use `/Users/souljoy/miniconda3/bin/python` for all Python commands. The default `python3` points to `~/.browser-use-env/bin/python3` which lacks `pyyaml`, `openai`, and other dependencies.

## Project Structure

```
src/
├── user_input_normalization/      # Module 1: input normalization
│   ├── server.py                  # FastAPI server - exposes POST /agent/intent (main) + sub-endpoints
│   ├── pipeline.py                # NormalizationPipeline - main orchestrator
│   ├── models.py                  # All Pydantic data models & enums
│   ├── config.py                  # Configuration center (thresholds, top-k, etc.)
│   ├── storage/
│   │   ├── base.py                # Abstract storage interfaces (5 stores)
│   │   └── memory.py              # In-memory implementations (for dev/test)
│   ├── llm/
│   │   ├── base.py                # LLMClient abstract interface
│   │   ├── mock.py                # MockLLMClient (for testing)
│   │   └── openai_client.py       # OpenAI-compatible LLM client (reads .env)
│   ├── classification/            # Input problem classification (6 categories)
│   ├── pre_normalization/         # Core: pre-normalization stage
│   ├── clarification/             # Low-confidence clarification mechanism
│   ├── deep_normalization/        # Deep stage (ReAct loop, quantification)
│   ├── quantification/            # Adjective -> tool params conversion
│   ├── context/                   # Three-layer context integration
│   ├── vocabulary/                # Self-iterating vocabulary table
│   └── attribute_resolution/      # Attribute + retrieval anaphora resolution
│
└── intent_recognition/            # Module 2: three-layer intent recognition
    ├── __init__.py                # Public exports (Pipeline, Registry, Models, Config)
    ├── pipeline.py                # IntentRecognitionPipeline - main orchestrator (D1, D2)
    ├── models.py                  # IntentRecognitionResult, IntentDefinition, SlotDefinition, Constraint
    ├── config.py                  # IntentRecognitionConfig (thresholds, weights, signals)
    ├── intent_registry.py         # IntentRegistry (hierarchical intent definitions, D13)
    ├── storage/
    │   ├── base.py                # Abstract: SlotStateStore, IntentHistoryStore, EvaluationStore
    │   └── memory.py              # In-memory implementations
    ├── code_layer/                # Layer 1: code-based recognition (D3)
    │   ├── page_guidance.py       # UI event -> intent mapping
    │   ├── keyword_matcher.py     # Keyword + regex matching
    │   ├── rule_engine.py         # Context state + normalized input rules
    │   ├── vector_matcher.py      # D21: Vector matching fallback
    │   └── classifier.py          # Orchestrator (returns first match, < 10ms, zero LLM)
    ├── lightweight_llm/           # Layer 2: lightweight LLM (D4, D5)
    │   ├── prompts.py             # Structured prompt builder (candidate intents, slots, few-shot)
    │   ├── classifier.py          # Flash/Mini model call + JSON parsing
    │   ├── confidence_router.py   # >=0.85 accept, 0.6-0.85 clarify, <0.6 escalate
    │   ├── multi_signal.py        # Multi-signal fusion (LLM + rule + vector + history)
    │   ├── candidate_retriever.py # D17: Retrieval-based candidate narrowing
    │   └── dynamic_fewshot.py     # D18: Dynamic few-shot injection
    ├── deep_llm/                  # Layer 3: deep reasoning LLM (D6)
    │   ├── prompts.py             # Deep reasoning prompt (5 complex scenarios)
    │   └── classifier.py          # Deep model call + intent validation
    ├── slot_filling/              # Slot extraction + constraints (D7, D8, D9)
    │   ├── extractor.py           # SlotExtractor (extract slots, detect missing)
    │   ├── constraints.py         # ConstraintExtractor (hard/soft separation, regex+LLM)
    │   └── cross_turn.py          # CrossTurnSlotMerger (latest-wins, conflict detection)
    ├── rejection_clarification/   # Dual exits (D10, D11)
    │   └── handler.py             # RejectionClarificationHandler (unsupported + unclear)
    ├── evaluation/                # Metrics & test set (D15, D16)
    │   ├── metrics.py             # MetricsCalculator (Top-K, per-intent, rejection, slot)
    │   ├── test_set.py            # TestSet (9 scenario types)
    │   └── runner.py              # TestRunner (run test set, generate report)
    ├── intent_reuse_strategy.py   # D22: Intent reuse with rollback
    ├── multi_recognizer_arbiter.py # D23: Multi-recognizer arbitration
    └── training_data_exporter.py  # D24: Training data export for fine-tuning

tests/                             # 650 tests across 23 files, all passing
docs/                              # API docs + technical principles (bilingual)
```

## Architecture

### Main Endpoint: `POST /agent/intent` (orchestrator)

The single unified entry point of the service. Defined in `user_input_normalization/server.py` along with its request/response models (`AgentIntentRequest`, `AgentIntentResponse`, `AgentIntentNormalizationDetail`).

Orchestration flow:
1. **Stage 1 — Normalization (internal, skippable)**: Calls `NormalizationPipeline.process()` on the raw input. If `request.skip_normalization=True`, this stage is bypassed and the raw input is fed directly to intent recognition. If normalization pauses for clarification (low-confidence pronoun, etc.), the endpoint returns early with `intent=null`, `paused_at_normalization=true`, and `pipeline_path=["normalization"]` — the client must resolve the clarification before retrying.
2. **Stage 2 — Intent recognition**: Calls `IntentRecognitionPipeline.recognize()` on the normalized text (or raw text if normalization was skipped). Produces structured intent + slots + constraints + layer metadata.
3. **Unified response**: Intent recognition result sits at the top level of `AgentIntentResponse`. The full normalization sub-result is nested under `normalization`. Pipeline meta (`pipeline_path`, `raw_input`, `normalized_input`, `skipped_normalization`, `paused_at_normalization`) describes what actually ran.

`pipeline_path` values:
- `["normalization", "intent-recognition"]` — full pipeline ran (default)
- `["normalization"]` — paused at normalization awaiting user clarification
- `["intent-recognition"]` — normalization skipped (`skip_normalization=true`)

### Module 1: Two-Stage Normalization Pipeline

```
User Input -> [InputClassifier] -> [PreNormalizer] -> Intent Recognition -> [DeepNormalizer] -> Output
                                      ↑                                      ↑
                                 Before intent                          Inside ReAct loop
```

- **Pre-normalization** (`PreNormalizer`): Pronoun resolution, ellipsis completion, sentence correction, term standardization. Runs before intent recognition. Enforces strict responsibility boundary (cannot answer questions, execute tools, or fabricate facts).
- **Deep-normalization** (`DeepNormalizer`): Adjective quantification, external fact resolution. Runs inside ReAct loop when `observation` is provided.

### Module 2: Three-Layer Intent Recognition Waterfall

```
Normalized Input -> [L1 Code Layer] --miss--> [L2 Lightweight LLM] --low conf--> [L3 Deep LLM]
                        |                              |                              |
                        +------------ hit -------------+---------- high conf --------+
                                                       |
                                              [Slot Filling] -> [Rejection / Clarification]
                                                       |
                                          IntentRecognitionResult
```

- **Layer 1 - Code Layer** (`code_layer/`): Page guidance + keyword/regex + rule engine. Zero LLM calls, < 10ms. Returns first match or "unmatched" signal.
- **Layer 2 - Lightweight LLM** (`lightweight_llm/`): Flash/Mini model with confidence routing (>=0.85 accept, 0.6-0.85 clarify, <0.6 escalate to L3). Multi-signal fusion combines LLM confidence + rule match + vector similarity + historical accuracy.
- **Layer 3 - Deep LLM** (`deep_llm/`): Deep reasoning model for 5 complex scenarios: complex expressions (病句/倒装), cross-turn context dependency, intent switching, multi-intent decomposition, implicit info completion.
- **Slot Filling** (`slot_filling/`): Parameter extraction, hard/soft constraint separation, cross-turn accumulation with latest-wins strategy and conflict detection.
- **Rejection/Clarification** (`rejection_clarification/`): Unsupported intent (not in candidate list) → rejection with reason. Unclear intent (low confidence or missing required slots) → clarification question. Implicit failure signal detection ("你理解错了") for next-turn evaluation.

## Key Design Decisions

### Normalization (user-input-normalization)

1. **Responsibility boundary**: `SYSTEM_PROMPT` with explicit `CAN_DO` / `CANNOT_DO` lists in `pre_normalization/prompts.py`. The normalizer must NOT answer questions, execute tools, or fabricate facts.
2. **Structured output**: `NormalizationResult` model in `models.py` with SPO, pronoun resolution table, quantifiable adjectives, term mappings, completions.
3. **Cross-turn reuse**: Pronoun resolutions persisted as `KeyFact` in `KeyFactStore`, reused across turns with same `session_id`.
4. **Named indexing**: Entities get semantic names (e.g., "TCC方案" not "方案一") via `named_entity` field.
5. **Clarification threshold**: `theta_clarify=0.6` (configurable). Confidence below threshold triggers clarification instead of guessing.
6. **Vocabulary self-iteration**: Offline analysis promotes candidate terms to vocabulary table when thresholds met (count > 100, discussants > 3, consecutive > 10).

### Intent Recognition (three-layer-intent-recognition)

- **D1 - Three-layer waterfall**: Code → Lightweight LLM → Deep LLM, escalating only on miss/low-confidence. Chosen over single-layer (cost/latency) and parallel (3× API cost, non-comparable confidence).
- **D2 - Independent stage**: Intent recognition runs before ReAct/TAO loop, not inside. Enables independent evaluation and affects Prompt/Context/Tools selection.
- **D3 - Code layer methods**: Page guidance + keyword/regex + rule engine. Returns first match or "unmatched" (not a default intent).
- **D4 - Lightweight LLM + confidence routing**: Flash/Mini model with `accept_threshold=0.85`, `clarify_threshold=0.6`.
- **D5 - Multi-signal confidence fusion**: LLM confidence (0.5) + rule match (0.2) + vector similarity (0.2) + historical accuracy (0.1), not just LLM confidence.
- **D6 - Deep LLM for 5 complex scenarios**: Complex expressions, cross-turn dependency, intent switching, multi-intent decomposition, implicit info completion.
- **D7 - Slot filling integrated**: Slot extraction is part of intent recognition, not a separate stage.
- **D8 - Hard vs soft constraint separation**: Hard constraints (不超过, 必须) vs soft constraints (最好, 优先) extracted separately.
- **D9 - Cross-turn slot accumulation**: Latest-wins for slot values, accumulate for constraints, conflict detection for contradictory expressions.
- **D10 - Unsupported + unclear dual exits**: Unsupported (not in candidate list) → rejection with reason. Unclear (low confidence/missing slots) → clarification question.
- **D11 - Next-turn implicit evaluation**: Detect failure signals ("你理解错了") and convergence tracking for clarification loop.
- **D12 - Intent boundary + few-shot**: Positive/negative examples per intent, few-shot injection for confusing boundaries.
- **D13 - Hierarchical intents**: Parent/child intent support for large intent spaces (`parent_intent` field).
- **D14 - Intent affects 3 dimensions**: Prompt loading, Context organization, Tool selection.
- **D15 - 4-tier evaluation metrics**: Top-K accuracy, per-intent breakdown, rejection/clarification rates, slot filling metrics.
- **D16 - 9-scenario test set**: Clear/simple, complex expr, multi-intent, cross-turn, intent switch, slot filling, rejection, clarification, hierarchical.
- **D17 - Retrieval-based candidate narrowing**: Optional vector/LLM-coarse retrieval before L2 to narrow candidates to Top-N (dynamic N based on context). Default OFF.
- **D18 - Dynamic few-shot injection**: Few-shot split into STATIC (always injected) + DYNAMIC (retrieved per input). Extends D12. Default OFF.
- **D19 - Intent orthogonality governance**: detect_overlap, split_intent, merge_with_param APIs on IntentRegistry. Default OFF.
- **D20 - True multi-intent detection**: L3 filters process descriptions, outputs relations + pending_intents, default sequential execution. Default ON (filter+sequential), configurable.
- **D21 - Vector matching fallback**: Code-layer vector matcher with pre-built mappings and offline accumulation. Interview highlight. Default OFF.
- **D22 - Intent reuse with rollback**: Reuse previous turn's intent, rollback on failure signal. Interview highlight. Default OFF.
- **D23 - Multi-recognizer arbitration**: Vote/weighted-score across multiple recognizers. Default OFF.
- **D24 - Fine-tuning integration point**: model_tier="fine_tuned", evaluate_fine_tuned, JSONL export. Design only, no training. Default OFF.
- **D25 - Five problems to solutions mapping**: Summary table mapping intent accuracy problems to D17-D24 solutions.
- **D26 - Evidence grading**: Every slot value tagged `verified` (current input/context/confirmed KeyFact) or `provisional` (history/profile/speculation/default). Hard operations require verified evidence or must disclose assumptions. Default ON (additive).
- **D27 - sub_tasks vs independent_intents boundary**: "意图的职责是选择业务流程" — only goals that independently trigger different business flows are `independent_intents`; intra-flow steps (比价、查配送) are `sub_tasks`. `sub_tasks` do NOT enter D20 multi-intent governance. Default ON (additive).
- **D28 - Five-factor routing arbitration**: Confidence is a signal not probability. Five factors in priority order: (1) rule validation, (2) slot completeness, (3) hard constraint risk, (4) candidate gap, (5) confidence + historical errors. Also includes L2/L3 disagreement arbitration. Default ON (additive).
- **D29 - Extended evaluation metrics**: 9 metrics (confusion matrix, hard-sample accuracy, rejection accuracy with false/missed reject, clarification convergence, slot recall/completeness, constraint identification, state update accuracy) + online feedback → offline test set loop via `import_online_samples` / `collect_online_failures`. Default ON (additive).
- **D30 - v0/v1/v2 solution evolution methodology**: v0 (single deep LLM) → v1 (lightweight LLM + code layer + vector + advanced mechanisms, 3 sub-stages) → v2 (D26-D31 interview insights). `assess_evolution_stage()` returns current stage. Default ON (methodology only).
- **D31 - Structured output protocol**: 10-field protocol (`normalized_query`, `intent`, `sub_tasks`, `independent_intents`, `slots`, `missing_slots`, `hard_constraints`, `soft_constraints`, `verified_evidence`, `provisional_evidence`) as handoff contract. `sub_intents` kept as deprecated alias of `independent_intents`. Default ON (additive).

## Code Conventions

- **Comments and docstrings**: English only (user requirement)
- **String literals**: Chinese is OK for enum values (`"指代问题"`), Field descriptions, prompt content, test data
- **Type hints**: Use Python 3.10+ syntax (`str | None`, `list[dict]`, not `Optional[str]`, `List[Dict]`)
- **Models**: Pydantic `BaseModel` for all data structures
- **Storage**: Abstract interfaces in `storage/base.py`, in-memory implementations in `storage/memory.py`. Production can implement Redis/MySQL/vector DB versions.
- **LLM**: All LLM calls go through `LLMClient` interface. Use `MockLLMClient` in tests, `OpenAICompatibleClient` in production. The client supports a `model_tier` parameter (`"flash"` for Layer 2, `"pro"` for Layer 3).
- **API versioning**: Single source of truth is `APP_VERSION` in `server.py`. Update both the FastAPI `app` init and the `/health` endpoint by changing this constant.

## Testing

- 650 tests across 23 test files, all passing in ~0.7s
  - Normalization: `test_pipeline`, `test_pre_normalization`, `test_deep_normalization`, `test_classification`, `test_clarification`, `test_quantification`, `test_context`, `test_vocabulary`, `test_attribute_resolution`
  - Intent Recognition: `test_intent_models`, `test_code_layer`, `test_lightweight_llm`, `test_deep_llm`, `test_slot_filling`, `test_rejection_clarification`, `test_intent_pipeline`, `test_evaluation`
  - Server (main endpoint + sub-endpoints): `test_server` (26 tests covering `POST /agent/intent` orchestration, early-exit on clarification, `skip_normalization`, UI event passthrough, rejection, sub-endpoints, D17/D21/D22 per-request overrides, config leak regression, D22 reuse end-to-end, and D31 protocol fields)
- Tests use `MockLLMClient` with registered handlers - no real API calls
- `tests/test_server.py` patches `dotenv.load_dotenv` before importing the server module (prevents `.env` API_KEY from leaking into tests) and uses an autouse fixture to swap `get_pipeline` / `get_intent_pipeline` with MockLLMClient-backed instances
- Test data contains Chinese strings (e.g., `"第二个适合生产吗？"`) - these are test inputs, keep as-is
- When adding new features, add corresponding tests in `tests/test_<module>.py`

## Configuration

### Normalization config (`user_input_normalization/config.py`)

| Config | Default | Purpose |
|--------|---------|---------|
| `clarify.theta_clarify` | 0.6 | Clarification trigger threshold |
| `clarify.max_consecutive_clarifications` | 3 | Max clarifications before degrade |
| `fewshot.top_k` | 3 | Few-shot examples to inject |
| `vocab_promotion.min_total_count` | 100 | Min occurrences for vocab promotion |
| `vocab_promotion.min_discussant_count` | 3 | Min discussants for public vocab |
| `pipeline.enable_deep_normalization` | True | Enable/disable deep stage |

### Intent Recognition config (`intent_recognition/config.py`)

| Config | Default | Purpose |
|--------|---------|---------|
| `confidence.accept_threshold` | 0.85 | Direct accept threshold (Layer 2) |
| `confidence.clarify_threshold` | 0.6 | Clarify threshold (below = escalate to L3) |
| `confidence.weight_llm_confidence` | 0.5 | LLM confidence weight in multi-signal fusion |
| `confidence.weight_rule_match` | 0.2 | Rule match weight |
| `confidence.weight_vector_similarity` | 0.2 | Vector similarity weight |
| `confidence.weight_historical_accuracy` | 0.1 | Historical accuracy weight |
| `clarification.max_consecutive_clarifications` | 3 | Max clarifications before degrade to best guess |
| `evaluation.top_k` | 3 | K value for Top-K accuracy |
| `evaluation.domain_specific_benchmark` | 0.99 | Domain-specific benchmark target |
| `evaluation.general_agent_benchmark` | 0.85 | General agent benchmark target |
| `enable_code_layer` / `enable_lightweight_llm` / `enable_deep_llm` | True | Layer enable flags |
| `fewshot_top_k` | 3 | Few-shot examples per intent |
| `enable_hierarchical` | False | Enable hierarchical intent dispatch (D13) |
| `enable_implicit_eval` | True | Enable next-turn failure signal detection (D11) |
| `failure_signals` | `["你理解错了", "不是这个意思", ...]` | Implicit failure signal phrases |

### D17-D24 extension configs (all default OFF)

| Config | Default | Purpose |
|--------|---------|---------|
| `retrieval.enable` | False | D17: Enable retrieval-based candidate narrowing |
| `retrieval.method` | "vector" | D17: "vector" / "llm_coarse" / "hybrid" |
| `retrieval.top_n` | 10 | D17: Number of candidates after narrowing |
| `dynamic_fewshot.dynamic_enabled` | False | D18: Enable dynamic few-shot injection |
| `orthogonality.enable_check` | False | D19: Enable overlap detection on register |
| `multi_intent.enable` | True | D20: Enable multi-intent detection in L3 |
| `multi_intent.sequential_execution` | True | D20: Execute one intent at a time |
| `vector_fallback.enable` | False | D21: Enable vector matching fallback in code layer |
| `reuse_strategy.enable` | False | D22: Enable intent reuse with rollback |
| `arbiter.enable` | False | D23: Enable multi-recognizer arbitration |
| `fine_tuning.enable` | False | D24: Enable fine-tuning integration point |
| `accuracy_preset` | "balanced" | One-click preset: "balanced" / "high_accuracy" / "low_cost" |

### D26-D31 extension configs (all default ON — additive)

| Config | Default | Purpose |
|--------|---------|---------|
| `evidence.enable_grading` | True | D26: Enable evidence grading (verified/provisional) |
| `evidence.require_verified_for_hard_ops` | True | D26: Hard operations require verified evidence |
| `evidence.high_risk_intents` | `[]` | D26: Intents requiring verified evidence (e.g. `["refund", "payment"]`) |
| `boundary.enable_sub_tasks` | True | D27: Enable sub_tasks vs independent_intents boundary |
| `boundary.strict_mode` | False | D27: Reject ambiguous classifications instead of best-guess |
| `arbitration.enable_five_factor` | True | D28: Enable five-factor routing arbitration in ambiguous zone |
| `arbitration.candidate_gap_threshold` | 0.1 | D28: Top1-Top2 gap below this escalates to L3 |
| `arbitration.risk_aware_clarify` | True | D28: High-risk intent + provisional slot triggers Clarify |
| `arbitration.high_risk_intents` | `[]` | D28: Intents subject to risk-aware Clarify |
| `protocol.enable_structured_output` | True | D31: Enable 10-field structured output protocol |
| `protocol.deprecate_sub_intents` | False | D31: When True, `sub_intents` field is omitted from responses |

## Environment

`.env` file (gitignored, not in repo):

```env
API_KEY=sk-...
FLASH_LLM_MODEL=<flash-model>   # Layer 2: lightweight intent recognition + normalization
PRO_LLM_MODEL=<pro-model>        # Layer 3: deep reasoning intent recognition
EMBEDDING_MODEL=<embedding-model>
RERANKER_MODEL=<reranker-model>  # Optional: for retrieval-based candidate narrowing
BASE_URL=https://api.openai.com/v1  # Any OpenAI-compatible provider

# Optional: LLM client timeout and retry (for reasoning models with long think time)
# LLM_TIMEOUT=120      # Request timeout in seconds (default: 120)
# LLM_MAX_RETRIES=1    # Max retries on failure (default: 1, set 0 to disable)
```

Without `API_KEY`, server falls back to `MockLLMClient` mode (both modules).

## API Endpoints

| # | Method | Path | Role | Purpose |
|---|--------|------|------|---------|
| 1 | `GET` | `/` | info | API info + version + endpoint list (highlights `main_endpoint`) |
| 2 | `GET` | `/health` | info | Health check + LLM backend label |
| **3** | **`POST`** | **`/agent/intent`** | **main** | **Main entry: raw input → normalization (internal) → intent recognition → unified `AgentIntentResponse`** |
| 4 | `POST` | `/normalize` | sub | Two-stage normalization only (debugging / single-stage) |
| 5 | `POST` | `/recognize` | sub | Three-layer intent recognition only (expects pre-normalized text) |
| 6 | `GET` | `/intents` | sub | List registered intent definitions |
| 7 | `GET` | `/docs` | info | Swagger UI (auto-generated) |
| 8 | `GET` | `/redoc` | info | ReDoc (auto-generated) |

> Production callers should prefer `POST /agent/intent`. Sub-endpoints exist for debugging, partial use, or when the caller already has normalized text.

## Common Tasks

### Call the main endpoint (most common use case)

```bash
curl -X POST http://localhost:8000/agent/intent \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "帮我推荐一款3000元以内的手机",
    "session_id": "s1",
    "turn": 1
  }'
```

The response has the intent recognition result at the top level, the normalization detail nested under `normalization`, and pipeline meta (`pipeline_path`, `raw_input`, `normalized_input`). Use `skip_normalization=true` when the input is already normalized. Pass `event` for code-layer page guidance, or `observation` to trigger deep normalization.

### Add a new quantification rule
1. Add rule to `quantification/rules.py` `DEFAULT_RULES` dict
2. Add test in `tests/test_quantification.py`

### Add a new classification rule
1. Add regex pattern to `classification/rules.py` (e.g., `SUBJECTIVE_RULES`)
2. Add test in `tests/test_classification.py`

### Register a new intent
1. Call `registry.register(IntentDefinition(...))` in `_build_default_registry()` in `server.py`
2. Provide `name`, `description`, `positive_examples`, `negative_examples`, `slots` (with `required` flag)
3. Optionally set `parent_intent` for hierarchical intents (D13)
4. Test via `POST /agent/intent` (preferred, full pipeline) or `POST /recognize` (intent recognition only)

### Add a new failure signal (D11)
1. Append phrase to `IntentRecognitionConfig.evaluation.failure_signals` list, or pass via constructor
2. The `RejectionClarificationHandler` will detect the phrase in next-turn input

### Add a new API endpoint
1. Add request/response Pydantic models in `server.py`
2. Add endpoint function with `@app` decorator
3. Update `docs/API.md` and `docs/API.zh-CN.md`
4. Update the `endpoints` dict in the `root()` function
5. If the new endpoint should be the main entry, update `main_endpoint` in `root()` and reorder the `endpoints` dict so it appears first

### Extend the main endpoint orchestration
1. The main endpoint lives in `agent_intent()` in `server.py`. Modify the orchestration flow there (e.g., add a new stage between normalization and intent recognition).
2. Update `AgentIntentRequest` / `AgentIntentResponse` Pydantic models in the same file.
3. Update `pipeline_path` possible values in both `docs/API.md` and `docs/API.zh-CN.md`.
4. Add tests in `tests/test_server.py` (follow the autouse fixture pattern that swaps `get_pipeline` / `get_intent_pipeline`).

### Implement a real storage backend
1. Implement the abstract interface from `storage/base.py` (e.g., `RedisKeyFactStore`, `RedisSlotStateStore`)
2. Replace `Memory*Store` in `server.py` `get_pipeline()` / `get_intent_pipeline()` with the real implementation

### Bump API version
1. Update `APP_VERSION` constant in `server.py` (single source of truth)
2. Both `/` and `/health` endpoints will reflect the new version automatically

### Enable D17-D24 accuracy improvements
1. Set the desired config flags in `IntentRecognitionConfig` (e.g., `config.retrieval.enable = True`)
2. Or use per-request overrides on `POST /agent/intent`: `enable_retrieval`, `enable_vector_fallback`, `reuse_previous_intent`
3. Or use `accuracy_preset = "high_accuracy"` for one-click preset
4. All flags default OFF; enabling doesn't break existing behavior

### Assess system evolution stage (D30)
1. Call `TestRunner.assess_evolution_stage(config)` to inspect the current system stage
2. Returns one of `"v0"`, `"v1-stage1"`, `"v1-stage2"`, `"v1-stage3"`, `"v2"`
3. Stage progression: `v0` (only deep LLM, no normalization) → `v1-stage1` (L2 on, L1 off) → `v1-stage2` (three-layer on, D17-D24 off) → `v1-stage3` (D17-D24 on, D26-D31 off) → `v2` (D26-D31 on)
4. D26-D31 features default ON, so the default stage is `"v2"`. To test earlier stages, explicitly disable the four D26-D31 flags (`evidence.enable_grading`, `arbitration.enable_five_factor`, `boundary.enable_sub_tasks`, `protocol.enable_structured_output`)
5. Use this in interviews or status reports to communicate system maturity, or to gate feature expectations per stage

## Pitfalls

- **`dict.get("key", "")` returns `None` when value is `None`**: Use `dict.get("key") or ""` instead. This caused a production bug when LLM returned `null` for `resolved_to`.
- **LLM may return JSON in ```json code blocks**: Use `_extract_json()` helper (in `pre_normalization/normalizer.py`, `quantification/engine.py`, and `deep_llm/classifier.py`) to parse.
- **Deep normalization is slow**: Involves 2 LLM calls (pre + quantification). Real API calls can take 60+ seconds. Set appropriate timeouts.
- **Intent recognition Layer 3 is slow**: Real deep LLM calls can take 10+ seconds. The pipeline falls back to Layer 2 result if L3 fails (graceful degradation).
- **Cross-turn slot conflicts**: `CrossTurnSlotMerger` uses latest-wins for slot values but accumulates constraints. Use `has_constraint_conflict()` to detect contradictory hard constraints before merging.
- **`_llm_backend` global**: The `/health` endpoint uses `_get_llm_backend_label()` which reads env vars lazily, so it returns an accurate backend even before any pipeline is built.
- **`python3` vs miniconda python**: Default `python3` lacks dependencies. Always use `/Users/souljoy/miniconda3/bin/python`.
- **Tests that import `server.py` will trigger real LLM calls unless `.env` is masked**: `server.py` calls `dotenv.load_dotenv()` at import time, which loads the real `API_KEY`. `tests/test_server.py` patches `dotenv.load_dotenv = lambda *a, **k: False` *before* importing the server module, and uses an autouse fixture to swap `get_pipeline` / `get_intent_pipeline` with `MockLLMClient`-backed instances. Any new test file that imports `server.py` must follow the same pattern, or tests will hang for 200+ seconds hitting the real LLM API.
- **Main endpoint early-exit on clarification**: When normalization pauses for clarification (low-confidence pronoun, etc.), `POST /agent/intent` returns `intent=null` with `paused_at_normalization=true` and `pipeline_path=["normalization"]`. The client must resolve the clarification and retry — the main endpoint does not auto-recover. In tests, set `config.pipeline.agent_type="domain"` to skip clarification pauses (domain agents have clearer intents).
- **D17-D24 all default OFF**: Enabling multiple features simultaneously may have performance impact. Use `accuracy_preset` for balanced presets.
- **D22 reuse can mask intent switches**: If reuse is ON, intent switch detection relies on `INTENT_SWITCH_MARKERS` in `intent_reuse_strategy.py`. Add domain-specific markers as needed.
- **Per-request override config leak (fixed)**: `_apply_extension_overrides()` mutates the singleton pipeline's config in-place. Without `_restore_extension_config()` in a `finally` block, overrides would leak across requests (e.g., a request with `reuse_previous_intent=true` would leave D22 enabled for all subsequent requests). Always pair apply + restore. `_init_extensions()` also resets all extension modules to `None` first, so toggling a flag from enabled back to disabled actually releases the previously created module.
- **`source` field when paused at normalization**: When the main endpoint pauses at normalization (clarification needed), `source` is `""` (empty string), NOT `"code-layer"`. The `source` field only reflects an intent-recognition layer when recognition actually ran.
- **Evidence grading cannot be downgraded (D26)**: Once a slot value is tagged `verified`, it stays `verified` even if a later turn provides weaker evidence. `_upgrade_provisional_to_verified()` is one-way. This prevents provisional noise from overwriting verified facts.
- **`sub_tasks` must NOT enter D20 multi-intent governance (D27)**: `sub_tasks` are intra-flow steps (比价、查配送); they are recorded for execution but never enter `relations`, `pending_intents`, or process-description filtering. Only `independent_intents` (goals that trigger different business flows) enter D20.
- **Five-factor arbitration priority order matters (D28)**: Factors are evaluated in strict priority order (rule → slot → risk → gap → history). The first failing factor that mandates Clarify/Escalate wins. Do not reorder — slot completeness must be checked before risk, otherwise a high-risk intent with missing slots would produce a confusing risk-related reason instead of the actionable "missing required slots" reason.
- **`arbitration_breakdown` is separate from `signals` (D28)**: `signals` is typed `dict[str, float]` and cannot hold nested factor results. The five-factor breakdown lives in `arbitration_breakdown: dict[str, Any]`.
- **D26-D31 all default ON (additive)**: Unlike D17-D24 (which default OFF for cost/performance reasons), D26-D31 are additive — they enrich output without changing existing behavior. Disabling them reverts to pre-D26 behavior but does not break anything.
- **`sub_intents` is a deprecated alias (D31)**: `independent_intents` is the primary field; `sub_intents` mirrors it via `model_validator(mode="after")`. Old clients reading `sub_intents` keep working. Set `protocol.deprecate_sub_intents=True` to omit `sub_intents` from responses entirely.

## Documentation

- [README.md](README.md) / [README.zh-CN.md](README.zh-CN.md) - Setup, Docker, API endpoints
- [docs/API.md](docs/API.md) / [docs/API.zh-CN.md](docs/API.zh-CN.md) - HTTP API call documentation
- [docs/TECHNICAL_PRINCIPLES.md](docs/TECHNICAL_PRINCIPLES.md) / [docs/TECHNICAL_PRINCIPLES.zh-CN.md](docs/TECHNICAL_PRINCIPLES.zh-CN.md) - Technical principles with implementation references
