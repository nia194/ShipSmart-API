# ShipSmart — FastAPI AI Service (`api-python`)

AI / orchestration service for the ShipSmart shipping platform. Owns no
transactional data; provides RAG-grounded shipping advice, tracking
guidance, recommendation scoring, and tool orchestration on top of a
multi-provider LLM router.

**Stack:** FastAPI 0.135.3 · Python 3.13 · uv · pgvector · slowapi · OpenAI / Anthropic / Gemini / Ollama / Echo

---

## What this service does

| Capability | Endpoint | Notes |
|---|---|---|
| RAG query | `POST /api/v1/rag/query` | Embed → similarity search → LLM synthesis. |
| RAG ingest | `POST /api/v1/rag/ingest` | Loads `data/documents/*` into the vector store. Auto-runs on first boot when pgvector is empty. |
| Shipping advisor | `POST /api/v1/advisor/shipping` | RAG + tool calls (`validate_address`, `get_quote_preview`) + LLM reasoning. |
| Tracking advisor | `POST /api/v1/advisor/tracking` | RAG + optional address validation + LLM guidance. Extracts next-step list. |
| Recommendation | `POST /api/v1/advisor/recommendation` | Deterministic scoring (cheapest/fastest/best_value/balanced). Hydrates from Java if `services` empty + `context.shipment_request_id` set. |
| Compare | `POST /api/v1/compare` | Decision-cockpit: compares 2–3 shipping options across scenarios (on-time, damage, price, speed) using LLM reasoning. |
| Tool orchestration | `POST /api/v1/orchestration/run` | Executes a registered tool. Auto-selects via regex first, then LLM-assisted fallback. |
| Tool catalog | `GET /api/v1/orchestration/tools` | JSON Schemas for all registered tools. |
| Service info | `GET /api/v1/info` | Returns service metadata (version, env, active providers). No secrets exposed. |
| Health | `GET /health` | Liveness probe. |

Interactive docs (dev only): `http://localhost:8000/docs`.

---

## Architecture inside this service

```
                          ┌─────────────────────────────────┐
   request ──► route ──►  │ service layer                   │
                          │  • shipping_advisor_service     │
                          │  • tracking_advisor_service     │
                          │  • recommendation_service       │
                          │  • compare_service              │
                          │  • orchestration_service        │
                          │  • rag_service                  │
                          │  • java_client (→ Java API)     │
                          └──┬─────────────┬─────────────┬──┘
                             │             │             │
                       ┌─────▼────┐ ┌──────▼─────┐ ┌─────▼─────┐
                       │ RAG      │ │ Tools      │ │ LLM       │
                       │ • embed  │ │ • registry │ │ • router  │
                       │ • store  │ │ • validate │ │ • prompts │
                       │   (pgvec │ │ • quote    │ │ • openai  │
                       │   /mcp   │ │            │ │ • claude  │
                       │   /mem)  │ │            │ │ • gemini  │
                       │ • chunk  │ │            │ │ • llama   │
                       │ • retrv  │ │            │ │ • echo    │
                       └──────────┘ └────────────┘ └───────────┘
```

### Key modules

| Path | Purpose |
|---|---|
| `app/main.py` | Lifespan: builds embedding provider, vector store (memory, pgvector, or mcp), LLM router, shipping provider, tool registry. Auto-ingests on first boot. |
| `app/mcp_server.py` | Standalone MCP HTTP server exposing tools (`validate_address`, `get_quote_preview`) for Claude Code and other MCP clients. |
| `app/core/config.py` | All settings (env-driven via pydantic-settings). |
| `app/core/cache.py` | TTL cache used by RAG, recommendation, and LLM tool selection. |
| `app/core/errors.py` | Centralized error handling: `AppError` exception class + global exception handlers returning consistent JSON error responses. |
| `app/core/logging.py` | Structured logging setup (`setup_logging()`) and named logger factory (`get_logger()`). |
| `app/core/middleware.py` | `RequestLoggingMiddleware` — logs method, path, status, duration; attaches `X-Request-Id` header. |
| `app/core/rate_limit.py` | Shared `slowapi` limiter (per IP). |
| `app/schemas/` | Pydantic request/response models (`advisor.py`, `compare.py`). |
| `app/llm/router.py` | Task-based router: each task → its own provider with fallback chain. |
| `app/llm/client.py` | `OpenAIClient`, `AnthropicClient`, `GeminiClient`, `LlamaClient`, `EchoClient`. |
| `app/llm/prompts.py` | Prompt templates for RAG queries and advisor flows (system instructions, context formatting). |
| `app/rag/embeddings.py` | `OpenAIEmbedding` + `LocalHashEmbedding` placeholder. |
| `app/rag/vector_store.py` | `VectorStore` ABC + `InMemoryVectorStore`. |
| `app/rag/pgvector_store.py` | Postgres + pgvector implementation (asyncpg, cosine via `<=>`). |
| `app/rag/mcp_vector_store.py` | MCP-based pgvector store via Supabase MCP server (alternative to direct asyncpg). |
| `app/rag/chunking.py` | Document chunking: splits text into overlapping chunks for embedding. |
| `app/rag/ingestion.py` · `retrieval.py` | Ingestion + retrieval pipeline. |
| `app/services/compare_service.py` | LLM-driven multi-scenario shipping comparison logic. |
| `app/services/orchestration_service.py` | Rule-based + LLM-assisted tool selection. |
| `app/services/java_client.py` | Thin async wrapper around the shared `httpx` client → calls Java for `quotes` / `saved-options`. |
| `app/providers/__init__.py` | Shipping provider factory. Loud WARN on mock; raises ValueError on missing carrier creds. |
| `app/providers/base.py` | Base provider interface (ABC for all external service providers). |
| `app/providers/shipping_provider.py` | Shipping provider abstraction (carrier-facing operations interface). |
| `app/providers/{mock,ups,fedex,dhl,usps}_provider.py` | Real carrier providers are currently **stubs**; only `mock` returns data. |
| `app/dependencies/__init__.py` | FastAPI dependency injection providers (`Depends()` helpers). |
| `app/tools/` | Tool ABC, registry, address + quote tools. |
| `scripts/perf_check.py` | Post-launch performance check: measures response times for key endpoints against thresholds. |

---

## Running locally

### Prerequisites

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) 0.6.5+

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env
# edit .env — see "Environment variables" below
```

### Run

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The boot logs are intentionally loud about degraded modes:

```
WARNING  EMBEDDING_PROVIDER unset — using LocalHashEmbedding…
INFO     Vector store backend: memory (InMemoryVectorStore)
WARNING  Task 'reasoning' provider='<unset>' unavailable — falling back to echo
WARNING  SHIPPING_PROVIDER=mock — All quote previews… return FAKE data.
INFO     Tool registry initialized: 2 tools, provider=mock
```

If you see all four warnings, the server still works end-to-end but is
running on placeholder data + placeholder LLM. Set the env vars below to
unlock real behavior.

---

## Environment variables

All flags live in `.env.example` with comments. Highlights:

### LLM routing

```env
LLM_PROVIDER=                  # legacy single-provider
LLM_PROVIDER_REASONING=        # advisors (shipping, tracking, orchestration tool selection)
LLM_PROVIDER_SYNTHESIS=        # /rag/query, recommendation summary
LLM_PROVIDER_FALLBACK=echo     # safety net
LLM_TIMEOUT=30
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.3
```

Each task picks its own provider. Empty inherits `LLM_PROVIDER`. Unknown
or missing-key providers fall through to `LLM_PROVIDER_FALLBACK`, then
to `EchoClient` (placeholder responses).

### Provider keys

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-5

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

LLAMA_BASE_URL=http://localhost:11434
LLAMA_MODEL=llama3.2
```

### Embeddings

```env
EMBEDDING_PROVIDER=            # "openai" or empty (= LocalHashEmbedding placeholder)
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

### Vector store

```env
VECTOR_STORE_TYPE=memory       # "memory", "pgvector", or "mcp"
DATABASE_URL=                  # required when VECTOR_STORE_TYPE=pgvector
PGVECTOR_TABLE=rag_chunks
```

**pgvector** — direct asyncpg connection to Postgres + pgvector:

1. Create a `rag_chunks` table with a `vector(1536)` embedding column (matching `text-embedding-3-small`). If you use a different embedding dimension, alter the column accordingly.
2. Set `VECTOR_STORE_TYPE=pgvector` and `DATABASE_URL=postgresql://…`.
3. Restart. The first boot auto-ingests `data/documents/*` if the table is empty.

**mcp** — connects to Supabase pgvector through an MCP HTTP endpoint instead of direct asyncpg:

```env
MCP_SERVER_URL=               # MCP server HTTP endpoint (required for "mcp" backend)
MCP_API_KEY=                  # Optional API key for MCP server auth
```

### RAG settings

```env
RAG_AUTO_INGEST=true           # auto-ingest data/documents/* on startup if store is empty
RAG_DOCUMENTS_PATH=data/documents
RAG_TOP_K=3                    # number of chunks returned per similarity search
RAG_CHUNK_SIZE=500             # characters per chunk
RAG_CHUNK_OVERLAP=50           # overlap between consecutive chunks
```

### Shipping provider

```env
SHIPPING_PROVIDER=mock         # mock | ups | fedex | dhl | usps

# UPS
UPS_CLIENT_ID=
UPS_CLIENT_SECRET=
UPS_ACCOUNT_NUMBER=
UPS_BASE_URL=https://onlinetools.ups.com

# FedEx
FEDEX_CLIENT_ID=
FEDEX_CLIENT_SECRET=
FEDEX_ACCOUNT_NUMBER=
FEDEX_BASE_URL=https://apis.fedex.com       # sandbox: https://apis-sandbox.fedex.com

# DHL
DHL_API_KEY=
DHL_API_SECRET=
DHL_ACCOUNT_NUMBER=
DHL_BASE_URL=https://express.api.dhl.com

# USPS
USPS_CLIENT_ID=
USPS_CLIENT_SECRET=
USPS_BASE_URL=https://api.usps.com
```

`mock` is the default and returns deterministic fake quotes/addresses
with a loud WARNING at boot. Selecting a real carrier (`ups`/`fedex`/
`dhl`/`usps`) **requires all of that carrier's env vars**; the factory
raises `ValueError` at startup listing missing keys instead of silently
falling back. The actual carrier API integrations are currently stubs.

### Tools

```env
ENABLE_TOOLS=true              # enable/disable tool registry
```

### Rate limiting

```env
RATE_LIMIT_ADVISOR=10/minute       # /advisor/* endpoints
RATE_LIMIT_ORCHESTRATION=20/minute # /orchestration/run
RATE_LIMIT_COMPARE=10/minute       # /compare endpoint
```

Per IP, via slowapi. Returns HTTP 429 when exceeded.

---

## Tool orchestration: how selection works

`POST /api/v1/orchestration/run` accepts `{ query, tool?, params }`.

1. **Explicit**: if `tool` is set, that tool runs directly.
2. **Auto / fast path**: deterministic regex rules in
   `orchestration_service._TOOL_PATTERNS`.
3. **Auto / slow path**: if regex misses *and* a reasoning LLM is
   configured, the orchestrator asks the LLM to pick exactly one tool
   from the registry (or `NONE`). Result is cached per query for 10
   minutes.

The `metadata.selection_method` field in the response tells you which
path fired (`rule` / `llm` / `none`).

---

## Recommendations + Java hydration

`POST /api/v1/advisor/recommendation` accepts a list of `services` and
`context`. If `services` is empty but `context.shipment_request_id` is
set, the route forwards the incoming `Authorization` header to the Java
API and pulls the actual quotes from
`GET /api/v1/quotes?shipmentRequestId=…` before scoring. This lets the
frontend ask for "ranked recommendations for shipment X" without
re-sending the quote list.

If Java is unreachable, the call degrades gracefully — empty
recommendations rather than a 500.

---

## MCP Server

A standalone MCP HTTP server (`app/mcp_server.py`) exposes the tool
registry over HTTP so that Claude Code, Spring Boot, and other MCP
clients can discover and execute tools without going through the main
FastAPI routes.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe (includes tool count). |
| `POST` | `/tools/list` | Returns MCP-compatible JSON Schemas for all registered tools. |
| `POST` | `/tools/call` | Executes a tool by name with provided arguments. |
| `GET` | `/` | Service discovery (name, version, tool count, endpoint list). |

### Run locally

```bash
uv run uvicorn app.mcp_server:app --reload --host 0.0.0.0 --port 8001
```

### Quick test

```bash
# list available tools
curl -X POST http://localhost:8001/tools/list

# call a tool
curl -X POST http://localhost:8001/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"name":"validate_address","arguments":{"street":"123 Main St","city":"NYC","state":"NY","zip_code":"10001"}}'
```

---

## Deployment (Render)

The repo ships a `render.yaml` Render Blueprint that deploys **two
services** from this codebase:

| Service | Entry point | Purpose |
|---|---|---|
| `shipsmart-api-python` | `app.main:app` | Main FastAPI AI/advisory service. |
| `shipsmart-mcp-tools` | `app.mcp_server:app` | MCP tools server for external tool consumers. |

Both services use `pip install uv && uv sync` as the build command.

To deploy: connect the repo to Render and apply the Blueprint. Set all
`sync: false` env vars (secrets like `DATABASE_URL`, `OPENAI_API_KEY`)
in the Render dashboard before the first deploy.

---

## Smoke tests

After boot, with no extra config:

```bash
# liveness
curl http://localhost:8000/health

# tool catalog
curl http://localhost:8000/api/v1/orchestration/tools

# explicit tool execution (mock provider)
curl -X POST http://localhost:8000/api/v1/orchestration/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"validate","tool":"validate_address","params":{"street":"1 Infinite Loop","city":"Cupertino","state":"CA","zip_code":"95014"}}'

# recommendation (deterministic scoring)
curl -X POST http://localhost:8000/api/v1/advisor/recommendation \
  -H 'Content-Type: application/json' \
  -d '{"services":[{"service":"Ground","price_usd":12.5,"estimated_days":5},{"service":"Express","price_usd":29,"estimated_days":1}],"context":{"urgent":true}}'
```

---

## Tests

```bash
uv run pytest
```

Tests live under `tests/` and use `pytest-asyncio` (async mode = auto).

---

## Operational notes

- **Rate limit 429**: someone is hammering an `/advisor` endpoint. Tune `RATE_LIMIT_ADVISOR` if legitimate.
- **Echo responses**: no LLM provider keys are set. Set `OPENAI_API_KEY` + `LLM_PROVIDER_REASONING=openai` (or equivalent) to enable real completions.
- **`is_valid: true` for any address**: you're on the mock provider. Set `SHIPPING_PROVIDER` + carrier credentials.
- **RAG returns nothing relevant**: you're on `LocalHashEmbedding`. Set `EMBEDDING_PROVIDER=openai`.
- **RAG cleared on restart**: you're on `VECTOR_STORE_TYPE=memory`. Switch to `pgvector` + `DATABASE_URL`.
