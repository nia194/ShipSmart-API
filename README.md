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
| `app/main.py` | Lifespan: builds embedding provider, vector store (memory, pgvector, or mcp), LLM router, and the remote `RemoteToolRegistry` backed by the ShipSmart-MCP service. Auto-ingests on first boot. |
| `app/services/mcp_client.py` | Thin HTTP client for the standalone ShipSmart-MCP server, plus `RemoteTool` / `RemoteToolRegistry` shims that ducktype the old in-process tool interface. |
| `app/core/config.py` | All settings (env-driven via pydantic-settings). |
| `app/core/cache.py` | TTL cache used by RAG, recommendation, and LLM tool selection. |
| `app/core/errors.py` | Centralized error handling: `AppError` exception class + global exception handlers returning consistent JSON error responses. |
| `app/core/logging.py` | Structured logging setup (`setup_logging()`) and named logger factory (`get_logger()`). |
| `app/core/middleware.py` | `RequestLoggingMiddleware` — logs method, path, status, duration; honors inbound `X-Request-Id` and W3C `traceparent` (mints them when missing), stores them in ContextVars, and echoes both back as response headers. |
| `app/core/correlation.py` | ContextVars (`request_id_var`, `traceparent_var`) + `outbound_headers()` helper. Lets outbound clients (Java API, MCP) forward the same correlation IDs on every hop. |
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
| `app/services/java_client.py` | Thin async wrapper around the shared `httpx` client → calls Java for `quotes` / `saved-options`. Forwards `X-Request-Id` / `traceparent` via `outbound_headers()` so requests stay correlated across the Java hop. |
| `app/dependencies/__init__.py` | FastAPI dependency injection providers (`Depends()` helpers). |
| `scripts/perf_check.py` | Post-launch performance check: measures response times for key endpoints against thresholds. |

> Tools and carrier providers no longer live in this repo. They are served by
> the standalone **ShipSmart-MCP** service — see the [MCP Server](#mcp-server-separate-repo)
> section below.

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
WARNING  SHIPSMART_MCP_URL is not set — advisor/orchestration routes will return 503…
INFO     Remote tool registry hydrated from MCP http://localhost:8001 (N tools)
```

If you see the first three warnings plus "no MCP URL", the server still
boots but the `/advisor/*` and `/orchestration/*` routes return 503 until
you point `SHIPSMART_MCP_URL` at a live ShipSmart-MCP instance. Set the
env vars below to unlock real behavior.

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

Carrier credentials (`SHIPPING_PROVIDER`, `UPS_*`, `FEDEX_*`, `DHL_*`,
`USPS_*`) no longer live in this service. They belong to the
**ShipSmart-MCP** repo, which owns all carrier-API calls. Configure
them there and point this service at its HTTP endpoint with
`SHIPSMART_MCP_URL` (below).

### Tools (delegated to ShipSmart-MCP)

```env
SHIPSMART_MCP_URL=http://localhost:8001   # standalone MCP tool server
SHIPSMART_MCP_API_KEY=                    # optional; must match MCP_API_KEY on the server
```

If `SHIPSMART_MCP_URL` is empty, the advisor and orchestration routes
return HTTP 503 (no tools available). See the **ShipSmart-MCP** repo for
how to run the tool server locally.

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

## MCP Server (separate repo)

The tool layer (`validate_address`, `get_quote_preview`, carrier
providers, MCP HTTP endpoints) lives in the separate **ShipSmart-MCP**
repo and is deployed as its own Render service.

This API calls that service through `app/services/mcp_client.py`:

- `McpClient` — async HTTP client for `/tools/list` and `/tools/call`.
- `RemoteTool` / `RemoteToolRegistry` — shims that implement the same
  interface the in-process tool layer used to expose, so
  `orchestration_service`, `shipping_advisor_service`, and
  `tracking_advisor_service` are unchanged.

Contract (defined by ShipSmart-MCP):

| Method | Path          | Purpose                                              |
|--------|---------------|------------------------------------------------------|
| `GET`  | `/health`     | Liveness probe.                                      |
| `POST` | `/tools/list` | MCP tool catalog as JSON Schemas.                    |
| `POST` | `/tools/call` | Execute a tool by name.                              |

If `MCP_API_KEY` is set on the MCP server, set the matching
`SHIPSMART_MCP_API_KEY` here so requests pass the `X-MCP-Api-Key` header.

---

## Deployment (Render)

The repo ships a `render.yaml` Render Blueprint for a single service:

| Service | Entry point | Purpose |
|---|---|---|
| `shipsmart-api-python` | `app.main:app` | FastAPI AI/advisory service. Tools are delegated to the `shipsmart-mcp` service deployed from the ShipSmart-MCP repo. |

Build command: `pip install uv && uv sync`.

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
- **`is_valid: true` for any address**: the MCP server is running on the mock carrier. Switch `SHIPPING_PROVIDER` on the **ShipSmart-MCP** service to a real carrier (it owns those env vars now).
- **`/advisor/*` or `/orchestration/*` return 503**: `SHIPSMART_MCP_URL` is empty or the MCP server is unreachable. Boot `ShipSmart-MCP` and re-check.
- **RAG returns nothing relevant**: you're on `LocalHashEmbedding`. Set `EMBEDDING_PROVIDER=openai`.
- **RAG cleared on restart**: you're on `VECTOR_STORE_TYPE=memory`. Switch to `pgvector` + `DATABASE_URL`.
