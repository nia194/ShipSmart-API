# ShipSmart — FastAPI AI Service (`api-python`)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-0.6%2B-DE5FE9?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![pgvector](https://img.shields.io/badge/pgvector-Postgres-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Claude](https://img.shields.io/badge/Claude-native%20tool--calling-D97757?logo=anthropic&logoColor=white)](#spotlight-the-concierge-agent)
[![Tests](https://img.shields.io/badge/tests-280%20passing-3FB950?logo=pytest&logoColor=white)](#tests)
[![Deploy: Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)](https://render.com/)
[![License](https://img.shields.io/badge/License-See%20LICENSE-blue)](./LICENSE)

> An async **AI-orchestration microservice** that turns a multi-provider LLM stack into a
> grounded shipping **concierge** — a model-driven **agent loop**, **hybrid + iterative RAG**,
> **prompt-injection guardrails**, and **per-task provider failover** — all behind a
> hermetic **280-test** suite.

AI / orchestration service for the ShipSmart shipping platform. Owns no transactional
data; provides RAG-grounded shipping advice, a tool-calling concierge agent, tracking
guidance, recommendation scoring, and tool orchestration on top of a multi-provider LLM
router. Every external dependency degrades gracefully — the service boots, answers, and
stays observable even with no API keys, no database, and no tool server.

**Stack:** FastAPI 0.135.3 · Python 3.13 (async) · uv · pgvector · slowapi · OpenAI / Anthropic (Claude) / Gemini / Ollama / Echo

---

## Table of contents

- [Engineering highlights](#engineering-highlights)
- [Spotlight: the Concierge agent](#spotlight-the-concierge-agent)
- [Retrieval modes](#retrieval-modes)
- [The ShipSmart ecosystem](#the-shipsmart-ecosystem)
- [What this service does](#what-this-service-does)
- [Architecture inside this service](#architecture-inside-this-service)
- [Running locally](#running-locally)
- [Environment variables](#environment-variables)
- [Tool orchestration: how selection works](#tool-orchestration-how-selection-works)
- [Recommendations + Java hydration](#recommendations--java-hydration)
- [MCP Server (separate repo)](#mcp-server-separate-repo)
- [Deployment (Render)](#deployment-render)
- [Smoke tests](#smoke-tests)
- [Tests](#tests)
- [Cross-service contracts](#cross-service-contracts)
- [Operational notes](#operational-notes)
- [License](#license)

---

## Engineering highlights

The parts worth a closer look — each maps to real, tested code in this repo:

| | Capability | Why it's interesting |
|---|---|---|
| 🤖 | **Model-driven Concierge agent** | A genuine reason→act→observe loop (`app/services/agent_service.py`). The model plans, calls read-only tools, and **conditionally re-retrieves** when knowledge-base coverage is weak — bounded by hard step + retrieval caps, guarded against degenerate query loops. |
| 🧰 | **Native tool-calling with graceful fallback** | Claude drives the loop with native function calling; providers without it fall through to a single-pass text tool-selection — and a keyless `scripted` stub runs the whole loop with **no API keys** for demos and CI. |
| 🔀 | **Multi-provider LLM router + failover** | Each task (`reasoning`, `synthesis`) routes to its own provider, with a request-time **failover chain** (retry → next provider → always-terminating `echo`). Per-task model/temperature/token overrides. The app never crashes on LLM config. |
| 🛡️ | **Prompt-injection guardrails + grounding** | All prompt assembly flows through one assembler (`app/llm/guardrails.py`): role separation, untrusted-data fencing, injection detection (block or neutralize), and **grounding** — answer only from retrieved data or refuse, never guess. |
| 🔎 | **Retrieval that scales by config** | Single-shot dense → **hybrid** (dense + BM25/Postgres-lexical fusion) → **iterative** (bounded plan→retrieve→assess loop). All behind flags; defaults reproduce the simple path. |
| 🔭 | **Observability built in** | W3C `traceparent` + `X-Request-Id` minted/propagated across every hop (Java, MCP), structured logging, a `decision_path` trace on every answer, and a `/ready` probe that reports the live wiring. |
| 🧪 | **Hermetic-by-construction tests** | **280 tests in ~4s**, zero network, zero real keys — an autouse fixture pins every test to a self-contained profile. Includes an agentic eval harness (`scripts/agentic_eval.py`). |
| 🧩 | **Polyglot microservice design** | One of five sibling services: this Python AI service alongside a Java/Spring transactional API, a React SPA, an MCP tool server, and a Supabase/Infra repo — communicating over typed HTTP contracts. |

---

## Spotlight: the Concierge agent

`POST /api/v1/agent/run` is the newest and most interesting surface: a **model-driven,
read-only** agent that plans and calls tools to answer free-text shipping questions, then
returns a grounded answer **plus its full reasoning trace**.

```mermaid
flowchart TD
    Q["Query + context"] --> L["Reasoning LLM<br/>plan next step"]
    L -->|"tool_use"| T{"Which tool?"}
    T -->|"retrieve_rag"| R["Single-shot RAG<br/>(dense / hybrid)"]
    T -->|"validate_address<br/>get_quote_preview"| M["MCP read-only tool"]
    R --> C["Coverage signal<br/>top_score · covered · chunks"]
    C -->|"weak → reformulate<br/>bounded by agent_max_retrievals"| L
    C -->|"strong"| L
    M -->|"observation"| L
    L -->|"final answer"| G["Guardrails<br/>fence · sanitize · ground"]
    G --> S["Synthesis via<br/>per-task failover chain"]
    S --> A["Answer + trace:<br/>steps · tools_used · sources · decisions · provider"]
```

**What makes it more than a tool-call wrapper:**

- **Conditional, bounded re-retrieval.** Each `retrieve_rag` result leads with a coverage
  signal (`top_score`, `covered`, `chunk_count`). On weak coverage the model reformulates
  with a *different, more specific* query and searches again — capped by `AGENT_MAX_RETRIEVALS`,
  guarded against repeating an identical query, and honest when a sub-area stays uncovered.
  A well-covered first hit stays single-shot.
- **The control flow is the model's; retrieval stays deterministic.** The agent owns the
  loop; the RAG layer underneath has no LLM in its control flow. (This is why the
  deterministic loop is named `iterative`, not "agentic" — see [Retrieval modes](#retrieval-modes).)
- **Reuse over reinvention.** MCP tools dispatch through the same `execute_tool` path as
  `/orchestration` (input validation, 502 handling); the final answer goes through the same
  guardrailed assembler and synthesis failover chain as the RAG path. A tool error becomes
  an *observation the model can recover from*, not a 500.
- **Degrades to keyless.** Providers without native tool-calling fall back to a single-pass
  text tool-selection; the `scripted` provider runs the full loop deterministically with no
  keys at all.
- **Read-only, day-1.** The agent plans, retrieves, and calls read-only tools. It never
  persists.

**Sample request / response:**

```bash
curl -X POST http://localhost:8000/api/v1/agent/run \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "Can I ship a power bank to Berlin, and what would it cost from 10001?",
        "context": {"origin_zip": "10001", "destination_zip": "10115", "weight_lbs": 2}
      }'
```

```jsonc
{
  "answer": "Power banks ship as lithium-battery dangerous goods … (grounded answer)",
  "tools_used": ["retrieve_rag", "get_quote_preview"],
  "sources": [{"source": "restrictions.md", "chunk_index": 3, "score": 0.81}],
  "steps": [
    {"step": 1, "tool": "retrieve_rag", "observation": "coverage: top_score=0.81 covered=true chunks=3 …"},
    {"step": 2, "tool": "get_quote_preview", "observation": "{\"service\":\"Express\", …}"}
  ],
  "decisions": ["agent:plan", "agent:step1", "agent:tool:retrieve_rag", "agent:retrieve:1",
                "agent:step2", "agent:tool:get_quote_preview"],
  "provider": "anthropic"
}
```

The `decisions[]` trace tags every branch the agent took, and `provider` reports who
actually answered — so the loop is debuggable without reading logs.

---

## Retrieval modes

Retrieval scales with config; defaults reproduce the simplest path. The naming is honest:
only the [Concierge agent](#spotlight-the-concierge-agent) is model-driven — the modes below
are deterministic.

| Mode | Flag | What it does |
|---|---|---|
| **Normal** (default) | `RAG_MODE=normal` | Single-shot dense similarity search → synthesis. |
| **Hybrid** | `RAG_HYBRID=true` | Dense (pgvector cosine) **+** sparse (BM25 / Postgres lexical) retrieval, fused by `RAG_HYBRID_ALPHA`. Catches exact tokens — carrier names, service codes — that pure embeddings miss. Degrades to dense-only when no sparse backend is available. |
| **Iterative** | `RAG_MODE=iterative` | A bounded, **deterministic** plan→retrieve→assess loop: reformulate and retry on weak coverage (≤ `RAG_ITERATIVE_MAX_STEPS`), optionally escalate to MCP tools for ground truth, then ground + answer — or refuse deterministically when nothing covers the question. No LLM in its control flow. |

All three reuse the same guardrailed assembler, context budget, and synthesis failover chain.

---

## The ShipSmart ecosystem

This service is one of five sibling repositories. Clone them as
siblings of this directory when working on the full system.

| Repo | Role | Stack |
|------|------|-------|
| [ShipSmart-Web](https://github.com/nia194/ShipSmart-Web) | React SPA — user-facing UI | React 19, Vite, TypeScript |
| [ShipSmart-Orchestrator](https://github.com/nia194/ShipSmart-Orchestrator) | Java transactional API — **single writer** to Supabase Postgres; quotes, bookings, saved options, carrier integration | Spring Boot 3.4, Java 17 |
| **[ShipSmart-API](https://github.com/nia194/ShipSmart-API)** _(this repo)_ | Python AI/orchestration service — agent, RAG, advisors, recommendations | FastAPI, Python 3.13 |
| [ShipSmart-MCP](https://github.com/nia194/ShipSmart-MCP) | MCP tool server — `validate_address`, `get_quote_preview` (provider-pluggable) | FastAPI + MCP |
| [ShipSmart-Infra](https://github.com/nia194/ShipSmart-Infra) | Supabase migrations + edge functions, deployment configs, docs | Supabase, Render blueprints |

```
            ┌──────────────────────────────┐
            │       ShipSmart-Web          │
            │       React SPA · Vite       │
            └──────────────┬───────────────┘
                           │  Authorization: Bearer <Supabase JWT>
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│  ShipSmart-Orchestrator      │◀──│  ShipSmart-API (this repo)   │
│  Java / Spring Boot          │   │  Python / FastAPI            │
│  Sole writer to Postgres     │   │  agent · RAG · advisors      │
│  Carrier integration (FedEx) │   │  Forwards JWT to Java for    │
│                              │   │  recommendation hydration    │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               │                                  │
               │                                  ▼
               │                   ┌──────────────────────────────┐
               │                   │        ShipSmart-MCP         │
               │                   │   shipping tools (HTTP/MCP)  │
               │                   │   validate_address, quotes   │
               │                   └──────────────────────────────┘
               ▼
┌──────────────────────────────┐
│   Supabase Postgres + Auth   │
└──────────────────────────────┘
```

This service owns no transactional data. It calls Java (`ShipSmart-Orchestrator`)
for quote hydration on the recommendation path, and MCP (`ShipSmart-MCP`) for
every tool execution. The same Supabase JWT the frontend sends here is
forwarded verbatim to Java so user-scoped queries continue to work without
re-issuing credentials.

---

## What this service does

| Capability | Endpoint | Notes |
|---|---|---|
| **Concierge agent** | `POST /api/v1/agent/run` | Model-driven reason→act→observe loop over MCP tools + `retrieve_rag`, with bounded conditional re-retrieval. Returns a grounded answer **+ full reasoning trace**. Read-only. See [spotlight](#spotlight-the-concierge-agent). |
| RAG query | `POST /api/v1/rag/query` | Embed → similarity search → LLM synthesis. Honors `RAG_MODE` / `RAG_HYBRID`. |
| RAG ingest | `POST /api/v1/rag/ingest` | Loads `data/documents/*` into the vector store. Auto-runs on first boot when pgvector is empty. |
| Shipping advisor | `POST /api/v1/advisor/shipping` | RAG + tool calls (`validate_address`, `get_quote_preview`) + LLM reasoning. |
| Tracking advisor | `POST /api/v1/advisor/tracking` | RAG + optional address validation + LLM guidance. Extracts next-step list. |
| Recommendation | `POST /api/v1/advisor/recommendation` | Deterministic scoring (cheapest/fastest/best_value/balanced). Hydrates from Java if `services` empty + `context.shipment_request_id` set. |
| Compare | `POST /api/v1/compare` | Decision-cockpit: compares 2–3 shipping options across scenarios (on-time, damage, price, speed) using LLM reasoning. |
| Tool orchestration | `POST /api/v1/orchestration/run` | Executes a registered tool. Auto-selects via regex first, then LLM-assisted fallback. |
| Tool catalog | `GET /api/v1/orchestration/tools` | JSON Schemas for all registered tools. |
| Service info | `GET /api/v1/info` | Returns service metadata (version, env, active providers). No secrets exposed. |
| Liveness | `GET /health` | Liveness probe. |
| Readiness | `GET /ready` | Reports resolved `rag_mode`, `rag_hybrid`, `guardrails_enabled`, `agent_enabled`, and per-task LLM failover chains — confirm the live wiring without reading logs. |

Interactive docs (dev only): `http://localhost:8000/docs`.

---

## Architecture inside this service

```
                          ┌──────────────────────────────────────┐
   request ──► route ──►  │ service layer                        │
                          │  • agent_service   (Concierge loop)  │
                          │  • shipping_advisor_service          │
                          │  • tracking_advisor_service          │
                          │  • recommendation_service            │
                          │  • compare_service                   │
                          │  • orchestration_service             │
                          │  • rag_service                       │
                          │  • java_client      (→ Java API)     │
                          └──┬──────────────┬──────────────┬─────┘
                             │              │              │
                       ┌─────▼─────┐ ┌──────▼─────┐ ┌──────▼──────┐
                       │ RAG       │ │ Tools (MCP)│ │ LLM         │
                       │ • embed   │ │ • registry │ │ • router    │
                       │ • store   │ │ • validate │ │ • failover  │
                       │   (pgvec  │ │ • quote    │ │ • guardrails│
                       │   /mcp    │ │            │ │ • budget    │
                       │   /mem)   │ │            │ │ • prompts   │
                       │ • chunk   │ │            │ │ • openai    │
                       │ • hybrid  │ │            │ │ • claude    │
                       │ • iterativ│ │            │ │ • gemini    │
                       │ • retrieve│ │            │ │ • llama·echo│
                       └───────────┘ └────────────┘ └─────────────┘
```

### Key modules

| Path | Purpose |
|---|---|
| `app/main.py` | Lifespan: builds embedding provider, vector store (memory, pgvector, or mcp), LLM router, and the remote `RemoteToolRegistry` backed by the ShipSmart-MCP service. Auto-ingests on first boot. |
| `app/services/agent_service.py` | **The Concierge agent** — model-driven reason→act→observe loop over the MCP tools + a `retrieve_rag` pseudo-tool, with bounded conditional re-retrieval on weak coverage and a keyless text-fallback for providers without native tool calling. |
| `app/api/routes/agent.py` · `app/schemas/agent.py` | `POST /api/v1/agent/run` route + request/response schemas (answer + reasoning trace). |
| `app/services/mcp_client.py` | Thin HTTP client for the standalone ShipSmart-MCP server, plus `RemoteTool` / `RemoteToolRegistry` shims that ducktype the old in-process tool interface. |
| `app/core/config.py` | All settings (env-driven via pydantic-settings). |
| `app/core/cache.py` | TTL cache used by RAG, recommendation, and LLM tool selection. |
| `app/core/errors.py` | Centralized error handling: `AppError` exception class + global exception handlers returning consistent JSON error responses. |
| `app/core/logging.py` | Structured logging setup (`setup_logging()`) and named logger factory (`get_logger()`). |
| `app/core/middleware.py` | `RequestLoggingMiddleware` — logs method, path, status, duration; honors inbound `X-Request-Id` and W3C `traceparent` (mints them when missing), stores them in ContextVars, and echoes both back as response headers. |
| `app/core/correlation.py` | ContextVars (`request_id_var`, `traceparent_var`) + `outbound_headers()` helper. Lets outbound clients (Java API, MCP) forward the same correlation IDs on every hop. |
| `app/core/rate_limit.py` | Shared `slowapi` limiter (per IP). |
| `app/schemas/` | Pydantic request/response models (`advisor.py`, `compare.py`, `agent.py`). |
| `app/llm/router.py` | Task-based router: each task → its own provider with a request-time failover chain. |
| `app/llm/client.py` | `OpenAIClient`, `AnthropicClient` (native tool calling), `GeminiClient`, `LlamaClient`, `EchoClient`, and the keyless `ScriptedToolCallingClient`. |
| `app/llm/guardrails.py` | Prompt-assembly guardrails: role separation, fencing of untrusted data, prompt-injection detection (block/neutralize), and grounding/refusal. Every decision is tagged for `decision_path`. |
| `app/llm/budget.py` | Token estimation + context-budget trimming (drops lowest-scoring chunks to fit the window) and temperature clamping. |
| `app/llm/prompts.py` | Prompt templates for RAG queries and advisor flows (system instructions, context formatting). |
| `app/rag/embeddings.py` | `OpenAIEmbedding` + `LocalHashEmbedding` placeholder. |
| `app/rag/vector_store.py` | `VectorStore` ABC + `InMemoryVectorStore`. |
| `app/rag/pgvector_store.py` | Postgres + pgvector implementation (asyncpg, cosine via `<=>`, plus lexical search for hybrid). |
| `app/rag/mcp_vector_store.py` | MCP-based pgvector store via Supabase MCP server (alternative to direct asyncpg). |
| `app/rag/hybrid.py` | Dense + sparse (BM25 / Postgres lexical) retrieval fused by `RAG_HYBRID_ALPHA`. |
| `app/rag/iterative.py` | Deterministic bounded plan→retrieve→assess→ground loop (`RAG_MODE=iterative`). Not model-driven — that's the agent. |
| `app/rag/chunking.py` | Document chunking: splits text into overlapping chunks for embedding. |
| `app/rag/ingestion.py` · `retrieval.py` | Ingestion + retrieval pipeline (`retrieve_auto` picks dense vs. hybrid per config). |
| `app/services/compare_service.py` | LLM-driven multi-scenario shipping comparison logic. |
| `app/services/orchestration_service.py` | Rule-based + LLM-assisted tool selection. |
| `app/services/java_client.py` | Thin async wrapper around the shared `httpx` client → calls Java for `quotes` / `saved-options`. Forwards `X-Request-Id` / `traceparent` via `outbound_headers()` so requests stay correlated across the Java hop. |
| `app/dependencies/__init__.py` | FastAPI dependency injection providers (`Depends()` helpers). |
| `scripts/perf_check.py` | Post-launch performance check: measures response times for key endpoints against thresholds. |
| `scripts/agentic_eval.py` | Offline eval harness for the agent / iterative retrieval (coverage + decision-path checks). |

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
boots but the `/advisor/*`, `/agent/*`, and `/orchestration/*` routes return
503 until you point `SHIPSMART_MCP_URL` at a live ShipSmart-MCP instance. Set
the env vars below to unlock real behavior.

> **Tip — drive the agent with no API keys.** Set `LLM_PROVIDER_REASONING=scripted`
> (non-production only) for a deterministic, keyless tool-calling stub that exercises the
> full agent loop. Combined with the in-memory store and `EchoClient` synthesis, the whole
> service runs end-to-end with zero external dependencies.

---

## Environment variables

All flags live in `.env.example` with comments. Highlights:

### LLM routing

```env
LLM_PROVIDER=                  # legacy single-provider
LLM_PROVIDER_REASONING=        # advisors + agent loop (set "scripted" for a keyless demo)
LLM_PROVIDER_SYNTHESIS=        # /rag/query, recommendation summary, grounded answers
LLM_PROVIDER_FALLBACK=echo     # safety net
LLM_TIMEOUT=30
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.3
```

Each task picks its own provider. Empty inherits `LLM_PROVIDER`. Unknown
or missing-key providers fall through to `LLM_PROVIDER_FALLBACK`, then
to `EchoClient` (placeholder responses).

### Request-time failover + context budget

```env
LLM_FALLBACK_CHAIN=            # csv tried after the primary errors, e.g. openai,gemini,echo
LLM_RETRY_MAX_ATTEMPTS=2       # retries per provider before the next in the chain
LLM_MAX_CONTEXT_TOKENS=8000    # token budget for retrieved context fed to the LLM
```

Empty `LLM_FALLBACK_CHAIN` keeps the single-client path (today's behavior). A non-empty
chain always terminates with `echo`, so a request can never dead-end with no answer.
Per-task model / temperature / token overrides (`LLM_MODEL_REASONING`,
`LLM_TEMPERATURE_SYNTHESIS`, `LLM_MAX_TOKENS_*`, …) inherit the global value when empty.

### Provider keys

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=             # Claude — the provider with native tool calling for the agent
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

### Retrieval modes (hybrid + iterative)

```env
RAG_MODE=normal                # normal (single-shot) | iterative  ("agentic" = deprecated alias)
RAG_HYBRID=false               # false = dense-only; true = dense + sparse (BM25 / lexical) fusion
RAG_HYBRID_ALPHA=0.5           # dense vs. sparse fusion weight (0..1; 1.0 = all dense)
RAG_ITERATIVE_MAX_STEPS=3      # max plan/retrieve steps when RAG_MODE=iterative
RAG_QUERY_LOG=false            # best-effort iterative-RAG traces to the rag_query_log table
```

See [Retrieval modes](#retrieval-modes). Defaults reproduce the simple dense, single-shot path.

### Guardrails

```env
GUARDRAILS_ENABLED=true              # fence/sanitize + detect prompt injection on advisor/RAG/agent calls
GUARDRAILS_BLOCK_ON_INJECTION=true   # block on detected injection (else neutralize and continue)
```

### Agent (Concierge)

```env
AGENT_ENABLED=true             # gate POST /api/v1/agent/run (404 when false)
AGENT_MAX_STEPS=5              # hard cost bound on the agent loop
AGENT_MAX_RETRIEVALS=2         # cap on retrieve_rag calls per run (1 = single-shot; >1 enables re-retrieval)
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

If `SHIPSMART_MCP_URL` is empty, the advisor, agent, and orchestration routes
return HTTP 503 (no tools available). See the **ShipSmart-MCP** repo for
how to run the tool server locally.

### Rate limiting

```env
RATE_LIMIT_ADVISOR=10/minute       # /advisor/* endpoints
RATE_LIMIT_ORCHESTRATION=20/minute # /orchestration/run
RATE_LIMIT_COMPARE=10/minute       # /compare endpoint
RATE_LIMIT_AGENT=10/minute         # /agent/run endpoint
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
path fired (`rule` / `llm` / `none`). The [Concierge agent](#spotlight-the-concierge-agent)
reuses this same `execute_tool` path for its MCP tool calls.

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
  `orchestration_service`, `agent_service`, `shipping_advisor_service`, and
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

# readiness — resolved retrieval/guardrail flags + LLM failover chains
curl http://localhost:8000/ready

# tool catalog
curl http://localhost:8000/api/v1/orchestration/tools

# explicit tool execution (mock provider)
curl -X POST http://localhost:8000/api/v1/orchestration/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"validate","tool":"validate_address","params":{"street":"1 Infinite Loop","city":"Cupertino","state":"CA","zip_code":"95014"}}'

# concierge agent (read-only reason→act→observe loop)
curl -X POST http://localhost:8000/api/v1/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"Can I ship a power bank to Berlin, and what would it cost from 10001?","context":{"origin_zip":"10001","destination_zip":"10115","weight_lbs":2}}'

# recommendation (deterministic scoring)
curl -X POST http://localhost:8000/api/v1/advisor/recommendation \
  -H 'Content-Type: application/json' \
  -d '{"services":[{"service":"Ground","price_usd":12.5,"estimated_days":5},{"service":"Express","price_usd":29,"estimated_days":1}],"context":{"urgent":true}}'
```

---

## Tests

```bash
uv run pytest          # 280 tests, ~4s, no network / no real keys
```

Tests live under `tests/` and use `pytest-asyncio` (async mode = auto).

**Hermetic by construction.** An autouse fixture in `tests/conftest.py`
(`_hermetic_settings`) pins every test to the self-contained profile —
`LocalHashEmbedding` + `InMemoryVectorStore` + `EchoClient`, no DATABASE_URL /
MCP URL — so the suite ignores the real provider config in your local `.env`
(OpenAI key, Supabase `DATABASE_URL`, pgvector). The MCP layer is served by a
`httpx.MockTransport`-backed `RemoteToolRegistry` (no live ShipSmart-MCP needed),
and the agent loop runs on the keyless `ScriptedToolCallingClient`.

Coverage spans the LLM router/fallback/budget, guardrails, dense/hybrid/iterative
RAG, the Concierge agent (loop, re-retrieval, route, native + text tool-calling),
the advisors, and `decision_path`. Service/seam coverage worth calling out:

| File | Focus |
| --- | --- |
| `test_agent_service.py` · `test_agent_route.py` | The agent loop end-to-end + the `/agent/run` route (404 when disabled, 503 when registry/router missing). |
| `test_agent_reretrieval.py` | Conditional, bounded re-retrieval: weak-coverage reformulation, degenerate-query rejection, and the per-run cap. |
| `test_agent_llm.py` | Native tool-calling vs. the keyless text fallback (`NotImplementedError` → `select_tool_with_llm`). |
| `test_guardrails.py` | Injection detection, fencing/neutralization, grounding, and output leak scanning. |
| `test_hybrid.py` | Dense + sparse fusion (BM25 / lexical) and graceful degradation to dense-only. |
| `test_agentic.py` · `test_agentic_eval.py` | Iterative RAG loop + the offline eval harness. |
| `test_compare.py` | `/api/v1/compare` end-to-end on Echo → the deterministic fallback scenarios + the cache. |
| `test_java_client.py` | `JavaApiClient` hydration over MockTransport: success + every graceful-`None` failure mode. |
| `test_mcp_client.py` | `RemoteToolRegistry` hydration, a tool call, and MCP-down degrading to `success=false` (never raising). |
| `test_orchestration_service.py` | Rule + LLM-assisted tool selection (with caching) and the `AppError` mapping in `execute_tool`. |
| `test_middleware.py` | `X-Request-Id` / `traceparent` minting + echo and `outbound_headers()` propagation. |
| `test_pgvector_store.py` | SQL-shape contract (cosine operator; `match_rag_chunks_lexical($1,$2)` selecting `source, chunk_index, text, score`) via a fake asyncpg pool — no DB. |

---

## Cross-service contracts

When the Java API or MCP server change shape, update these files in
lockstep:

| Caller | Endpoint | Used by |
|---|---|---|
| **Web → Python** | `POST /api/v1/agent/run` | Concierge agent — free-text shipping tasks; returns answer + reasoning trace. |
| **Web → Python** | `POST /api/v1/advisor/shipping` | Shipping advisor page. |
| **Web → Python** | `POST /api/v1/advisor/tracking` | Tracking advisor page. |
| **Web → Python** | `POST /api/v1/advisor/recommendation` | Recommendations widget. Frontend may send `services[]` directly **or** just `context.shipment_request_id` and let this service hydrate from Java. |
| **Web → Python** | `POST /api/v1/compare` | Decision-cockpit compare page. |
| **Web → Python** | `POST /api/v1/rag/query` | RAG q&a over the shipping knowledge base. |
| **Python → Java** | `GET /api/v1/quotes?shipmentRequestId=…` | Recommendation hydration — forwards inbound `Authorization` header. See `app/services/java_client.py`. |
| **Python → MCP** | `POST /tools/list`, `POST /tools/call` | Every advisor/agent/orchestration tool call. See `app/services/mcp_client.py`. Auth via `X-MCP-Api-Key` when `SHIPSMART_MCP_API_KEY` is set. |

Schemas live in `app/schemas/` (`advisor.py`, `compare.py`, `agent.py`). Java DTO
changes for the recommendation hydration path should be mirrored in
`app/services/java_client.py`. MCP tool-catalog changes are picked up
automatically at boot — the `RemoteToolRegistry` hydrates from `/tools/list`.

Correlation: `RequestLoggingMiddleware` honours inbound `X-Request-Id`
and W3C `traceparent` (minting them when missing) and stashes them in
ContextVars. `outbound_headers()` (in `app/core/correlation.py`) propagates
them to both Java and MCP, so a single request can be `grep`'d end-to-end.

---

## Operational notes

- **Agent returns 404**: `AGENT_ENABLED=false`. Set it `true` to expose `POST /api/v1/agent/run`.
- **`/agent/*`, `/advisor/*`, or `/orchestration/*` return 503**: `SHIPSMART_MCP_URL` is empty or the MCP server is unreachable (the tool registry / LLM router is not initialized). Boot `ShipSmart-MCP` and re-check.
- **Reading the agent trace**: `decisions[]` tags every branch (`agent:step2`, `agent:retrieve:reformulate`, `agent:retrieve:uncovered`, `guardrail:blocked`…) and `provider` reports who actually answered — debug the loop without log diving.
- **Rate limit 429**: someone is hammering an `/advisor`/`/agent` endpoint. Tune `RATE_LIMIT_*` if legitimate.
- **Echo / scripted responses**: no real LLM provider keys are set. Set `OPENAI_API_KEY` + `LLM_PROVIDER_REASONING=openai` (or `ANTHROPIC_*` for native agent tool-calling) to enable real completions.
- **`is_valid: true` for any address**: the MCP server is running on the mock carrier. Switch `SHIPPING_PROVIDER` on the **ShipSmart-MCP** service to a real carrier (it owns those env vars now).
- **RAG returns nothing relevant**: you're on `LocalHashEmbedding`. Set `EMBEDDING_PROVIDER=openai`. For exact-token misses (carrier/service codes), try `RAG_HYBRID=true`.
- **RAG cleared on restart**: you're on `VECTOR_STORE_TYPE=memory`. Switch to `pgvector` + `DATABASE_URL`.
- **CORS errors from the frontend**: the web origin (e.g. `http://localhost:5173`) must be in `CORS_ALLOWED_ORIGINS`.

---

## License

See [LICENSE](./LICENSE) for the full text.
