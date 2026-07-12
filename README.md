# ShipSmart — FastAPI AI Service (`api-python`)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-0.6%2B-DE5FE9?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![pgvector](https://img.shields.io/badge/pgvector-hybrid%20RAG-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![SSE](https://img.shields.io/badge/Streaming-SSE-FF8A5B)](#spotlight-streaming--the-typed-assistant-contract)
[![Tests](https://img.shields.io/badge/tests-499%20hermetic%2C%20no%20keys-3FB950?logo=pytest&logoColor=white)](#tests--quality-gates)
[![Live](https://img.shields.io/badge/Live-%2Fready-46E3B7?logo=render&logoColor=white)](https://shipsmart-api-python.onrender.com/ready)
[![License](https://img.shields.io/badge/License-See%20LICENSE-blue)](./LICENSE)

> The **AI layer** of the ShipSmart platform: a task-routed, failover-hardened
> multi-provider LLM stack behind **one guardrail choke point** — hybrid +
> iterative RAG with grounding-or-refuse, **typed structured outputs** the UI
> renders (never parsed prose), a read-only tool-calling agent, durable
> human-in-the-loop workflows, SSE token streaming, and an append-only,
> pseudonymized audit trail with runtime kill-switches. **Deterministic core,
> model at the edges.**

Owns no transactional data; provides grounded shipping advice, a slot-filling
concierge, compliance review, recommendation scoring, and multi-agent workflows
on top of a multi-provider LLM router. Every external dependency degrades
gracefully — the service boots, answers, and stays observable with **no API
keys, no database, and no tool server**.

**Stack:** FastAPI 0.135.3 · Python 3.13 (async) · uv · Pydantic v2 · pgvector ·
slowapi · OpenAI / Anthropic / Gemini / Llama / Scripted / Echo

**Live:** [`GET /ready`](https://shipsmart-api-python.onrender.com/ready) shows
the resolved feature flags and LLM chains ·
[`/health`](https://shipsmart-api-python.onrender.com/health) ·
[`/api/v1/info`](https://shipsmart-api-python.onrender.com/api/v1/info)
*(Render free tier — first hit may take ~30–60 s to wake).*

---

## Table of contents

- [The ShipSmart ecosystem](#the-shipsmart-ecosystem)
- [Engineering highlights](#engineering-highlights)
- [Architecture](#architecture)
- [Spotlight: the LLM router](#spotlight-the-llm-router)
- [Spotlight: RAG done properly](#spotlight-rag-done-properly)
- [Spotlight: the guardrail control plane](#spotlight-the-guardrail-control-plane)
- [Spotlight: streaming + the typed assistant contract](#spotlight-streaming--the-typed-assistant-contract)
- [Agent surfaces](#agent-surfaces)
- [Observability, audit & kill-switches](#observability-audit--kill-switches)
- [Endpoint surface](#endpoint-surface)
- [Running locally](#running-locally)
- [Configuration](#configuration)
- [Tests & quality gates](#tests--quality-gates)
- [License](#license)

---

## The ShipSmart ecosystem

This service is one of six sibling repositories. Clone them as siblings of this
directory when working on the full system. All six are also mirrored together in
**[ShipSmart](https://github.com/nia194/ShipSmart)** — the umbrella repository
that snapshots each component at a pinned commit (see its `COMPONENTS.yml`).

| Repo | Role | Stack |
|------|------|-------|
| [ShipSmart-Web](https://github.com/nia194/ShipSmart-Web) | React SPA — search-first UI, typed AI rendering | React 19, Vite, TS 5.9 |
| [ShipSmart-Orchestrator](https://github.com/nia194/ShipSmart-Orchestrator) | Java system of record — **single writer** to Postgres; quotes, bookings, the AI trust boundary | Spring Boot 3.4, Java 17 |
| **[ShipSmart-API](https://github.com/nia194/ShipSmart-API)** *(this repo)* | Python AI layer — RAG, guardrails, agents, streaming, audit | FastAPI, Python 3.13 |
| [ShipSmart-MCP](https://github.com/nia194/ShipSmart-MCP) | Read-only MCP tool server (boot-enforced allowlist) | FastAPI + MCP |
| [ShipSmart-Infra](https://github.com/nia194/ShipSmart-Infra) | Supabase schema, RLS, WORM audit ledger, pgvector, edge functions | Supabase, Deno |
| [ShipSmart-Test](https://github.com/nia194/ShipSmart-Test) | Cross-repo contracts + six-layer evals + live e2e | Python 3.13, pytest |

---

## Engineering highlights

| | Capability | Why it's interesting |
|---|---|---|
| 🔀 | **Task-routed LLM failover** | `reasoning`/`synthesis`/`fallback` each bind their own provider; retryable errors retry → fail over down a chain that terminates in a keyless echo. A provider outage degrades — it never 500s. |
| 🌊 | **True SSE streaming** | `POST /api/v1/assistant/stream` emits token deltas and closes with a **typed envelope**; the router fails over **only before the first token**. |
| 🔎 | **Hybrid + iterative RAG** | pgvector dense + Postgres lexical (`tsvector`/GIN/`ts_rank_cd`) fusion; a bounded reason→retrieve→reformulate loop; **grounding-or-refuse** — never guess past the corpus. |
| 🛡️ | **One guardrail choke point** | Every prompt flows through `assemble()`: untrusted-data fencing, injection detection (block/neutralize/warn), NFKC + homoglyph + zero-width hygiene, output leak scan. |
| 📦 | **Typed structured outputs** | The model returns `AssistantResponse` (a discriminated union) + `FormPatch` + grid actions — validated, one corrective retry, then safe refusal. The UI renders types, never prose. |
| 🔐 | **State integrity + PII discipline** | Client conversation state is **HMAC-signed** (forged approvals rejected); identity is pseudonymized and free text redacted before the **WORM** audit ledger. |
| 🤖 | **Four agent surfaces, one discipline** | Slot-filling concierge, read-only tool agent (step/retrieval caps + tool policy), advisory-only compliance critic, checkpointed workflow with human-in-the-loop review. |
| 🧮 | **Deterministic domain core** | Landed cost, duty rates, HS codes are code (`app/domain/`) behind ports — workflow agents compute numbers, models only explain them. |
| 🚨 | **Runtime kill-switches** | `agent · concierge · workflow · compliance · rag` can be disabled live (audited actor + reason); **guardrails are never killable**. |
| 🧪 | **499 hermetic tests** | Zero network, zero keys — EchoClient + in-memory stores; the entire suite is CI-safe by construction. |

---

## Architecture

```
  HTTP ──▶ middleware (X-Request-Id · CORS · rate limit)
       ──▶ routers (/api/v1: advisor · agent · assistant/stream · compare ·
            compliance · concierge · feedback · orchestration · rag · workflow · admin)
       ──▶ services ──▶ security gates (normalization · injection · pii ·
            state_integrity · misuse · budgets · tool_policy)
       ──▶ LLM layer (LLMRouter + 6 providers)   ──▶ RAG layer (dense/hybrid/
            iterative + grounding + 3 vector stores)
       ──▶ agents (concierge · tool agent · compliance · workflow + domain core)
       ──▶ AIEvent audit (WORM) · guardrail metrics · kill-switch registry
  all singletons wired once in bootstrap.py → app.state (composition root)
```

**Patterns:** hexagonal ports-and-adapters (vector store, conversation store,
domain providers, audit sink) · composition root · strategy (per-task provider)
· chain of responsibility (failover, gates).

---

## Spotlight: the LLM router

- Per-task chains from env (`LLM_PROVIDER_REASONING`, `LLM_PROVIDER_SYNTHESIS`);
  chain = `[primary, *fallbacks, echo]`.
- Typed error taxonomy: retryable (transient/network) retries up to
  `LLM_RETRY_MAX_ATTEMPTS` then fails over; terminal (auth/context/content
  filter) fails fast. Every hop logged with provider + error class + request id.
- Six clients behind one ABC: OpenAI, Anthropic, Gemini, Llama,
  ScriptedToolCalling (deterministic native function-calling), Echo (keyless).
- Native tool-calling with graceful text-mode fallback; per-request budgets
  (LLM/tool/token caps, temperature clamping).

## Spotlight: RAG done properly

- `retrieve_auto` dispatches dense vs hybrid by config; hybrid fuses pgvector
  cosine with BM25-style lexical (`match_rag_chunks_lexical` — its SQL column
  shape is CI-asserted in ShipSmart-Infra).
- Three swappable stores: in-memory (dev/CI), pgvector (asyncpg, pooled), MCP
  (remote retrieval).
- **Ingestion guard**: source allowlist + scan/quarantine so a poisoned document
  can't enter the corpus; embedding-version metadata + a **fail-closed startup
  compatibility check** (no silent mixed vector spaces).
- Grounding signal (`coverage_of`) → the advisor **refuses** when the corpus
  can't support the ask.

## Spotlight: the guardrail control plane

`app/llm/guardrails.py` is the single prompt-assembly choke point: labeled
fences (`<user_input>`, `<retrieved_chunk>`, `<tool_results>`) with fence-token
neutralization; a pattern battery for direct-override / fence-spoof / role-play
/ encoded injection; `sanitize_user_input`; Unicode normalization
(`app/security/normalization.py`); post-response leak scanning; and a
deterministic **apply-policy** — whether a form patch auto-applies, needs
confirmation, or never mutates is decided by **rule-derived confidence + field
risk, never model self-report**. Every decision emits a `guardrail:*` tag — the
same vocabulary the eval suite's coverage gate joins on.

## Spotlight: streaming + the typed assistant contract

`POST /api/v1/assistant/stream` returns `text/event-stream`: guardrail checks
and retrieval run first, then token deltas stream as generated
(`{"delta": …}`), closing with `{"done": true, "assistant": AssistantResponse}`
— a **discriminated union** (`shipping_option | comparison | missing_info |
policy_answer`) mirrored field-for-field by the Web's TypeScript types and
parity-tested in CI. Even refusals stream.

## Agent surfaces

| Surface | What it is | Containment |
|---|---|---|
| **Concierge** (`app/agents/concierge/`) | multi-turn slot-filling chat; deterministic intent extraction with LLM fallback; corrections, compound intents | client-owned **HMAC-signed** state; server-side recall store |
| **Tool agent** (`app/services/agent_service.py`) | model-driven plan→retrieve→call loop over MCP tools + RAG | step caps · retrieval caps · pre-execution tool policy · read-only |
| **Compliance** (`app/agents/compliance/`) | areas → structural rules → critic | **advisory-only**: "uncovered ⇒ unverified", never a fabricated clearance |
| **Workflow** (`app/workflow/`) | checkpointed multi-agent run (classification, documentation, landed-cost, routing) | suspends to a **human review queue** on high-risk areas; resumes across requests |

## Observability, audit & kill-switches

- `X-Request-Id` + W3C `traceparent` minted/propagated across Web → API → Java →
  MCP; ContextVar-scoped into every log line.
- **AIEvent**: one append-only record per model/tool call — HMAC-pseudonymized
  identity, redacted text, `prompt_version`/`schema_version`/
  `embedding_version`, decisions + tool calls + guardrail events — persisted to
  the WORM Postgres ledger (retention: 30d / 13mo / 24mo classes).
- Guardrail metrics with volume-floored alerts; the SQL twin
  (`ai_guardrail_daily`) lives in ShipSmart-Infra.
- `GET|POST /api/v1/admin/ai-controls`: token-gated runtime kill-switches for
  the five AI features — every flip audited with actor + reason.

## Endpoint surface

| Group | Endpoints |
|---|---|
| Advisory | `POST /advisor/shipping` · `/advisor/tracking` · `/advisor/recommendation` |
| Assistant | `POST /concierge/chat` · `GET /concierge/{session_id}` · `POST /assistant/stream` (SSE) |
| Agent | `POST /agent/run` · `GET /agent/tools` |
| Compliance / Workflow | `POST /compliance/check` · `POST /workflow/process` · `POST /workflow/{id}/review` · `GET /workflow/{id}` |
| RAG / Compare | `POST /rag/query` · `POST /rag/ingest` · `POST /compare` |
| Ops | `GET /health` · `GET /ready` · `GET /info` · `POST /feedback` · `GET|POST /admin/ai-controls` |

## Running locally

```bash
uv sync                                                    # deps (lockfile)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
curl localhost:8000/ready                                  # resolved wiring
```

Runs fully keyless by default (echo LLM, in-memory vector store, mock domain
adapters). Point it at siblings with `SHIPSMART_MCP_URL=http://127.0.0.1:8001`
and `INTERNAL_JAVA_API_URL=http://127.0.0.1:8080` — or boot everything at once
with ShipSmart-Test's `scripts/run-stack.sh`.

## Configuration

12-factor via `pydantic-settings`. The interesting dials:

| Env | Effect |
|---|---|
| `LLM_PROVIDER_REASONING` / `LLM_PROVIDER_SYNTHESIS` | per-task provider (+ failover chain) |
| `AGENT_ENABLED` · `CONCIERGE_ENABLED` · `WORKFLOW_ENABLED` · `COMPLIANCE_ENABLED` | AI surfaces — **404 when off**, conservative defaults |
| `GUARDRAILS_ENABLED` · `FEEDBACK_ENABLED` | control plane + feedback loop |
| `RAG_AUTO_INGEST` | first-boot corpus ingestion |
| `DATABASE_URL` | pgvector store (absent ⇒ in-memory) |
| `SHIPSMART_MCP_URL` / `SHIPSMART_MCP_API_KEY` | tool server hop |

## Tests & quality gates

```bash
uv run pytest            # 499 tests / 71 files — hermetic, zero keys
uv run ruff check .      # lint (E,F,I,N,W,UP)
```

Local eval harnesses: `scripts/agentic_eval.py`, `scripts/compliance_eval.py`,
`scripts/workflow_eval.py`, `scripts/perf_check.py`. Cross-repo: the typed
contract, decision-tag registry, trust boundary, and guardrail coverage are
asserted by **ShipSmart-Test** (10 contract suites + a six-layer eval system) —
drift fails CI before it can break a neighbor.

## License

See [LICENSE](./LICENSE).
