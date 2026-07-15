# ShipSmart — FastAPI AI Service (`api-python`)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-0.6%2B-DE5FE9?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![pgvector](https://img.shields.io/badge/pgvector-hybrid%20RAG-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![SSE](https://img.shields.io/badge/Streaming-SSE-FF8A5B)](#streaming--the-typed-assistant-contract)
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

> **Metric convention:** structural counts (tests, providers, routers) are
> facts verified against source; latency/availability figures are **(target)**
> budgets, never measured production metrics.

---

## Table of contents

- [The ShipSmart ecosystem](#the-shipsmart-ecosystem)
- [Architecture (HLD)](#architecture-hld)
- [Request flow](#request-flow)
- [The LLM router](#the-llm-router)
- [RAG done properly](#rag-done-properly)
- [The guardrail control plane](#the-guardrail-control-plane)
- [Streaming & the typed assistant contract](#streaming--the-typed-assistant-contract)
- [Agent surfaces](#agent-surfaces)
- [Object design (OOD)](#object-design-ood)
- [Data flow & privacy](#data-flow--privacy)
- [Observability, audit & kill-switches](#observability-audit--kill-switches)
- [Performance & availability](#performance--availability)
- [Endpoint surface](#endpoint-surface)
- [Deployment topology](#deployment-topology)
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

## Architecture (HLD)

**Figure 1 — container/component view.** The model never sits on a trust path —
LLM output must re-enter through the output validator and apply-policy before
anything renders. The deterministic domain core (HS codes, duty and carrier
rates) means numeric answers are computed, not generated.

```mermaid
flowchart TB
    WEB["ShipSmart-Web (React SPA)"] -->|Bearer JWT| MW
    subgraph API["ShipSmart-API (FastAPI, async)"]
        MW["Middleware: X-Request-Id + traceparent · CORS · slowapi rate limit"]
        RT["13 routers (/api/v1): advisor · agent · assistant/stream · compare · compliance · concierge · feedback · orchestration · rag · workflow · admin · health · info"]
        SVC["Services: advisors · recommendation · compare · rag_service · agent_service · orchestration"]
        subgraph GATES["Security gates (app/security)"]
            G1["normalization (NFKC)"]
            G2[injection_gate]
            G3["pii (HMAC pseudonymize + redact)"]
            G4["state_integrity (HMAC state)"]
            G5["misuse · budgets · tool_policy"]
        end
        subgraph LLM["LLM layer"]
            RTR["LLMRouter: per-task chains + failover + stream()"]
            PROV["6 clients: OpenAI · Anthropic · Gemini · Llama · Scripted · Echo"]
        end
        subgraph RAG["RAG layer"]
            RA["retrieve_auto: dense | hybrid"]
            IT["iterative loop (bounded)"]
            GR["grounding: coverage_of / refuse"]
            VS["vector-store port: InMemory · PGVector · MCPVectorStore"]
            ING["ingestion + ingestion_guard + trusted_sources"]
        end
        subgraph AGENTS["Agent surfaces"]
            CON["concierge (slot-filling)"]
            AGT["read-only tool agent"]
            CMP["compliance critic (advisory-only)"]
            WKF["workflow engine + checkpointer + HITL review_queue"]
        end
        DOM["Deterministic domain core: hs_codes · duty_rates · carrier_rates"]
        AUD["AIEvent audit (WORM) + guardrail metrics + kill-switch registry"]
    end
    JAVA["ShipSmart-Orchestrator (Java)"]
    MCP["ShipSmart-MCP (read-only tools)"]
    PG[("Supabase Postgres + pgvector")]
    MW --> RT --> SVC
    SVC --> GATES
    SVC --> LLM
    SVC --> RAG
    RT --> AGENTS
    AGENTS --> LLM
    AGENTS --> MCP
    WKF --> DOM
    RAG --> PG
    SVC -->|"httpx, shared client"| JAVA
    SVC --> AUD --> PG
```

**Patterns:** hexagonal ports-and-adapters (vector store, conversation store,
domain providers, audit sink) · composition root (`bootstrap.py` → `app.state`)
· strategy (per-task provider) · chain of responsibility (failover, gates).

---

## Request flow

**Figure 2 — `/assistant` and `/assistant/stream`, with four refusal exits.**
The pipeline refuses on injection, forged state, weak grounding, and
schema-invalid output — before and after the model call. The happy path always
ends typed and audited.

```mermaid
flowchart LR
    A[Ingress] --> B["mint X-Request-Id · rate limit"]
    B --> C["normalize: NFKC, zero-width strip, homoglyph fold, language-ID"]
    C --> D{"injection_gate severity"}
    D -->|block| R1["safe refusal + guardrail tag"]
    D -->|allow / neutralize| E["pii: redact + pseudonymize"]
    E --> F["state_integrity: verify HMAC state"]
    F -->|unsigned/forged| R2["state treated as empty / approval rejected"]
    F --> G["RAG: retrieve_auto + grounding check"]
    G -->|coverage too low| R3["grounded refusal (no hallucination)"]
    G --> H["assemble(): fenced prompt"]
    H --> I["LLMRouter: task chain, failover"]
    I --> J["output_validator: schema check, 1 corrective retry"]
    J -->|invalid twice| R4["safe fallback + guardrail:structured_output_invalid"]
    J --> K["apply_policy: auto / confirm / never (rule-derived)"]
    K --> L["AIEvent audit (WORM)"]
    L --> M["typed AssistantResponse | SSE stream"]
```

---

## The LLM router

**Figure 3 — task routing + failover: retryable errors fail over; terminal
errors fail fast; the chain ends at a keyless echo.** Swap providers by env; a
provider outage degrades — it never 500s. `GET /ready` shows the resolved
chains in production.

```mermaid
sequenceDiagram
    participant S as Service
    participant R as LLMRouter
    participant P1 as Primary (e.g. OpenAI)
    participant P2 as Fallback (e.g. Gemini)
    participant E as EchoClient
    S->>R: complete(task=reasoning, prompt)
    R->>P1: complete()
    P1--xR: transient/network error (retryable)
    R->>P1: retry (≤ LLM_RETRY_MAX_ATTEMPTS)
    P1--xR: still failing
    R->>P2: failover: next in chain
    alt P2 succeeds
        P2-->>R: completion
        R-->>S: result (provider + hops logged with request_id)
    else P2 terminal error (auth / context / content-filter)
        P2--xR: terminal -> no retry
        R->>E: last resort (keyless echo)
        E-->>R: deterministic fallback
        R-->>S: degraded-but-typed result
    end
```

- Per-task chains from env (`LLM_PROVIDER_REASONING`, `LLM_PROVIDER_SYNTHESIS`).
- Typed error taxonomy (`classify_provider_error`); every hop logged.
- Native tool-calling (`complete_with_tools`) with text-mode fallback.
- Per-request budgets: LLM/tool/token ceilings, temperature clamping.

---

## RAG done properly

**Figure 4 — hybrid retrieval with grounding-or-refuse.**

```mermaid
sequenceDiagram
    participant Q as Query
    participant RA as retrieve_auto
    participant D as PGVectorStore (dense)
    participant L as match_rag_chunks_lexical (SQL)
    participant G as grounding
    Q->>RA: retrieve(question)
    par dense
        RA->>D: cosine top-k (pgvector)
        D-->>RA: chunks + scores
    and lexical
        RA->>L: websearch_to_tsquery + ts_rank_cd (GIN)
        L-->>RA: chunks + scores
    end
    RA->>RA: alpha-fusion (rank/linear)
    RA->>G: coverage_of(answer area, chunks)
    alt coverage sufficient
        G-->>Q: grounded context (cited)
    else weak coverage
        G-->>Q: REFUSE — corpus cannot support the ask
    end
```

**Figure 5 — the bounded iterative loop (state machine).** The model may
reformulate a weak query, but it cannot run away.

```mermaid
stateDiagram-v2
    [*] --> Retrieve
    Retrieve --> Assess: coverage_of()
    Assess --> Generate: grounded
    Assess --> Reformulate: weak AND steps left
    Reformulate --> Retrieve: model rewrites query
    Assess --> Refuse: weak AND budget exhausted
    Generate --> [*]
    Refuse --> [*]: honest cannot-support
```

Plus **supply-chain hygiene**: `ingestion_guard` + `trusted_sources` (source
allowlist + scan/quarantine — a poisoned document can't enter the corpus) and
**embedding-version governance** (`embedding_compat` — a fail-closed startup
check; no silent mixed vector spaces).

---

## The guardrail control plane

**Figure 6 — one prompt-assembly choke point for every feature.**

```mermaid
sequenceDiagram
    participant F as Feature (advisor/agent/concierge)
    participant AS as assemble()
    participant DI as detect_injection
    participant M as Model
    participant OV as output_validator
    F->>AS: user text + chunks + tool results
    AS->>AS: fence regions + neutralize fence/role tokens inside data
    AS->>DI: pattern battery (override / fence-spoof / role-play / encoded)
    DI-->>AS: block | neutralize | warn | allow (tagged)
    AS->>M: fenced prompt
    M-->>OV: raw output
    OV->>OV: schema validate
    alt invalid
        OV->>M: ONE corrective retry
        M--xOV: still invalid
        OV-->>F: safe refusal (guardrail tag emitted)
    else valid
        OV-->>F: typed structured output
    end
```

Input hygiene (NFKC + zero-width strip + homoglyph fold + language-ID) runs
before the fences; `scan_output` leak-scans after the model. The deterministic
**apply-policy** decides auto/confirm/never from **rule-derived confidence +
field risk — never model self-report**. Every decision emits a `guardrail:*`
tag — the same vocabulary the eval suite's coverage gate joins on.

---

## Streaming & the typed assistant contract

**Figure 7 — SSE: deltas stream; failover only before the first token.**

```mermaid
sequenceDiagram
    participant W as Web (fetch + ReadableStream)
    participant A as /assistant/stream
    participant R as LLMRouter.stream()
    W->>A: POST (question + signed state)
    A->>A: gates + RAG (as Fig. 2)
    A->>R: stream(task=synthesis)
    alt provider fails BEFORE first delta
        R->>R: failover to next provider
    end
    loop token deltas
        R-->>A: delta
        A-->>W: data: {"delta": "..."}
    end
    A->>A: validate typed envelope
    A-->>W: data: {"done": true, "assistant": AssistantResponse}
    Note over W: onDelta renders progressively, then onDone swaps in typed cards
```

The envelope is a **discriminated union** (`shipping_option | comparison |
missing_info | policy_answer`) mirrored field-for-field by the Web's TypeScript
types and parity-tested in CI. Even refusals stream.

---

## Agent surfaces

**Figure 8 — the read-only tool agent loop.**

```mermaid
sequenceDiagram
    participant U as User
    participant AG as agent_service
    participant TP as tool_policy
    participant MC as MCP (/tools/call)
    U->>AG: POST /agent/run
    loop steps ≤ step cap
        AG->>AG: plan next action (model)
        AG->>TP: validate tool + args + route + confirmation
        alt denied
            TP-->>AG: refusal (tagged)
        else allowed
            AG->>MC: tools/call (X-MCP-Api-Key)
            MC-->>AG: read-only result
        end
    end
    AG-->>U: typed answer + tool_calls audit trail
```

**Figure 9 — the durable workflow with human-in-the-loop (state machine).**

```mermaid
stateDiagram-v2
    [*] --> Running
    Running --> Checkpointed: after each node (classification / docs / landed-cost / routing)
    Checkpointed --> Running: resume across requests
    Running --> AwaitingReview: high-risk compliance area -> review_queue
    AwaitingReview --> Running: human approves (POST /workflow/id/review)
    AwaitingReview --> Rejected: human rejects
    Running --> Done
    Done --> [*]
    Rejected --> [*]
```

| Surface | What it is | Containment |
|---|---|---|
| **Concierge** | multi-turn slot-filling chat; deterministic intent extraction with LLM fallback | client-owned **HMAC-signed** state; server-side recall store |
| **Tool agent** | model-driven plan→retrieve→call loop | step/retrieval caps · pre-execution tool policy · read-only |
| **Compliance** | areas → structural rules → critic | **advisory-only**: never a fabricated clearance |
| **Workflow** | checkpointed multi-agent run over the deterministic domain core | suspends to a human review queue |

---

## Object design (OOD)

**Figure 10 — the LLM layer and typed outputs.** The seven gate modules are
pure-Python units with no framework coupling — which is why 499 tests run
keylessly in minutes.

```mermaid
classDiagram
    class LLMClient {
        <<abstract>>
        +complete(prompt) str
        +complete_with_tools(prompt, tools)
        +stream(prompt) AsyncIterator
    }
    class LLMRouter {
        +complete(task, prompt)
        +stream(task, prompt)
        -chains: dict~task, list~
        -classify_provider_error()
    }
    LLMClient <|-- OpenAIClient
    LLMClient <|-- AnthropicClient
    LLMClient <|-- GeminiClient
    LLMClient <|-- LlamaClient
    LLMClient <|-- ScriptedToolCallingClient
    LLMClient <|-- EchoClient
    LLMRouter o-- LLMClient : per-task failover chain
    class VectorStore {
        <<interface>>
        +add(chunks)
        +search_dense(q, k)
        +search_lexical(q, k)
    }
    VectorStore <|.. InMemoryVectorStore
    VectorStore <|.. PGVectorStore
    VectorStore <|.. MCPVectorStore
    class AssistantResponse {
        +intent
        +apply_policy
        +confidence
        +result: ResultUnion
        +grid_actions
        +tool_calls
        +audit
    }
    class ResultUnion {
        <<discriminated union>>
        shipping_option | comparison | missing_info | policy_answer
    }
    AssistantResponse *-- ResultUnion
```

---

## Data flow & privacy

**Figure 11 — where identity and free text are protected.** Raw identity never
reaches the ledger — `session_id_hash` is an HMAC pseudonym; free text is
redacted before persistence; `prompt_version` / `schema_version` /
`embedding_version` make any answer reproducible.

```mermaid
flowchart LR
    U["user text"] --> N[normalize]
    N --> P["pii.redact: emails, phones, tracking, addresses"]
    P --> F["fenced assembly"]
    RC["retrieved chunks"] --> F
    TR["tool results"] --> F
    F --> MODEL["LLM"]
    MODEL --> V["typed validation"]
    V --> RESP["response to client"]
    P -. "session_id -> HMAC pseudonym" .-> AE["AIEvent"]
    V -. "decisions + tags + versions" .-> AE
    AE --> WORM[("WORM ledger: UPDATE never, DELETE via retention job only")]
```

---

## Observability, audit & kill-switches

- `X-Request-Id` + W3C `traceparent` across Web → API → Java → MCP;
  ContextVar-scoped into every log line.
- **AIEvent** per model/tool call → the WORM Postgres ledger (retention
  30d/13mo/24mo classes); guardrail metrics with volume-floored alerts (the SQL
  twin `ai_guardrail_daily` lives in ShipSmart-Infra).
- `GET|POST /api/v1/admin/ai-controls`: token-gated runtime kill-switches for
  `agent · concierge · workflow · compliance · rag` — every flip audited with
  actor + reason; **guardrails are never killable**.

| Threat | Control |
|---|---|
| Prompt injection / fence-spoof | fencing + neutralization + `detect_injection` |
| Obfuscated / multilingual jailbreak | NFKC + zero-width + homoglyph + language-ID |
| Forged approvals / tampered context | HMAC-signed state; unsigned ⇒ empty |
| Fabricated structure / leakage | schema validation + retry + `scan_output` |
| Unauthorized / costly tool calls | `tool_policy` + budgets |
| PII in logs | redaction + pseudonymization + WORM ledger |
| Runaway feature | audited kill-switch (capability only) |

---

## Performance & availability

**Latency budget (target):**

| Stage | Budget *(target)* |
|---|---|
| Gates (normalize + injection + PII + state) | 40 ms |
| Retrieval (hybrid, pooled) | 120 ms |
| LLM first token (streamed) | 700 ms |
| Output validation + apply-policy | 30 ms |
| **First token, end-to-end** | **< 1 s** |

```mermaid
xychart-beta
    title "Stage latency budget in ms (target)"
    x-axis ["gates", "retrieval", "first-token", "validate"]
    y-axis "ms (target)" 0 --> 800
    bar [40, 120, 700, 30]
```

**Degradation matrix (coded behaviors, facts):**

| Dependency down | Behavior |
|---|---|
| Primary LLM provider | retry → failover chain → EchoClient (typed, deterministic) |
| Embedding key absent | `LocalHashEmbedding` (keyless) |
| MCP unreachable | 503 for tool paths — never a crash |
| Conversation store | recall disabled; boot continues |
| Feature flagged off | endpoint 404s |
| Feature kill-switched | runtime off, audited; guardrails stay on |

---

## Endpoint surface

| Group | Endpoints |
|---|---|
| Advisory | `POST /advisor/shipping` · `/advisor/tracking` · `/advisor/recommendation` |
| Assistant | `POST /concierge/chat` · `GET /concierge/{session_id}` · `POST /assistant/stream` (SSE) |
| Agent | `POST /agent/run` · `GET /agent/tools` |
| Compliance / Workflow | `POST /compliance/check` · `POST /workflow/process` · `POST /workflow/{id}/review` · `GET /workflow/{id}` |
| RAG / Compare | `POST /rag/query` · `POST /rag/ingest` · `POST /compare` |
| Ops | `GET /health` · `GET /ready` · `GET /info` · `POST /feedback` · `GET|POST /admin/ai-controls` |

---

## Deployment topology

**Figure 12 — production layout.** `/docs` dev-only; features env-flagged;
`GET /ready` is the wiring inspection.

```mermaid
flowchart LR
    U[Browser] --> W["Render: shipsmart-web"]
    W -->|"JWT + X-Request-Id"| A["Render: shipsmart-api-python (this service)"]
    A -->|X-MCP-Api-Key| M["Render: shipsmart-mcp"]
    A -->|httpx| J["Render: shipsmart (Java)"]
    A --> S[("Supabase Postgres + pgvector: rag_chunks, ai_audit_log, conversations")]
    A -. "GET /ready = resolved flags + LLM chains" .-> OPS[Operator]
```

---

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
asserted by **ShipSmart-Test** (10 contract suites + a six-layer eval system).

## License

See [LICENSE](./LICENSE).
