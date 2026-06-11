# ShipSmart Concierge — AI Agent + Agentic RAG Implementation Plan

**Status:** Ready to implement
**Date:** 2026-06-11
**Scope:** ShipSmart-API (day-1 core) · ShipSmart-Test (verification). Web/Orchestrator/MCP touched only in fast-follow.
**Builds on:** [`agent-integration-design.md`](./agent-integration-design.md) (concept/routing decision) and the
existing `app/rag/agentic.py` (a deterministic multi-pass retrieval loop — today mislabeled "agentic"; see §1a).

---

## 1. TL;DR

We already have every primitive an agent needs: a multi-provider LLM layer with a
task router (`app/llm/router.py`), a remote MCP tool registry (`app/services/mcp_client.py`),
hybrid RAG retrieval (`app/rag/`), guardrails + grounding (`app/llm/guardrails.py`), and a
decision-path tagging convention (`decisions[]`). What's missing is a single, user-visible
**feature** that ties them into one coherent, multi-step experience.

This plan defines that feature — the **ShipSmart Concierge** — and the concrete work to ship a
**read-only** version of it in **one focused day**. It picks the open decisions left in
`agent-integration-design.md`:

- **Routing: Option A** — a new additive `POST /api/v1/agent/run`. Existing
  `/orchestration/run` and `/advisor/*` stay unchanged (clean rollback; A/B-able).
- **Substrate: native function calling**, **Anthropic-first** (`claude-sonnet-4-5` is the
  configured default), with the existing text-based selection (`select_tool_with_llm`) as the
  non-native fallback. OpenAI parity is a fast-follow.
- **Naming fix:** today's `app/rag/agentic.py` is **not** agentic (see §1a). Rename it
  **Iterative RAG** and expose it to the new agent as a first-class tool (`retrieve_rag`).
  The word "agentic" is reserved for the model-driven loop introduced here.

The **Iterative RAG** layer (deterministic retrieve → assess → reformulate → ground/refuse)
and the **AI Agent** layer (model-driven reason → act → observe → repeat over MCP tools)
become two layers of the same request: the agent *decides* when to retrieve and whether the
result is good enough; the Iterative RAG just retrieves well and grounds honestly.

**Day-1 boundary:** the agent plans, calls read-only tools, retrieves, and **proposes** a
shipment — it does **not** persist. Creating the shipment is a deliberate, separately-authorized
fast-follow (see §7). This keeps the whole day-1 surface read-only and fully testable.

---

## 1a. Review note — "the 'agentic' in the current RAG is vague / unnecessary"

This is a correct critique, and the plan acts on it rather than defending the label.

**What `app/rag/agentic.py` actually does today:** a *bounded, deterministic* multi-pass
retrieval. The control flow has **no LLM in it** — by its own docstring, *"reformulation +
coverage are deterministic heuristics (no hidden LLM calls in the control flow); the LLM is
used only for the final grounded answer."* Concretely:

- `_covered()` = "any chunk with `score > 0`" — a fixed threshold, not a judgment.
- `_reformulate()` = appends a **canned suffix string** per step — not a model rewrite.
- `_maybe_escalate_tools()` fires **iff** specific context keys are present — a presence
  check, not a decision.

By the very definition an agent needs (*the model's own output produces an observation that
is fed back to inform its next decision* — `agent-integration-design.md` §2), this loop is
**not agentic**. It is honest, cheap, testable *iterative retrieval with grounding*. The
"agentic" name oversells it and invites exactly this objection.

**The fix (two parts):**

1. **Rename it to what it is — "Iterative RAG."** Keep the feature (it's genuinely useful:
   multi-pass recall + deterministic refusal when nothing covers the question). It is a *pure
   rename* with behavior unchanged; the exact, bounded set of touch-points is in §4.1(4).
   This alone removes the "vague" critique.

2. **Make retrieval *actually* agentic by promotion, not by naming.** In the new
   `AgentService`, retrieval becomes a tool (`retrieve_rag`) that the **LLM chooses to
   call**, reads the returned chunks/scores, and then **decides** — answer now, retrieve
   again with a *model-written* reformulation, or call a different tool. That is the
   feedback loop the old code lacked. "Agentic RAG" is then a true, earned description of
   the agent invoking retrieval under its own reasoning — and the deterministic Iterative
   RAG sits underneath as the dependable retrieval primitive it always was.

So the answer to the review is: the adjective wasn't wrong to *want* — it was attached to
the wrong layer. This plan moves it up one layer, where a model is genuinely driving.

---

## 2. The use case — "Ship-it Concierge"

A single free-text request that no single existing endpoint can satisfy today:

> *"Send a 5 lb box from 10001 to 90210, cheapest option that still arrives by Friday.
> Destination is 123 Main St, Beverly Hills CA 90210 — is that deliverable? It has a
> power bank in it, any restrictions I should know about?"*

This is realistic, and it forces **every** capability into one flow:

| Need in the sentence | Capability exercised | Where it lives |
|---|---|---|
| "power bank … restrictions?" | **Iterative RAG**, model-invoked via `retrieve_rag` — retrieve hazmat/lithium policy, ground or refuse | `app/rag/iterative.py` (renamed from `agentic.py`) |
| "is 123 Main St … deliverable?" | **Tool: `validate_address`** (read-only, MCP), self-correct on failure | ShipSmart-MCP |
| "5 lb … cheapest" | **Tool: `get_quote_preview`** (read-only, MCP) | ShipSmart-MCP |
| "…arrives by Friday" | **Synthesis-side reasoning** over `services[].estimated_days` (the tool takes no deadline param) | `AgentService` final pass |
| chain all of the above | **Agent loop** — sequence tools, react to results, no hardcoded order | `app/services/agent_service.py` (new) |
| "actually create it" (follow-up) | **Propose** a draft now (day-1); persist on explicit confirm (**fast-follow**) | `propose_shipment` (in-API) → Java `POST /shipments` |
| show the reasoning | **decision_path + step trace** in the response | reuse `decisions[]` convention |

### Why this use case is the right one
- It is **genuinely multi-step**: address must be validated *before* a quote is trustworthy,
  and the policy answer is independent — an agent that plans beats fixed `if` branches.
- It exercises **self-correction**: a bad address normalizes and re-validates; a thin RAG
  pass reformulates (both already modeled in `iterative.py`).
- It ends at a **proposed** state-changing action, which lets us demonstrate the safety model
  (§3.1) — and the create itself is the natural fast-follow once the read-only loop is proven.

---

## 3. Architecture

```
 POST /api/v1/agent/run                                   ShipSmart-API
 { query, context?, confirm?, proposed_action? }
        │
        ▼
 ┌─────────────────── AgentService.run() ───────────────────────────────────┐
 │  reasoning_llm = llm_router.for_task(TASK_REASONING)   # the tool-caller  │
 │  tools = registry.list_schemas()                                          │
 │        + retrieve_rag        (model-invoked Iterative RAG)                │
 │        + propose_shipment    (read-only: assembles a draft)              │
 │  ┌──────────── loop (agent_max_steps, default 5) ───────────────────┐    │
 │  │ 1. out = reasoning_llm.complete_with_tools(messages, tools)        │    │
 │  │      └─ NotImplementedError (Echo/Gemini/Llama, no native tools)   │    │
 │  │         → fall back to select_tool_with_llm (single-pass, §3.2)    │    │
 │  │ 2. final?  → break                                                 │    │
 │  │ 3. tool_use → dispatch:                                            │    │
 │  │      retrieve_rag     → iterative_rag(...)     (RAG layer) ─────────┼─┐ │
 │  │      validate_address → registry.execute(...)  (MCP, read-only)    │ │ │
 │  │      get_quote_preview→ registry.execute(...)  (MCP, read-only)    │ │ │
 │  │      propose_shipment → assemble CreateShipmentRequest draft       │ │ │
 │  │ 4. append observation → messages (trim to budget); tag decisions[] │─┘ │
 │  └─────────────────────────────────────────────────────────────────────┘  │
 │  final answer: llm_router.execute(TASK_SYNTHESIS, assembled.messages)     │
 │                (failover + guardrails; pattern reused from iterative.py)   │
 └────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 AgentResult { answer, steps[], tools_used[], sources[], decisions[],
               provider, proposed_action?, requires_confirmation }

 ── fast-follow (NOT day-1) ─────────────────────────────────────────────────
 confirm=true + proposed_action →  java_client.create_shipment(body,
   auth_token=<forwarded user JWT>, idempotency_key=<fresh>)
   → POST /api/v1/shipments → 201 ShipmentSummaryDto       (ShipSmart-Orchestrator)
```

### 3.1 The safety boundary (do not violate)
ShipSmart-MCP **refuses to start** if any registered tool "writes state, moves money, or
books anything" (`ShipSmart-MCP/app/main.py:55-69`). The Concierge keeps that invariant:

- **All MCP tools stay read-only** (`validate_address`, `get_quote_preview`).
- **Day-1 the agent is read-only end to end.** The only "action" tool, `propose_shipment`,
  is an **in-API pseudo-tool** (never registered on MCP) that just assembles a
  `CreateShipmentRequest` draft from the accumulated observations and returns it as
  `proposed_action`. It persists nothing. The LLM never holds a tool that can mutate state.
- **Persistence (fast-follow) is a deliberate, separately-authorized action outside the
  loop:** the client re-calls `/agent/run` with `confirm=true` + the echoed `proposed_action`;
  the API forwards the user's bearer token to Java `POST /api/v1/shipments` with a fresh
  `Idempotency-Key`. Ownership is JWT-scoped in Java; the confirm is an explicit human
  decision. **No bespoke confirmation token is needed** — user-JWT scoping + explicit confirm
  + idempotency are the gate. (Optional hardening: bind the proposal server-side in a
  `TTLCache` so the client can't tamper with the body; not required for v1.)

### 3.2 LLM handles — which client does what (correctness-critical)
- **Tool-calling loop:** `llm_router.for_task(TASK_REASONING)` (`router.py:65`) returns a
  *single* `LLMClient`; the loop calls the **new** `complete_with_tools(messages, tools)` on
  it. **Tool-calling does not get the failover chain in v1** — `LLMRouter.execute()`
  (`router.py:85`) only wraps `complete()` (text). Failover-with-tools is a named fast-follow.
- **Final grounded answer:** goes through `llm_router.execute(TASK_SYNTHESIS,
  assembled.messages, request_id=...)` — failover **and** guardrails — reusing the exact
  `assemble(...) → router.execute(...)` pattern at `agentic.py:168-184` (→ `iterative.py`).
- **Context budget:** the message list grows each step; trim it to
  `settings.llm_max_context_tokens` using `app/llm/budget.py` before each call so a long loop
  can't overflow the context window.
- **Non-native providers:** `complete_with_tools` default raises `NotImplementedError`; the
  loop catches it and falls back to `select_tool_with_llm` (`orchestration_service.py:62`) —
  a single-pass text selection that works with Echo/Gemini/Llama. This is the keyless default
  (see §3.3).

### 3.3 Scripted stub provider — exercising the real loop, keyless
- The local stack runs **keyless** (`run-stack.sh` sets `LLM_PROVIDER=` etc.), so the
  reasoning client is `EchoClient`, which has **no native tool-calling**. A live e2e therefore
  cannot show the model *choosing* tools — unaided it would only ever hit the §3.2 text
  fallback, never a genuine multi-tool plan.
- Add a deterministic **`ScriptedToolCallingClient`** in the API, modeled on ShipSmart-MCP's
  mock shipping provider (`SHIPPING_PROVIDER=mock`). Its `complete_with_tools` emits a fixed
  sequence for the canonical query: `retrieve_rag` → `validate_address` → `get_quote_preview`
  → final. Selected by env (`LLM_PROVIDER_REASONING=scripted`), gated to non-production.
- `run-stack.sh` sets that env so `e2e/test_agent_e2e.py` exercises the **genuine** multi-step
  loop deterministically, with no API keys — consistent with the stack's "self-contained, no
  real keys" design. Unit tests still mock `complete_with_tools` directly.

---

## 4. Work breakdown

Ordered so each step is independently testable and the existing suites stay green
(today: MCP 93 · API 246 · Orchestrator clean · Web 30 · Infra exit 0 · Test 26).

### 4.1 ShipSmart-API (all of the day-1 work)
1. **LLM tool-calling primitive** — `app/llm/client.py`
   - `ToolCallResult { kind: "final"|"tool_calls", text, calls[] }` + `async def
     complete_with_tools(messages, tools) -> ToolCallResult` on the `LLMClient` ABC, default
     raises `NotImplementedError`.
   - Implement for **`AnthropicClient`** (native `tool_use` blocks) and add the
     **`ScriptedToolCallingClient`** (§3.3). Echo/Gemini/Llama keep the default → §3.2 text
     fallback. **OpenAI parity is fast-follow.**
2. **AgentService** — `app/services/agent_service.py` (new)
   - `run_agent(query, context, *, registry, llm_router, embedding_provider, vector_store,
     max_steps=settings.agent_max_steps)`.
   - reasoning client = `llm_router.for_task(TASK_REASONING)`; tool surface =
     `registry.list_schemas()` + `RETRIEVE_RAG_SCHEMA` + `PROPOSE_SHIPMENT_SCHEMA`.
   - `_dispatch`: `retrieve_rag` → `iterative_rag(...)`; `validate_address`/`get_quote_preview`
     → reuse `execute_tool(...)` (input validation + 502 handling already there);
     `propose_shipment` → assemble a `CreateShipmentRequest` draft (no persistence).
   - Bounded by `max_steps`; trim messages to budget each step; on cap, force one final
     synthesis pass via `router.execute(TASK_SYNTHESIS, …)`.
   - Reuse guardrails (`assemble`) and `decisions[]` tags (`agent:plan`, `agent:step{n}`,
     `agent:tool:{name}`, `agent:propose`, `guardrail:blocked`).
3. **Route** — `app/api/routes/agent.py` (new), registered in `app/main.py`
   - `POST /api/v1/agent/run`. Request `{ query, context?, confirm?, proposed_action? }`
     (`confirm`/`proposed_action` are accepted but **inert on day-1** — wired in fast-follow).
   - Response schema in `app/schemas/agent.py`: `answer, steps[], tools_used[], sources[],
     decisions[], provider, proposed_action?, requires_confirmation`.
   - Rate-limit with the slowapi `@limiter.limit(settings.rate_limit_agent)` + `request:
     Request` pattern (`app/core/rate_limit.py`); 503 when `app.state.tool_registry is None`
     (same as orchestration). Surface `agent_enabled` on `/ready` (`health.py` already
     reports `rag_mode`).
4. **Iterative-RAG rename** (pure rename — §1a). Exact touch-points:
   - `app/core/config.py:135-137` — `rag_mode` value `"agentic"`→`"iterative"`;
     `rag_agentic_max_steps`→`rag_iterative_max_steps`; fix the `rag_query_log` comment.
   - `app/api/routes/health.py:22,49` — the reported `rag_mode` string.
   - `app/services/rag_service.py:84-90` — the `== "agentic"` branch + the
     `from app.rag.agentic import agentic_rag, make_retriever` import.
   - `app/rag/agentic.py` → `app/rag/iterative.py`; `agentic_rag`→`iterative_rag`,
     `AgenticResult`→`IterativeRagResult`, decision tags `agentic:*`→`rag:*` (keep
     `_UNCOVERED_REFUSAL`). Move the existing tests in lockstep.
   - **Back-compat:** accept legacy `rag_mode=agentic` and `RAG_AGENTIC_MAX_STEPS` via a
     one-line alias for one release so existing `.env` files and the run-stack envs don't break.
5. **Config** — `app/core/config.py` + `.env.example`: `agent_enabled: bool = True`,
   `agent_max_steps: int = 5`, `rate_limit_agent: str = "10/minute"` (slowapi format, matches
   the existing `rate_limit_*`).

### 4.2 ShipSmart-MCP — no change for day-1
`validate_address` + `get_quote_preview` already cover the use case, and the read-only
invariant must hold. *(Optional, later: a read-only `check_service_availability` tool — still
read-only, registers cleanly.)*

### 4.3 ShipSmart-Orchestrator (Java) — no change; used only in fast-follow
The write endpoint already exists and is tested: `POST /api/v1/shipments` (`@Idempotent`,
JWT-scoped, rate-limited) → `201 ShipmentSummaryDto` (`ShipmentController.java:65-73`,
`dto/CreateShipmentRequest.java`). Day-1 does not call it; the create fast-follow does.

### 4.4 ShipSmart-Web — fast-follow (not day-1)
A **ConciergePanel** calling `/agent/run` and rendering the step trace
(`tools_used[]`/`decisions[]`/`sources[]`), with a "Create shipment?" confirm button. Lands
with the create fast-follow so the panel has a write path to confirm.

### 4.5 ShipSmart-Test
See the companion plan:
[`ShipSmart-Test/docs/agent-concierge-e2e-plan.md`](../../ShipSmart-Test/docs/agent-concierge-e2e-plan.md).
Day-1: stub-driven live e2e (canonical flow, step-cap, refusal, text-fallback, MCP-read-only
regression) **propose-only**, plus the `proposed_action` ↔ Java `CreateShipmentRequest`
contract. The propose→confirm→persist e2e lands with the create fast-follow.

---

## 5. Phasing for the day (read-only Concierge)

Each phase leaves all suites green and is mergeable on its own.

- **Phase 1 — LLM primitive.** `ToolCallResult` + `complete_with_tools` (`AnthropicClient`) +
  `ScriptedToolCallingClient`; unit tests with a mocked SDK `tool_use` response. *No
  route/behavior change.*
- **Phase 2 — AgentService loop.** `run_agent` + `_dispatch` + budget-trim + forced-final via
  `execute(TASK_SYNTHESIS)`; unit tests: single-tool, multi-tool chain, step-cap, tool-error
  recovery, pure-RAG, refusal, **text-fallback (no native tools)**. Mock the LLM + registry
  (the suite already mocks both).
- **Phase 3 — Iterative-RAG rename + `retrieve_rag` wiring.** Apply §4.1(4); expose
  `iterative_rag` as the `retrieve_rag` tool; verify reformulate/refuse tags flow into
  `decisions[]` as `rag:*`.
- **Phase 4 — Route + schema + rate limit.** `POST /api/v1/agent/run`, `app/schemas/agent.py`,
  register in `main.py`, `rate_limit_agent`, `/ready` surfacing, OpenAPI docs.
- **Phase 5 — Config + observability.** `agent_*` settings + `.env.example`; per-step
  structured logs (tool, latency, success) under the correlation ID; expose `steps[]`.
- **Phase 6 — Cross-repo tests.** Wire `LLM_PROVIDER_REASONING=scripted` into `run-stack.sh`;
  add `e2e/test_agent_e2e.py` (stub-driven) + the `proposed_action` contract. Flip this doc's
  status to "Implemented (read-only)".

---

## 6. Risks & guardrails

| Risk | Mitigation |
|---|---|
| Runaway loop / cost | Hard `agent_max_steps`; reuse the per-request token budget (`app/llm/budget.py`) and `TTLCache` for idempotent tool results |
| Context-window overflow over many steps | Trim the growing message list to `llm_max_context_tokens` each step (§3.2) |
| Latency (N sequential LLM calls) | Cap steps; keep `/orchestration/run` as the fast single-shot path; stream the final answer in a fast-follow |
| Accidental writes by the model | **MCP stays read-only**; day-1 the agent only ever *proposes* (`propose_shipment` persists nothing); the create path is out-of-loop, JWT-scoped, explicit-confirm (§3.1) |
| Tool-calling has no failover in v1 | By design (§3.2); the **final synthesis** still uses the failover chain via `execute(TASK_SYNTHESIS)`; failover-with-tools is a fast-follow |
| Hallucinated tool args | Reuse `RemoteTool.validate_input`; feed validation errors back into the loop as observations (already how `execute_tool` raises 422) |
| Ungrounded policy answers | Iterative RAG's deterministic refusal (`_UNCOVERED_REFUSAL`) when no chunk covers the question |
| Provider without native tools | `complete_with_tools` default → fall back to `select_tool_with_llm` text mode; the scripted stub (§3.3) gives the keyless e2e a real loop |
| MCP unavailable | Same 503 path as orchestration (`app.state.tool_registry is None`) |

---

## 7. Decisions & fast-follow

**Resolved (locked for this build):**
1. **Persistence path:** API-side create that **forwards the user's bearer token** to Java
   `POST /shipments` (reuses the advisor forward-auth pattern at `advisor.py:39-44, 147-172`),
   gated by explicit `confirm` + `Idempotency-Key`. No bespoke token. → **fast-follow.**
2. **Session memory:** out of day-1; stateless `/agent/run`. A `session_id`-keyed `TTLCache`
   of the running message list is a fast-follow.
3. **`agent_max_steps` = 5:** comfortable headroom for the 3-tool use case (allows one
   tool retry + a final pass) while bounding latency/cost. Configurable.
4. **Streaming:** single final response in v1; token streaming is a fast-follow.

**Fast-follow backlog (explicitly not day-1):** create/persist handshake (§3.1) + Web
ConciergePanel (§4.4) + the propose→confirm→persist e2e · OpenAI `complete_with_tools` parity ·
session memory · streaming · failover-with-tools.

**Still open (decide when starting the create fast-follow):** whether `propose_shipment` should
also pre-call `get_quote_preview`/`validate_address` to guarantee a complete draft, or rely on
the model having already gathered them earlier in the loop.
