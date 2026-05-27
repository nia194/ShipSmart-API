# Agent Integration — Design Proposal

**Status:** Draft for review (not yet implemented)
**Author:** _(fill in)_
**Date:** 2026-05-26
**Decision needed:** Approve/reject the concept, then choose **Routing Option A vs B** (see §6).

---

## 1. TL;DR

ShipSmart-API already has every building block an agent needs — a multi-provider
LLM layer, RAG retrieval, a remote MCP tool registry, and a task-based LLM router.
What it does **not** have is an **agentic loop**: today the LLM is used as a
*classifier* (pick one tool) and a *summarizer* (write the answer), but it never
*drives* a multi-step **reason → act → observe → repeat** process.

This document proposes adding that loop. It covers:
- why it matters for ShipSmart's use case (§3),
- exactly where the current code stops short of being agentic (§4),
- the proposed design and integration points (§5),
- **two routing options to choose between after review** (§6),
- a phased development plan (§7), and
- risks, guardrails, and testing (§8–9).

No code is changed by this document. It exists to be reviewed first.

---

## 2. Definitions

- **Single-shot orchestration (today):** one pass — select at most one tool, run
  it, return. The LLM never sees tool output and then decides to act again.
- **Agent (proposed):** a controller loop where the LLM, given the available
  tools and the observations so far, decides the *next* action on each turn —
  call a tool, retrieve more context, or finish — up to a bounded step limit.

The defining property of an agent is the **feedback loop**: the model's own output
(a tool call) produces an observation (the tool result) that is fed back into the
model to inform its next decision.

---

## 3. Why add an agent — the need

### 3.1 The limitation today
The current advisor flow (`shipping_advisor_service.py`) only runs a tool when the
*caller pre-supplies the exact parameters*, and it runs each tool **at most once**
via fixed `if` branches:

```python
# shipping_advisor_service.py — paraphrased
if all(k in context for k in ["origin_zip", "destination_zip", "weight_lbs"]):
    execute_tool("get_quote_preview", ...)   # runs once, only if params present
if all(k in context for k in ["street", "city", "state", "zip_code"]):
    execute_tool("validate_address", ...)    # runs once, only if params present
```

`orchestration_service.run_orchestration` is similar: regex (or an LLM classifier)
picks **one** tool, it runs **once**, done.

### 3.2 What an agent unlocks
Consider a realistic ShipSmart query:

> "Can I ship this 5lb box to 90210 cheaply, and is 123 Main St, Beverly Hills CA
> a valid delivery address?"

| Step | Single-shot today | Agent (proposed) |
|------|-------------------|------------------|
| Pick tool | Picks **one** tool only | Plans: validate address **and** get quote |
| Bad address | Returns failure as-is | Sees `is_valid=false`, re-tries with normalized address |
| Quote | Only if params pre-parsed | Extracts `weight=5lb`, `dest=90210` from the *sentence* and calls the tool |
| Synthesize | Answers from whatever ran | Answers from the **full** chain of observations |

Concretely, an agent adds:
1. **Multi-tool sequencing** — chain `validate_address` → `get_quote_preview`
   without hardcoding the order.
2. **Self-correction** — react to a failed/empty tool result and try again.
3. **Parameter extraction from natural language** — no need for the caller to
   pre-fill `context["origin_zip"]` etc.
4. **Iterative retrieval** — fetch more RAG context if the first pass was thin.

### 3.3 Why this is low-risk to add here
- The MCP tool layer (`mcp_client.py`) already exposes a clean
  `list_schemas()` / `get(name).execute(...)` interface — a ready-made agent
  tool surface.
- The LLM router already has a dedicated `TASK_REASONING` client
  (`llm/router.py`), separate from synthesis.
- `anthropic>=0.40` and `openai>=1.60` (already in `pyproject.toml`) both support
  **native function calling**, the proper substrate for an agent loop.

---

## 4. Where the current code stops short

| Capability | Where it lives today | Agentic? |
|------------|---------------------|----------|
| Tool registry / execution | `services/mcp_client.py` (`RemoteToolRegistry`) | ✅ reusable as-is |
| Tool selection | `services/orchestration_service.py` (regex + LLM classifier) | ❌ single tool, one pass |
| LLM completion | `llm/client.py` `LLMClient.complete()` → **text only** | ❌ no `tool_use` |
| Task routing | `llm/router.py` (`TASK_REASONING`, `TASK_SYNTHESIS`) | ✅ reusable |
| RAG retrieval | `rag/retrieval.py` `retrieve()` | ✅ reusable as a tool |
| Advisor flows | `services/shipping_advisor_service.py`, `tracking_advisor_service.py` | ❌ fixed `if` branches |
| App wiring | `main.py` lifespan → `app.state.tool_registry`, `app.state.llm_router` | ✅ reusable |

**The single missing primitive:** `LLMClient.complete()` returns a string. An agent
needs the model to be able to return *structured tool-call requests*. That is the
one capability we must add to the LLM layer.

---

## 5. Proposed design

### 5.1 Component overview

```
                 ┌──────────────────────────────────────────────┐
   POST /agent   │            AgentService.run()                 │
   (or /orch)    │                                               │
      │          │   ┌───────── loop (max_steps) ────────────┐   │
      ▼          │   │ 1. LLM.complete_with_tools(msgs,tools) │   │
  query + ctx ───┼──▶│ 2. model returns: tool_use | final     │   │
                 │   │ 3. if tool_use → registry.execute(...)  │   │
                 │   │ 4. append observation → messages        │   │
                 │   │ 5. repeat until final or step cap       │◀──┐
                 │   └─────────────────────────────────────────┘  │ │
                 │                       │                          │
                 │            tools = registry.list_schemas()       │
                 │            + a "retrieve_rag" pseudo-tool ────────┘
                 └──────────────────────────────────────────────┘
                              │
                              ▼
                  AgentResult { answer, steps[], tools_used[], sources[] }
```

### 5.2 Changes required

**(a) Extend the LLM abstraction — `app/llm/client.py`**

Add a new optional method to the `LLMClient` ABC with a safe default so existing
providers (Echo, Llama, Gemini-via-REST) don't break:

```python
class LLMClient(ABC):
    ...
    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolCallResult:
        """Return either final text or a list of requested tool calls.

        Default raises NotImplementedError; providers that support native
        function calling (Anthropic, OpenAI) override it.
        """
        raise NotImplementedError
```

- Implement for `AnthropicClient` (native `tool_use` blocks) and `OpenAIClient`
  (native `tools=` / `tool_calls`).
- For providers without native tool calling, the agent falls back to the
  existing text-based selection style (see §6 substrate note), or the route
  returns a clear 501/"agent unsupported for this provider" message.

A small `ToolCallResult` dataclass carries `{ kind: "final"|"tool_calls", text, calls[] }`.

**(b) New `AgentService` — `app/services/agent_service.py`**

A sibling to the existing advisor services. Owns the loop:

```python
async def run_agent(
    query: str,
    context: dict | None,
    *,
    registry: RemoteToolRegistry,
    llm_client: LLMClient,          # llm_router.for_task(TASK_REASONING)
    embedding_provider, vector_store,
    max_steps: int = 5,
) -> AgentResult:
    tools = registry.list_schemas() + [RETRIEVE_RAG_SCHEMA]
    messages = [system_prompt, user(query, context)]
    steps = []
    for _ in range(max_steps):
        out = await llm_client.complete_with_tools(messages, tools)
        if out.kind == "final":
            return AgentResult(answer=out.text, steps=steps, ...)
        for call in out.calls:
            obs = await _dispatch(call, registry, embedding_provider, vector_store)
            messages.append(tool_result_message(call, obs))
            steps.append(...)
    # step cap hit → force a final synthesis pass
    return AgentResult(answer=await _force_final(messages, llm_client), ...)
```

- `_dispatch` routes `retrieve_rag` → `rag.retrieval.retrieve(...)`, everything
  else → `registry.get(name).execute(...)`. Reuses existing error handling.
- Bounded by `max_steps` (config-driven) to cap latency and cost.

**(c) Wiring — `app/main.py`**

No new app.state needed: the agent reuses `app.state.tool_registry`,
`app.state.llm_router`, and `app.state.rag`. (If Option A is chosen, just register
the new router.)

**(d) Config — `app/core/config.py` + `.env.example`**

Add: `AGENT_MAX_STEPS` (default 5), `AGENT_ENABLED` (feature flag),
`RATE_LIMIT_AGENT` (reuse the slowapi pattern already used for orchestration).

### 5.3 What is reused unchanged
- `RemoteToolRegistry` / MCP transport
- `LLMRouter` and `TASK_REASONING`
- `rag.retrieval.retrieve`, embeddings, vector store
- `AppError` handling, rate limiter, request-logging middleware, correlation IDs

---

## 6. Routing — two options to decide between (post-review)

Both are fully viable. The recommendation is **Option A** for the first iteration,
but this is exactly the decision to make at review time.

### Option A — New `/agent` endpoint (additive)

Add `app/api/routes/agent.py` → `POST /api/v1/agent/run`, registered alongside the
existing routers in `main.py`. `run_orchestration` stays as the fast single-shot path.

```
POST /api/v1/orchestration/run   → unchanged, fast, deterministic single-shot
POST /api/v1/agent/run           → new, multi-step agent loop
```

| Pros | Cons |
|------|------|
| Lowest risk — nothing existing changes | Two endpoints to document/maintain |
| Easy A/B comparison of cost & quality | Caller must choose which to hit |
| Fast path stays cheap for simple queries | Some conceptual overlap between the two |
| Clean rollback (just don't call it) | |

### Option B — Replace orchestration (unified)

Make the agent loop the engine behind `POST /api/v1/orchestration/run`. Keep the
regex fast-path as a *step-0 shortcut* inside the loop (if a rule matches with all
params present, answer in one pass; otherwise enter the loop).

| Pros | Cons |
|------|------|
| One endpoint, one mental model | Higher risk — changes a shipped route's behavior |
| Regex stays as an optimization, not a separate path | Latency/cost rise for queries that used to be single-shot |
| Naturally smarter over time | Harder rollback; needs the flag (`AGENT_ENABLED`) to fall back |
| | Existing orchestration tests must be revisited |

### Decision criteria
- If we want to **measure** agent value before committing → **Option A**.
- If we're confident and want a **single clean surface** → **Option B** behind a flag.
- Hybrid: ship **Option A** now, migrate to **Option B** once metrics justify it.

### Tool-calling substrate (orthogonal sub-decision)
- **Native function calling** (recommended): robust, standard; requires the
  `complete_with_tools` work in §5.2(a). Anthropic/OpenAI already support it.
- **Text-based selection**: loop on top of the existing string-parsing approach
  (`select_tool_with_llm`). Less new code; works with every provider including
  Echo/Llama; more brittle parsing. Good fallback for non-native providers.

Recommendation: **native for Anthropic/OpenAI, text-based fallback** for the rest.

---

## 7. Development plan (phased)

Each phase is independently reviewable/mergeable.

**Phase 0 — Spec & contract (this doc + sign-off)**
- Approve concept, pick Option A/B, confirm substrate. _Deliverable: this file._

**Phase 1 — LLM tool-calling primitive**
- Add `ToolCallResult` + `LLMClient.complete_with_tools` (default raises).
- Implement for `AnthropicClient`, then `OpenAIClient`.
- Unit tests with a mocked SDK response containing a `tool_use` block.
- _No behavior change to existing routes._

**Phase 2 — AgentService loop**
- Implement `run_agent` with `max_steps`, `_dispatch`, forced-final fallback.
- Reuse `registry`, RAG, `TASK_REASONING`.
- Tests: single-tool, multi-tool chain, step-cap hit, tool-error recovery,
  no-tool (pure RAG) path. Mock the LLM and the registry.

**Phase 3 — Route exposure**
- **Option A:** add `agent.py` route + register in `main.py`.
- **Option B:** thread the loop into `orchestration.run_workflow` behind
  `AGENT_ENABLED`, keeping the regex fast-path.
- Rate limit + request schema + OpenAPI docs.

**Phase 4 — Config, observability, docs**
- `AGENT_MAX_STEPS`, `AGENT_ENABLED`, `RATE_LIMIT_AGENT` in config + `.env.example`.
- Per-step structured logging (tool name, latency, success) under the existing
  correlation ID; expose `steps[]` in the response for debuggability.
- README section; update this doc's status to "Implemented".

**Phase 5 — Evaluation (decides A→B migration)**
- Compare agent vs single-shot on a fixed query set: answer quality, tool-call
  count, p50/p95 latency, token cost. Feed results back into the A-vs-B call.

---

## 8. Risks & guardrails

| Risk | Mitigation |
|------|------------|
| **Unbounded loops / runaway cost** | Hard `max_steps` cap; per-request token budget; cache (reuse `TTLCache`) |
| **Latency** (N sequential LLM calls) | Cap steps; keep regex fast-path; stream final answer later |
| **Provider lock-in** | `complete_with_tools` on the existing abstraction; text fallback for non-native providers |
| **Tool misuse / hallucinated args** | Reuse `RemoteTool.validate_input`; surface validation errors back into the loop as observations |
| **Behavior regression (Option B)** | Feature flag `AGENT_ENABLED`; keep single-shot reachable; revisit orchestration tests |
| **MCP unavailable** | Same 503 path as today (`app.state.tool_registry is None`) |
| **Cost of LLM tool-selection** | Reuse `_tool_selection_cache` pattern; cache idempotent tool results |

---

## 9. Testing strategy
- **Unit:** mock `complete_with_tools` to script tool-call sequences; assert the
  loop dispatches, observes, and terminates correctly.
- **Failure scenarios:** tool returns error, empty result, step cap reached,
  provider lacks native tool calling (extends `tests/test_failure_scenarios.py`).
- **Integration:** end-to-end with a fake MCP transport (the codebase already
  supports `httpx` transport injection in `McpClient`) — extends
  `tests/test_integration_full_flow.py`.
- **Regression:** existing orchestration/advisor tests must stay green (critical
  for Option B).

---

## 10. Open questions for reviewers
1. **Option A or B** for the first release? (See §6.)
2. Native-only, or include the text-based fallback for Llama/Echo/Gemini-REST?
3. Default `AGENT_MAX_STEPS` — is 5 right for our latency budget?
4. Should `retrieve_rag` be a first-class agent tool, or should RAG always run
   once up front (as today) and only tools loop?
5. Do we want streaming responses in v1, or is a single final answer acceptable?
```
