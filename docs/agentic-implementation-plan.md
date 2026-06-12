# ShipSmart Concierge — Agentic RAG (Conditional Re-Retrieval) Implementation Plan

**Status:** Implemented (read-only, agentic re-retrieval)
**Date:** 2026-06-12
**Scope:** ShipSmart-API. Web/Orchestrator/MCP untouched.
**Supersedes (for the agentic-RAG slice):** [`agent-concierge-implementation-plan.md`](./agent-concierge-implementation-plan.md)
— the prior concierge record is preserved unchanged; this document is the self-contained
upgrade that incorporates the agentic-RAG work.

---

## 1. TL;DR

The Concierge agent already plans, calls read-only MCP tools (`validate_address`,
`get_quote_preview`), and retrieves from the knowledge base through a `retrieve_rag` tool —
a model-driven reason → act → observe loop (`app/services/agent_service.py`) grounded through
the same guardrailed assembler + synthesis failover chain as every other LLM path.

But one thing was still **"RAG as a tool," not agentic RAG**: the agent called `retrieve_rag`
once and used whatever came back, with **no visibility into how good the retrieval was**. A
compound question whose first retrieval returned thin coverage got answered from thin coverage.

This upgrade closes that gap with the smallest justified change: the agent now **sees the
coverage of what it retrieved** and, **only when coverage is weak**, retrieves **again** with a
different, model-written query — decomposing a hard, compound question into sub-areas. Most
queries stay single-shot; re-retrieval is a **conditional fallback**, bounded and auditable.

Three additions, each independently testable, all additive (the 266-test baseline stayed green
throughout; the suite is now **280 passing**):

1. **A coverage signal** surfaced in the `retrieve_rag` observation (`top_score`, `covered`,
   `chunk_count`) so the model can reason over retrieval quality.
2. **Conditional, bounded re-retrieval** in `run_agent`: a different query only after weak
   coverage, capped by a new `agent_max_retrievals` (default 2), guarded against
   identical-query loops, with honest flagging of any sub-area left uncovered.
3. **A comparison eval** (`scripts/agentic_eval.py`) that runs hard compound queries through
   single-shot vs the agentic loop and reports, with numbers, that the loop earns its cost.

---

## 2. Changes from the concierge plan — and why they're justified

The concierge plan introduced the agent loop and exposed retrieval as a `retrieve_rag` tool.
That made retrieval **model-invoked**, which is necessary but not sufficient for the label
"agentic RAG." The agent chose *when* to retrieve, but never assessed *what it got back*.

**The justification, stated as a constraint:** agentic RAG must **require conditional
re-retrieval driven by a coverage signal** — it is *not* an always-on multi-pass loop. The
distinction matters:

- An **always-on** re-retrieval loop spends extra LLM turns and retrievals on *every* query,
  including the simple ones that were already well covered. That is cost without benefit, and
  it is exactly what the deterministic `app/rag/agentic.py` loop already does cheaply and
  without an LLM in its control flow.
- A **conditional** loop spends the extra cost **only when the coverage signal says the first
  retrieval was thin** — which is precisely the class of question (hard, compound,
  multi-policy) where a second, decomposed retrieval changes a partial refusal into a grounded
  answer. The cost is incurred where, and only where, it buys grounding.

So the earned version of "agentic RAG" here is: **the model observes retrieval quality and
decides to retry** — under a hard budget, never on identical queries, and honest about
sub-areas it still could not cover. Simple questions stay single-shot and behave exactly as
before.

> **Note on the codebase vs the prior plan.** The concierge plan proposed renaming
> `app/rag/agentic.py` → `iterative.py`, adding `iterative_rag` as the `retrieve_rag` backend,
> and a `propose_shipment` pseudo-tool. In the **shipped** code, `retrieve_rag` wraps the
> single-shot `retrieve_auto` (not the multi-pass deterministic loop), the deterministic loop
> is still `app/rag/agentic.py` with `_covered()` as its grounding-threshold notion, and there
> is no `propose_shipment`. This plan describes what is actually true now. The agentic
> re-retrieval added here lives entirely in the **agent layer**, reusing the deterministic
> layer's `_covered()` threshold so the two agree on what "covered" means — it does **not**
> change the deterministic layer's behavior or control flow.

---

## 3. The justifying use case (drives the eval)

**Single-shot (stays single-shot, no loop):**

> *"Can I ship a power bank?"*  → one retrieval, good coverage → answer.

A power bank is a single, well-covered policy area (lithium battery). The first retrieval
clears the grounding threshold; the agent sees `covered=true` and answers. No second retrieval,
no extra LLM turn — identical to today.

**Agentic (requires re-retrieval):**

> *"I'm sending a drone to Germany — any restrictions I should know about?"*

A drone spans multiple policy areas at once — **lithium battery** + **electronics export** +
**Germany import**. A single retrieval on "drone Germany" returns thin coverage: no single
chunk covers the compound question. The agent must **see the weak coverage**, decompose, and
retrieve each sub-area:

```
retrieve "drone Germany"                  → weak   (top_score below threshold → triggers re-retrieval)
retrieve "lithium battery intl shipping"  → strong
retrieve "electronics import Germany"     → strong
→ synthesize a grounded answer, and HONESTLY flag any sub-area still uncovered
```

This is genuinely multi-step and self-correcting, and it is the exact shape the eval (§6)
measures.

---

## 4. The coverage signal (Phase 1)

The deterministic RAG layer already computes per-chunk similarity scores and a
coverage/threshold notion: `app/rag/agentic.py:_covered()` = "at least one retrieved chunk with
`score > 0`". We **read** that — we do not change it — and surface it as an **observable signal**
in the `retrieve_rag` tool result the model sees.

`app/services/agent_service.py`:

```python
@dataclass
class CoverageSignal:
    top_score: float     # highest similarity among retrieved chunks
    covered: bool        # did anything clear the grounding threshold (reuses _covered)
    chunk_count: int     # how many chunks came back

def coverage_of(results) -> CoverageSignal: ...
```

The `retrieve_rag` observation now **leads with a coverage line** before the chunk bodies, so
the model reasons over quality first:

```
coverage: top_score=0.182 covered=false chunks=3
[hazmat score=0.182] …
[customs score=0.041] …
```

`covered` reuses the deterministic layer's `_covered()` exactly, so the agent and the pure RAG
loop share one definition of "grounded enough." The signal is intentionally **generic** (a
future tool could surface the same shape), but not over-built: it is a plain dataclass plus one
helper, RAG-focused, with no hard-coded assumptions about the query.

**Tested:** `tests/test_agent_coverage.py` — `coverage_of` for strong / empty / below-threshold
results, the observation rendering, and the live `retrieve_rag` dispatch path for both a
well-covered and a poorly-covered query.

---

## 5. Bounded, conditional re-retrieval (Phase 2)

The existing loop in `run_agent` is **extended, not rewritten**. When the model calls
`retrieve_rag`, the agent layer applies a small policy before/after the retrieval:

- **Conditional retry.** The agent may call `retrieve_rag` more than once per run, with a
  **different, model-written query**, after a prior retrieval returned weak coverage. The model
  decides (it sees the coverage signal); the agent layer enforces the bounds. A second
  retrieval that follows any weak-coverage retrieval is tagged
  `agent:retrieve:reformulate` — the justified reformulation (e.g. decomposing the drone
  question into lithium / electronics sub-areas).
- **Degenerate-retry guard.** A retrieval whose (normalized) query is **unchanged** from a
  prior retrieval this run is **rejected** — no identical-query loops. The rejection is not
  silently dropped: a clear observation (*"this query is unchanged… reformulate or proceed"*)
  is fed back so the model can recover, tagged `agent:retrieve:rejected`.
- **Hard budget.** A new config **`agent_max_retrievals` (default 2)**, **separate from
  `agent_max_steps`**, bounds total `retrieve_rag` calls per run independent of overall steps.
  When the cap is hit, no further retrieval runs; the agent is told to proceed to synthesis,
  tagged `agent:retrieve:capped`. Setting `agent_max_retrievals=1` reproduces pure single-shot
  retrieval — the config gate that keeps the old behavior available.
- **Honest gaps.** If, after retrying, the agent's most recent retrieval is still weak (a
  sub-area it could not cover), the run is tagged `agent:retrieve:uncovered` and the synthesis
  prompt instructs the model to say so rather than guess.

**Decision tags** (auditable in `decisions[]`): `agent:retrieve:{n}` per executed retrieval,
`agent:retrieve:reformulate` on a weak-coverage retry, `agent:retrieve:rejected` on a degenerate
retry, `agent:retrieve:capped` at the budget, `agent:retrieve:uncovered` for an honest gap. The
backward-compatible `agent:tool:retrieve_rag` tag is still emitted for every executed retrieval.

**Single-shot path unchanged.** A query whose first retrieval is well-covered triggers no
re-retrieval, no reformulate/rejected/capped/uncovered tag, and no extra LLM turn — verified by
a regression test. All model-driven decisions live in the agent layer; the deterministic RAG
layer stays pure (no LLM in its control flow). Input validation, the token-budget trim,
guardrails, and the synthesis failover chain (`router.execute(TASK_SYNTHESIS)`) are reused
exactly as the current loop does. MCP tools stay read-only; the agent only reads, never persists.

**Tested:** `tests/test_agent_reretrieval.py` — weak coverage triggers one re-retrieval;
identical-query retry rejected; `agent_max_retrievals` cap enforced; well-covered query stays
single-shot (regression); uncovered-after-retries flags the honest gap.

---

## 6. Comparison eval — does the loop earn its cost? (Phase 3)

`scripts/agentic_eval.py` (in the runnable-script style of `scripts/perf_check.py`) runs a small
set of hard, compound queries (the drone-to-Germany class) through **both** strategies:

- **single-shot** — one broad retrieval, then answer; and
- **the agentic loop** — retrieve → assess coverage → reformulate → retrieve sub-areas →
  synthesize.

It is driven **deterministically and keyless** via `ScriptedToolCallingClient` (no API keys, no
network): one script re-retrieves on weak coverage, one does not. Coverage is forced with
fixed-vector embeddings so the numbers are reproducible. Per query it measures coverage **before**
vs **after** re-retrieval, whether the final answer moved from a partial refusal to a grounded
answer, and the **added** retrieval/step count (the cost).

### Headline numbers

```
query                         cover@1     cover@N      grounded  +retr  +steps
------------------------------------------------------------------------------
drone -> Germany             NO  0.00   yes 1.00    refuse>grnd      2       2
hoverboard -> Germany        NO  0.00   yes 1.00    refuse>grnd      2       2
radio transmitter abroad     NO  0.00   yes 1.00    refuse>grnd      2       2
lithium battery (control)    yes 1.00   yes 1.00            n/a      0       0

hard compound queries               : 3
grounding improved (refusal->grounded): 3/3
added cost (extra retrievals)       : +6 retrievals, +6 agent steps total
honestly-flagged uncovered sub-areas: 1
control query stayed single-shot    : yes
```

**Read:** on all **3/3** hard compound queries the agentic loop lifted the answer from a partial
refusal (weak coverage, `top_score 0.00`, nothing clears the threshold) to a **grounded** answer
(`top_score 1.00`, covered) — at a cost of **+2 retrievals / +2 agent steps each** (**+6 / +6**
total). The **control** "lithium battery shipping" query was already covered on the first pass,
so the loop **did not fire** — zero extra cost, single-shot preserved. One sub-area (a radio
transmitter's frequency-license rules, deliberately absent from the KB) stayed uncovered after
retrying and was **honestly flagged** rather than hallucinated.

So we can state with numbers: **agentic re-retrieval improves grounding on exactly the hard
queries, at a bounded, proportional cost, and adds nothing on simple ones.**

**Tested:** `tests/test_agentic_eval.py` asserts the eval runs end to end, that every hard query
improves refusal→grounded with at least one added retrieval and an `agent:retrieve:reformulate`
tag, and that the control query spends zero extra retrievals.

---

## 7. Architecture (where the new policy sits)

```
 POST /api/v1/agent/run  { query, context? }
        │
        ▼
 ┌─────────────────── run_agent() loop (agent_max_steps) ────────────────────┐
 │ out = reasoning_llm.complete_with_tools(messages, tools)                   │
 │   └─ NotImplementedError (no native tools) → single-pass text fallback     │
 │ final? → break                                                             │
 │ tool_use → dispatch:                                                       │
 │   retrieve_rag → retrieve_auto(...)  ── coverage_of(...) ──┐               │
 │                                                            │ NEW: the      │
 │     • degenerate query?  → agent:retrieve:rejected         │ coverage      │
 │     • retrievals == cap? → agent:retrieve:capped           │ signal +      │
 │     • else execute, tag agent:retrieve:{n}                 │ bounded       │
 │       (after weak coverage → agent:retrieve:reformulate)   │ conditional   │
 │   validate_address / get_quote_preview → execute_tool(...) │ re-retrieval  │
 │ append observation (incl. coverage line) → messages; trim ─┘ (agent layer) │
 │ after loop: still-weak last retrieval → agent:retrieve:uncovered           │
 │ final answer: router.execute(TASK_SYNTHESIS, assemble(...))  (guardrails + │
 │               failover, reused unchanged)                                  │
 └────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 AgentResult { answer, steps[], tools_used[], sources[], decisions[], provider }
```

The deterministic `app/rag/agentic.py` loop is untouched and still available for the RAG/advisor
paths via `RAG_MODE`. The agentic re-retrieval added here is purely in the agent layer and
reuses that layer's `_covered()` grounding threshold.

---

## 8. Safety / invariants (held)

- **MCP tools stay read-only.** The agent only reads, never persists.
- **Re-retrieval is bounded** by `agent_max_retrievals`; the degenerate-query guard prevents
  identical-query loops. No unbounded loops.
- **The deterministic RAG layer stays pure** — no LLM in its control flow. All model-driven
  decisions live in the agent layer.
- **Reuse, not reinvention:** existing input validation, the token-budget trim
  (`app/llm/budget.py`), guardrails (`assemble`), and the synthesis failover chain
  (`router.execute(TASK_SYNTHESIS)`) are reused exactly as the current loop does.
- **General where reasonable, not over-engineered:** the coverage signal + "assess result →
  decide to retry" pattern is a small, reusable shape (a future tool could surface the same
  `CoverageSignal`), but the implementation is a clean, RAG-focused addition with no hard-coded
  query assumptions.

---

## 9. Considered and deliberately excluded

Agentic RAG (conditional re-retrieval) is scoped to the **advisor / agent path only** — the
free-text, retrieval-grounded experience. It is deliberately **not** applied to:

- **The recommendation flow** (`/advisor/recommendation`). It ranks **structured** service data
  with **no retrieval** at all — there is nothing to re-retrieve and no coverage signal to act
  on. Adding a retrieval loop there would be cost with no grounding to improve.
- **Transactional paths** (quotes, shipment create). These are **deterministic, no-LLM** flows
  with their own validation and idempotency. A model-driven retrieval loop has no place in a
  money-moving or state-changing path.

**Future fast-follow (noted, not built now):** a **proactive pre-shipment compliance check** —
before a shipment is proposed/created, run the destination + contents through the same coverage
signal to surface restriction gaps up front (e.g. "this lithium item to Germany needs X"),
rather than only answering when asked. This reuses the coverage signal and the bounded
re-retrieval policy from this work, but it touches the transactional path's UX and authorization
boundary, so it is a deliberate, separately-scoped follow-up — not part of this change.

---

## 10. Config & file summary

| Change | Where |
|---|---|
| `agent_max_retrievals: int = 2` (separate from `agent_max_steps`) | `app/core/config.py` |
| `CoverageSignal`, `coverage_of`, coverage-leading observation, coverage in `_dispatch` | `app/services/agent_service.py` |
| Bounded conditional re-retrieval + decision tags in `run_agent` (`max_retrievals` param) | `app/services/agent_service.py` |
| Coverage-signal unit tests | `tests/test_agent_coverage.py` |
| Re-retrieval behavior tests (trigger / reject / cap / single-shot / honest-gap) | `tests/test_agent_reretrieval.py` |
| Comparison eval (single-shot vs agentic) | `scripts/agentic_eval.py` |
| Eval test (runs + improves hard queries) | `tests/test_agentic_eval.py` |

**Suite:** 266 → **280 passing** (additive). Single-shot agent behavior is unchanged for queries
that don't trigger re-retrieval.
