"""End-to-end tests for the UC3 workflow orchestrator + engine.

Hermetic + keyless: default mock domain adapters, an EchoClient LLM router, a
deterministic FixedEmbedding, and an empty InMemoryVectorStore. Asserts the full
stage graph runs, the decision trail is in deterministic order, and the parallel
fork merges reproducibly.
"""

from __future__ import annotations

from app.domain.adapters import default_providers
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore
from app.workflow.engine import StateMachineEngine
from app.workflow.orchestrator import DurableWorkflow, WorkflowDeps
from app.workflow.state import WorkflowState


class _FixedEmbedding(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    @property
    def dimensions(self) -> int:
        return 3


def _deps() -> WorkflowDeps:
    echo = EchoClient()
    return WorkflowDeps(
        providers=default_providers(),
        llm_router=LLMRouter(
            clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
            fallback=echo,
        ),
        embedding_provider=_FixedEmbedding(),
        vector_store=InMemoryVectorStore(),
    )


def _state(origin="US", destination="DE", desc="a 20000mAh power bank") -> WorkflowState:
    return WorkflowState(
        workflow_id="wf-test",
        origin_country=origin, destination_country=destination,
        declared_value_usd=1000.0, weight_lbs=5.0, description=desc,
    )


def _index(decisions: list[str], needle: str) -> int:
    return next(i for i, d in enumerate(decisions) if d == needle or d.startswith(needle))


# ── full run ──────────────────────────────────────────────────────────────────


async def test_process_runs_all_stages_to_completion():
    wf = DurableWorkflow(engine=StateMachineEngine(), deps=_deps())
    state = await wf.process(_state())

    assert state.status == "completed"
    assert state.hs_code == "8507.60"          # power bank
    assert state.hs_title
    assert state.landed_cost is not None
    assert state.carrier_quotes and state.recommended_carrier is not None
    assert state.compliance is not None and state.compliance.verdict
    # International US -> DE → packing list + commercial invoice + CN23.
    assert len(state.documents) == 3


async def test_decision_trail_is_in_deterministic_stage_order():
    wf = DurableWorkflow(engine=StateMachineEngine(), deps=_deps())
    d = (await wf.process(_state())).decisions

    assert d[0] == "workflow:start"
    assert d[-1] == "workflow:complete"
    # classify → landed_cost → routing → compliance → docs, in order.
    order = [
        _index(d, "workflow:classify:"),
        _index(d, "workflow:landed_cost:"),
        _index(d, "workflow:routing:"),
        _index(d, "compliance:plan"),
        _index(d, "workflow:docs:generated:"),
    ]
    assert order == sorted(order)


async def test_process_is_reproducible():
    runs = []
    for _ in range(2):
        wf = DurableWorkflow(engine=StateMachineEngine(), deps=_deps())
        runs.append((await wf.process(_state())).decisions)
    assert runs[0] == runs[1]


async def test_domestic_shipment_single_document():
    wf = DurableWorkflow(engine=StateMachineEngine(), deps=_deps())
    state = await wf.process(_state(origin="US", destination="US"))
    assert [doc.doc_type for doc in state.documents] == ["packing_list"]


async def test_explicit_compliance_off_skips_stage_and_completes():
    deps = WorkflowDeps(
        providers=_deps().providers,
        llm_router=_deps().llm_router,
        embedding_provider=_FixedEmbedding(),
        vector_store=InMemoryVectorStore(),
        compliance_explicit_enabled=False,
        # Even with a high-risk area configured, no interrupt can fire (no findings).
        high_risk_areas=frozenset({"lithium_battery"}),
    )
    wf = DurableWorkflow(engine=StateMachineEngine(), deps=deps)
    state = await wf.process(_state())

    assert state.status == "completed"          # straight through, no interrupt
    assert state.compliance is None             # explicit pass never ran
    assert "workflow:compliance:explicit_skipped" in state.decisions
    assert not any(d == "compliance:plan" for d in state.decisions)
    assert len(state.documents) == 3            # docs still generated


# ── engine: parallel fork merges deterministically ────────────────────────────


async def test_engine_parallel_merges_fields_and_decisions_in_node_order():
    engine = StateMachineEngine()
    base = _state()

    async def node_a(state: WorkflowState) -> WorkflowState:
        state.hs_code = "AAAA"
        state.decisions.append("a")
        return state

    async def node_b(state: WorkflowState) -> WorkflowState:
        state.weight_lbs = 99.0
        state.decisions.append("b")
        return state

    merged = await engine.run_parallel(base, [node_a, node_b])
    assert merged.hs_code == "AAAA"            # node_a's write
    assert merged.weight_lbs == 99.0           # node_b's write (disjoint field)
    assert merged.decisions[-2:] == ["a", "b"]  # deterministic: node order
