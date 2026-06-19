"""End-to-end tests for the compliance service (UC2).

Hermetic + keyless: a deterministic FixedEmbedding (every text → the same unit
vector) plus an InMemoryVectorStore make coverage exactly controllable — an empty
store yields uncovered (→ unverified) areas, a seeded store yields covered (→ info)
areas. The LLM summary runs through the keyless EchoClient.
"""

from __future__ import annotations

from app.agents.compliance.models import Shipment
from app.agents.compliance.service import check_compliance
from app.core.audit import InMemoryAuditSink
from app.llm.client import EchoClient, ScriptedToolCallingClient, ToolCall, ToolCallResult
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore, StoredChunk


class _FixedEmbedding(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    @property
    def dimensions(self) -> int:
        return 3


def _router(reasoning=None) -> LLMRouter:
    echo = EchoClient()
    return LLMRouter(
        clients={
            TASK_REASONING: reasoning or echo,
            TASK_SYNTHESIS: echo,
            TASK_FALLBACK: echo,
        },
        fallback=echo,
    )


async def _seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.add([
        StoredChunk(
            text="Lithium batteries are Class 9 dangerous goods; declare UN3480.",
            source="compliance/lithium-batteries-dangerous-goods.md",
            chunk_index=0, embedding=[1.0, 0.0, 0.0],
        ),
    ])
    return store


# ── deterministic spine: structural flags + uncovered areas → unverified ──────


async def test_uncovered_areas_become_unverified_and_flags_drive_verdict():
    result = await check_compliance(
        Shipment("US", "DE", declared_value_usd=0, description="20000mAh power bank"),
        llm_router=_router(),
        embedding_provider=_FixedEmbedding(),
        vector_store=InMemoryVectorStore(),  # empty → every area uncovered
    )
    statuses = {f.area: f.status for f in result.findings}
    # Structural flags fired (missing customs value + dangerous goods).
    assert statuses.get("customs_value_missing") == "flag"
    assert statuses.get("dangerous_goods_declaration") == "flag"
    # All four fixed areas investigated but uncovered → unverified (never fabricated).
    for area in ("lithium_battery", "customs_docs", "import_restriction", "value_threshold"):
        assert statuses.get(area) == "unverified"
    assert result.verdict == "action_required"  # a flag is present
    assert "compliance:plan" in result.decisions
    assert "compliance:investigate:lithium_battery" in result.decisions


async def test_covered_areas_become_info_and_clean_shipment_is_advisory():
    result = await check_compliance(
        Shipment("US", "US", declared_value_usd=20, description="a hardcover book"),
        llm_router=_router(),
        embedding_provider=_FixedEmbedding(),
        vector_store=await _seeded_store(),  # every area retrieves the seeded chunk
    )
    statuses = {f.status for f in result.findings}
    assert statuses == {"info"}              # no structural flags, all areas covered
    assert result.verdict == "advisory"      # advisory — never "compliant"/"cleared"
    assert result.sources                    # grounded citations present
    assert result.provider == "echo"


# ── critic OFF by default ─────────────────────────────────────────────────────


async def test_critic_off_by_default_emits_no_critique_decisions():
    result = await check_compliance(
        Shipment("US", "BR", declared_value_usd=600, description="camera drone"),
        llm_router=_router(),
        embedding_provider=_FixedEmbedding(),
        vector_store=InMemoryVectorStore(),
        critique_max_rounds=0,
    )
    assert not any(d.startswith("critique:") for d in result.decisions)
    assert result.critique_rounds == 0


# ── UC2 critic ON: proposes a gap, grounds it; uncovered gap → unverified ─────


async def test_critic_adds_grounded_gap_uncovered_stays_unverified():
    scripted = ScriptedToolCallingClient([
        ToolCallResult(
            kind="tool_calls",
            calls=[ToolCall(id="c1", name="propose_gaps",
                            arguments={"areas": "destination_drone_import_rules",
                                       "rationale": "drone into Brazil"})],
        ),
    ])
    result = await check_compliance(
        Shipment(
            "US", "BR", declared_value_usd=600,
            description="camera drone with lithium battery",
        ),
        llm_router=_router(scripted),
        embedding_provider=_FixedEmbedding(),
        vector_store=InMemoryVectorStore(),  # empty → the proposed gap can't be grounded
        critique_max_rounds=1,
    )
    critic_findings = [f for f in result.findings if f.kind == "critic"]
    assert len(critic_findings) == 1
    gap = critic_findings[0]
    assert gap.area == "destination_drone_import_rules"
    # Load-bearing invariant: an uncovered proposal is unverified, NEVER a flag.
    assert gap.status == "unverified"
    assert "critique:round:1" in result.decisions
    assert "critique:gap:destination_drone_import_rules" in result.decisions
    assert result.critique_rounds == 1


async def test_audit_event_emitted_for_verdict():
    sink = InMemoryAuditSink()
    await check_compliance(
        Shipment("US", "DE", declared_value_usd=0, description="power bank"),
        llm_router=_router(),
        embedding_provider=_FixedEmbedding(),
        vector_store=InMemoryVectorStore(),
        audit_sink=sink,
    )
    assert any(e.event.startswith("compliance:verdict:") for e in sink.events)
    event = sink.events[-1]
    assert event.actor == "agent" and event.actor_name == "compliance"
