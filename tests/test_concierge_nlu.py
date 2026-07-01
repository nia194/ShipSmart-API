"""Dialog-manager tests — robust NLU, corrections, disambiguation, proactive replies.

Keyless: the LLM NLU/polish are skipped for EchoClient, so these exercise the
deterministic floor plus the integration of the new pieces (monkeypatching the NLU
where a model-only signal — corrections/ambiguities — is required)."""

from __future__ import annotations

from app.agents.concierge.extract import NluResult, extract_nlu
from app.agents.concierge.models import ConversationState
from app.agents.concierge.reply import (
    compose_gathering_reply,
    compose_ready_summary,
    correction_note,
    summarize_slots,
)
from app.agents.concierge.service import run_concierge
from app.agents.concierge.state import apply_corrections, choose_intent
from app.llm.client import EchoClient
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter
from app.rag.embeddings import LocalHashEmbedding
from app.rag.vector_store import InMemoryVectorStore


def _deps():
    echo = EchoClient()
    return {
        "llm_router": LLMRouter(
            clients={TASK_REASONING: echo, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
            fallback=echo,
        ),
        "embedding_provider": LocalHashEmbedding(dims=16),
        "vector_store": InMemoryVectorStore(),
    }


# ── extract_nlu (keyless = deterministic floor) ──────────────────────────────
async def test_extract_nlu_keyless_is_deterministic():
    nlu = await extract_nlu("ship from Atlanta to Seattle, 12 lb", {}, None)
    assert nlu.intent == "quote"
    assert nlu.intents == ["quote"]
    assert nlu.slots["origin"] == "Atlanta"
    assert nlu.slots["destination"] == "Seattle"
    assert nlu.slots["weight_lbs"] == 12.0
    assert nlu.corrections == {} and nlu.ambiguities == []


# ── pure policy helpers ──────────────────────────────────────────────────────
def test_choose_intent_precedence_and_fallback():
    assert choose_intent(["quote", "compliance"], None) == "compliance"
    assert choose_intent(["tracking", "quote"], None) == "tracking"
    assert choose_intent([], "quote") == "quote"
    assert choose_intent([], None) == "advice"


def test_apply_corrections_overwrites_present_slot():
    state = ConversationState(slots={"weight_lbs": 12.0})
    out = apply_corrections(state, {"weight_lbs": 15.0})
    assert out.slots["weight_lbs"] == 15.0
    assert apply_corrections(state, {}) is state  # no-op


# ── proactive reply templates ────────────────────────────────────────────────
def test_summary_and_correction_note():
    s = summarize_slots({"origin": "Atlanta", "destination": "Seattle", "weight_lbs": 12.0})
    assert s == "from Atlanta, to Seattle, 12.0 lb"
    assert correction_note({"weight_lbs": 15.0}) == "Updated weight to 15.0. "
    assert correction_note({}) == ""


async def test_compose_gathering_confirms_then_asks():
    reply = await compose_gathering_reply(
        "Where's it going?", {"origin": "Atlanta"}, llm_router=None,
    )
    assert "Atlanta" in reply and reply.endswith("Where's it going?")


async def test_compose_ready_summary_suggests_next_action():
    reply = await compose_ready_summary(
        {"origin": "Atlanta", "destination": "Seattle", "weight_lbs": 12.0},
        "quote", llm_router=None,
    )
    assert "Atlanta" in reply and "options" in reply


async def test_polish_keeps_clarifying_question():
    # A degenerate rephrase that drops the question must fall back to the template,
    # so the user always gets an actual question (not an apology).
    class DropsQuestion:
        provider_name = "openai"

        async def complete(self, messages, **kw):
            return "I'm sorry, I don't have enough information."

    router = LLMRouter(clients={TASK_SYNTHESIS: DropsQuestion()}, fallback=EchoClient())
    out = await compose_gathering_reply(
        "Where are you shipping from?", {}, llm_router=router,
    )
    assert out.endswith("Where are you shipping from?")


# ── service integration (monkeypatched NLU for model-only signals) ────────────
async def test_correction_overwrites_and_is_acknowledged(monkeypatch):
    import app.agents.concierge.service as svc

    async def fake_nlu(message, prior, router, *, reference_block="", request_id=""):
        return NluResult(intent="quote", intents=["quote"], slots={},
                         corrections={"weight_lbs": 15.0}, ambiguities=[])

    monkeypatch.setattr(svc, "extract_nlu", fake_nlu)
    state = ConversationState(
        slots={"origin": "Atlanta", "destination": "Seattle", "weight_lbs": 12.0},
        intent="quote",
    )
    res = await run_concierge("actually make it 15 lb", state, **_deps())
    assert res.state.slots["weight_lbs"] == 15.0
    assert "Updated" in res.reply
    assert "concierge:correction" in res.decisions


async def test_disambiguation_when_required_slot_is_vague(monkeypatch):
    import app.agents.concierge.service as svc

    async def fake_nlu(message, prior, router, *, reference_block="", request_id=""):
        return NluResult(intent="quote", intents=["quote"], slots={},
                         corrections={}, ambiguities=["origin"])

    monkeypatch.setattr(svc, "extract_nlu", fake_nlu)
    state = ConversationState(
        slots={"origin": "Springfield", "destination": "Seattle", "weight_lbs": 10.0},
        intent="quote",
    )
    res = await run_concierge("from Springfield", state, **_deps())
    assert any(d.startswith("concierge:disambiguate:origin") for d in res.decisions)
    assert res.state.status == "gathering"


async def test_compound_intent_prefers_compliance(monkeypatch):
    import app.agents.concierge.service as svc

    async def fake_nlu(message, prior, router, *, reference_block="", request_id=""):
        return NluResult(intent="quote", intents=["quote", "compliance"], slots={},
                         corrections={}, ambiguities=[])

    monkeypatch.setattr(svc, "extract_nlu", fake_nlu)
    state = ConversationState(
        slots={"destination_country": "BR", "description": "camera drone"},
    )
    res = await run_concierge("quote me, and is it allowed?", state, **_deps())
    assert res.state.intent == "compliance"
    assert res.dispatched_to == "compliance"


# ── lowercase routes + city→country resolution (regression) ───────────────────
async def test_lowercase_city_route_parses_and_resolves_country():
    # Real users type lowercase; the route + its countries must still resolve.
    nlu = await extract_nlu("atlanta to seattle, 12 lb", {}, None)
    assert nlu.slots["origin"] == "atlanta"
    assert nlu.slots["destination"] == "seattle"
    assert nlu.slots["origin_country"] == "US"
    assert nlu.slots["destination_country"] == "US"
    assert nlu.slots["weight_lbs"] == 12.0


async def test_city_route_resolves_country_to_enable_workflow():
    # Without city→country, the international multi-agent workflow can never fire from chat.
    nlu = await extract_nlu("ship a drone from New York to Berlin", {}, None)
    assert nlu.slots["origin_country"] == "US"
    assert nlu.slots["destination_country"] == "DE"


async def test_non_place_route_not_overparsed():
    # "my shipment to brazil" must NOT be read as a route origin "my shipment";
    # the standalone-country rule still sets the destination country.
    nlu = await extract_nlu("my shipment to brazil", {}, None)
    assert "origin" not in nlu.slots
    assert nlu.slots.get("destination_country") == "BR"
