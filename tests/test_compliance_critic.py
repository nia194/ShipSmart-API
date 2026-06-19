"""Tests for the UC2 compliance critic (the only model-in-the-loop step).

Covers honest degradation (no native tool calling → deterministic no-op),
parsing/validation/capping of proposed gaps, and exclusion of already-investigated
areas. Uses the keyless EchoClient and the deterministic ScriptedToolCallingClient
so no API keys are needed.
"""

from __future__ import annotations

from app.agents.compliance.critic import propose_gaps
from app.agents.compliance.models import Shipment
from app.llm.client import EchoClient, ScriptedToolCallingClient, ToolCall, ToolCallResult
from app.llm.router import TASK_FALLBACK, TASK_REASONING, TASK_SYNTHESIS, LLMRouter

_SHIPMENT = Shipment(
    "US", "BR", declared_value_usd=600, description="camera drone with lithium battery",
)


def _router(reasoning) -> LLMRouter:
    echo = EchoClient()
    return LLMRouter(
        clients={TASK_REASONING: reasoning, TASK_SYNTHESIS: echo, TASK_FALLBACK: echo},
        fallback=echo,
    )


def _gap_turn(areas: str, rationale: str = "") -> ToolCallResult:
    return ToolCallResult(
        kind="tool_calls",
        calls=[ToolCall(id="c1", name="propose_gaps",
                        arguments={"areas": areas, "rationale": rationale})],
    )


async def test_critic_noop_without_native_tool_calling():
    # EchoClient.complete_with_tools raises NotImplementedError → deterministic no-op.
    gaps = await propose_gaps(
        _SHIPMENT, [], already_investigated=set(),
        llm_router=_router(EchoClient()), max_gap_areas=3,
    )
    assert gaps == []


async def test_critic_zero_budget_returns_empty():
    scripted = ScriptedToolCallingClient([_gap_turn("destination_drone_import_rules")])
    gaps = await propose_gaps(
        _SHIPMENT, [], already_investigated=set(),
        llm_router=_router(scripted), max_gap_areas=0,
    )
    assert gaps == []


async def test_critic_parses_slugifies_and_caps():
    scripted = ScriptedToolCallingClient([
        _gap_turn("Destination Drone Import Rules; battery watt-hour limit; extra area", "why"),
    ])
    gaps = await propose_gaps(
        _SHIPMENT, [], already_investigated=set(),
        llm_router=_router(scripted), max_gap_areas=2,
    )
    assert [g.area for g in gaps] == ["destination_drone_import_rules", "battery_watt_hour_limit"]
    assert all(g.rationale == "why" for g in gaps)


async def test_critic_excludes_already_investigated_and_dedupes():
    scripted = ScriptedToolCallingClient([
        _gap_turn(
            "lithium_battery; destination_drone_import_rules; destination_drone_import_rules",
        ),
    ])
    gaps = await propose_gaps(
        _SHIPMENT, [], already_investigated={"lithium_battery"},
        llm_router=_router(scripted), max_gap_areas=5,
    )
    assert [g.area for g in gaps] == ["destination_drone_import_rules"]


async def test_critic_empty_areas_returns_empty():
    scripted = ScriptedToolCallingClient([_gap_turn("   ;  ; ")])
    gaps = await propose_gaps(
        _SHIPMENT, [], already_investigated=set(),
        llm_router=_router(scripted), max_gap_areas=3,
    )
    assert gaps == []
