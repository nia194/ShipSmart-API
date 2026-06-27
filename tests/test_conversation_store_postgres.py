"""Integration test for PostgresConversationStore against a real Postgres.

Gated on TEST_DATABASE_URL (a Postgres with the conversations migration applied) so
the hermetic suite is unaffected. Validates the asyncpg round-trip the in-memory
adapter can't: jsonb slots, ::uuid casts (hex session id from the route), append +
bounded oldest-first load.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.conversations.store import ConversationMessage, PostgresConversationStore

DSN = os.getenv("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set TEST_DATABASE_URL to a Postgres with the conversations migration",
)


async def test_postgres_conversation_round_trip():
    store = PostgresConversationStore(DSN)
    await store.connect()
    try:
        sid = uuid.uuid4().hex  # matches how the route mints session ids (no dashes)
        await store.upsert_state(
            sid, status="answered", intent="quote",
            slots={"origin": "Atlanta", "weight_lbs": 12.0}, turns=2,
            last_dispatched_to="agent",
        )
        await store.append_messages(sid, [
            ConversationMessage(role="user", content="ship a 12 lb box"),
            ConversationMessage(role="assistant", content="Where to?"),
        ])
        rec = await store.load(sid)
        assert rec is not None
        assert rec.intent == "quote"
        assert rec.slots["weight_lbs"] == 12.0          # jsonb round-trip
        assert rec.last_dispatched_to == "agent"
        assert [m.role for m in rec.messages] == ["user", "assistant"]  # oldest-first

        # upsert is idempotent on the conversation row
        await store.upsert_state(
            sid, status="answered", intent="quote", slots={"origin": "Boston"}, turns=3,
            last_dispatched_to="workflow",
        )
        rec2 = await store.load(sid)
        assert rec2.turns == 3 and rec2.slots["origin"] == "Boston"

        assert await store.load(uuid.uuid4().hex) is None  # unknown session
    finally:
        await store.disconnect()
