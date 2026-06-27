"""Conversation store — durability behind a swappable port (concierge recall).

The concierge persists its merged shipment state + transcript by ``session_id`` so
a chat can be **recalled after a page reload** (and, with the Postgres adapter,
after a process restart / on another device). Two adapters implement the port:

  * :class:`InMemoryConversationStore` (default, tests) — a dict, process-lifetime.
  * :class:`PostgresConversationStore` (``CONVERSATION_STORE=postgres``) — ``asyncpg``,
    reusing the connection pattern of :class:`app.rag.pgvector_store.PGVectorStore`.
    Writes the tables created in
    ``supabase/migrations/<ts>_create_conversations.sql``.

This is the Python-owned, assistive-memory data plane (same access model as
``rag_chunks`` / ``rag_query_log``): no FK into user/business tables, never a
source of truth for a booking. Every method is **best-effort at the call site** —
the route wraps writes so persistence can never break a turn — but the adapters
themselves only swallow nothing; callers decide. A ``PostgresConversationStore``
is a swap behind the same port — no architecture change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from app.rag.vector_store import validate_table_name

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass(frozen=True)
class ConversationMessage:
    """One transcript turn (append-only)."""

    role: str  # "user" | "assistant"
    content: str
    slots_delta: dict = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "slots_delta": dict(self.slots_delta),
            "decisions": list(self.decisions),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ConversationRecord:
    """The recall snapshot for one session — merged state + (bounded) transcript."""

    session_id: str
    status: str = "gathering"
    intent: str | None = None
    slots: dict = field(default_factory=dict)
    turns: int = 0
    last_dispatched_to: str | None = None
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@runtime_checkable
class ConversationStore(Protocol):
    """Port for persisting + restoring a concierge conversation by session id."""

    async def load(self, session_id: str, *, limit: int = 50) -> ConversationRecord | None:
        """Return the recall snapshot (state + up to ``limit`` recent messages)."""
        ...

    async def upsert_state(
        self,
        session_id: str,
        *,
        status: str,
        intent: str | None,
        slots: dict,
        turns: int,
        last_dispatched_to: str | None,
    ) -> None:
        """Create/update the per-session recall snapshot (no messages)."""
        ...

    async def append_messages(
        self, session_id: str, messages: list[ConversationMessage],
    ) -> None:
        """Append transcript turns for a session (parent row must exist)."""
        ...


class InMemoryConversationStore:
    """Process-lifetime store (default / tests). Mirrors InMemoryCheckpointer."""

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}
        self._messages: dict[str, list[ConversationMessage]] = {}

    async def load(self, session_id: str, *, limit: int = 50) -> ConversationRecord | None:
        snap = self._state.get(session_id)
        if snap is None:
            return None
        msgs = self._messages.get(session_id, [])
        recent = msgs[-limit:] if limit and limit > 0 else list(msgs)
        return ConversationRecord(messages=list(recent), **snap)

    async def upsert_state(
        self, session_id, *, status, intent, slots, turns, last_dispatched_to,
    ) -> None:
        existing = self._state.get(session_id)
        created_at = existing["created_at"] if existing else _now()
        self._state[session_id] = {
            "session_id": session_id,
            "status": status,
            "intent": intent,
            "slots": dict(slots),
            "turns": turns,
            "last_dispatched_to": last_dispatched_to,
            "created_at": created_at,
            "updated_at": _now(),
        }

    async def append_messages(self, session_id, messages) -> None:
        self._messages.setdefault(session_id, []).extend(messages)


class PostgresConversationStore:
    """Durable store backed by Postgres (asyncpg). ``CONVERSATION_STORE=postgres``.

    Connection lifecycle mirrors :class:`app.rag.pgvector_store.PGVectorStore`:
    ``connect()`` once at startup, ``disconnect()`` at shutdown.
    """

    def __init__(
        self,
        dsn: str,
        conversations_table: str = "conversations",
        messages_table: str = "conversation_messages",
    ) -> None:
        if not dsn:
            raise ValueError(
                "PostgresConversationStore requires a non-empty DSN. "
                "Set DATABASE_URL when CONVERSATION_STORE=postgres."
            )
        self._dsn = dsn
        self._conversations = validate_table_name(conversations_table)
        self._messages = validate_table_name(messages_table)
        self._pool = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        import asyncpg

        self._pool = await asyncpg.create_pool(
            dsn=self._dsn, min_size=1, max_size=5, command_timeout=30,
        )
        logger.info("PostgresConversationStore connected (table=%s)", self._conversations)

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgresConversationStore disconnected")

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError(
                "PostgresConversationStore is not connected. Call connect() during startup."
            )
        return self._pool

    async def load(self, session_id: str, *, limit: int = 50) -> ConversationRecord | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT session_id, status, intent, slots, turns, last_dispatched_to, "
                f"created_at, updated_at FROM {self._conversations} WHERE session_id = $1::uuid",
                session_id,
            )
            if row is None:
                return None
            msg_rows = await conn.fetch(
                f"SELECT role, content, slots_delta, decisions, created_at "
                f"FROM {self._messages} WHERE session_id = $1::uuid "
                f"ORDER BY created_at DESC, id DESC LIMIT $2",
                session_id, max(0, limit),
            )
        messages = [
            ConversationMessage(
                role=m["role"],
                content=m["content"],
                slots_delta=_loads(m["slots_delta"]),
                decisions=list(m["decisions"] or []),
                created_at=str(m["created_at"]),
            )
            for m in reversed(msg_rows)  # DESC fetch → oldest-first for replay
        ]
        return ConversationRecord(
            session_id=str(row["session_id"]),
            status=row["status"],
            intent=row["intent"],
            slots=_loads(row["slots"]),
            turns=int(row["turns"]),
            last_dispatched_to=row["last_dispatched_to"],
            messages=messages,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    async def upsert_state(
        self, session_id, *, status, intent, slots, turns, last_dispatched_to,
    ) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {self._conversations} "
                "(session_id, status, intent, slots, turns, last_dispatched_to, updated_at) "
                "VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6, now()) "
                "ON CONFLICT (session_id) DO UPDATE SET "
                "status = EXCLUDED.status, intent = EXCLUDED.intent, "
                "slots = EXCLUDED.slots, turns = EXCLUDED.turns, "
                "last_dispatched_to = EXCLUDED.last_dispatched_to, updated_at = now()",
                session_id, status, intent, json.dumps(slots or {}),
                int(turns), last_dispatched_to,
            )

    async def append_messages(self, session_id, messages) -> None:
        if not messages:
            return
        pool = self._require_pool()
        rows = [
            (
                session_id, m.role, m.content,
                json.dumps(m.slots_delta or {}), list(m.decisions or []),
            )
            for m in messages
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                f"INSERT INTO {self._messages} "
                "(session_id, role, content, slots_delta, decisions) "
                "VALUES ($1::uuid, $2, $3, $4::jsonb, $5)",
                rows,
            )


def _loads(value) -> dict:
    """asyncpg returns jsonb as str (no codec registered); be tolerant."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            out = json.loads(value)
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def create_conversation_store(kind: str = "memory", dsn: str = "") -> ConversationStore:
    """Factory: build the configured conversation store.

    ``memory`` (default) → :class:`InMemoryConversationStore`; ``postgres`` →
    :class:`PostgresConversationStore` (requires ``dsn``). Unknown values fall back
    to memory with a warning (config never crashes the app).
    """
    normalized = (kind or "memory").strip().lower()
    if normalized == "postgres":
        return PostgresConversationStore(dsn)
    if normalized != "memory":
        logger.warning("Unknown CONVERSATION_STORE %r — defaulting to memory", kind)
    return InMemoryConversationStore()
