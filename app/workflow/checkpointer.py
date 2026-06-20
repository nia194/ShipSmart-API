"""
Workflow checkpointer (UC4) — durability behind a swappable port.

The orchestrator persists ``WorkflowState`` by ``workflow_id`` at the interrupt
(and at completion) so a suspended workflow can be **resumed in a different
request — or after a process restart**. Two adapters implement the port:

  * :class:`InMemoryCheckpointer` (default, tests) — a dict, process-lifetime.
  * :class:`SqliteCheckpointer` (``workflow_durable=true``) — stdlib ``sqlite3``,
    file-backed; survives restarts. This is the honest "kill & resume" backend.

Both round-trip the state through JSON (``WorkflowState`` is fully serializable),
so resume reproduces the exact state regardless of adapter. A ``PostgresCheckpointer``
would be a future adapter behind the same port — no architecture change.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Protocol, runtime_checkable

from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


@runtime_checkable
class WorkflowCheckpointer(Protocol):
    """Port for persisting and restoring workflow state by id."""

    def save(self, state: WorkflowState) -> None: ...

    def load(self, workflow_id: str) -> WorkflowState | None: ...


class InMemoryCheckpointer:
    """Process-lifetime checkpointer (default / tests).

    Stores the JSON serialization (not the live object) so load returns an
    independent copy — mirroring a real backend and proving resume reconstructs
    state rather than sharing a reference.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def save(self, state: WorkflowState) -> None:
        self._store[state.workflow_id] = state.model_dump_json()

    def load(self, workflow_id: str) -> WorkflowState | None:
        raw = self._store.get(workflow_id)
        return WorkflowState.model_validate_json(raw) if raw is not None else None


class SqliteCheckpointer:
    """File-backed checkpointer using the standard-library ``sqlite3``.

    One row per workflow (``workflow_id`` primary key) holding the JSON state.
    A fresh instance pointed at the same file sees previously-saved workflows —
    that is the durable "kill & resume" property. Connections are opened per
    operation (simple + thread-safe); zero new dependencies.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS workflow_checkpoints ("
                "workflow_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, "
                "updated_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def save(self, state: WorkflowState) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workflow_checkpoints "
                "(workflow_id, state_json, updated_at) VALUES (?, ?, ?)",
                (state.workflow_id, state.model_dump_json(), state.updated_at),
            )

    def load(self, workflow_id: str) -> WorkflowState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM workflow_checkpoints WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
        return WorkflowState.model_validate_json(row[0]) if row else None


def create_checkpointer(durable: bool, path: str) -> WorkflowCheckpointer:
    """Factory: ``SqliteCheckpointer`` when durable, else ``InMemoryCheckpointer``."""
    if durable:
        logger.info("Workflow checkpointer: SqliteCheckpointer at %s", path)
        return SqliteCheckpointer(path)
    logger.info("Workflow checkpointer: InMemoryCheckpointer (non-durable)")
    return InMemoryCheckpointer()
