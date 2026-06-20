"""
Review queue (UC4) — the human-in-the-loop inbox behind a swappable port.

When the workflow interrupts on an unverified high-risk shipment, it enqueues a
:class:`ReviewItem` here so an officer can list pending reviews and resolve them.
The authoritative state lives in the checkpointer; this queue is the index of
*open questions* (and the record of their resolution).

:class:`InMemoryReviewQueue` is the default adapter; a persistent queue would be
a future adapter behind the same port.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

ReviewStatus = Literal["pending", "resolved"]
Determination = Literal["cleared", "blocked"]


class ReviewItem(BaseModel):
    """One pending (or resolved) human-review question for a workflow."""

    workflow_id: str
    question: str
    high_risk_areas: list[str] = Field(default_factory=list)
    status: ReviewStatus = "pending"
    determination: Determination | None = None
    note: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


@runtime_checkable
class ReviewQueue(Protocol):
    """Port for the human-review inbox."""

    def enqueue(self, item: ReviewItem) -> None: ...

    def list_pending(self) -> list[ReviewItem]: ...

    def peek(self, workflow_id: str) -> ReviewItem | None: ...

    def resolve(
        self, workflow_id: str, determination: Determination, note: str = "",
    ) -> ReviewItem | None: ...


class InMemoryReviewQueue:
    """Process-lifetime review inbox (default / tests)."""

    def __init__(self) -> None:
        self._items: dict[str, ReviewItem] = {}

    def enqueue(self, item: ReviewItem) -> None:
        self._items[item.workflow_id] = item

    def list_pending(self) -> list[ReviewItem]:
        return [i for i in self._items.values() if i.status == "pending"]

    def peek(self, workflow_id: str) -> ReviewItem | None:
        return self._items.get(workflow_id)

    def resolve(
        self, workflow_id: str, determination: Determination, note: str = "",
    ) -> ReviewItem | None:
        item = self._items.get(workflow_id)
        if item is None:
            return None
        resolved = item.model_copy(
            update={"status": "resolved", "determination": determination, "note": note}
        )
        self._items[workflow_id] = resolved
        return resolved
