"""Tests for the UC4 in-memory review queue."""

from __future__ import annotations

from app.workflow.review_queue import InMemoryReviewQueue, ReviewItem


def _item(workflow_id: str = "wf-1") -> ReviewItem:
    return ReviewItem(
        workflow_id=workflow_id,
        question="Unverified high-risk area: lithium_battery",
        high_risk_areas=["lithium_battery"],
    )


def test_enqueue_and_list_pending():
    q = InMemoryReviewQueue()
    q.enqueue(_item("a"))
    q.enqueue(_item("b"))
    pending = q.list_pending()
    assert {i.workflow_id for i in pending} == {"a", "b"}
    assert all(i.status == "pending" for i in pending)


def test_peek_returns_item_or_none():
    q = InMemoryReviewQueue()
    q.enqueue(_item("a"))
    assert q.peek("a").workflow_id == "a"
    assert q.peek("missing") is None


def test_resolve_marks_resolved_and_records_determination():
    q = InMemoryReviewQueue()
    q.enqueue(_item("a"))
    resolved = q.resolve("a", "cleared", note="ok to ship")
    assert resolved.status == "resolved"
    assert resolved.determination == "cleared"
    assert resolved.note == "ok to ship"
    # No longer pending after resolution.
    assert q.list_pending() == []


def test_resolve_missing_returns_none():
    assert InMemoryReviewQueue().resolve("nope", "blocked") is None
