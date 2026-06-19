"""
Workflow orchestration + durability (UC3/UC4).

A deterministic, hand-rolled state machine (a ``WorkflowEngine`` Protocol +
``StateMachineEngine``) sequences the specialist agents; durability and the
human-in-the-loop interrupt/resume (checkpointer + review-queue ports) wrap the
graph. The engine is swappable — a LangGraph adapter could be added later behind
the same Protocol — but control flow stays deterministic code, never a model.

Populated in Phases 2–3.
"""
