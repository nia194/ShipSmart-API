"""
Specialist agents — the reasoning layer (first-class).

Each agent reasons about one domain, depends on a domain port (not a concrete
adapter) and, where it reasons, the LLM router. All are built on the shared
grounding primitive (``app.rag.grounding``). The compliance agent (UC2) lands
first (Phase 1); classification / landed-cost / routing / documentation follow
(Phase 2). The existing Concierge advisor (``app.services.agent_service``) stays
the standalone advisory door.
"""
