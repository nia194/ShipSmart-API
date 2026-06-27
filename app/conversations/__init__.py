"""Server-side conversation memory for the Conversational Concierge.

A swappable persistence port (mirrors ``app.core.audit`` / ``app.workflow.checkpointer``)
that lets a chat be recalled after a page reload via an anonymous session id. The
default in-memory adapter keeps the keyless dev/test stack working; the Postgres
adapter is the durable backend. See :mod:`app.conversations.store`.
"""

from app.conversations.store import (
    ConversationMessage,
    ConversationRecord,
    ConversationStore,
    InMemoryConversationStore,
    PostgresConversationStore,
    create_conversation_store,
)

__all__ = [
    "ConversationMessage",
    "ConversationRecord",
    "ConversationStore",
    "InMemoryConversationStore",
    "PostgresConversationStore",
    "create_conversation_store",
]
