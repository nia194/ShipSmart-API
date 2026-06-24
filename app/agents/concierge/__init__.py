"""Conversational Concierge — a stateful, slot-filling chat over shipping.

A thin deterministic shell on top of the existing workers. It carries a typed
:class:`ConversationState` (the shipment ``slots``), merges each turn with the
pure :func:`app.agents.concierge.state.fold_turn` reducer, asks only for the
slots an intent still needs — never re-asking for ones already present, whether
the client sent them from a form or a prior turn — then dispatches to an existing
deterministic worker (compliance / the read-only agent). The model may help
*extract* entities; it never quotes, books, or decides — those stay in code.
"""
