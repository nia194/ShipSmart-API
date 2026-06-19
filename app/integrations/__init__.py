"""
Outbound integration clients (infrastructure layer).

Thin async HTTP clients for the services ShipSmart-API talks *out* to — the Java
orchestrator (`java_client`) and the standalone ShipSmart-MCP tool server
(`mcp_client`). Kept separate from `app.services` (application/use-case logic) so
the integration boundary is explicit and each external dependency is swappable.

Back-compat: `app.services.java_client` / `app.services.mcp_client` remain as
re-export shims, so existing imports keep working.
"""
