"""
Application configuration.
Settings are loaded from environment variables using pydantic-settings.
Set values in .env for local dev.
In production (Render), set them via the Render dashboard.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_name: str = "shipsmart-api-python"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    # Audit + tracing sink: "logging" (default — structured log lines) or "memory"
    # (in-process capture, mainly for tests). A persistent backend is a future adapter.
    audit_sink: str = "logging"
    # AI-event observability sink: "logging" (default) or "memory". Durable
    # (ai_audit_log) is a future adapter — see app/core/ai_events.py.
    ai_event_sink: str = "logging"
    # Secret used to pseudonymize identity at write time (§6.1). Dev default —
    # OVERRIDE in production; rotating/deleting it unlinks pseudonymized events.
    pseudonym_secret: str = "dev-pseudonym-secret-change-me"
    # Secret used to HMAC-sign client-echoed conversation state (§7.2). Dev
    # default — OVERRIDE in production.
    state_secret: str = "dev-state-secret-change-me"
    # Runtime AI-controls admin token (§12 incident response). EMPTY (default)
    # means /admin/ai-controls does not exist — fail-closed until configured.
    admin_api_token: str = ""
    # Explicit user feedback endpoint (§6.6 / Layer-6 online loop). Gates
    # POST /api/v1/feedback (404 when false).
    feedback_enabled: bool = True

    # ── Shipping scope (platform policy) ─────────────────────────────────────
    # Deployment-level policy: does this deployment ship internationally or only
    # within one country? "worldwide" (default) = today's behavior (cross-border
    # allowed; international derived per shipment). "domestic" = only deliveries
    # within DOMESTIC_COUNTRY are possible; the API rejects cross-border requests
    # (422) and the concierge degrades to a domestic-only reply. This is the single
    # source of truth; the API publishes it on GET /api/v1/info and the siblings
    # (Web/Orchestrator/MCP) read/enforce the same value.
    shipping_scope: str = "worldwide"   # "worldwide" | "domestic"
    domestic_country: str = "US"        # ISO-3166 alpha-2 home country when domestic

    # ── Internal service-to-service ─────────────────────────────────────────
    internal_java_api_url: str = "http://localhost:8080"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_allowed_origins: str = "http://localhost:5173"

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Legacy single-provider selector. Kept for back-compat — task-based
    # routing below takes precedence when set.
    llm_provider: str = ""  # "openai", "gemini", "llama", "" (empty = EchoClient)
    llm_timeout: int = 30  # seconds
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.3

    # ── Task-based LLM routing ──────────────────────────────────────────────
    # Each task picks an underlying provider. Empty string = inherit
    # llm_provider (legacy behaviour). Unknown / missing-key providers
    # fall through to LLM_PROVIDER_FALLBACK, then to EchoClient.
    llm_provider_reasoning: str = ""   # advisors (shipping, tracking)
    llm_provider_synthesis: str = ""   # RAG q&a, recommendation summary
    llm_provider_fallback: str = "echo"  # safety net

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Google Gemini ────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── Llama (local / Ollama) ───────────────────────────────────────────────
    llama_base_url: str = "http://localhost:11434"
    llama_model: str = "llama3.2"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: str = ""  # "openai" or "" (empty = local placeholder)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Vector store ─────────────────────────────────────────────────────────
    vector_store_type: str = "memory"  # "memory", "pgvector", or "mcp"
    vector_store_path: str = ""
    database_url: str = ""              # Postgres connection string for pgvector backend
    pgvector_table: str = "rag_chunks"  # table name used by PGVectorStore / MCPVectorStore
    rag_auto_ingest: bool = True        # auto-ingest at startup if store is empty

    # ── MCP Vector Store (Supabase MCP Server) ────────────────────────────────
    mcp_server_url: str = ""            # MCP server HTTP endpoint (for "mcp" backend)
    mcp_api_key: str = ""               # Optional API key for MCP server authentication

    # ── Anthropic / Claude ───────────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # ── Rate limiting ────────────────────────────────────────────────────────
    rate_limit_advisor: str = "10/minute"
    rate_limit_orchestration: str = "20/minute"
    rate_limit_compare: str = "10/minute"
    rate_limit_agent: str = "10/minute"
    rate_limit_compliance: str = "10/minute"
    rate_limit_workflow: str = "10/minute"
    rate_limit_concierge: str = "60/minute"  # interactive chat: a single user easily exceeds 10/min

    # ── Agent (Concierge) ────────────────────────────────────────────────────
    # Model-driven, read-only tool-calling loop over the MCP tools + retrieve_rag.
    agent_enabled: bool = True          # gate POST /api/v1/agent/run
    agent_max_steps: int = 5            # hard cost bound on the agent loop
    # Total retrieve_rag calls allowed per run, independent of agent_max_steps.
    # 1 = single-shot retrieval only (today's effective behavior); >1 enables the
    # bounded, conditional re-retrieval the agent triggers ONLY on weak coverage.
    agent_max_retrievals: int = 2       # cap on retrieve_rag calls per agent run

    # ── Conversational Concierge (stateful chat) ─────────────────────────────
    # Multi-turn, slot-filling chat (POST /api/v1/concierge/chat), distinct from
    # the one-shot Agent above. Gathers shipment slots, never re-asks for known
    # ones, then dispatches to an existing worker (compliance / the agent). The
    # model only helps extract entities; decisions stay deterministic. OFF by default.
    concierge_enabled: bool = False     # gate POST /api/v1/concierge/chat (404 when false)
    concierge_max_turns: int = 12       # soft bound on a single conversation
    # Server-side conversation memory so a chat can be RECALLED after a page reload
    # (anonymous session id). "memory" (default) = process-lifetime in-memory store —
    # keyless, keeps today's hermetic stack working. "postgres" = durable backend via
    # asyncpg + DATABASE_URL, writing the tables in the conversations migration. This
    # is the Python-owned assistive-memory data plane (same access model as rag_chunks);
    # it is never the source of truth for a booking.
    conversation_store: str = "memory"  # "memory" | "postgres"
    conversation_max_messages: int = 50  # recall window: max transcript turns loaded

    # ── Compliance (UC2) ─────────────────────────────────────────────────────
    # Deterministic compliance analysis (structural rules + grounded, coverage-
    # gated investigation of fixed areas) with an OPTIONAL model-driven critic
    # that proposes gaps a single pass may have missed. The deterministic path
    # has no LLM in its control flow; only the critic and the final summary call
    # a model. Advisory only — never a legal/customs clearance.
    compliance_enabled: bool = True     # gate POST /api/v1/compliance/check (404 when false)
    # Additive feature switch for the EXPLICIT compliance pass when it is reachable
    # from the chat (concierge) and durable workflow paths. True ⇒ the hard UC2 pass
    # runs as an extra layer on top of the normal flow. False ⇒ the normal flow runs
    # by itself: the explicit pass is skipped, but the always-on lightweight checks
    # (guardrails + RAG grounding over the compliance corpus) still apply. Distinct
    # from compliance_enabled, which only gates the standalone /compliance/check
    # endpoint. NOTE: when false, the workflow's high-risk HITL interrupt cannot fire
    # (it lives inside the explicit pass) — intentional, with a startup warning.
    compliance_explicit_enabled: bool = True
    # Rounds of the UC2 critic. 0 (default) = critic OFF: deterministic structural
    # + fixed-area investigation only. >0 = the model proposes additional areas to
    # investigate; an uncovered proposal becomes an honest "unverified" finding,
    # never a fabricated flag (the load-bearing invariant).
    compliance_critique_max_rounds: int = 0
    # Max gap areas accepted from the critic per round (cost + blast-radius bound).
    compliance_max_gap_areas: int = 3
    # Declared value (USD) at/above which a commercial invoice is flagged for an
    # international shipment (US EEI/AES filing threshold by default).
    compliance_value_threshold_usd: float = 2500.0

    # ── Workflow (UC3 / UC4) ─────────────────────────────────────────────────
    # Multi-agent durable workflow: classify → (landed-cost ‖ routing) →
    # compliance(+UC2) → documentation, with a durable human-in-the-loop
    # interrupt on unverified high-risk shipments. OFF by default.
    workflow_enabled: bool = False      # gate POST /api/v1/workflow/* (404 when false)
    # Durability (UC4). false ⇒ InMemoryCheckpointer (process-lifetime);
    # true ⇒ SqliteCheckpointer at workflow_checkpoint_path (survives restarts).
    workflow_durable: bool = False
    workflow_checkpoint_path: str = "workflow_checkpoints.db"
    # Interrupt predicate: a compliance "unverified" area in this set suspends the
    # workflow for human review. Empty ⇒ never interrupt (straight-through).
    workflow_high_risk_areas: str = "lithium_battery,import_restriction"

    # ── ShipSmart MCP (tool server) ──────────────────────────────────────────
    # HTTP endpoint of the standalone ShipSmart-MCP service. Empty = no tools
    # (advisor/orchestration routes return 503).
    shipsmart_mcp_url: str = ""
    # Optional shared secret sent as X-MCP-Api-Key when calling the MCP server.
    shipsmart_mcp_api_key: str = ""

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_provider: str = ""
    rag_top_k: int = 3
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_documents_path: str = "data/documents"

    # ── Request-time LLM fallback chain (A) ──────────────────────────────────
    # CSV of providers tried in order AFTER the task's primary client errors on a
    # *retryable* failure, e.g. "openai,gemini,echo". Empty (default) = today's
    # single-client behavior: the primary is called once and its error propagates.
    llm_fallback_chain: str = ""
    # Retries against ONE provider before moving to the next in the chain. Only
    # consulted when a fallback chain is configured (keeps today's path retry-free).
    llm_retry_max_attempts: int = 2

    # ── LLM context budget + per-task overrides (B) ──────────────────────────
    llm_max_context_tokens: int = 8000
    # Per-task model / temperature / max-token overrides. Empty string = inherit
    # the global value (today's behavior). Stored as str so an empty env value
    # (the documented .env.example default) never fails to parse; numbers are
    # parsed lazily where used. Advisor/synthesis temperature is clamped <= 0.3.
    llm_model_reasoning: str = ""
    llm_model_synthesis: str = ""
    llm_temperature_reasoning: str = ""
    llm_temperature_synthesis: str = ""
    llm_max_tokens_reasoning: str = ""
    llm_max_tokens_synthesis: str = ""

    # ── Guardrails (C) ───────────────────────────────────────────────────────
    # NOTE: .env.example documents these as `true` (recommended). The code
    # default here follows ShipSmart-API task C ("default true"): guardrails are
    # on unless explicitly disabled. Set GUARDRAILS_ENABLED=false for the legacy
    # passthrough.
    guardrails_enabled: bool = True
    guardrails_block_on_injection: bool = True

    # ── Hybrid retrieval (F) ─────────────────────────────────────────────────
    rag_hybrid: bool = False            # false = dense-only (today); true = dense + sparse
    rag_hybrid_alpha: float = 0.5       # dense weight in fusion (0..1; 1.0 = all dense)

    # ── Iterative RAG (G) ─────────────────────────────────────────────────────
    rag_mode: str = "normal"            # normal (single-shot, today) | iterative
                                        # ("agentic" is a deprecated alias for "iterative")
    # Cost bound on the iterative loop. RAG_AGENTIC_MAX_STEPS is a deprecated env
    # alias accepted for legacy .env compatibility.
    rag_iterative_max_steps: int = Field(
        3, validation_alias=AliasChoices("rag_iterative_max_steps", "rag_agentic_max_steps"),
    )
    rag_query_log: bool = False         # write iterative-RAG traces to rag_query_log (best-effort)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def home_country(self) -> str:
        """The configured domestic home country, normalized to upper ISO-2."""
        return (self.domestic_country or "US").strip().upper()

    @property
    def is_domestic_scope(self) -> bool:
        """True when the deployment only ships within the home country."""
        return self.shipping_scope.strip().lower() == "domestic"

    @property
    def workflow_high_risk_areas_set(self) -> frozenset[str]:
        """Parse WORKFLOW_HIGH_RISK_AREAS (CSV) into a normalized set."""
        return frozenset(
            a.strip().lower()
            for a in self.workflow_high_risk_areas.split(",")
            if a.strip()
        )


# Singleton — import this wherever config is needed
settings = Settings()
