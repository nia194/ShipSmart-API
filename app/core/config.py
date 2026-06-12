"""
Application configuration.
Settings are loaded from environment variables using pydantic-settings.
Set values in .env for local dev.
In production (Render), set them via the Render dashboard.
"""

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

    # ── Agent (Concierge) ────────────────────────────────────────────────────
    # Model-driven, read-only tool-calling loop over the MCP tools + retrieve_rag.
    agent_enabled: bool = True          # gate POST /api/v1/agent/run
    agent_max_steps: int = 5            # hard cost bound on the agent loop

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

    # ── Agentic RAG (G) ──────────────────────────────────────────────────────
    rag_mode: str = "normal"            # normal (single-shot, today) | agentic
    rag_agentic_max_steps: int = 3      # cost bound on the agentic loop
    rag_query_log: bool = False         # write agentic traces to rag_query_log (best-effort)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


# Singleton — import this wherever config is needed
settings = Settings()
