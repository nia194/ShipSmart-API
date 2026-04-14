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

    # ── Providers ────────────────────────────────────────────────────────────
    shipping_provider: str = "mock"  # "mock", "ups", "fedex", "dhl", "usps"

    # ── UPS ──────────────────────────────────────────────────────────────────
    ups_client_id: str = ""
    ups_client_secret: str = ""
    ups_account_number: str = ""
    ups_base_url: str = "https://onlinetools.ups.com"

    # ── FedEx ────────────────────────────────────────────────────────────────
    fedex_client_id: str = ""
    fedex_client_secret: str = ""
    fedex_account_number: str = ""
    fedex_base_url: str = "https://apis.fedex.com"

    # ── DHL ──────────────────────────────────────────────────────────────────
    dhl_api_key: str = ""
    dhl_api_secret: str = ""
    dhl_account_number: str = ""
    dhl_base_url: str = "https://express.api.dhl.com"

    # ── USPS ─────────────────────────────────────────────────────────────────
    usps_client_id: str = ""
    usps_client_secret: str = ""
    usps_base_url: str = "https://api.usps.com"

    # ── Tools ─────────────────────────────────────────────────────────────────
    enable_tools: bool = True

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_provider: str = ""
    rag_top_k: int = 3
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_documents_path: str = "data/documents"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


# Singleton — import this wherever config is needed
settings = Settings()
