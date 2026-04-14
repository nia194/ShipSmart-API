"""Tests for configuration loading."""

from app.core.config import Settings


def test_config_defaults():
    """Settings should load with sensible defaults even without .env file."""
    s = Settings(_env_file=None)
    assert s.app_env == "development"
    assert s.app_port == 8000
    assert s.app_name == "shipsmart-api-python"
    assert s.log_level == "INFO"
    assert s.llm_provider == ""
    assert s.rag_provider == ""


def test_cors_origins_list():
    s = Settings(_env_file=None, cors_allowed_origins="http://a.com, http://b.com")
    assert s.cors_origins_list == ["http://a.com", "http://b.com"]


def test_is_production():
    s = Settings(_env_file=None, app_env="production")
    assert s.is_production is True

    s2 = Settings(_env_file=None, app_env="development")
    assert s2.is_production is False
