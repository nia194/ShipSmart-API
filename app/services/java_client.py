"""Back-compat shim — the Java API client moved to
:mod:`app.integrations.java_client`. New first-party code imports from there.
"""

from app.integrations.java_client import JavaApiClient

__all__ = ["JavaApiClient"]
