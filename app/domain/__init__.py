"""
Domain ports & adapters — the swappable mock boundary.

Each external domain (classification, duty rates, carrier quotes, document
rendering) is a ``Protocol`` (port) in ``app.domain.ports`` with a deterministic
``Mock*Adapter`` in ``app.domain.adapters``. Agents depend on the port, never a
concrete adapter (Dependency Inversion); wiring picks the adapter in
``app.bootstrap``. Swapping a mock for a real backend is an adapter change, not an
architecture change.

Populated in Phase 2 (UC3).
"""
