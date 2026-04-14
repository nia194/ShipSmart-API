"""
Performance check script for post-launch monitoring.
Measures response times for key endpoints on both Python and Java APIs.

Usage:
    python scripts/perf_check.py [python_url] [java_url]

Defaults:
    python_url: http://localhost:8000
    java_url:   http://localhost:8080
"""

import sys
import time

import httpx

PYTHON_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
JAVA_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"

THRESHOLDS = {
    "health": 500,
    "recommendation": 200,
    "advisor": 2000,
    "quote": 3000,
}


def measure(
    base_url: str, method: str, path: str, json: dict | None = None,
) -> tuple[int, float]:
    """Make a request and return (status, duration_ms)."""
    url = f"{base_url}{path}"
    start = time.perf_counter()
    with httpx.Client(timeout=15) as client:
        if method == "GET":
            r = client.get(url)
        else:
            r = client.post(url, json=json)
    duration = (time.perf_counter() - start) * 1000
    return r.status_code, duration


def check_endpoint(
    label: str, base_url: str, method: str, path: str,
    json: dict | None = None, threshold_ms: float = 2000,
) -> bool:
    """Check one endpoint and print result."""
    try:
        status, ms = measure(base_url, method, path, json)
        ok = status < 500
        slow = ms > threshold_ms
        marker = "SLOW" if slow else ("OK" if ok else "FAIL")
        print(f"  {label:<42} {status:>4}  {ms:>8.1f} ms  {marker}")
        return ok and not slow
    except Exception as e:
        print(f"  {label:<42}  ERR  {'---':>8}     {e}")
        return False


def main():
    print(f"ShipSmart Performance Check")
    print(f"  Python API: {PYTHON_URL}")
    print(f"  Java API:   {JAVA_URL}")
    print()

    all_pass = True

    # ── Python API ──
    print("Python API:")
    all_pass &= check_endpoint(
        "GET /health", PYTHON_URL, "GET", "/health",
        threshold_ms=THRESHOLDS["health"],
    )
    all_pass &= check_endpoint(
        "GET /ready", PYTHON_URL, "GET", "/ready",
        threshold_ms=THRESHOLDS["health"],
    )
    all_pass &= check_endpoint(
        "GET /api/v1/info", PYTHON_URL, "GET", "/api/v1/info",
        threshold_ms=THRESHOLDS["health"],
    )

    rec_body = {
        "services": [
            {"service": "Ground", "price_usd": 9.99, "estimated_days": 5},
            {"service": "Express", "price_usd": 19.99, "estimated_days": 2},
            {"service": "Overnight", "price_usd": 49.99, "estimated_days": 1},
        ],
        "context": {"fragile": True},
    }
    all_pass &= check_endpoint(
        "POST /advisor/recommendation", PYTHON_URL, "POST",
        "/api/v1/advisor/recommendation", rec_body,
        threshold_ms=THRESHOLDS["recommendation"],
    )
    all_pass &= check_endpoint(
        "POST /advisor/shipping", PYTHON_URL, "POST",
        "/api/v1/advisor/shipping", {"query": "What carriers are available?"},
        threshold_ms=THRESHOLDS["advisor"],
    )
    all_pass &= check_endpoint(
        "POST /advisor/tracking", PYTHON_URL, "POST",
        "/api/v1/advisor/tracking", {"issue": "My package is delayed"},
        threshold_ms=THRESHOLDS["advisor"],
    )

    # ── Cache test ──
    print("\nCache test (repeat recommendation):")
    _, ms1 = measure(PYTHON_URL, "POST", "/api/v1/advisor/recommendation", rec_body)
    _, ms2 = measure(PYTHON_URL, "POST", "/api/v1/advisor/recommendation", rec_body)
    print(f"  First call:  {ms1:.1f} ms")
    print(f"  Second call: {ms2:.1f} ms {'(cached)' if ms2 < ms1 else ''}")

    # ── Java API ──
    print("\nJava API:")
    all_pass &= check_endpoint(
        "GET /api/v1/health", JAVA_URL, "GET", "/api/v1/health",
        threshold_ms=THRESHOLDS["health"],
    )

    print()
    print(f"Thresholds: health={THRESHOLDS['health']}ms, "
          f"recommendation={THRESHOLDS['recommendation']}ms, "
          f"advisor={THRESHOLDS['advisor']}ms")
    print()

    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks FAILED or SLOW — review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
