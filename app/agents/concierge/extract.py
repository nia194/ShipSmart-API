"""Entity + intent extraction for the Conversational Concierge.

Deterministic-first (regex/keyword) so the whole loop runs keyless — the
hermetic test profile and the Echo/Scripted providers exercise exactly this
path. An OPTIONAL structured LLM call fills gaps when a real reasoning provider
is wired; it is best-effort and fully guarded (any failure → the deterministic
result stands). Extraction returns ONLY what the current message adds.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agents.concierge.models import SLOT_KEYS, Slots
from app.llm.guardrails import assemble
from app.llm.router import TASK_REASONING, LLMRouter

# ── tiny, US-centric country recognition ────────────────────────────────────
_COUNTRIES = {
    "us": "US", "usa": "US", "united states": "US", "america": "US",
    "brazil": "BR", "germany": "DE", "canada": "CA", "mexico": "MX",
    "uk": "GB", "united kingdom": "GB", "england": "GB", "france": "FR",
    "india": "IN", "china": "CN", "japan": "JP", "australia": "AU",
}
_US_STATE = re.compile(r",\s*[A-Za-z]{2}\b")
_VERB_HEAD = re.compile(
    r"^(ship|shipping|send|sending|mail|mailing|move|moving|want|need|deliver)\b", re.I,
)

# A place-like token: 1–2 words, optional ", ST" suffix. Place-likeness (below)
# decides whether a "X to Y" really names a route vs. an unrelated phrase.
_CITY = r"[A-Za-z][A-Za-z.'-]+(?:\s+[A-Za-z][A-Za-z.'-]+)?(?:\s*,\s*[A-Za-z]{2})?"
_ROUTE_END = r"(?=[,.;!?]|\s+(?:by|for|weighing|with|at|on)\b|$)"
_ROUTE_FROM = re.compile(rf"\bfrom\s+({_CITY})\s+to\s+({_CITY}){_ROUTE_END}", re.I)
_ROUTE_TO = re.compile(rf"\b({_CITY})\s+(?:to|->|→)\s+({_CITY}){_ROUTE_END}", re.I)

_WEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(kgs?|kilograms?|lbs?|pounds?)", re.I)
_DIMS = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", re.I)
_VALUE = re.compile(r"\$\s?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:usd|dollars)", re.I)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DESC = re.compile(
    r"\b(?:ship(?:ping)?|send(?:ing)?|mail(?:ing)?)\s+(?:a|an|some|my|the)\s+"
    r"([a-z][a-z0-9 \-]{1,40}?)(?=\s+(?:from|to|by|weighing|that|which|worth)\b|[.?!,]|$)",
    re.I,
)
_TRACKING = re.compile(r"\b(?:tracking(?:\s+(?:number|no\.?|#))?\s*[:#]?\s*)?([A-Z0-9]{10,})\b")

_PRIORITY = [
    (re.compile(r"\b(cheap(?:est)?|lowest price|budget)\b", re.I), "price"),
    (re.compile(r"\b(fast(?:est)?|urgent|asap|quick(?:est)?|overnight)\b", re.I), "speed"),
    (re.compile(r"\b(on[- ]?time|guarantee[d]?|deadline)\b", re.I), "ontime"),
    (re.compile(r"\b(fragile|breakable|damage)\b", re.I), "damage"),
]
_INTENT = [
    (
        re.compile(
            r"\b(complian|customs|prohibit|restrict|allowed|legal|declare|duty|"
            r"hs\s?code|lithium|hazmat|dangerous)\b",
            re.I,
        ),
        "compliance",
    ),
    (re.compile(r"\b(track|where\s+is|status\s+of|delivered\s+yet|delayed)\b", re.I), "tracking"),
    (
        re.compile(
            r"\b(cheap|fast|option|quote|cost|how\s+much|price|rate|compare|ship)\b", re.I,
        ),
        "quote",
    ),
]


def _country(text: str) -> str | None:
    t = " ".join(text.strip().lower().split())
    if t in _COUNTRIES:
        return _COUNTRIES[t]
    if _US_STATE.search(text):
        return "US"
    return None


def _looks_like_place(token: str) -> bool:
    """A route endpoint is trusted only when it reads like a place — a known
    country, a "City, ST", or Title-cased word(s) — so "my shipment to Brazil"
    doesn't get parsed as a route."""
    t = " ".join(token.strip().split())
    if not t:
        return False
    if t.lower() in _COUNTRIES:
        return True
    if _US_STATE.search(t):
        return True
    return all(w[:1].isupper() for w in re.split(r"[\s,]+", t) if w)


def extract_deterministic(message: str) -> tuple[str | None, Slots]:
    """Pure regex/keyword extraction — returns ``(intent, new_entities)``."""
    out: Slots = {}
    msg = message.strip()

    route = _ROUTE_FROM.search(msg) or _ROUTE_TO.search(msg)
    if route:
        origin, dest = route.group(1).strip(" ,"), route.group(2).strip(" ,")
        if _looks_like_place(origin) and _looks_like_place(dest):
            out["origin"] = origin
            if _country(origin):
                out["origin_country"] = _country(origin)
            out["destination"] = dest
            if _country(dest):
                out["destination_country"] = _country(dest)

    # standalone country mentions ("... to Brazil", "from Germany ...")
    for word, code in _COUNTRIES.items():
        if re.search(rf"\bto\s+{re.escape(word)}\b", msg, re.I):
            out.setdefault("destination_country", code)
        if re.search(rf"\bfrom\s+{re.escape(word)}\b", msg, re.I):
            out.setdefault("origin_country", code)

    if (w := _WEIGHT.search(msg)):
        val = float(w.group(1))
        if w.group(2).lower().startswith(("kg", "kilo")):
            val = round(val * 2.20462, 2)
        out["weight_lbs"] = val

    if (d := _DIMS.search(msg)):
        out["length_in"] = float(d.group(1))
        out["width_in"] = float(d.group(2))
        out["height_in"] = float(d.group(3))

    if (v := _VALUE.search(msg)):
        raw = (v.group(1) or v.group(2) or "").replace(",", "")
        if raw:
            out["declared_value_usd"] = float(raw)

    if (dt := _ISO_DATE.search(msg)):
        out["expected_delivery_date"] = dt.group(1)

    if (desc := _DESC.search(msg)):
        text = desc.group(1).strip()
        if text and not _VERB_HEAD.match(text) and not re.match(
            r"^(from|to|by|at|with)\b", text, re.I,
        ):
            out["description"] = text

    if (tr := _TRACKING.search(message)):  # case-sensitive: codes are upper
        out["tracking_reference"] = tr.group(1)

    for pat, label in _PRIORITY:
        if pat.search(msg):
            out["priority"] = label
            break

    intent: str | None = None
    for pat, label in _INTENT:
        if pat.search(msg):
            intent = label
            break

    return intent, out


async def extract(
    message: str,
    llm_router: LLMRouter | None = None,
    *,
    request_id: str = "",
) -> tuple[str | None, Slots]:
    """Deterministic extraction, with an optional LLM assist filling any gaps."""
    intent, slots = extract_deterministic(message)
    for key, value in (await _llm_extract(message, llm_router, request_id=request_id)).items():
        slots.setdefault(key, value)  # deterministic wins; the model only fills gaps
    return intent, slots


async def _llm_extract(
    message: str, llm_router: LLMRouter | None, *, request_id: str = "",
) -> Slots:
    """Best-effort structured extraction. {} for keyless/echo or on any error."""
    if llm_router is None:
        return {}
    try:
        client = llm_router.for_task(TASK_REASONING)
    except Exception:
        return {}
    if getattr(client, "provider_name", "") in ("", "echo", "scripted"):
        return {}
    system = (
        "Extract shipping entities from the user message as STRICT JSON using only "
        "these keys when clearly present: " + ", ".join(SLOT_KEYS) + ". Use yyyy-mm-dd "
        "for dates, numbers for weight/dims/value, ISO-3166 alpha-2 for countries. "
        "Return {} when nothing is found. Output JSON only, no prose."
    )
    assembled = assemble(
        system_prompt=system, user_text=message, contexts=[], request_id=request_id,
    )
    if assembled.blocked:
        return {}
    try:
        text = (await client.complete(assembled.messages) or "").strip()
        data: Any = json.loads(text)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in SLOT_KEYS and v not in (None, "")}
