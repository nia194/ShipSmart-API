"""
Compare service for decision-cockpit shipping comparisons.

Real facts only. No fabricated data.
LLM generates structured decision-support content grounded in quote facts.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings
from app.llm.client import LLMClient
from app.schemas.compare import (
    CompareOption,
    CompareRequest,
    CompareResponse,
    ComparisonDimension,
    DecisionFactors,
    OptionInsight,
    Scenario,
    Verdict,
)

logger = logging.getLogger(__name__)


class CompareCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 900):
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[datetime, Any]] = {}

    def get(self, key: str) -> Any | None:
        if key not in self.cache:
            return None
        stored_at, value = self.cache[key]
        if datetime.now(timezone.utc) - stored_at > timedelta(seconds=self.ttl_seconds):
            del self.cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (datetime.now(timezone.utc), value)


_compare_cache = CompareCache(ttl_seconds=900)


def _make_cache_key(shipment_hash: str, option_ids_sorted: list[str]) -> str:
    ids_str = ",".join(option_ids_sorted)
    combined = f"{shipment_hash}:{ids_str}"
    return hashlib.md5(combined.encode()).hexdigest()


def _hash_shipment(shipment_dict: dict[str, Any]) -> str:
    key_fields = {
        k: v
        for k, v in shipment_dict.items()
        if k not in ["id", "timestamp"]
    }
    json_str = json.dumps(key_fields, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()


def _clean_service_name(carrier: str, service_name: str) -> str:
    """Remove redundant carrier prefix from service name.

    e.g. carrier='FedEx', service_name='FedEx Express Saver' -> 'Express Saver'
         carrier='UPS', service_name='UPS 2nd Day Air' -> '2nd Day Air'
         carrier='LuggageToShip', service_name='LuggageToShip Economy' -> 'Economy'
         carrier='USPS', service_name='Priority Mail' -> 'Priority Mail' (no prefix)
    """
    if service_name.startswith(carrier + " "):
        return service_name[len(carrier) + 1:]
    return service_name


def _build_decision_context(
    request: CompareRequest, options: list[CompareOption]
) -> dict[str, Any]:
    """Build JSON context for LLM with real facts only."""
    options_data = []
    for opt in options:
        clean_name = _clean_service_name(opt.carrier, opt.service_name)
        options_data.append(
            {
                "id": opt.id,
                "carrier": opt.carrier,
                "service_name": clean_name,
                "display_name": f"{opt.carrier} {clean_name}",
                "price_usd": opt.price_usd,
                "arrival_date": opt.arrival_date,
                "arrival_label": opt.arrival_label,
                "transit_days": opt.transit_days,
                "guaranteed": opt.guaranteed,
                "carrier_type": opt.carrier_type,
            }
        )

    return {
        "shipment": {
            "item": request.shipment.item_description,
            "origin_zip": request.shipment.origin_zip,
            "destination_zip": request.shipment.destination_zip,
            "deadline_date": request.shipment.deadline_date,
            "weight_lb": request.shipment.weight_lb,
        },
        "options": options_data,
        "user_priority": request.selected_priority,
    }


def _build_llm_prompt(context: dict[str, Any]) -> str:
    """Build the LLM prompt for rich decision-support content."""
    json_context = json.dumps(context, indent=2)
    user_priority = context.get("user_priority", "ontime")

    priority_descriptions = {
        "ontime": "on-time delivery (meeting the deadline reliably)",
        "damage": "damage protection (safe handling, guaranteed service as a proxy since no real damage data exists)",
        "price": "lowest total cost",
        "speed": "earliest possible arrival",
    }
    priority_desc = priority_descriptions.get(user_priority, user_priority)

    return f"""You are the decision engine for ShipSmart's compare section. You produce structured comparison content that helps users choose between 2-3 shipping options. Your output must be grounded entirely in the real quote data provided.

The user's selected priority is: {priority_desc}

For EACH of the 4 scenarios (ontime, damage, price, speed), produce:

1. verdict:
   - purpose: "Best for [priority]" (6 words max)
   - pick_name: Use the display_name from the data for the winner. IMPORTANT: Never repeat the carrier name twice (wrong: "FedEx FedEx Express Saver", correct: "FedEx Express Saver").
   - reason: 1-2 sentences. State what makes this option win AND what it costs. Use specific dollar amounts, dates, and transit days from the data.
   - context_note: 1 sentence. Explain what about THIS specific shipment (the item being shipped, the weight, the route, the deadline) makes this recommendation particularly relevant. Be specific — reference the actual deadline date, weight, or route.
   - override_note: 1 sentence. State when the user should reasonably pick a DIFFERENT option instead. Reference a concrete condition. Example: "If your deadline is flexible past Apr 25, the $12 savings with USPS becomes the smarter call." Do not write a vague caveat.

2. option_insights: For EACH option, produce a rich, multi-sentence decision brief shown directly under that option's column. This is the most important text the user reads — it must feel like a shipping advisor explaining the option in context.
   - option_id: the option ID
   - role_label: A short positioning label. Examples: "Best for urgency", "Budget pick", "Best balanced", "Best if certainty matters", "Best if deadline is flexible". 2-5 words. Each option must get a DISTINCT label.
   - strength: 2-3 sentences. Explain what this option does well for THIS specific shipment. Reference concrete facts — price, transit days, arrival date, guarantee status — and connect them to the user's situation (the deadline, the route, the weight). Avoid generic phrases like "good choice" or "reliable service"; every sentence should carry a fact or a tradeoff the user can act on.
   - consideration: 2-3 sentences. The honest tradeoff. What does the user give up by choosing this? Use exact numbers (e.g., "$11.70 more than USPS", "one day later than FedEx Priority"). Explain why the tradeoff matters (or doesn't) for this deadline and priority.
   - choose_when: 1 sentence starting with a condition. Example: "Choose this when your deadline is firm and you need guaranteed arrival."
   - skip_when: 1 sentence starting with a condition. Example: "Skip this when budget matters more than speed — you'd pay $15 extra for 1 day faster."
   - card_tag: 4-8 word factual tag using middle-dot separator (e.g., "Guaranteed · Apr 19 · $48.50")

3. comparison_dimensions: Emit EXACTLY these six dimensions, in this order, with these exact names:

   ANCHORS (high-level summary):
   - "Price": Actual dollar amount (e.g., "$45.20"). Set winner_id to the cheapest option.
   - "Speed": Transit days + arrival label (e.g., "2 days · Fri, Apr 19"). Set winner_id to the fastest option.
   - "Reliability": "Guaranteed" if the carrier guarantees the delivery date, otherwise "Not guaranteed". Set winner_id to the guaranteed option if exactly one option is guaranteed; leave winner_id empty otherwise.

   DETAILS (granular specs — use well-known, standard carrier service terms only, NEVER invent numbers):
   - "Insurance": Standard included coverage language per carrier. Examples: FedEx/UPS/DHL → "Up to $100 included"; USPS Priority Mail → "Up to $100 included"; USPS Ground Advantage → "Up to $100 included"; Specialty/private carriers → "Carrier-specific coverage" or "Declared-value based" if unknown. Use a short phrase, no invented claim rates.
   - "Tracking": Describe the standard tracking experience. Examples: FedEx/UPS/DHL Express → "Real-time tracking"; USPS → "Scan-based tracking"; Private specialists → "Milestone tracking" or "Carrier tracking". Short phrase only.
   - "Handling": Describe handling character. Examples: FedEx Express/Priority Overnight, UPS Next Day Air → "Priority handling"; standard ground services → "Standard parcel handling"; luggage/specialty shippers → "Specialist handling". Short phrase only.

   For each dimension, "values" must include every option_id with a short value string. Leave winner_id empty ("") on Insurance, Tracking, and Handling UNLESS one option is clearly superior by standard service definition. Add a short "note" (optional, 1 sentence) contextualizing this dimension for the CURRENT PRIORITY scenario. Omit the note if it would be filler.

4. decision_summary: 2-3 sentences. Explain HOW these options differ in the context of this priority. Not a restatement of the verdict. Help the user understand the tradeoff landscape. Be specific — use dollar differences, day differences, guarantee status.

5. decision_factors:
   - primary_driver: 1 sentence. The single most important factor for this shipment + priority. Example: "Your Apr 23 deadline is 5 days away, making guaranteed delivery the decisive factor."
   - key_tradeoff: 1 sentence. The core tension. Example: "You're choosing between $45.20 guaranteed arrival on Apr 21 vs. $33.50 non-guaranteed arrival on Apr 22 — one day and $11.70 apart."
   - what_would_change: 1 sentence. What would flip the recommendation. Example: "If your deadline moved to Apr 28, the price difference would matter more than the speed difference."

Rules:
- Use ONLY specific numbers, dates, and facts from the data provided.
- NEVER invent reliability rates, insurance details, coverage data, historical performance, claims statistics, or confidence metrics.
- NEVER repeat the carrier name redundantly. The display_name field already has "Carrier ServiceName" — use it as-is.
- If two options are effectively tied on a dimension (within $2 AND same arrival), say so.
- Sentence case. No ALL CAPS. No marketing fluff. No "AI recommends" language. No "our analysis shows" language.
- Write like a knowledgeable shipping advisor talking directly to the user.
- The damage scenario should use guaranteed delivery and carrier type as proxies for care. Be honest that actual damage/claims data is not available.

Return ONLY valid JSON matching this schema (no preamble, no markdown fencing):
{{
  "ontime": {{
    "winner_id": "string",
    "verdict": {{"purpose": "string", "pick_name": "string", "reason": "string", "context_note": "string", "override_note": "string"}},
    "option_insights": [
      {{"option_id": "string", "role_label": "string", "strength": "string", "consideration": "string", "choose_when": "string", "skip_when": "string", "card_tag": "string"}}
    ],
    "comparison_dimensions": [
      {{"dimension": "string", "values": {{"option_id": "value_string"}}, "winner_id": "string_or_empty", "note": "string"}}
    ],
    "decision_summary": "string",
    "decision_factors": {{
      "primary_driver": "string",
      "key_tradeoff": "string",
      "what_would_change": "string"
    }}
  }},
  "damage": {{...same structure...}},
  "price": {{...same structure...}},
  "speed": {{...same structure...}}
}}

Decision context:
{json_context}"""


async def generate_compare_response(
    request: CompareRequest,
    options: list[CompareOption],
    llm_client: LLMClient,
) -> CompareResponse:
    """Generate a complete compare response with all 4 scenarios."""
    shipment_hash = _hash_shipment(request.shipment.dict())
    sorted_ids = sorted(request.option_ids)
    cache_key = _make_cache_key(shipment_hash, sorted_ids)

    cached = _compare_cache.get(cache_key)
    if cached:
        logger.info("Compare cache hit: %s", cache_key)
        return cached

    context = _build_decision_context(request, options)
    prompt = _build_llm_prompt(context)

    logger.info("Calling LLM for compare: %d options", len(options))
    try:
        response_text = await llm_client.complete(
            messages=[
                {"role": "system", "content": "You are a shipping decision engine. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ]
        )
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        llm_response = json.loads(cleaned)
    except Exception as e:
        logger.error("LLM call or JSON parse failed, returning fallback: %s", e)
        llm_response = _fallback_scenarios_dict(options, request)

    scenarios = _validate_and_merge_scenarios(llm_response, options)

    shipment_summary = (
        f"{request.shipment.item_description} · "
        f"{request.shipment.origin_zip} → {request.shipment.destination_zip} · "
        f"must arrive by {request.shipment.deadline_date}"
    )

    response = CompareResponse(
        shipment_summary=shipment_summary,
        scenarios=scenarios,
    )

    _compare_cache.set(cache_key, response)
    logger.info("Compare response cached: %s", cache_key)

    return response


def _validate_and_merge_scenarios(
    llm_data: dict[str, Any], options: list[CompareOption]
) -> dict[str, Scenario]:
    """Validate LLM output and fill gaps with deterministic fallbacks."""
    option_ids = {opt.id for opt in options}
    scenarios: dict[str, Scenario] = {}

    for priority in ["ontime", "damage", "price", "speed"]:
        if priority not in llm_data:
            logger.warning("Missing scenario: %s, using fallback", priority)
            scenarios[priority] = _build_fallback_scenario(options, priority)
            continue

        sd = llm_data[priority]

        # If the LLM returned an already-validated Scenario dict (from fallback),
        # handle that case.
        if isinstance(sd, Scenario):
            scenarios[priority] = sd
            continue

        # winner_id
        winner_id = sd.get("winner_id")
        if winner_id not in option_ids:
            logger.warning("Invalid winner_id %s for %s", winner_id, priority)
            winner_id = options[0].id

        # verdict
        vd = sd.get("verdict", {})
        verdict = Verdict(
            purpose=vd.get("purpose", f"Best for {priority}"),
            pick_name=vd.get("pick_name", ""),
            reason=vd.get("reason", ""),
            context_note=vd.get("context_note", ""),
            override_note=vd.get("override_note", ""),
        )

        # option_insights
        raw_insights = sd.get("option_insights", [])
        insights: list[OptionInsight] = []
        seen_ids: set[str] = set()
        for ins in raw_insights:
            oid = ins.get("option_id")
            if oid in option_ids and oid not in seen_ids:
                seen_ids.add(oid)
                insights.append(
                    OptionInsight(
                        option_id=oid,
                        role_label=ins.get("role_label", ""),
                        strength=ins.get("strength", ""),
                        consideration=ins.get("consideration", ""),
                        choose_when=ins.get("choose_when", ""),
                        skip_when=ins.get("skip_when", ""),
                        card_tag=ins.get("card_tag", ""),
                    )
                )
        for opt in options:
            if opt.id not in seen_ids:
                clean = _clean_service_name(opt.carrier, opt.service_name)
                insights.append(
                    OptionInsight(
                        option_id=opt.id,
                        role_label=f"{opt.carrier} {clean}",
                        strength=f"${opt.price_usd:.2f} with {opt.transit_days}-day transit.",
                        consideration="See comparison details.",
                        choose_when="Consider when comparing overall value.",
                        skip_when="",
                        card_tag=f"{opt.carrier} · ${opt.price_usd:.2f}",
                    )
                )

        # comparison_dimensions
        raw_dims = sd.get("comparison_dimensions", [])
        dimensions: list[ComparisonDimension] = []
        for dim in raw_dims:
            dim_name = dim.get("dimension", "")
            values = dim.get("values", {})
            for opt in options:
                if opt.id not in values:
                    values[opt.id] = "\u2014"
            dim_winner = dim.get("winner_id", "")
            if dim_winner and dim_winner not in option_ids:
                dim_winner = ""
            dimensions.append(
                ComparisonDimension(
                    dimension=dim_name,
                    values=values,
                    winner_id=dim_winner,
                    note=dim.get("note", ""),
                )
            )
        if not dimensions:
            dimensions = _build_fallback_dimensions(options)

        decision_summary = sd.get("decision_summary", "")

        # decision_factors
        raw_factors = sd.get("decision_factors")
        decision_factors = None
        if isinstance(raw_factors, dict):
            decision_factors = DecisionFactors(
                primary_driver=raw_factors.get("primary_driver", ""),
                key_tradeoff=raw_factors.get("key_tradeoff", ""),
                what_would_change=raw_factors.get("what_would_change", ""),
            )

        scenarios[priority] = Scenario(
            winner_id=winner_id,
            verdict=verdict,
            option_insights=insights,
            comparison_dimensions=dimensions,
            decision_summary=decision_summary,
            decision_factors=decision_factors,
        )

    return scenarios


_MAJOR_CARRIERS = {"FedEx", "UPS", "DHL", "USPS"}


def _insurance_value(opt: CompareOption) -> str:
    if opt.carrier in _MAJOR_CARRIERS:
        return "Up to $100 included"
    return "Declared-value based"


def _tracking_value(opt: CompareOption) -> str:
    if opt.carrier in {"FedEx", "UPS", "DHL"}:
        return "Real-time tracking"
    if opt.carrier == "USPS":
        return "Scan-based tracking"
    return "Carrier tracking"


def _handling_value(opt: CompareOption) -> str:
    if opt.guaranteed and opt.carrier_type == "private":
        return "Priority handling"
    if opt.carrier_type == "private":
        return "Standard parcel handling"
    return "Specialist handling"


def _build_fallback_dimensions(options: list[CompareOption]) -> list[ComparisonDimension]:
    """Deterministic comparison dimensions from real quote data (Apple-style hierarchy)."""
    cheapest = min(options, key=lambda o: o.price_usd)
    price_winner = cheapest.id if len(set(o.price_usd for o in options)) > 1 else ""

    fastest = min(options, key=lambda o: o.transit_days)
    speed_winner = fastest.id if len(set(o.transit_days for o in options)) > 1 else ""

    guaranteed_opts = [o for o in options if o.guaranteed]
    reliability_winner = guaranteed_opts[0].id if len(guaranteed_opts) == 1 else ""

    return [
        ComparisonDimension(
            dimension="Price",
            values={o.id: f"${o.price_usd:.2f}" for o in options},
            winner_id=price_winner,
            note="",
        ),
        ComparisonDimension(
            dimension="Speed",
            values={o.id: f"{o.transit_days} days \u00b7 {o.arrival_label}" for o in options},
            winner_id=speed_winner,
            note="",
        ),
        ComparisonDimension(
            dimension="Reliability",
            values={o.id: "Guaranteed" if o.guaranteed else "Not guaranteed" for o in options},
            winner_id=reliability_winner,
            note="",
        ),
        ComparisonDimension(
            dimension="Insurance",
            values={o.id: _insurance_value(o) for o in options},
            winner_id="",
            note="",
        ),
        ComparisonDimension(
            dimension="Tracking",
            values={o.id: _tracking_value(o) for o in options},
            winner_id="",
            note="",
        ),
        ComparisonDimension(
            dimension="Handling",
            values={o.id: _handling_value(o) for o in options},
            winner_id="",
            note="",
        ),
    ]


def _build_fallback_scenario(
    options: list[CompareOption], priority: str
) -> Scenario:
    """Single fallback scenario using real facts only."""
    if priority == "price":
        winner = min(options, key=lambda o: o.price_usd)
    elif priority == "speed":
        winner = min(options, key=lambda o: (o.transit_days, o.price_usd))
    else:
        winner = (
            next((o for o in options if o.guaranteed), None)
            or min(options, key=lambda o: (o.transit_days, o.price_usd))
        )

    clean_winner = _clean_service_name(winner.carrier, winner.service_name)

    insights = []
    for opt in options:
        is_w = opt.id == winner.id
        clean = _clean_service_name(opt.carrier, opt.service_name)
        insights.append(
            OptionInsight(
                option_id=opt.id,
                role_label="Recommended" if is_w else f"{opt.carrier} {clean}",
                strength=(
                    f"{'Guaranteed delivery' if opt.guaranteed else f'{opt.transit_days}-day transit'} at ${opt.price_usd:.2f}."
                ),
                consideration=(
                    f"${opt.price_usd:.2f} total."
                    if is_w
                    else f"${opt.price_usd - winner.price_usd:+.2f} vs recommended."
                ),
                choose_when="When this option fits your priority." if is_w else "",
                skip_when="" if is_w else "When the recommended option is available.",
                card_tag=f"{opt.carrier} \u00b7 ${opt.price_usd:.2f} \u00b7 {opt.arrival_label}",
            )
        )

    return Scenario(
        winner_id=winner.id,
        verdict=Verdict(
            purpose=f"Best for {priority}",
            pick_name=f"{winner.carrier} {clean_winner}",
            reason=f"{winner.carrier} {clean_winner} at ${winner.price_usd:.2f}, arriving {winner.arrival_label}.",
            context_note="",
            override_note="",
        ),
        option_insights=insights,
        comparison_dimensions=_build_fallback_dimensions(options),
        decision_summary="",
        decision_factors=None,
    )


def _fallback_scenarios_dict(
    options: list[CompareOption], request: CompareRequest
) -> dict[str, Any]:
    """Build raw dict fallback for all 4 priorities when LLM fails."""
    result = {}
    for priority in ["ontime", "damage", "price", "speed"]:
        scenario = _build_fallback_scenario(options, priority)
        result[priority] = scenario
    return result
