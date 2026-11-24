"""Tool implementations shared by (a) the MCP server and (b) agents calling tools in-process.

Live mode (SURGESHIELD_LIVE_SIGNALS=1) calls public APIs:
  - NWS active alerts:  https://api.weather.gov/alerts/active?area=NY
  - NYC permitted events (NYC Open Data, Socrata): https://data.cityofnewyork.us/resource/tvpp-9vvx.json
Offline mode serves scenario signals registered on the MarketWorld.
All free text returned by tools is screened for prompt injection before an LLM sees it.
"""
from __future__ import annotations

import os
from typing import Any

from surgeshield.config import get_settings
from surgeshield.geo import ZONE_BY_ID, zone_name
from surgeshield.llm.guardrails import screen_untrusted

LIVE = os.environ.get("SURGESHIELD_LIVE_SIGNALS") == "1"


def get_zone_snapshot(zone_id: int, at: str) -> dict[str, Any]:
    """Demand, supply proxy and rider wait for a TLC zone in the 15-min bin containing `at`."""
    from surgeshield.world import get_world
    snap = get_world().snapshot(at)
    row = snap[snap["zone_id"] == zone_id]
    if row.empty:
        return {"zone_id": zone_id, "found": False}
    r = row.iloc[0]
    return {"zone_id": zone_id, "zone_name": zone_name(zone_id), "bin_start": str(r.get("bin_start", at)),
            "requests": int(r["requests"]), "dropoffs": int(r["dropoffs"]), "wait_med_min": round(float(r["wait_med"]), 2),
            "wav_requests": int(r["wav_requests"])}


def get_weather_alerts(at: str) -> list[dict]:
    """Active weather alerts for NYC (NWS). Offline: scenario-registered alerts."""
    if LIVE:
        import httpx
        r = httpx.get("https://api.weather.gov/alerts/active", params={"area": "NY"},
                      headers={"User-Agent": "surgeshield-demo (ops@example.com)"}, timeout=20)
        feats = r.json().get("features", [])
        return [{"event": f["properties"].get("event"), "severity": f["properties"].get("severity"),
                 "text": screen_untrusted(f["properties"].get("headline") or "").text,
                 "declared_emergency": f["properties"].get("severity") in ("Extreme", "Severe")} for f in feats[:10]]
    from surgeshield.world import get_world
    return [{"event": s.source, "text": screen_untrusted(s.text).text, "zones": s.zones,
             "declared_emergency": s.declared_emergency}
            for s in get_world().signals_for(at) if s.source == "nws_alerts"]


def get_events(at: str, zone_ids: list[int] | None = None) -> list[dict]:
    """Permitted / venue events near the zones around `at`. Text is screened (untrusted)."""
    if LIVE:
        import httpx
        r = httpx.get("https://data.cityofnewyork.us/resource/tvpp-9vvx.json",
                      params={"$limit": 50, "$where": f"start_date_time <= '{at}' AND end_date_time >= '{at}'",
                              "event_borough": "Manhattan"}, timeout=20)
        return [{"event": e.get("event_name"), "location": e.get("event_location"), "type": e.get("event_type"),
                 "text": screen_untrusted(str(e.get("event_name", ""))).text} for e in r.json()]
    from surgeshield.world import get_world
    out = []
    for s in get_world().signals_for(at, zone_ids):
        if s.source in ("nyc_events", "venue_calendar", "flight_status", "mta_status"):
            scr = screen_untrusted(s.text)
            out.append({"source": s.source, "zones": s.zones, "text": scr.text, "injection_flagged": scr.flagged,
                        "declared_emergency": s.declared_emergency})
    return out


def search_policies(query: str, k: int = 4, doc_type: str | None = None) -> list[dict]:
    """Hybrid (BM25 + dense, RRF) search over regulations, internal policy and postmortems."""
    from surgeshield.rag.index import get_index
    hits = get_index().search(query, k=k, filters={"doc_type": doc_type} if doc_type else None)
    return [{"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "section": h.chunk.section,
             "source_url": h.chunk.source_url, "doc_type": h.chunk.meta.get("doc_type"), "text": h.chunk.text,
             "score": round(h.score, 4)} for h in hits]


def tlc_min_driver_pay(miles: float, minutes: float, wav: bool = False) -> dict:
    """NYC TLC HVFHS per-trip minimum (rates from config; source: nyc.gov driver-pay-rates page)."""
    s = get_settings().policy
    per_mile = float(s.get("tlc_min_per_mile_wav" if wav else "tlc_min_per_mile", 1.601 if wav else 1.283))
    per_min = float(s.get("tlc_min_per_minute", 0.681))
    return {"min_pay_usd": round(per_mile * miles + per_min * minutes, 2), "per_mile": per_mile, "per_minute": per_min}


def zone_info(zone_id: int) -> dict:
    z = ZONE_BY_ID[zone_id]
    return {"zone_id": z.id, "name": z.name, "borough": z.borough, "venue": z.venue, "lat": z.lat, "lon": z.lon}


TOOLS = {f.__name__: f for f in (get_zone_snapshot, get_weather_alerts, get_events, search_policies,
                                 tlc_min_driver_pay, zone_info)}
