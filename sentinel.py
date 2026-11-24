"""Sentinel agent - perception. Detects marketplace disruptions with the anomaly model and
classifies them using external signals pulled through MCP tools (events, weather alerts)."""
from __future__ import annotations

import re

import pandas as pd

from surgeshield.a2a.protocol import AgentSkill
from surgeshield.agents.base import BaseAgent
from surgeshield.geo import ZONE_BY_ID, neighbors, zone_name
from surgeshield.mcp_server import tools
from surgeshield.ml.anomaly import score_frame
from surgeshield.world import get_world


class SentinelAgent(BaseAgent):
    name = "sentinel"
    description = "Detects demand/supply disruptions in NYC ride-hail zones and classifies their cause."
    skills = [
        AgentSkill(id="detect", name="Detect disruptions", tags=["anomaly-detection", "ml"],
                   description="Scan a time window of the zone x 15-min panel and return disruption incidents.",
                   examples=['{"window_start": "2024-03-16T22:00", "window_end": "2024-03-16T23:00"}']),
        AgentSkill(id="classify", name="Classify disruption", tags=["signals", "mcp"],
                   description="Classify cause (arena let-out, airport bank, storm, transit outage) from signals."),
    ]

    def skill_detect(self, window_start: str, window_end: str) -> dict:
        w = get_world()
        scored = score_frame(w.window(window_start, window_end), w.baseline)
        alerts = w.detector.detect(scored).sort_values(["bin_start", "z"], ascending=[True, False])
        incidents: list[dict] = []
        used: set[tuple] = set()
        for _, a in alerts.iterrows():
            key = (int(a["zone_id"]), a["bin_start"])
            if any(int(a["zone_id"]) in i["affected_zones"] and
                   abs(pd.Timestamp(i["detected_at"]) - a["bin_start"]) <= pd.Timedelta(minutes=60) for i in incidents):
                continue
            if key in used:
                continue
            same_bin = scored[(scored["bin_start"] == a["bin_start"])]
            nb = set(neighbors(int(a["zone_id"]))) | {int(a["zone_id"])}
            hot = same_bin[(same_bin["demand_vs_base"] > 1.4) & same_bin["zone_id"].isin(nb)]
            affected = sorted(set(hot["zone_id"].astype(int)) | {int(a["zone_id"])})
            # city-wide: many zones alerted in the same bin -> treat as one incident
            alerted_now = alerts[alerts["bin_start"] == a["bin_start"]]["zone_id"].astype(int).tolist()
            if len(alerted_now) >= 6:
                affected = sorted(set(affected) | set(alerted_now))
            incidents.append({
                "incident_id": f"inc-{int(a['zone_id'])}-{a['bin_start']:%Y%m%d%H%M}",
                "epicenter_zone": int(a["zone_id"]), "affected_zones": affected,
                "detected_at": str(a["bin_start"]), "z_score": round(float(a["z"]), 2),
                "demand_vs_base": round(float(a["demand_vs_base"]), 2), "wait_med": round(float(a["wait_med"]), 2),
                "kind": "unknown", "signals": [f"anomaly z={a['z']:.1f}", f"iso={a['iso_score']:.2f}"]})
            used.add(key)
        return {"output": {"incidents": incidents},
                "summary": f"{len(incidents)} disruption(s) in {window_start}..{window_end}"}

    def skill_classify(self, disruption: dict) -> dict:
        d = dict(disruption)
        at, zones = d["detected_at"], d["affected_zones"]
        events = tools.get_events(at, zones)
        alerts = tools.get_weather_alerts(at)
        texts = [e["text"] for e in events] + [a["text"] for a in alerts]
        blob = " ".join(texts).lower()
        venue = ZONE_BY_ID[d["epicenter_zone"]].venue
        if re.search(r"subway|signal|suspend|outage|service change", blob):
            kind = "transit_outage"
        elif venue == "airport" or "flight" in blob or "arrival" in blob:
            kind = "airport_bank"
        elif re.search(r"rain|storm|snow|flood|wind", blob) or len(zones) >= 6:
            kind = "rainstorm"
        elif venue == "arena" or re.search(r"concert|game|garden|arena", blob):
            kind = "arena_letout"
        else:
            kind = "demand_spike"
        d["kind"] = kind
        d["declared_emergency"] = any(x.get("declared_emergency") for x in events + alerts)
        d["context_notes"] = " | ".join(texts)[:1500]
        d["signals"] = list(d.get("signals", [])) + [f"{e.get('source', 'event')}" for e in events] + \
            [f"weather:{a.get('event')}" for a in alerts]
        if any(e.get("injection_flagged") for e in events):
            d["signals"].append("injection_flagged")
        return {"output": {"disruption": d},
                "summary": f"{kind} at {zone_name(d['epicenter_zone'])} (emergency={d['declared_emergency']})"}
