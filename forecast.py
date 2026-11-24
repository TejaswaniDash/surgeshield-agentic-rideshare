"""Forecast agent - deep learning demand forecast (LSTM) + ML wait-time model + episodic recall."""
from __future__ import annotations

import numpy as np
import pandas as pd

from surgeshield.a2a.protocol import AgentSkill
from surgeshield.agents.base import BaseAgent
from surgeshield.geo import ZONE_INDEX, neighbors, zone_name
from surgeshield.mcp_server import tools
from surgeshield.memory.store import EpisodicMemory
from surgeshield.ml.rl.market_sim import wait_minutes
from surgeshield.ml.wait_model import predict_wait
from surgeshield.schemas import Assessment, Disruption, ZoneForecast
from surgeshield.world import get_world


def assessed_zones(d: Disruption) -> list[int]:
    zs = list(dict.fromkeys(d.affected_zones + neighbors(d.epicenter_zone)))
    return zs[:10]


class ForecastAgent(BaseAgent):
    name = "forecaster"
    description = "Forecasts next-hour demand per zone (LSTM) and predicted rider wait (GBM); rates severity."
    skills = [AgentSkill(id="assess", name="Assess disruption", tags=["forecasting", "deep-learning"],
                         description="Multi-horizon demand forecast + wait prediction + similar past incidents.")]

    def skill_assess(self, disruption: dict) -> dict:
        w = get_world()
        d = Disruption.model_validate(disruption)
        at = pd.Timestamp(d.detected_at)
        live = w.live.get(d.incident_id)
        base = w.baseline_demand(at)
        snap = w.snapshot(at).set_index("zone_id") if not w.snapshot(at).empty else None
        forecasts = []
        for z in assessed_zones(d):
            hist = w.history(z, at, w.forecaster.lookback)
            exp = w.forecaster.predict_zone(hist, z) if len(hist) >= w.forecaster.lookback else np.repeat(base[ZONE_INDEX[z]], 4)
            current = float(snap.loc[z, "requests"]) if snap is not None and z in snap.index else exp[0]
            # nowcast blending: the LSTM learned seasonality; an in-progress shock is better captured by
            # persistence of the latest observation decaying toward the model forecast
            persist = current * 0.85 ** np.arange(1, len(exp) + 1)
            exp = np.maximum(exp, persist)
            # available next interval = idle now + drivers finishing trips (busy queue head)
            idle = float(live.sim.idle[ZONE_INDEX[z]] + live.sim.busy[0][ZONE_INDEX[z]]) if live \
                else float(base[ZONE_INDEX[z]] * 1.25)
            row = pd.DataFrame([{"zone_id": z, "requests": max(exp[0], current), "dropoffs": idle,
                                 "dow": at.dayofweek, "slot": at.hour * 4 + at.minute // 15, "hour": at.hour}])
            ml_wait = float(predict_wait(w.wait_model, row, w.baseline)[0])
            sim_wait = float(wait_minutes(np.array([max(exp[0], current)]), np.array([idle]))[0])
            forecasts.append(ZoneForecast(
                zone_id=z, zone_name=zone_name(z), expected_requests=[round(float(x), 1) for x in exp],
                baseline_requests=round(float(base[ZONE_INDEX[z]]), 1), idle_supply=round(idle, 1),
                gap_next=round(float(exp[0] - idle), 1), predicted_wait_min=round(min(max(ml_wait, sim_wait), 45.0), 1)))
        peak = max(f.predicted_wait_min for f in forecasts)
        unmet = float(sum(max(0.0, sum(f.expected_requests) - 4 * f.idle_supply) for f in forecasts))
        severity = "low" if peak < 6 else "medium" if peak < 10 else "high" if peak < 15 else "critical"
        similar = [f"{s['incident_id']} ({s['kind']}): " + "; ".join(s["lessons"][:2])
                   for s in EpisodicMemory().similar(d.kind, d.affected_zones, d.demand_vs_base)]
        similar += [f"{h['title']}: {h['text'][-220:]}" for h in
                    tools.search_policies(f"{d.kind} {zone_name(d.epicenter_zone)} lessons", k=2, doc_type="postmortem")]
        a = Assessment(severity=severity, forecasts=forecasts, predicted_peak_wait_min=round(peak, 1),
                       expected_unmet_next_hour=round(unmet, 1), similar_incidents=similar[:4],
                       model_versions={"forecaster": "lstm-v1", "wait_model": "hgb-v1"})
        return {"output": {"assessment": a.model_dump()},
                "summary": f"severity={severity}, peak wait {peak:.1f} min, unmet≈{unmet:.0f}/h"}
