"""Pricing agent - sets per-zone surge multipliers from the forecast supply/demand ratio, under the
normal cap or the emergency cap (60-day range excluding the 3 highest multiples)."""
from __future__ import annotations

import re

import numpy as np

from surgeshield.a2a.protocol import AgentSkill
from surgeshield.agents.base import BaseAgent
from surgeshield.config import get_settings
from surgeshield.mcp_server import tools
from surgeshield.schemas import Assessment, Disruption, Plan
from surgeshield.world import get_world


def emergency_cap(zone_id: int, at: str) -> float:
    s = get_settings().policy
    hist = np.sort(get_world().surge_history(zone_id, at))[::-1]
    normal_range_max = float(hist[3:].max()) if len(hist) > 3 else 1.0
    return round(max(1.0, min(float(s.emergency_surge_fallback_cap), normal_range_max)), 2)


def applicable_cap(d: Disruption, zone_id: int) -> float:
    s = get_settings().policy
    return emergency_cap(zone_id, d.detected_at) if d.declared_emergency else float(s.surge_cap_normal)


class PricingAgent(BaseAgent):
    name = "pricer"
    description = "Sets capped surge multipliers and validates driver minimum pay for the plan."
    skills = [AgentSkill(id="price", name="Price plan", tags=["pricing", "optimization"],
                         description="Add surge multipliers to a plan under normal/emergency caps.")]

    def skill_price(self, plan: dict, disruption: dict, assessment: dict, feedback: list[str] | None = None) -> dict:
        p, d, a = Plan.model_validate(plan), Disruption.model_validate(disruption), Assessment.model_validate(assessment)
        override = None
        for fb in feedback or []:
            if m := re.search(r"Lower surge to ≤ ([\d.]+)x", fb):
                override = float(m.group(1))
        fc = {f.zone_id: f for f in a.forecasts}
        caps = []
        for act in p.actions:
            f = fc[act.zone_id]
            supply = f.idle_supply + act.expected_drivers_in
            ratio = f.expected_requests[0] / max(supply, 1.0)
            cap = applicable_cap(d, act.zone_id)
            if override is not None:
                cap = min(cap, override)
            caps.append(cap)
            act.surge_multiplier = round(float(np.clip(1 + 0.3 * (ratio - 1.1), 1.0, cap)), 2)
        p.surge_cap_applied = min(caps) if caps else 1.0
        trip = tools.tlc_min_driver_pay(miles=2.6, minutes=14)
        w = get_world()
        est_pay = float(w.panel["driver_pay_per_min"].median() * 14)
        notes = {"tlc_min_pay_typical_trip": trip["min_pay_usd"], "est_driver_pay_typical_trip": round(est_pay, 2),
                 "crz_fee_passthrough_usd": float(get_settings().policy.crz_fee_hvfhv_usd),
                 "cap_basis": "emergency_60d_range" if d.declared_emergency else "normal"}
        return {"output": {"plan": p.model_dump(), "pricing_notes": notes},
                "summary": f"max surge {max((x.surge_multiplier for x in p.actions), default=1):.2f}x (cap basis {notes['cap_basis']})"}
