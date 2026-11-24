"""Comms agent - GenAI drafting of driver/rider/ops messages grounded in style-guide RAG, checked by
output guardrails, delivered as A2UI surfaces (generative UI) rather than raw text."""
from __future__ import annotations

import json

import pandas as pd

from surgeshield.a2a.protocol import AgentSkill
from surgeshield.a2ui import builder as ui
from surgeshield.agents.base import BaseAgent
from surgeshield.geo import zone_name
from surgeshield.llm.client import get_llm
from surgeshield.llm.guardrails import check_driver_message, check_rider_message, wrap_untrusted
from surgeshield.llm.prompts import COMMS_SYSTEM, COMMS_USER
from surgeshield.mcp_server import tools
from surgeshield.schemas import Assessment, CommsBundle, Disruption, Plan

KIND_LABEL = {"arena_letout": "Arena let-out", "airport_bank": "Airport arrival surge", "rainstorm": "Weather surge",
              "transit_outage": "Transit outage", "demand_spike": "Demand spike", "unknown": "Disruption"}


class CommsAgent(BaseAgent):
    name = "comms"
    description = "Drafts driver, rider and ops communications and renders them as A2UI surfaces."
    uses_a2ui = True
    skills = [AgentSkill(id="compose", name="Compose communications", tags=["genai", "a2ui", "guardrails"],
                         description="Driver push + card, rider banner, ops approval surface (A2UI v0.8).",
                         outputModes=["application/json", "application/json+a2ui"])]

    def skill_compose(self, plan: dict, disruption: dict, assessment: dict, review: dict,
                      needs_approval: bool) -> dict:
        p, d, a = Plan.model_validate(plan), Disruption.model_validate(disruption), Assessment.model_validate(assessment)
        window_end = (pd.Timestamp(d.detected_at) + pd.Timedelta(minutes=45)).strftime("%-I:%M %p")
        epic = [{"zone_id": z, "zone_name": zone_name(z),
                 "surge": next((x.surge_multiplier for x in p.actions if x.zone_id == z), 1.0)} for z in sorted(d.affected_zones, key=lambda z: z != d.epicenter_zone)]
        eta = {str(z["zone_id"]): [max(2, round(p.expected_wait_min * 0.8)), round(p.expected_wait_min * 1.3) + 1]
               for z in epic}
        excerpts = [h["text"] for h in tools.search_policies("driver push notification style safety bonus", k=2)]
        excerpts += [h["text"] for h in tools.search_policies("rider message disruption wait alternatives price", k=2)]
        ctx = {"actions": [x.model_dump() for x in p.actions], "window_end": window_end, "epicenter_zones": epic,
               "eta": eta, "transit_ok": d.kind != "transit_outage", "kind_label": KIND_LABEL[d.kind],
               "epicenter_name": zone_name(d.epicenter_zone), "demand_vs_base": d.demand_vs_base,
               "plan_version": p.version, "budget": p.incentive_budget_usd, "surge_cap": p.surge_cap_applied,
               "expected_wait": p.expected_wait_min, "expected_fulfillment": p.expected_fulfillment}
        prompt = COMMS_USER.format(plan=p.model_dump_json(), disruption=json.dumps(
            {"kind": d.kind, "epicenter": zone_name(d.epicenter_zone), "window_end": window_end}),
            forecast=f"peak wait {a.predicted_peak_wait_min} min; expected with plan {p.expected_wait_min} min; eta {eta}",
            excerpts="\n\n".join(wrap_untrusted("style_guides", e) for e in excerpts))
        bundle: CommsBundle = get_llm().generate("comms", COMMS_SYSTEM, prompt, schema=CommsBundle, context=ctx).parsed

        # ---- output guardrails: block/repair anything that violates the style guides
        issues: dict[str, list[str]] = {}
        bonus_by_zone = {x.zone_id: x.relocation_bonus_usd for x in p.actions}
        for m in bundle.driver_messages:
            iss = check_driver_message(m.push_text, bonus_by_zone.get(m.zone_id, 0.0))
            if iss:
                issues[f"driver:{m.zone_id}"] = iss
                m.push_text = f"Higher demand near {zone_name(m.zone_id)}. ${bonus_by_zone.get(m.zone_id, 0):.0f} bonus."[:140]
            if "Drive safely" not in m.details:
                m.details += " Drive safely - follow all traffic laws."
        for r in bundle.rider_messages:
            surge = next((x["surge"] for x in epic if x["zone_id"] == r.zone_id), 1.0)
            iss = check_rider_message(r.banner_text, surge)
            if iss:
                issues[f"rider:{r.zone_id}"] = iss
                r.banner_text += f" Pickup in about {r.eta_range_min[0]}-{r.eta_range_min[1]} min." + \
                    (" Prices are higher than usual." if surge > 1.05 else "")

        title = f"{KIND_LABEL[d.kind]} · {zone_name(d.epicenter_zone)} · {d.demand_vs_base:.1f}x demand"
        a2ui = ui.ops_incident_surface(d.incident_id, title, p.model_dump(), review, needs_approval)
        for m in bundle.driver_messages:
            a2ui += ui.driver_card_surface(d.incident_id, m.model_dump())
        for r in bundle.rider_messages:
            a2ui += ui.rider_banner_surface(d.incident_id, r.model_dump())
        bad = [e for m in a2ui for e in ui.validate(m)]
        if bad:
            raise ValueError(f"invalid A2UI: {bad}")
        return {"output": {"comms": bundle.model_dump(), "guardrail_repairs": issues}, "a2ui": a2ui,
                "summary": f"{len(bundle.driver_messages)} driver / {len(bundle.rider_messages)} rider messages, "
                           f"{len(a2ui)} A2UI messages, {len(issues)} guardrail repairs"}
