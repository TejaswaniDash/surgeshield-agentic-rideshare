"""Analyst agent - writes the postmortem and closes the learning loop (episodic memory + RAG index)."""
from __future__ import annotations

import json

from surgeshield.a2a.protocol import AgentSkill
from surgeshield.agents.base import BaseAgent
from surgeshield.geo import zone_name
from surgeshield.llm.client import get_llm
from surgeshield.llm.prompts import POSTMORTEM_SYSTEM, POSTMORTEM_USER
from surgeshield.memory.store import EpisodicMemory
from surgeshield.rag.index import get_index
from surgeshield.schemas import Postmortem


class AnalystAgent(BaseAgent):
    name = "analyst"
    description = "Writes blameless postmortems and stores lessons in episodic + semantic memory."
    skills = [AgentSkill(id="postmortem", name="Write postmortem", tags=["genai", "memory"],
                         description="Summarise incident timeline, KPIs and lessons; persist to memory.")]

    def skill_postmortem(self, disruption: dict, plan: dict, kpis: list[dict], events: list[str], resolved: bool,
                         replans: int, similar: list[str] | None = None) -> dict:
        ctx = {"kpis": kpis or [{"wait": 0}], "events": events, "spend": kpis[-1]["incentive_spend_usd"] if kpis else 0,
               "budget": plan["incentive_budget_usd"], "kind": disruption["kind"], "epicenter": disruption["epicenter_zone"],
               "epicenter_name": zone_name(disruption["epicenter_zone"]), "resolved": resolved, "replans": replans}
        ctx["kpis"] = [{"wait": k["wait_min"], **k} for k in kpis] or [{"wait": 0.0}]
        prompt = POSTMORTEM_USER.format(disruption=json.dumps(disruption), plan=json.dumps(plan),
                                        kpis=json.dumps(kpis), events="\n".join(events), similar=similar or [])
        pm: Postmortem = get_llm().generate("postmortem", POSTMORTEM_SYSTEM, prompt, schema=Postmortem, context=ctx).parsed
        outcome = {"resolved": resolved, "final_wait": kpis[-1]["wait_min"] if kpis else None,
                   "spend": ctx["spend"], "cycles": len(kpis)}
        EpisodicMemory().record(disruption["incident_id"], disruption["kind"], disruption["epicenter_zone"],
                                disruption["affected_zones"], disruption["demand_vs_base"], plan, outcome, pm.model_dump())
        md = f"# {pm.title}\n\n{pm.summary}\n\n## Lessons\n" + "\n".join(f"- {x}" for x in pm.lessons)
        get_index().add_document(pm.title, md, {"doc_type": "postmortem", "incident_kind": disruption["kind"],
                                                "doc_id": disruption["incident_id"], "source_url": ""})
        return {"output": {"postmortem": pm.model_dump(), "outcome": outcome}, "summary": pm.summary}
