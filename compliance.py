"""Compliance agent - RAG + LLM review layered on a deterministic rule engine.

Defense in depth: hard rules are computed in code (authoritative); the LLM explains, cites the
retrieved regulation/policy excerpts and may only make the verdict *stricter*, never looser.
"""
from __future__ import annotations

import json

from surgeshield.a2a.protocol import AgentSkill
from surgeshield.agents.base import BaseAgent
from surgeshield.agents.pricing import applicable_cap
from surgeshield.config import get_settings
from surgeshield.llm.client import get_llm
from surgeshield.llm.guardrails import wrap_untrusted
from surgeshield.llm.prompts import COMPLIANCE_SYSTEM, COMPLIANCE_USER
from surgeshield.mcp_server import tools
from surgeshield.schemas import ComplianceReview, Disruption, Plan
from surgeshield.world import get_world

RULE_QUERIES = {
    "R1_SURGE_CAP": "surge cap abnormal disruption emergency 60 days price gouging",
    "R2_BUDGET_HARD": "incentive budget approval limits rejected over",
    "R3_AUTO_APPROVE": "auto-approved human Marketplace Ops approver incentive commitment",
    "R4_DRIVER_MIN_PAY": "TLC minimum per-trip payment per mile per minute incentives on top",
    "R5_WAV_COVERAGE": "WAV accessible vehicle incentive hotspot requests plan",
    "R6_EMERGENCY_CLASSIFICATION": "transit outage abnormal disruption classification uncertain human approval",
    "R7_FAIR_TARGETING": "incentives offered to all eligible drivers no individual targeting",
    "R8_UNTRUSTED_SIGNALS": "messages must not encourage unsafe behaviour policy communications",
}
STRICTNESS = {"approved": 0, "needs_human_approval": 1, "needs_changes": 2, "rejected": 3}


def rule_engine(p: Plan, d: Disruption) -> list[dict]:
    s = get_settings().policy
    w = get_world()
    out = []
    viol = [(a.zone_id, a.surge_multiplier, applicable_cap(d, a.zone_id)) for a in p.actions
            if a.surge_multiplier > applicable_cap(d, a.zone_id) + 1e-6]
    cap_txt = "emergency 60-day-range cap" if d.declared_emergency else f"normal cap {s.surge_cap_normal}x"
    out.append({"rule_id": "R1_SURGE_CAP", "status": "fail" if viol else "pass",
                "detail": (f"Surge exceeds {cap_txt} in zones {[v[0] for v in viol]}" if viol
                           else f"All multipliers within {cap_txt}"),
                "fix": f"Lower surge to ≤ {min(v[2] for v in viol):.2f}x" if viol else None})
    hard = float(s.incentive_hard_budget_usd)
    out.append({"rule_id": "R2_BUDGET_HARD", "status": "fail" if p.incentive_budget_usd > hard else "pass",
                "detail": f"Budget ${p.incentive_budget_usd:,.0f} vs hard limit ${hard:,.0f}",
                "fix": f"Reduce incentive budget to ≤ ${hard:,.0f}" if p.incentive_budget_usd > hard else None})
    auto = float(s.incentive_auto_approve_usd)
    out.append({"rule_id": "R3_AUTO_APPROVE", "status": "warn" if p.incentive_budget_usd > auto else "pass",
                "needs_human": p.incentive_budget_usd > auto,
                "detail": f"Budget ${p.incentive_budget_usd:,.0f} {'exceeds' if p.incentive_budget_usd > auto else 'within'} "
                          f"auto-approve limit ${auto:,.0f}"})
    trip = tools.tlc_min_driver_pay(2.6, 14)
    est = float(w.panel["driver_pay_per_min"].median() * 14)
    out.append({"rule_id": "R4_DRIVER_MIN_PAY", "status": "pass" if est >= trip["min_pay_usd"] else "warn",
                "detail": f"Incentives are additive; est. base pay ${est:.2f} vs TLC minimum ${trip['min_pay_usd']:.2f} "
                          f"for a 2.6 mi / 14 min trip"})
    recent = w.window_back(d.detected_at, 60, d.affected_zones)
    wav_zones = sorted(int(z) for z in recent[recent["wav_requests"] > 0]["zone_id"].unique())
    covered = {a.zone_id for a in p.actions if a.wav_bonus_usd > 0}
    missing = [z for z in wav_zones if z not in covered]
    out.append({"rule_id": "R5_WAV_COVERAGE", "status": "fail" if missing else "pass",
                "detail": f"WAV requests in last hour in zones {wav_zones}; WAV incentive missing for {missing}"
                if missing else "WAV demand covered",
                "fix": f"Add WAV bonus in zones [{', '.join(map(str, missing))}]" if missing else None})
    uncertain = d.kind in ("transit_outage", "rainstorm") and not d.declared_emergency
    out.append({"rule_id": "R6_EMERGENCY_CLASSIFICATION", "status": "warn" if uncertain else "pass",
                "needs_human": uncertain,
                "detail": f"{d.kind} without a declared emergency - classification needs human confirmation"
                if uncertain else f"classification {d.kind}, declared_emergency={d.declared_emergency}"})
    out.append({"rule_id": "R7_FAIR_TARGETING", "status": "pass",
                "detail": "Incentives are zone-level and offered to all eligible drivers in range"})
    inj = "injection_flagged" in d.signals
    out.append({"rule_id": "R8_UNTRUSTED_SIGNALS", "status": "warn" if inj else "pass", "needs_human": inj,
                "detail": "External feed contained instruction-like text; it was neutralised and ignored - "
                          "human confirmation required before acting on this incident's signals"
                if inj else "No instruction-like content in external signals"})
    return out


class ComplianceAgent(BaseAgent):
    name = "compliance"
    description = "Reviews plans against NYC/NY regulations and internal policy using RAG with citations."
    skills = [AgentSkill(id="review", name="Compliance review", tags=["rag", "llm", "policy"],
                         description="Return verdict, findings with citations and required changes.")]

    def skill_review(self, plan: dict, disruption: dict) -> dict:
        p, d = Plan.model_validate(plan), Disruption.model_validate(disruption)
        rules = rule_engine(p, d)
        citations, excerpts = {}, []
        for r in rules:
            hits = tools.search_policies(RULE_QUERIES[r["rule_id"]], k=2)
            if hits:
                citations[r["rule_id"]] = {"doc_title": hits[0]["title"], "source_url": hits[0]["source_url"],
                                           "chunk_id": hits[0]["chunk_id"]}
                excerpts += [f"[{h['title']} | {h['section']}]\n{h['text']}" for h in hits]
        excerpts = list(dict.fromkeys(excerpts))
        prompt = COMPLIANCE_USER.format(
            plan=p.model_dump_json(indent=1),
            disruption=json.dumps({k: v for k, v in d.model_dump().items() if k != "context_notes"})
            + "\n" + wrap_untrusted("external_signals", d.context_notes),
            rule_results=json.dumps(rules, indent=1),
            excerpts="\n\n".join(wrap_untrusted("policy_corpus", e) for e in excerpts[:10]))
        res = get_llm().generate("compliance_review", COMPLIANCE_SYSTEM, prompt, schema=ComplianceReview,
                                 context={"rule_results": rules, "citations": citations})
        review: ComplianceReview = res.parsed
        # --- enforce: the LLM can only tighten the deterministic verdict
        floor = "approved"
        if any(r["status"] == "fail" for r in rules):
            floor = "needs_changes"
        elif any(r.get("needs_human") for r in rules):
            floor = "needs_human_approval"
        if STRICTNESS[review.verdict] < STRICTNESS[floor]:
            review.verdict = floor
            review.required_changes = [r["fix"] for r in rules if r.get("fix")]
        for f in review.findings:  # rule engine status is authoritative per rule
            rr = next((r for r in rules if r["rule_id"] == f.rule_id), None)
            if rr and rr["status"] == "fail":
                f.status = "fail"
        return {"output": {"review": review.model_dump(), "rule_results": rules},
                "summary": f"{review.verdict}: {review.summary}"}
