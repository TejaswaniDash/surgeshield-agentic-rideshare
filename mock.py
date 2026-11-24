"""Deterministic offline LLM. Produces schema-valid JSON from the structured `context` the
agent passes alongside the prompt. Real providers ignore `context` and read the prompt."""
from __future__ import annotations

import json


def mock_response(task: str, ctx: dict) -> str:
    fn = _HANDLERS.get(task)
    if fn is None:
        return json.dumps({"text": f"[mock:{task}]"})
    return fn(ctx)


def _compliance(ctx: dict) -> str:
    rules = ctx["rule_results"]
    cites = ctx.get("citations", {})
    findings = []
    for r in rules:
        c = cites.get(r["rule_id"])
        findings.append({"rule_id": r["rule_id"], "status": r["status"], "detail": r["detail"], "citation": c})
    fails = [r for r in rules if r["status"] == "fail"]
    needs_human = any(r.get("needs_human") for r in rules)
    if any(r.get("unfixable") for r in fails):
        verdict = "rejected"
    elif fails:
        verdict = "needs_changes"
    elif needs_human:
        verdict = "needs_human_approval"
    else:
        verdict = "approved"
    changes = [r["fix"] for r in fails if r.get("fix")]
    why = "; ".join(r["rule_id"] for r in rules if r.get("needs_human"))
    summary = {"approved": "Plan complies with all checked regulations and internal policies.",
               "needs_changes": f"{len(fails)} rule(s) failed; plan must be revised before execution.",
               "needs_human_approval": f"Plan is compliant; human approval required ({why}).",
               "rejected": "Plan violates a non-fixable constraint."}[verdict]
    return json.dumps({"verdict": verdict, "findings": findings, "required_changes": changes, "summary": summary})


def _comms(ctx: dict) -> str:
    drivers, riders = [], []
    for a in ctx["actions"]:
        if a["relocation_bonus_usd"] <= 0:
            continue
        bonus = int(a["relocation_bonus_usd"])
        push = f"High demand near {a['zone_name']} until {ctx['window_end']}. Earn a ${bonus} bonus for heading there."
        drivers.append({"zone_id": a["zone_id"], "push_text": push[:140], "language": "en",
                        "details": (f"Requests near {a['zone_name']} are above normal. Arrive by {ctx['window_end']} "
                                    f"to receive a one-time ${bonus} relocation bonus on top of your regular pay. "
                                    + ("Follow airport staging instructions; do not wait at the curb. "
                                       if a["zone_id"] in (132, 138) else "")
                                    + "Drive safely - follow all traffic laws.")})
    for z in ctx["epicenter_zones"]:
        lo, hi = ctx["eta"][str(z["zone_id"])]
        alt = ["Wait & Save: lower price if you can wait 10-15 min", "Shared ride where available"]
        if ctx.get("transit_ok", True):
            alt.append("Subway/bus: check MTA service status")
        surge_note = " Prices are higher than usual." if z["surge"] > 1.05 else ""
        riders.append({"zone_id": z["zone_id"], "eta_range_min": [lo, hi], "alternatives": alt,
                       "banner_text": f"It's busier than usual near {z['zone_name']}. Pickup in about {lo}-{hi} min.{surge_note}"})
    brief = (f"{ctx['kind_label']} detected at {ctx['epicenter_name']} ({ctx['demand_vs_base']:.1f}x normal demand). "
             f"Plan v{ctx['plan_version']}: ${ctx['budget']:,.0f} relocation incentives across {len(drivers)} zones, "
             f"surge capped at {ctx['surge_cap']:.2f}x. Expected median wait {ctx['expected_wait']:.1f} min "
             f"(fulfillment {ctx['expected_fulfillment']:.0%}).")
    return json.dumps({"driver_messages": drivers, "rider_messages": riders, "ops_brief": brief})


def _postmortem(ctx: dict) -> str:
    k = ctx["kpis"]
    first, last = k[0], k[-1]
    worked = [f"Median wait fell from {first['wait']:.1f} to {last['wait']:.1f} min across affected zones."]
    if ctx.get("replans"):
        worked.append(f"Compliance loop caught {ctx['replans']} issue(s) before execution.")
    didnt = []
    if ctx["spend"] > 0.8 * ctx["budget"]:
        didnt.append("Incentive spend approached the committed budget; consider tapering earlier.")
    if len(k) > 4:
        didnt.append(f"Recovery took {len(k)} monitoring cycles.")
    if not didnt:
        didnt.append("No significant issues observed.")
    lessons = [f"For {ctx['kind']} at zone {ctx['epicenter']}, pre-position ~30 min before forecast peak.",
               "Size spillover-zone bonuses below the epicenter bonus."]
    return json.dumps({"title": f"Postmortem - {ctx['kind']} at {ctx['epicenter_name']}",
                       "summary": f"{ctx['kind']} at {ctx['epicenter_name']}; peak {first['wait']:.1f} min wait; "
                                  f"resolved={ctx['resolved']}; spend ${ctx['spend']:,.0f}.",
                       "timeline": ctx["events"][-12:], "what_worked": worked, "what_did_not": didnt,
                       "lessons": lessons})


def _rationale(ctx: dict) -> str:
    zones = ", ".join(f"{a['zone_name']} (${a['relocation_bonus_usd']:.0f})" for a in ctx["actions"]
                      if a["relocation_bonus_usd"] > 0)
    fb = f" Addressed feedback: {'; '.join(ctx['feedback'])}." if ctx.get("feedback") else ""
    return (f"The {ctx['policy']} policy pre-positions drivers into the forecast shortfall: {zones or 'no zones'}. "
            f"Bonuses scale with the predicted gap and are capped to keep total commitment at "
            f"${ctx['budget']:,.0f}. Expected median wait {ctx['expected_wait']:.1f} min.{fb}")


_HANDLERS = {"compliance_review": _compliance, "comms": _comms, "postmortem": _postmortem,
             "plan_rationale": _rationale}
