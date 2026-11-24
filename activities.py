"""Side-effecting steps of the incident workflow. Each is idempotent (keyed by incident id/version)
so Temporal can retry them safely. The same implementations back the local runtime."""
from __future__ import annotations

from surgeshield.a2ui.builder import data_update
from surgeshield.agents import registry
from surgeshield.orchestration.bus import BUS
from surgeshield.orchestration.planning_graph import run_planning_cycle
from surgeshield.world import get_world


def classify(p: dict) -> dict:
    return registry().call("classify", {"disruption": p["disruption"]}, p["disruption"]["incident_id"]).output["disruption"]


def start_twin(p: dict) -> dict:
    live = get_world().start_live(p["incident_id"], p["at"], p["zones"])
    return {"clock": str(live.clock)}


def planning_cycle(p: dict) -> dict:
    out = run_planning_cycle(p["incident_id"], p["disruption"], p["fsm"], p.get("feedback"), p.get("cycle", 0))
    if out.get("a2ui"):
        BUS.publish_ui(p["incident_id"], out["a2ui"])
    keep = ("assessment", "plan", "counterfactual", "pricing_notes", "review", "comms", "feedback", "replans", "fsm",
            "events", "outcome", "a2a_tasks")
    return {k: out.get(k) for k in keep}


def execute_plan(p: dict) -> dict:
    """Commit incentives in the marketplace (twin) and deliver messages. Idempotent per plan_id."""
    plan = p["plan"]
    bonuses = {a["zone_id"]: a["relocation_bonus_usd"] for a in plan["actions"] if a["relocation_bonus_usd"] > 0}
    get_world().set_incentives(p["incident_id"], bonuses)
    for m in (p.get("comms") or {}).get("driver_messages", []):
        BUS.deliver("driver_push", {"incident": p["incident_id"], "plan": plan["plan_id"], **m})
    for m in (p.get("comms") or {}).get("rider_messages", []):
        BUS.deliver("rider_banner", {"incident": p["incident_id"], "plan": plan["plan_id"], **m})
    return {"committed_zones": sorted(bonuses), "plan_id": plan["plan_id"]}


def measure(p: dict) -> dict:
    k = get_world().advance(p["incident_id"], 1)
    BUS.publish_ui(p["incident_id"], [data_update(f"ops-{p['incident_id']}", {
        "status": p.get("status", "MONITORING"), "wait": f"{k['wait_min']:.1f} min",
        "fulfillment": f"{k['fulfillment']:.0%}", "spend": f"${k['incentive_spend_usd']:,.0f}"})])
    return k


def unwind(p: dict) -> dict:
    """Compensation (saga): withdraw all incentives for the incident."""
    w = get_world()
    if p["incident_id"] in w.live:
        w.set_incentives(p["incident_id"], {})
        return {"spend": round(w.live[p["incident_id"]].spend, 2)}
    return {"spend": 0.0}


def postmortem(p: dict) -> dict:
    return registry().call("postmortem", p, p["disruption"]["incident_id"]).output


def publish_state(p: dict) -> dict:
    BUS.put_state(p["incident_id"], p)
    if p.get("ui"):
        BUS.publish_ui(p["incident_id"], p["ui"])
    return {"ok": True}


ACTIVITIES = {f.__name__: f for f in (classify, start_twin, planning_cycle, execute_plan, measure, unwind,
                                       postmortem, publish_state)}
