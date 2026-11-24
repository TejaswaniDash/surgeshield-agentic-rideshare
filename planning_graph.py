"""LangGraph planning cycle: assess -> plan (RL) -> price -> compliance (RAG+LLM) -> [replan loop] -> comms (A2UI).

Every node talks to its specialist over A2A (registry routes by skill id). The graph is
checkpointed (thread_id = incident:cycle) so a crashed cycle can resume from the last node.
The FSM is advanced inside the nodes so the graph can never take an illegal lifecycle step.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from surgeshield.agents import registry
from surgeshield.config import get_settings
from surgeshield.orchestration.fsm import IncidentFSM, S


class PlanningState(TypedDict, total=False):
    incident_id: str
    disruption: dict
    assessment: dict
    plan: dict
    counterfactual: dict
    pricing_notes: dict
    review: dict
    comms: dict
    a2ui: list[dict]
    feedback: list[str]
    replans: int
    cycle: int
    fsm: dict
    events: Annotated[list[str], operator.add]
    outcome: str          # "ready" | "needs_approval" | "escalated"
    a2a_tasks: Annotated[list[dict], operator.add]


def _fsm(state: PlanningState, new: S, reason: str) -> dict:
    f = IncidentFSM.load(state.get("fsm"))
    if f.state != new:
        f.to(new, reason, state["disruption"].get("detected_at", ""))
    return f.dump()


def _call(state: PlanningState, skill: str, payload: dict[str, Any]):
    res = registry().call(skill, payload, context_id=state["incident_id"])
    tinfo = {"skill": skill, "task_id": res.task["id"], "state": res.task["status"]["state"],
             "transitions": res.task.get("metadata", {}).get("transitions", [])}
    return res, tinfo


def assess(state: PlanningState) -> dict:
    fsm = _fsm(state, S.ASSESSING, "assessment started")
    res, t = _call(state, "assess", {"disruption": state["disruption"]})
    a = res.output["assessment"]
    return {"assessment": a, "fsm": fsm, "a2a_tasks": [t],
            "events": [f"forecaster: severity={a['severity']} peak_wait={a['predicted_peak_wait_min']}min"]}


def plan(state: PlanningState) -> dict:
    fsm = _fsm({**state}, S.PLANNING, f"plan v{state.get('replans', 0) + 1}")
    res, t = _call(state, "plan", {"disruption": state["disruption"], "assessment": state["assessment"],
                                   "feedback": state.get("feedback", []), "version": state.get("replans", 0) + 1,
                                   "cycle": state.get("cycle", 0)})
    p = res.output["plan"]
    return {"plan": p, "counterfactual": res.output["counterfactual"], "fsm": fsm, "a2a_tasks": [t],
            "events": [f"repositioner: plan v{p['version']} budget=${p['incentive_budget_usd']:,.0f} "
                       f"exp_wait={p['expected_wait_min']} (no-action {res.output['counterfactual']['no_action_wait']})"]}


def price(state: PlanningState) -> dict:
    res, t = _call(state, "price", {"plan": state["plan"], "disruption": state["disruption"],
                                    "assessment": state["assessment"], "feedback": state.get("feedback", [])})
    p = res.output["plan"]
    mx = max((a["surge_multiplier"] for a in p["actions"]), default=1.0)
    return {"plan": p, "pricing_notes": res.output["pricing_notes"], "a2a_tasks": [t],
            "events": [f"pricer: max surge {mx:.2f}x, cap basis {res.output['pricing_notes']['cap_basis']}"]}


def review(state: PlanningState) -> dict:
    fsm = _fsm(state, S.COMPLIANCE_REVIEW, "compliance review")
    res, t = _call(state, "review", {"plan": state["plan"], "disruption": state["disruption"]})
    r = res.output["review"]
    return {"review": r, "fsm": fsm, "a2a_tasks": [t],
            "events": [f"compliance: {r['verdict']} - {r['summary']}"
                       + (f" | changes: {r['required_changes']}" if r["required_changes"] else "")]}


def route_after_review(state: PlanningState) -> str:
    v = state["review"]["verdict"]
    if v in ("approved", "needs_human_approval"):
        return "comms"
    if v == "needs_changes" and state.get("replans", 0) < get_settings().workflow.max_replans:
        return "replan"
    return "escalate"


def replan(state: PlanningState) -> dict:
    fb = list(dict.fromkeys(state.get("feedback", []) + state["review"]["required_changes"]))
    return {"feedback": fb, "replans": state.get("replans", 0) + 1,
            "events": [f"supervisor: re-planning with feedback {state['review']['required_changes']}"]}


def escalate(state: PlanningState) -> dict:
    return {"fsm": _fsm(state, S.ESCALATED, "compliance could not be satisfied"), "outcome": "escalated",
            "events": ["supervisor: escalated to on-call lead"]}


def comms(state: PlanningState) -> dict:
    needs = state["review"]["verdict"] == "needs_human_approval"
    res, t = _call(state, "compose", {"plan": state["plan"], "disruption": state["disruption"],
                                      "assessment": state["assessment"], "review": state["review"],
                                      "needs_approval": needs})
    fsm = _fsm(state, S.AWAITING_APPROVAL if needs else S.READY, "needs human approval" if needs else "auto-approved")
    return {"comms": res.output["comms"], "a2ui": res.a2ui, "fsm": fsm, "a2a_tasks": [t],
            "outcome": "needs_approval" if needs else "ready",
            "events": [f"comms: {res.task['history'][-1]['parts'][0]['text']}"]}


def build_graph():
    g = StateGraph(PlanningState)
    for name, fn in [("assess", assess), ("plan", plan), ("price", price), ("review", review), ("replan", replan),
                     ("escalate", escalate), ("comms", comms)]:
        g.add_node(name, fn)
    g.add_edge(START, "assess")
    g.add_edge("assess", "plan")
    g.add_edge("plan", "price")
    g.add_edge("price", "review")
    g.add_conditional_edges("review", route_after_review, {"comms": "comms", "replan": "replan", "escalate": "escalate"})
    g.add_edge("replan", "plan")
    g.add_edge("comms", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=MemorySaver())


_graph = None


def run_planning_cycle(incident_id: str, disruption: dict, fsm: dict, feedback: list[str] | None = None,
                       cycle: int = 0) -> dict:
    global _graph
    _graph = _graph or build_graph()
    init: PlanningState = {"incident_id": incident_id, "disruption": disruption, "fsm": fsm,
                           "feedback": feedback or [], "replans": 0, "cycle": cycle, "events": [], "a2a_tasks": []}
    out = _graph.invoke(init, config={"configurable": {"thread_id": f"{incident_id}:{cycle}"}, "recursion_limit": 40})
    return dict(out)


def graph_mermaid() -> str:
    global _graph
    _graph = _graph or build_graph()
    return _graph.get_graph().draw_mermaid()
