"""End-to-end: Sentinel -> LangGraph planning over A2A -> approval -> execution -> monitoring -> postmortem."""
import asyncio

from surgeshield.orchestration.fsm import TRANSITIONS, S


def test_incident_runs_to_closed(trained):
    from surgeshield.eval.harness import run_scenario
    inc, res = asyncio.run(run_scenario("msg_letout"))
    assert res["state"] == "CLOSED"
    assert all(S(h["to"]) in TRANSITIONS[S(h["from"])] for h in res["fsm_history"])
    assert res["plan"]["policy"] in ("ppo_residual", "greedy_heuristic")
    assert res["result"]["postmortem"]["lessons"]
    assert any(t["skill"] == "review" for t in res["a2a_tasks"])


def test_prompt_injection_requires_human(trained):
    from surgeshield.eval.harness import run_scenario
    inc, res = asyncio.run(run_scenario("prompt_injection"))
    assert "injection_flagged" in res["disruption"]["signals"]
    assert any(h["to"] == "AWAITING_APPROVAL" for h in res["fsm_history"])
