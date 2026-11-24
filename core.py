"""IncidentOrchestrator - the long-running, durable control loop (temporal execution).

Written against a tiny `Runtime` interface so the *same* deterministic logic runs:
  - inside a Temporal workflow (TemporalRuntime: durable timers, signals, retries, replay), or
  - in-process (LocalRuntime: asyncio, used for tests, notebooks and the offline demo).
No I/O, clocks or randomness here - all side effects go through `rt.activity(...)`.
"""
from __future__ import annotations

from typing import Any, Protocol

from surgeshield.a2ui.builder import data_update
from surgeshield.orchestration.fsm import IncidentFSM, S


class Runtime(Protocol):
    async def activity(self, name: str, payload: dict) -> dict: ...
    async def sleep(self, minutes: float) -> None: ...
    async def wait_for_approval(self, timeout_minutes: float) -> dict | None: ...
    def cancelled(self) -> bool: ...


class IncidentOrchestrator:
    def __init__(self, rt: Runtime, wf_cfg: dict, policy_cfg: dict):
        self.rt, self.cfg, self.pol = rt, wf_cfg, policy_cfg
        self.fsm = IncidentFSM()
        self.events: list[str] = []
        self.kpis: list[dict] = []
        self.plan: dict | None = None
        self.review: dict | None = None
        self.comms: dict | None = None
        self.assessment: dict | None = None
        self.disruption: dict = {}
        self.replans = 0
        self.cycle = 0
        self.decision: dict | None = None
        self.a2a_tasks: list[dict] = []
        self.result: dict = {}

    # ---------------------------------------------------------------- helpers
    def _to(self, s: S, reason: str) -> None:
        if self.fsm.state != s:
            self.fsm.to(s, reason)
        self.events.append(f"[{s.value}] {reason}")

    def snapshot(self) -> dict[str, Any]:
        return {"incident_id": self.disruption.get("incident_id"), "state": self.fsm.state.value,
                "fsm_history": self.fsm.history, "disruption": self.disruption, "assessment": self.assessment,
                "plan": self.plan, "review": self.review, "comms": self.comms, "kpis": self.kpis,
                "events": self.events, "replans": self.replans, "decision": self.decision,
                "a2a_tasks": self.a2a_tasks[-20:], "result": self.result}

    async def _publish(self, ui: list[dict] | None = None) -> None:
        await self.rt.activity("publish_state", {**self.snapshot(), "ui": ui or []})

    async def _plan_cycle(self, feedback: list[str]) -> str:
        out = await self.rt.activity("planning_cycle", {"incident_id": self.disruption["incident_id"],
                                                        "disruption": self.disruption, "fsm": self.fsm.dump(),
                                                        "feedback": feedback, "cycle": self.cycle})
        self.cycle += 1
        self.fsm = IncidentFSM.load(out["fsm"])
        self.events += out["events"]
        self.a2a_tasks += out.get("a2a_tasks") or []
        self.replans += out.get("replans") or 0
        for k in ("assessment", "plan", "review", "comms"):
            if out.get(k) is not None:
                setattr(self, k, out[k])
        await self._publish()
        return out["outcome"]

    # ---------------------------------------------------------------- main loop
    async def run(self, disruption: dict) -> dict:
        self.disruption = disruption
        iid = disruption["incident_id"]
        self.events.append(f"[DETECTED] {disruption['epicenter_zone']} z={disruption['z_score']} "
                           f"{disruption['demand_vs_base']}x baseline")
        try:
            self.disruption = await self.rt.activity("classify", {"disruption": disruption})
            self._to(S.TRIAGED, f"classified as {self.disruption['kind']} "
                                f"(emergency={self.disruption['declared_emergency']})")
            await self.rt.activity("start_twin", {"incident_id": iid, "at": self.disruption["detected_at"],
                                                  "zones": self.disruption["affected_zones"]})
            await self._publish()

            outcome = await self._plan_cycle([])
            outcome = await self._approval_gate(outcome)
            if outcome == "escalated":
                return await self._close(resolved=False)

            await self._execute()
            resolved = await self._monitor()
            return await self._close(resolved=resolved)
        except Exception as e:  # any failure -> compensate, still produce a postmortem
            if self.rt.cancelled():
                self.events.append("[ABORTED] workflow cancelled")
                if self.fsm.can(S.ABORTED):
                    self.fsm.to(S.ABORTED, "cancelled")
            else:
                self.events.append(f"[ERROR] {type(e).__name__}: {e}")
                if self.fsm.can(S.ESCALATED):
                    self.fsm.to(S.ESCALATED, f"error: {type(e).__name__}")
                elif self.fsm.can(S.ABORTED):
                    self.fsm.to(S.ABORTED, f"error: {type(e).__name__}")
            if self.fsm.state in (S.RESOLVING, S.POSTMORTEM, S.CLOSED):
                raise
            return await self._close(resolved=False)

    async def _approval_gate(self, outcome: str) -> str:
        """Human-in-the-loop: wait for an ops decision (A2UI button -> API -> signal) with a durable timeout."""
        while outcome == "needs_approval":
            self.decision = await self.rt.wait_for_approval(self.cfg["approval_timeout_minutes"])
            budget = self.plan["incentive_budget_usd"]
            if self.decision is None:
                if budget <= self.pol["incentive_auto_approve_usd"]:
                    self._to(S.READY, "approval timeout - within auto-approve limit")
                    return "ready"
                self._to(S.ESCALATED, f"approval timeout - ${budget:,.0f} exceeds auto-approve limit")
                return "escalated"
            if self.decision.get("approved"):
                self._to(S.READY, f"approved by {self.decision.get('by', 'ops')}")
                return "ready"
            if self.replans >= self.cfg["max_replans"]:
                self._to(S.ESCALATED, "rejected and replan budget exhausted")
                return "escalated"
            fb = [f"Ops rejected: {self.decision.get('reason', 'no reason')}"]
            if self.decision.get("max_budget"):
                fb.append(f"Reduce incentive budget to ≤ ${float(self.decision['max_budget']):,.0f}")
            self.events.append(f"[AWAITING_APPROVAL] rejected: {fb}")
            self.decision = None
            outcome = await self._plan_cycle(fb)
        return outcome

    async def _execute(self) -> None:
        self._to(S.EXECUTING, f"committing plan {self.plan['plan_id']}")
        await self.rt.activity("execute_plan", {"incident_id": self.disruption["incident_id"], "plan": self.plan,
                                                "comms": self.comms})
        self._to(S.MONITORING, "incentives live; monitoring KPIs")
        await self._publish([data_update(f"ops-{self.disruption['incident_id']}", {"status": "EXECUTING"})])

    async def _monitor(self) -> bool:
        iid = self.disruption["incident_id"]
        for _ in range(self.cfg["max_monitor_cycles"]):
            await self.rt.sleep(self.cfg["monitor_interval_minutes"])  # durable timer
            k = await self.rt.activity("measure", {"incident_id": iid, "status": "MONITORING"})
            self.kpis.append(k)
            self.events.append(f"[MONITORING] {k['clock']} wait={k['wait_min']}min fulfillment={k['fulfillment']:.0%} "
                               f"spend=${k['incentive_spend_usd']:,.0f}")
            if (k["wait_min"] <= self.cfg["recovery_wait_minutes"]
                    and k["fulfillment"] >= self.cfg["recovery_min_fulfillment"] and len(self.kpis) >= 2):
                self._to(S.RESOLVING, f"recovered: wait {k['wait_min']} min, fulfillment {k['fulfillment']:.0%}")
                await self._publish()
                return True
            # adaptive re-planning: KPIs much worse than the plan expected -> new cycle with fresh forecast
            degraded = len(self.kpis) >= 2 and (k["wait_min"] > 1.5 * self.plan["expected_wait_min"] + 2
                                                or k["fulfillment"] < self.plan["expected_fulfillment"] - 0.15)
            if degraded and self.replans < self.cfg["max_replans"]:
                self.events.append("[MONITORING] KPIs degraded vs plan - re-assessing")
                self.disruption = {**self.disruption, "detected_at": k["clock"]}
                carry = [f for f in self.plan.get("feedback_addressed", []) if not f.startswith("Previous plan")]
                outcome = await self._plan_cycle(carry + [f"Previous plan under-delivered: wait {k['wait_min']} min, "
                                                          f"fulfillment {k['fulfillment']:.0%}"])
                if outcome == "needs_approval" and self.plan["incentive_budget_usd"] > self.pol["incentive_auto_approve_usd"]:
                    outcome = await self._approval_gate(outcome)
                elif outcome == "needs_approval":
                    self._to(S.READY, "in-flight replan within auto-approve limit")
                if outcome == "escalated":
                    return False
                await self._execute()
            await self._publish()
        self._to(S.ESCALATED, f"not recovered after {len(self.kpis)} intervals")
        return False

    async def _close(self, resolved: bool) -> dict:
        iid = self.disruption["incident_id"]
        if self.fsm.state != S.RESOLVING:
            self._to(S.RESOLVING, "unwinding (compensation)")
        spend = (await self.rt.activity("unwind", {"incident_id": iid}))["spend"]
        self.events.append(f"[RESOLVING] incentives withdrawn; total spend ${spend:,.0f}")
        self._to(S.POSTMORTEM, "writing postmortem")
        pm = await self.rt.activity("postmortem", {
            "disruption": self.disruption, "plan": self.plan or {"incentive_budget_usd": 0, "actions": []},
            "kpis": self.kpis, "events": self.events[-30:], "resolved": resolved, "replans": self.replans,
            "similar": (self.assessment or {}).get("similar_incidents", [])})
        self._to(S.CLOSED, "incident closed")
        self.result = {"resolved": resolved, "spend": spend, "postmortem": pm["postmortem"],
                       "final_wait": self.kpis[-1]["wait_min"] if self.kpis else None}
        await self._publish([data_update(f"ops-{iid}", {"status": "CLOSED" if resolved else "CLOSED (escalated)"})])
        return self.snapshot()
