"""Temporal durable execution for incidents.

DisruptionResponseWorkflow wraps the deterministic IncidentOrchestrator with:
  - durable timers           (workflow.sleep between monitoring intervals - survives worker restarts)
  - signals                  (approve: human decision from the A2UI approval card)
  - queries                  (state: live snapshot for dashboards)
  - retries / timeouts       (per-activity RetryPolicy, start-to-close timeouts)
  - saga compensation        (on cancel/failure: withdraw incentives, still write the postmortem)
  - idempotent start         (workflow id = incident id -> a duplicate detection can't double-spend)
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from surgeshield.orchestration.core import IncidentOrchestrator
    from surgeshield.orchestration.fsm import S

RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), backoff_coefficient=2.0, maximum_attempts=4,
                    non_retryable_error_types=["IllegalTransition", "ValidationError"])
TIMEOUTS = {"planning_cycle": timedelta(minutes=5), "postmortem": timedelta(minutes=3)}


class _TemporalRuntime:
    def __init__(self, wf: "DisruptionResponseWorkflow"):
        self.wf = wf

    async def activity(self, name: str, payload: dict) -> dict:
        return await workflow.execute_activity(name, payload, start_to_close_timeout=TIMEOUTS.get(name, timedelta(minutes=1)),
                                               retry_policy=RETRY)

    async def sleep(self, minutes: float) -> None:
        await workflow.sleep(timedelta(minutes=minutes))

    async def wait_for_approval(self, timeout_minutes: float) -> dict | None:
        try:
            await workflow.wait_condition(lambda: self.wf.decision is not None, timeout=timedelta(minutes=timeout_minutes))
        except asyncio.TimeoutError:
            return None
        d, self.wf.decision = self.wf.decision, None
        return d

    def cancelled(self) -> bool:
        return self.wf.cancel_requested


@workflow.defn(name="DisruptionResponseWorkflow")
class DisruptionResponseWorkflow:
    def __init__(self) -> None:
        self.decision: dict | None = None
        self.cancel_requested = False
        self.orch: IncidentOrchestrator | None = None

    @workflow.run
    async def run(self, disruption: dict, wf_cfg: dict, policy_cfg: dict) -> dict:
        self.orch = IncidentOrchestrator(_TemporalRuntime(self), wf_cfg, policy_cfg)
        try:
            return await self.orch.run(disruption)
        except asyncio.CancelledError:
            # compensation must still run after a cancellation request
            self.cancel_requested = True
            self.orch.events.append("[ABORTED] cancellation requested")
            if self.orch.fsm.can(S.ABORTED):
                self.orch.fsm.to(S.ABORTED, "cancelled")
            await self.orch._close(resolved=False)
            raise

    @workflow.signal
    def approve(self, decision: dict) -> None:
        self.decision = decision

    @workflow.query
    def state(self) -> dict:
        return self.orch.snapshot() if self.orch else {"state": "STARTING"}
