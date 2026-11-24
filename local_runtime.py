"""In-process runtime for the orchestrator (no Temporal server needed).

Activities run in a thread (like Temporal's sync activity executor) with retry + exponential backoff;
timers are scaled by `time_scale` (0 = instant replay); approvals arrive via `signal_approval`.
"""
from __future__ import annotations

import asyncio
import logging

from surgeshield.config import get_settings
from surgeshield.observability.tracing import span
from surgeshield.orchestration.activities import ACTIVITIES
from surgeshield.orchestration.core import IncidentOrchestrator

log = logging.getLogger(__name__)
RUNNING: dict[str, "LocalRuntime"] = {}


class LocalRuntime:
    def __init__(self, time_scale: float | None = None, auto_decision: dict | None = None, max_attempts: int = 3,
                 approval_time_scale: float | None = None):
        self.time_scale = get_settings().workflow.time_scale if time_scale is None else time_scale
        self.approval_time_scale = self.time_scale if approval_time_scale is None else approval_time_scale
        self.auto_decision = auto_decision  # e.g. {"approved": True, "by": "demo"} to simulate an approver
        self.max_attempts = max_attempts
        self._approval: asyncio.Queue = asyncio.Queue()
        self._cancel = False

    async def activity(self, name: str, payload: dict) -> dict:
        delay = 0.2
        for attempt in range(1, self.max_attempts + 1):
            try:
                with span(f"activity.{name}", attempt=attempt):
                    return await asyncio.to_thread(ACTIVITIES[name], payload)
            except Exception as e:
                if attempt == self.max_attempts:
                    raise
                log.warning("activity %s failed (%s) - retry %d", name, e, attempt)
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    async def sleep(self, minutes: float) -> None:
        if self._cancel:
            raise asyncio.CancelledError()
        await asyncio.sleep(minutes * 60 * self.time_scale)

    async def wait_for_approval(self, timeout_minutes: float) -> dict | None:
        if self.auto_decision is not None:
            return self.auto_decision
        try:
            return await asyncio.wait_for(self._approval.get(), timeout=max(timeout_minutes * 60 * self.approval_time_scale, 0.01))
        except asyncio.TimeoutError:
            return None

    def signal_approval(self, decision: dict) -> None:
        self._approval.put_nowait(decision)

    def cancel(self) -> None:
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel


async def run_incident_local(disruption: dict, **rt_kwargs) -> dict:
    s = get_settings()
    rt = LocalRuntime(**rt_kwargs)
    RUNNING[disruption["incident_id"]] = rt
    try:
        return await IncidentOrchestrator(rt, dict(s.workflow), dict(s.policy)).run(disruption)
    finally:
        RUNNING.pop(disruption["incident_id"], None)
