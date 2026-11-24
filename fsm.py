"""Incident lifecycle state machine - the contract every runtime (LangGraph, Temporal, local) obeys.

            ┌──────────────── replan (compliance needs_changes / ops reject / KPI degradation) ─────────┐
            ▼                                                                                            │
DETECTED → TRIAGED → ASSESSING → PLANNING → COMPLIANCE_REVIEW ─┬─► AWAITING_APPROVAL ─┬─► READY → EXECUTING → MONITORING
                                                              ├─► READY               └─► ESCALATED            │   │
                                                              └─► ESCALATED                                    │   │
MONITORING ─ recovered ─► RESOLVING → POSTMORTEM → CLOSED  ◄── ESCALATED / ABORTED (compensation: unwind) ────┘   │
           └─ degraded ─► ASSESSING ───────────────────────────────────────────────────────────────────────────┘
Pure python, no I/O, deterministic -> safe to run inside a Temporal workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class S(str, Enum):
    DETECTED = "DETECTED"
    TRIAGED = "TRIAGED"
    ASSESSING = "ASSESSING"
    PLANNING = "PLANNING"
    COMPLIANCE_REVIEW = "COMPLIANCE_REVIEW"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    READY = "READY"
    EXECUTING = "EXECUTING"
    MONITORING = "MONITORING"
    ESCALATED = "ESCALATED"
    ABORTED = "ABORTED"
    RESOLVING = "RESOLVING"
    POSTMORTEM = "POSTMORTEM"
    CLOSED = "CLOSED"


TRANSITIONS: dict[S, set[S]] = {
    S.DETECTED: {S.TRIAGED, S.ABORTED},
    S.TRIAGED: {S.ASSESSING, S.ABORTED},
    S.ASSESSING: {S.PLANNING, S.ESCALATED, S.ABORTED},
    S.PLANNING: {S.COMPLIANCE_REVIEW, S.ESCALATED, S.ABORTED},
    S.COMPLIANCE_REVIEW: {S.PLANNING, S.AWAITING_APPROVAL, S.READY, S.ESCALATED, S.ABORTED},
    S.AWAITING_APPROVAL: {S.READY, S.PLANNING, S.ASSESSING, S.ESCALATED, S.ABORTED},
    S.READY: {S.EXECUTING, S.ABORTED},
    S.EXECUTING: {S.MONITORING, S.ESCALATED, S.ABORTED},
    S.MONITORING: {S.MONITORING, S.ASSESSING, S.RESOLVING, S.ESCALATED, S.ABORTED},
    S.ESCALATED: {S.RESOLVING},
    S.ABORTED: {S.RESOLVING},
    S.RESOLVING: {S.POSTMORTEM},
    S.POSTMORTEM: {S.CLOSED},
    S.CLOSED: set(),
}
TERMINAL = {S.CLOSED}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class IncidentFSM:
    state: S = S.DETECTED
    history: list[dict] = field(default_factory=list)

    def can(self, new: S) -> bool:
        return S(new) in TRANSITIONS[self.state]

    def to(self, new: S | str, reason: str = "", at: str = "") -> S:
        new = S(new)
        if not self.can(new):
            raise IllegalTransition(f"{self.state.value} -> {new.value} not allowed ({reason})")
        self.history.append({"from": self.state.value, "to": new.value, "reason": reason, "at": at})
        self.state = new
        return new

    def dump(self) -> dict:
        return {"state": self.state.value, "history": list(self.history)}

    @classmethod
    def load(cls, d: dict | None) -> "IncidentFSM":
        if not d:
            return cls()
        return cls(S(d["state"]), list(d.get("history", [])))
