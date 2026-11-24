"""Typed contracts shared by agents, A2A messages, the workflow and the UI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DisruptionKind = Literal["arena_letout", "airport_bank", "rainstorm", "transit_outage", "demand_spike", "unknown"]
Severity = Literal["low", "medium", "high", "critical"]
Verdict = Literal["approved", "needs_changes", "needs_human_approval", "rejected"]


class Disruption(BaseModel):
    incident_id: str
    kind: DisruptionKind = "unknown"
    epicenter_zone: int
    affected_zones: list[int]
    detected_at: str
    z_score: float
    demand_vs_base: float
    wait_med: float
    declared_emergency: bool = False
    signals: list[str] = Field(default_factory=list)
    context_notes: str = ""  # untrusted external text (event feeds) - always screened


class ZoneForecast(BaseModel):
    zone_id: int
    zone_name: str
    expected_requests: list[float]
    baseline_requests: float
    idle_supply: float
    gap_next: float
    predicted_wait_min: float


class Assessment(BaseModel):
    severity: Severity
    forecasts: list[ZoneForecast]
    predicted_peak_wait_min: float
    expected_unmet_next_hour: float
    similar_incidents: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)


class ZoneAction(BaseModel):
    zone_id: int
    zone_name: str
    relocation_bonus_usd: float
    expected_drivers_in: float
    surge_multiplier: float = 1.0
    wav_bonus_usd: float = 0.0


class Plan(BaseModel):
    plan_id: str
    version: int = 1
    actions: list[ZoneAction]
    incentive_budget_usd: float
    surge_cap_applied: float
    policy: str
    expected_fulfillment: float
    expected_wait_min: float
    rationale: str = ""
    feedback_addressed: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    doc_title: str
    source_url: str = ""
    chunk_id: str = ""


class ComplianceFinding(BaseModel):
    rule_id: str
    status: Literal["pass", "warn", "fail"]
    detail: str
    citation: Citation | None = None


class ComplianceReview(BaseModel):
    verdict: Verdict
    findings: list[ComplianceFinding]
    required_changes: list[str] = Field(default_factory=list)
    summary: str


class DriverMessage(BaseModel):
    zone_id: int
    push_text: str = Field(max_length=140)
    details: str
    language: str = "en"


class RiderMessage(BaseModel):
    zone_id: int
    banner_text: str
    eta_range_min: list[int]
    alternatives: list[str]


class CommsBundle(BaseModel):
    driver_messages: list[DriverMessage]
    rider_messages: list[RiderMessage]
    ops_brief: str


class Postmortem(BaseModel):
    title: str
    summary: str
    timeline: list[str]
    what_worked: list[str]
    what_did_not: list[str]
    lessons: list[str]
