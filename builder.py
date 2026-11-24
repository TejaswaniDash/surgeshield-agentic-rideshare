"""A2UI (Agent-to-UI) v0.8 message builder.

Agents never ship HTML/JS. They emit declarative JSONL messages that the client renders with its own
trusted component catalog:
  {"surfaceUpdate":   {"surfaceId", "components": [{"id", "component": {<Type>: {...}}}]}}
  {"dataModelUpdate": {"surfaceId", "path", "contents": [{"key", "valueString"|"valueNumber"|...}]}}
  {"beginRendering":  {"surfaceId", "root", "styles"}}
  {"deleteSurface":   {"surfaceId"}}
User interactions come back as {"userAction": {"name", "surfaceId", "sourceComponentId", "context"}}.
Spec: https://a2ui.org/specification/v0.8-a2ui/
"""
from __future__ import annotations

from typing import Any

ALLOWED_COMPONENTS = {"Text", "Column", "Row", "Card", "Button", "Divider", "List", "Icon"}


def lit(v: str) -> dict:
    return {"literalString": str(v)}


def bound(path: str) -> dict:
    return {"path": path}


def _value(v: Any) -> dict:
    if isinstance(v, bool):
        return {"valueBoolean": v}
    if isinstance(v, (int, float)):
        return {"valueNumber": v}
    if isinstance(v, dict):
        return {"valueMap": [{"key": k, **_value(x)} for k, x in v.items()]}
    return {"valueString": str(v)}


class Surface:
    def __init__(self, surface_id: str):
        self.id = surface_id
        self.components: list[dict] = []
        self._n = 0

    def _add(self, ctype: str, props: dict, cid: str | None = None) -> str:
        assert ctype in ALLOWED_COMPONENTS
        self._n += 1
        cid = cid or f"{self.id}-{ctype.lower()}-{self._n}"
        self.components.append({"id": cid, "component": {ctype: props}})
        return cid

    def text(self, value: str | dict, hint: str = "body", cid: str | None = None) -> str:
        return self._add("Text", {"text": value if isinstance(value, dict) else lit(value), "usageHint": hint}, cid)

    def column(self, children: list[str], cid: str | None = None) -> str:
        return self._add("Column", {"children": {"explicitList": children}}, cid)

    def row(self, children: list[str], cid: str | None = None) -> str:
        return self._add("Row", {"children": {"explicitList": children}, "distribution": "spaceBetween"}, cid)

    def card(self, child: str, cid: str | None = None) -> str:
        return self._add("Card", {"child": child}, cid)

    def divider(self) -> str:
        return self._add("Divider", {})

    def button(self, label: str, action: str, context: dict[str, str], primary: bool = False) -> str:
        lbl = self.text(label, "body")
        ctx = [{"key": k, "value": lit(v)} for k, v in context.items()]
        return self._add("Button", {"child": lbl, "primary": primary, "action": {"name": action, "context": ctx}})

    def kv(self, label: str, value: str | dict) -> str:
        return self.row([self.text(label, "caption"), self.text(value, "body")])

    def messages(self, root: str, data: dict | None = None, styles: dict | None = None) -> list[dict]:
        msgs: list[dict] = [{"surfaceUpdate": {"surfaceId": self.id, "components": self.components}}]
        if data:
            msgs.append({"dataModelUpdate": {"surfaceId": self.id, "path": "/",
                                             "contents": [{"key": k, **_value(v)} for k, v in data.items()]}})
        msgs.append({"beginRendering": {"surfaceId": self.id, "root": root,
                                        "styles": styles or {"primaryColor": "#000000", "font": "Inter"}}})
        return msgs


def data_update(surface_id: str, data: dict, path: str = "/") -> dict:
    return {"dataModelUpdate": {"surfaceId": surface_id, "path": path,
                                "contents": [{"key": k, **_value(v)} for k, v in data.items()]}}


def validate(msg: dict) -> list[str]:
    """Light schema check so agents can't emit components outside the client's catalog."""
    errs = []
    keys = set(msg) & {"surfaceUpdate", "dataModelUpdate", "beginRendering", "deleteSurface"}
    if len(keys) != 1:
        return [f"message must have exactly one A2UI type, got {list(msg)}"]
    body = msg[next(iter(keys))]
    if "surfaceId" not in body:
        errs.append("missing surfaceId")
    for c in body.get("components", []):
        (ctype,) = c["component"].keys()
        if ctype not in ALLOWED_COMPONENTS:
            errs.append(f"component {ctype} not in catalog")
    return errs


# ---------------------------------------------------------------------- domain surfaces
def ops_incident_surface(incident_id: str, title: str, plan: dict, review: dict, needs_approval: bool) -> list[dict]:
    s = Surface(f"ops-{incident_id}")
    header = s.text(title, "h2")
    status = s.kv("Status", bound("/status"))
    wait = s.kv("Median wait (affected zones)", bound("/wait"))
    fulfil = s.kv("Fulfillment", bound("/fulfillment"))
    spend = s.kv("Incentive spend", bound("/spend"))
    rows = [s.kv(f"{a['zone_name']}", f"${a['relocation_bonus_usd']:.0f} relocation bonus · surge {a['surge_multiplier']:.2f}x")
            for a in plan["actions"] if a["relocation_bonus_usd"] > 0 or a["surge_multiplier"] > 1.0]
    plan_hdr = s.text(f"Plan v{plan['version']} · budget ${plan['incentive_budget_usd']:,.0f} · policy {plan['policy']}", "h4")
    rationale = s.text(plan.get("rationale", ""), "caption")
    verdict = s.text(f"Compliance: {review['verdict']} - {review['summary']}", "body")
    cites = sorted({f['citation']['doc_title'] for f in review["findings"] if f.get("citation")})
    cite_txt = s.text("Sources: " + "; ".join(cites), "caption") if cites else None
    children = [header, status, wait, fulfil, spend, s.divider(), plan_hdr, *rows, rationale, s.divider(), verdict]
    if cite_txt:
        children.append(cite_txt)
    if needs_approval:
        approve = s.button("Approve plan", "approve_plan", {"incidentId": incident_id, "planVersion": str(plan["version"])}, True)
        reject = s.button("Reject", "reject_plan", {"incidentId": incident_id, "planVersion": str(plan["version"])})
        children.append(s.row([approve, reject]))
    root = s.card(s.column(children), cid=f"ops-{incident_id}-root")
    return s.messages(root, {"status": "AWAITING_APPROVAL" if needs_approval else "PLANNED", "wait": "-",
                             "fulfillment": "-", "spend": "$0"})


def driver_card_surface(incident_id: str, msg: dict) -> list[dict]:
    s = Surface(f"driver-{incident_id}-{msg['zone_id']}")
    root = s.card(s.column([s.text(msg["push_text"], "h4"), s.text(msg["details"], "body"),
                            s.button("Navigate", "navigate_to_zone", {"zoneId": str(msg["zone_id"])}, True),
                            s.button("Not now", "dismiss", {"zoneId": str(msg["zone_id"])})]))
    return s.messages(root)


def rider_banner_surface(incident_id: str, msg: dict) -> list[dict]:
    s = Surface(f"rider-{incident_id}-{msg['zone_id']}")
    alts = [s.text("• " + a, "caption") for a in msg["alternatives"]]
    root = s.card(s.column([s.text(msg["banner_text"], "body"), *alts,
                            s.button("Wait & Save", "choose_wait_and_save", {"zoneId": str(msg["zone_id"])})]))
    return s.messages(root)
