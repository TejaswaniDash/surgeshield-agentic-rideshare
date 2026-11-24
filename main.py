"""SurgeShield API: A2A agent endpoints, incident control plane, A2UI streaming + user actions.

  uvicorn surgeshield.api.main:app --port 8000            # local runtime (no Temporal needed)
  SURGESHIELD_RUNTIME=temporal uvicorn surgeshield.api.main:app   # durable workflows + embedded worker
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from surgeshield.a2a.server import fastapi_router
from surgeshield.agents import a2a_servers, registry
from surgeshield.config import get_settings, path
from surgeshield.observability.tracing import COUNTERS, SPANS
from surgeshield.orchestration.bus import BUS
from surgeshield.scenarios import SCENARIOS, setup
from surgeshield.world import get_world

log = logging.getLogger(__name__)
RUNTIME = os.environ.get("SURGESHIELD_RUNTIME", "local")
STATE: dict = {"temporal": None, "tasks": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_world()
    registry()
    worker_task = None
    if RUNTIME == "temporal":
        from surgeshield.temporal_app.worker import build_worker, connect
        STATE["temporal"] = await connect()
        if os.environ.get("SURGESHIELD_EMBED_WORKER", "1") == "1":
            worker_task = asyncio.create_task(build_worker(STATE["temporal"]).run())
    yield
    if worker_task:
        worker_task.cancel()


app = FastAPI(title="SurgeShield - agentic marketplace disruption response", version="1.0.0", lifespan=lifespan)
app.include_router(fastapi_router(a2a_servers()))


# --------------------------------------------------------------------------- incidents
class DetectRequest(BaseModel):
    window_start: str
    window_end: str


async def _start(disruption: dict) -> str:
    iid = disruption["incident_id"]
    if RUNTIME == "temporal":
        from surgeshield.temporal_app.worker import start_incident
        try:
            await start_incident(STATE["temporal"], disruption)
        except Exception as e:  # duplicate workflow id => already handling this incident (idempotent)
            log.info("start skipped for %s: %s", iid, e)
    else:
        from surgeshield.orchestration.local_runtime import run_incident_local
        if iid not in STATE["tasks"] or STATE["tasks"][iid].done():
            BUS.ui.pop(iid, None)  # demo re-run of a finished incident starts with a clean UI stream
            ts = float(os.environ.get("SURGESHIELD_DEMO_TIME_SCALE", "0.005"))
            STATE["tasks"][iid] = asyncio.create_task(run_incident_local(disruption, time_scale=ts,
                                                                         approval_time_scale=0.2))
    return iid


@app.post("/incidents/detect")
async def detect(req: DetectRequest):
    res = await asyncio.to_thread(registry().call, "detect", req.model_dump())
    ids = [await _start(i) for i in res.output["incidents"]]
    return {"incidents": ids, "runtime": RUNTIME}


@app.post("/scenarios/{name}/run")
async def run_scenario(name: str):
    if name not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario; choose from {list(SCENARIOS)}")
    ws, we, shock = setup(get_world(), name)
    res = await asyncio.to_thread(registry().call, "detect", {"window_start": ws, "window_end": we})
    if not res.output["incidents"]:
        return {"incidents": []}
    target = max(res.output["incidents"], key=lambda i: i["z_score"])
    return {"incidents": [await _start(target)], "shock": shock, "runtime": RUNTIME}


@app.get("/scenarios")
def scenarios():
    return list(SCENARIOS)


@app.get("/incidents")
def incidents():
    return [{"incident_id": k, "state": v.get("state"), "kind": (v.get("disruption") or {}).get("kind"),
             "events": len(v.get("events", []))} for k, v in BUS.incidents.items()]


@app.get("/incidents/{iid}")
async def incident(iid: str):
    if RUNTIME == "temporal" and STATE["temporal"]:
        try:
            return await STATE["temporal"].get_workflow_handle(iid).query("state")
        except Exception:
            pass
    if iid not in BUS.incidents:
        raise HTTPException(404)
    return BUS.incidents[iid]


@app.post("/incidents/{iid}/cancel")
async def cancel(iid: str):
    if RUNTIME == "temporal":
        await STATE["temporal"].get_workflow_handle(iid).cancel()
    else:
        from surgeshield.orchestration.local_runtime import RUNNING
        if iid in RUNNING:
            RUNNING[iid].cancel()
    return {"cancel_requested": iid}


# --------------------------------------------------------------------------- A2UI
@app.get("/a2ui/stream")
async def a2ui_stream(incident: str = "*"):
    """Server-sent JSONL stream of A2UI messages (all incidents by default)."""
    async def gen():
        idx = 0
        while True:
            for m in BUS.ui_since(incident, idx):
                idx += 1
                yield {"data": json.dumps(m)}
            await asyncio.sleep(0.25)
    return EventSourceResponse(gen())


class UserActionEnvelope(BaseModel):
    userAction: dict


@app.post("/a2ui/action")
async def a2ui_action(env: UserActionEnvelope):
    """A2UI client -> agent event. approve/reject become workflow signals (human-in-the-loop)."""
    ua = env.userAction
    ctx = ua.get("context", {})
    name = ua.get("name")
    if name in ("approve_plan", "reject_plan"):
        decision = {"approved": name == "approve_plan", "by": ctx.get("user", "ops-ui"),
                    "plan_version": ctx.get("planVersion")}
        if name == "reject_plan":
            decision["reason"] = ctx.get("reason", "rejected in ops console")
        iid = ctx["incidentId"]
        if RUNTIME == "temporal":
            await STATE["temporal"].get_workflow_handle(iid).signal("approve", decision)
        else:
            from surgeshield.orchestration.local_runtime import RUNNING
            if iid not in RUNNING:
                raise HTTPException(409, "incident is not awaiting a decision")
            RUNNING[iid].signal_approval(decision)
        return {"ok": True, "signalled": decision}
    BUS.deliver("ui_action", ua)  # driver/rider actions (navigate, wait & save) are logged for analytics
    return {"ok": True}


# --------------------------------------------------------------------------- ops
@app.get("/health")
def health():
    return {"ok": True, "runtime": RUNTIME, "llm": get_settings().llm.provider, "agents": list(a2a_servers())}


@app.get("/metrics")
def metrics():
    p = path("artifacts/metrics.json")
    return {"counters": COUNTERS, "models": json.loads(p.read_text()) if p.exists() else {}}


@app.get("/traces")
def traces(limit: int = 200):
    return SPANS[-limit:]


@app.get("/graph")
def graph():
    from surgeshield.orchestration.planning_graph import graph_mermaid
    return {"mermaid": graph_mermaid()}


@app.get("/")
def root():
    return RedirectResponse("/ui")


@app.get("/ui")
def ui():
    return FileResponse(path("ui/index.html"))
