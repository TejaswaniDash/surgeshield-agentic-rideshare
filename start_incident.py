"""Detect disruptions for a scenario and start one durable workflow per incident.

  python scripts/start_incident.py --scenario msg_letout --runtime temporal [--with-worker] [--approve]
  python scripts/start_incident.py --scenario msg_letout --runtime local
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgeshield.agents import registry  # noqa: E402
from surgeshield.scenarios import SCENARIOS, setup  # noqa: E402
from surgeshield.world import get_world  # noqa: E402


async def main(a) -> None:
    ws, we, shock = setup(get_world(), a.scenario)
    incidents = registry().call("detect", {"window_start": ws, "window_end": we}).output["incidents"]
    target = max(incidents, key=lambda i: i["z_score"])
    print("detected:", target["incident_id"])
    if a.runtime == "local":
        from surgeshield.orchestration.local_runtime import run_incident_local
        r = await run_incident_local(target, auto_decision={"approved": True, "by": "cli"} if a.approve else None)
        print(json.dumps({"state": r["state"], "result": {k: v for k, v in r["result"].items() if k != "postmortem"}}))
        return
    from surgeshield.temporal_app.worker import build_worker, connect, start_incident
    client = await connect()
    worker_task = None
    if a.with_worker:
        worker = build_worker(client)
        worker_task = asyncio.create_task(worker.run())
    handle = await start_incident(client, target)
    print("started workflow", handle.id)
    if a.approve:
        for _ in range(120):
            st = await handle.query("state")
            if st.get("state") == "AWAITING_APPROVAL":
                await handle.signal("approve", {"approved": True, "by": "cli"})
                print("signalled approval")
                break
            if st.get("state") in ("READY", "EXECUTING", "MONITORING", "CLOSED"):
                break
            await asyncio.sleep(0.5)
    result = await handle.result()
    print(json.dumps({"state": result["state"], "events": result["events"][-8:],
                      "result": {k: v for k, v in result["result"].items() if k != "postmortem"}}, indent=1))
    if worker_task:
        worker_task.cancel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="msg_letout")
    ap.add_argument("--runtime", choices=["temporal", "local"], default="temporal")
    ap.add_argument("--with-worker", action="store_true", help="run an embedded worker in this process")
    ap.add_argument("--approve", action="store_true", help="auto-approve when the workflow waits for a human")
    asyncio.run(main(ap.parse_args()))
