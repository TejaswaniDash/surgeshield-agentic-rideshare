"""End-to-end demo without any servers: detect -> triage -> plan (LangGraph + A2A agents) -> approval ->
execute -> monitor (closed-loop twin) -> unwind -> postmortem.

  python scripts/run_demo.py --scenario msg_letout
  python scripts/run_demo.py --scenario storm_emergency --approve reject
  LLM_PROVIDER=anthropic python scripts/run_demo.py --scenario prompt_injection
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgeshield.agents import registry  # noqa: E402
from surgeshield.config import path  # noqa: E402
from surgeshield.orchestration.bus import BUS  # noqa: E402
from surgeshield.orchestration.local_runtime import run_incident_local  # noqa: E402
from surgeshield.scenarios import SCENARIOS, setup  # noqa: E402
from surgeshield.world import get_world  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


async def main(scenario: str, approve: str) -> dict:
    world = get_world()
    ws, we, shock = setup(world, scenario)
    print(f"\n=== Scenario {scenario}: {shock['description']} ({shock['start']}) ===")
    found = registry().call("detect", {"window_start": ws, "window_end": we}).output["incidents"]
    if not found:
        print("no disruption detected")
        return {}
    target = max(found, key=lambda i: i["z_score"])
    print(f"Sentinel: {len(found)} incident(s); handling {target['incident_id']} "
          f"(z={target['z_score']}, {target['demand_vs_base']}x baseline)")
    decision = {"auto": None, "approve": {"approved": True, "by": "ops-demo"},
                "reject": {"approved": False, "reason": "budget too high", "max_budget": 1500, "by": "ops-demo"}}[approve]
    result = await run_incident_local(target, auto_decision=decision, time_scale=0)
    if approve == "reject" and result["decision"] is None:
        pass
    print("\n--- Event log ---")
    for e in result["events"]:
        print(" ", e)
    print("\n--- FSM path ---\n ", " -> ".join([h["from"] for h in result["fsm_history"]] + [result["state"]]))
    p = result["plan"] or {}
    print(f"\n--- Final plan v{p.get('version')} ({p.get('policy')}) ---")
    for a in p.get("actions", []):
        if a["relocation_bonus_usd"] or a["surge_multiplier"] > 1 or a["wav_bonus_usd"]:
            print(f"  {a['zone_name']:<32} bonus ${a['relocation_bonus_usd']:<4.0f} surge {a['surge_multiplier']:.2f}x "
                  f"wav ${a['wav_bonus_usd']:.0f}  ~{a['expected_drivers_in']} drivers in")
    if result.get("comms"):
        print("\n--- Sample driver push ---\n ", result["comms"]["driver_messages"][0]["push_text"]
              if result["comms"]["driver_messages"] else "(none)")
        print("--- Sample rider banner ---\n ", result["comms"]["rider_messages"][0]["banner_text"])
    pm = result["result"]["postmortem"]
    print(f"\n--- Postmortem ---\n  {pm['summary']}\n  lessons: {pm['lessons']}")
    print(f"\nA2UI messages streamed: {len(BUS.ui.get(target['incident_id'], []))}; "
          f"A2A tasks: {len(result['a2a_tasks'])}; outbox: {path('artifacts/outbox.jsonl')}")
    out = path(f"artifacts/demo_{scenario}.json")
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"full incident record -> {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="msg_letout")
    ap.add_argument("--approve", choices=["auto", "approve", "reject"], default="approve",
                    help="simulated ops decision when human approval is required (auto = wait for timeout)")
    a = ap.parse_args()
    asyncio.run(main(a.scenario, a.approve))
