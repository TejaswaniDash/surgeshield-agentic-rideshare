"""Temporal worker + client helpers.

  temporal server start-dev                     # local Temporal (or docker compose up temporal)
  python -m surgeshield.temporal_app.worker     # run a worker
  python scripts/start_incident.py --scenario msg_letout --runtime temporal
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from surgeshield.config import get_settings
from surgeshield.orchestration.activities import ACTIVITIES
from surgeshield.temporal_app.workflows import DisruptionResponseWorkflow

log = logging.getLogger(__name__)


def _wrap(name, fn):
    @activity.defn(name=name)
    def _act(payload: dict) -> dict:
        return fn(payload)
    return _act


TEMPORAL_ACTIVITIES = [_wrap(n, f) for n, f in ACTIVITIES.items()]


async def connect() -> Client:
    t = get_settings().temporal
    return await Client.connect(t.address, namespace=t.namespace)


def build_worker(client: Client) -> Worker:
    return Worker(client, task_queue=get_settings().temporal.task_queue, workflows=[DisruptionResponseWorkflow],
                  activities=TEMPORAL_ACTIVITIES, activity_executor=ThreadPoolExecutor(max_workers=8))


async def start_incident(client: Client, disruption: dict):
    from temporalio.common import WorkflowIDReusePolicy
    s = get_settings()
    return await client.start_workflow(
        DisruptionResponseWorkflow.run, args=[disruption, dict(s.workflow), dict(s.policy)],
        id=disruption["incident_id"], task_queue=s.temporal.task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    client = await connect()
    worker = build_worker(client)
    log.info("worker listening on task queue %s", get_settings().temporal.task_queue)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
