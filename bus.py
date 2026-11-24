"""Thread-safe in-process bus for incident snapshots and A2UI streams (consumed by the API's SSE
endpoints). Swap for Redis Streams / Kafka when the worker and API run in separate processes."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from surgeshield.config import path


class Bus:
    def __init__(self):
        self._lock = threading.Lock()
        self.ui: dict[str, list[dict]] = {}
        self.incidents: dict[str, dict] = {}
        self.decisions: dict[str, dict] = {}
        self.outbox = path("artifacts/outbox.jsonl")

    def publish_ui(self, incident_id: str, messages: list[dict]) -> int:
        with self._lock:
            self.ui.setdefault(incident_id, []).extend(messages)
            self.ui.setdefault("*", []).extend({"incidentId": incident_id, **m} for m in messages)
            return len(self.ui[incident_id])

    def ui_since(self, incident_id: str, idx: int) -> list[dict]:
        with self._lock:
            return list(self.ui.get(incident_id, [])[idx:])

    def put_state(self, incident_id: str, state: dict) -> None:
        with self._lock:
            self.incidents[incident_id] = {**state, "updated": time.time()}

    def deliver(self, channel: str, payload: dict) -> None:
        """'Sends' a driver/rider message. Demo: append to an outbox file (no real notifications)."""
        Path(self.outbox).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.outbox, "a") as f:
            f.write(json.dumps({"channel": channel, "ts": time.time(), **payload}) + "\n")


BUS = Bus()
