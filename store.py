"""Agent memory.

- Working memory : the typed incident state carried through LangGraph / Temporal (see orchestration/state.py)
- Episodic memory: SQLite table of past incidents (features, plan, outcome, postmortem). Retrieved by
                   similarity (same kind, overlapping zones, similar magnitude) to ground new plans.
- Semantic memory: postmortems are also appended to the RAG index (rag/index.py: add_document).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from surgeshield.config import path

_lock = threading.Lock()


class EpisodicMemory:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or path("artifacts/memory.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY, kind TEXT, epicenter INTEGER, zones TEXT, magnitude REAL,
                plan TEXT, outcome TEXT, postmortem TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record(self, incident_id: str, kind: str, epicenter: int, zones: list[int], magnitude: float,
               plan: dict, outcome: dict, postmortem: dict) -> None:
        with _lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO incidents (incident_id, kind, epicenter, zones, magnitude, plan, outcome,"
                      " postmortem) VALUES (?,?,?,?,?,?,?,?)",
                      (incident_id, kind, epicenter, json.dumps(zones), magnitude, json.dumps(plan),
                       json.dumps(outcome), json.dumps(postmortem)))

    def similar(self, kind: str, zones: list[int], magnitude: float, k: int = 3) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT incident_id, kind, epicenter, zones, magnitude, outcome, postmortem "
                             "FROM incidents").fetchall()
        scored = []
        for r in rows:
            z = set(json.loads(r[3]))
            score = (2.0 if r[1] == kind else 0.0) + len(z & set(zones)) / max(len(z | set(zones)), 1) \
                - abs(r[4] - magnitude) / 10
            scored.append((score, {"incident_id": r[0], "kind": r[1], "epicenter": r[2], "magnitude": r[4],
                                   "outcome": json.loads(r[5]), "lessons": json.loads(r[6]).get("lessons", [])}))
        return [s[1] for s in sorted(scored, key=lambda x: -x[0])[:k] if s[0] > 0.5]
