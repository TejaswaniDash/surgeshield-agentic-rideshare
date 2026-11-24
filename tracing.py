"""Minimal tracing + metrics: nested spans with trace ids, emitted as JSON lines.

Drop-in compatible with OpenTelemetry concepts (trace_id/span_id/parent). If the
`opentelemetry` package is installed and OTEL_EXPORTER_OTLP_ENDPOINT is set you
can bridge `span()` to a real tracer; we keep zero hard dependencies here.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any

log = logging.getLogger("surgeshield.trace")
_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("span", default=None)
SPANS: list[dict] = []  # in-memory sink (exposed at /traces for the demo)
COUNTERS: dict[str, float] = {}


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def span(name: str, **attrs: Any):
    parent = _current.get()
    rec = {"name": name, "span_id": uuid.uuid4().hex[:8], "parent_id": parent["span_id"] if parent else None,
           "trace_id": (parent or {}).get("trace_id") or attrs.pop("trace_id", None) or new_trace_id(),
           "start": time.time(), "attrs": attrs, "status": "ok"}
    token = _current.set(rec)
    try:
        yield rec
    except Exception as e:
        rec["status"] = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        rec["duration_ms"] = round((time.time() - rec["start"]) * 1000, 2)
        _current.reset(token)
        SPANS.append(rec)
        if len(SPANS) > 5000:
            del SPANS[:1000]
        log.debug(json.dumps({k: v for k, v in rec.items() if k != "attrs"} | {"attrs": str(rec["attrs"])[:300]}))


def incr(metric: str, value: float = 1.0) -> None:
    COUNTERS[metric] = COUNTERS.get(metric, 0.0) + value
