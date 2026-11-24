"""Typed configuration loaded from configs/settings.yaml with env-var overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("SURGESHIELD_CONFIG", ROOT / "configs" / "settings.yaml"))
ENV_PREFIX = "SURGESHIELD__"


class Section(dict):
    """dict with attribute access: cfg.llm.provider"""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(item) from e
        return Section(v) if isinstance(v, dict) else v


def _coerce(value: str, like: Any) -> Any:
    if isinstance(like, bool):
        return value.lower() in {"1", "true", "yes"}
    if isinstance(like, (int, float)):
        f = float(value)
        return int(f) if isinstance(like, int) and f.is_integer() else f
    if isinstance(like, list):
        return [v.strip() for v in value.split(",")]
    return value


@lru_cache(maxsize=1)
def get_settings() -> Section:
    with open(CONFIG_PATH) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX):].lower().split("__")
        if len(parts) != 2:
            continue
        section, field = parts
        if section in raw:
            raw[section][field] = _coerce(value, raw[section].get(field, ""))
    return Section(raw)


def path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
