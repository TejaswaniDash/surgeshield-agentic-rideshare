"""Trip-level records -> zone x 15-minute marketplace panel.

Columns per (zone_id, bin_start):
  requests      ride requests (demand)
  dropoffs      trips ending in the zone (drivers becoming free = supply proxy)
  wait_med/p90  minutes from request to pickup (the rider-experience KPI)
  fare_per_mile, driver_pay_per_min, wav_requests
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from surgeshield.geo import ZONE_IDS


def build_panel(trips: pd.DataFrame, bin_minutes: int = 15, zone_ids: list[int] | None = None) -> pd.DataFrame:
    zone_ids = zone_ids or ZONE_IDS
    freq = f"{bin_minutes}min"
    t = trips.copy()
    t["wait_min"] = (t["pickup_datetime"] - t["request_datetime"]).dt.total_seconds() / 60
    t = t[(t["wait_min"] >= 0) & (t["wait_min"] < 90)]
    t["bin"] = t["request_datetime"].dt.floor(freq)
    pu = t[t["PULocationID"].isin(zone_ids)]
    g = pu.groupby(["PULocationID", "bin"])
    demand = pd.DataFrame({
        "requests": g.size(),
        "wait_med": g["wait_min"].median(),
        "wait_p90": g["wait_min"].quantile(0.9),
        "fare_per_mile": g.apply(lambda x: x["base_passenger_fare"].sum() / max(x["trip_miles"].sum(), 0.1),
                                 include_groups=False),
        "driver_pay_per_min": g.apply(lambda x: x["driver_pay"].sum() / max(x["trip_time"].sum() / 60, 0.1),
                                      include_groups=False),
        "wav_requests": g["wav_request_flag"].apply(lambda s: (s == "Y").sum()),
    })
    demand.index.names = ["zone_id", "bin_start"]
    do = t[t["DOLocationID"].isin(zone_ids)].copy()
    do["bin"] = do["dropoff_datetime"].dt.floor(freq)
    supply = do.groupby(["DOLocationID", "bin"]).size().rename("dropoffs")
    supply.index.names = ["zone_id", "bin_start"]

    full_idx = pd.MultiIndex.from_product(
        [zone_ids, pd.date_range(t["bin"].min(), t["bin"].max(), freq=freq)], names=["zone_id", "bin_start"])
    panel = demand.join(supply, how="outer").reindex(full_idx)
    panel[["requests", "dropoffs", "wav_requests"]] = panel[["requests", "dropoffs", "wav_requests"]].fillna(0)
    for c in ("wait_med", "wait_p90", "fare_per_mile", "driver_pay_per_min"):
        panel[c] = panel.groupby(level=0)[c].transform(lambda s: s.ffill().bfill())
    panel = panel.reset_index()
    panel["hour"] = panel["bin_start"].dt.hour
    panel["dow"] = panel["bin_start"].dt.dayofweek
    panel["slot"] = panel["hour"] * 4 + panel["bin_start"].dt.minute // bin_minutes
    panel["supply_ratio"] = panel["requests"] / np.maximum(panel["dropoffs"], 1)
    return panel


def seasonal_baseline(panel: pd.DataFrame) -> pd.DataFrame:
    """Median requests per (zone, day-of-week-type, slot) - the 'normal' reference used for anomaly scoring."""
    p = panel.copy()
    p["weekend"] = (p["dow"] >= 5).astype(int)
    base = p.groupby(["zone_id", "weekend", "slot"])["requests"].agg(
        base_med="median", base_mad=lambda s: np.median(np.abs(s - np.median(s))) + 1.0).reset_index()
    return base
