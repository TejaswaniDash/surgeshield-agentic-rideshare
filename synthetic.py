"""Synthetic trips in the exact HVFHV schema, for offline development and tests.

Mirrors realistic daily/weekly seasonality and injects *labelled* demand shocks
(arena let-outs, airport arrival banks, rainstorms, transit outages) so the
anomaly detector and agents can be evaluated against ground truth. When real
TLC data is available the pipeline is identical - only the source changes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from surgeshield.geo import ZONES, ZONE_INDEX, distance_matrix

HOURLY = np.array([  # relative demand by hour of day (weekday)
    0.45, 0.30, 0.20, 0.15, 0.15, 0.25, 0.50, 0.85, 1.00, 0.85, 0.70, 0.70,
    0.75, 0.75, 0.75, 0.80, 0.90, 1.00, 1.00, 0.95, 0.85, 0.80, 0.75, 0.60,
])
WEEKEND_HOURLY = np.array([
    0.80, 0.70, 0.55, 0.40, 0.25, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85,
    0.90, 0.90, 0.85, 0.85, 0.85, 0.90, 0.95, 0.95, 0.95, 1.00, 1.00, 0.95,
])
BASE_PER_HOUR = {"airport": 140.0, "arena": 70.0, "theatre": 75.0, "nightlife": 55.0, None: 45.0}
MIDTOWN = {161, 162, 163, 164, 230, 48, 100}


@dataclass
class Shock:
    kind: str
    zones: list[int]
    start: str
    end: str
    multiplier: float
    spill_zones: list[int]
    spill_multiplier: float
    description: str


def _profile(ts: pd.DatetimeIndex) -> np.ndarray:
    weekend = ts.dayofweek >= 5
    return np.where(weekend, WEEKEND_HOURLY[ts.hour], HOURLY[ts.hour])


def plan_shocks(start: pd.Timestamp, days: int, rng: np.random.Generator) -> list[Shock]:
    shocks: list[Shock] = []
    day_list = [start + pd.Timedelta(days=d) for d in range(days)]
    for d in rng.choice(days, size=min(6, days), replace=False):  # arena let-outs (MSG = zone 186)
        t0 = day_list[d] + pd.Timedelta(hours=22, minutes=int(rng.choice([30, 45])))
        shocks.append(Shock("arena_letout", [186], str(t0), str(t0 + pd.Timedelta(minutes=45)),
                            float(rng.uniform(4.0, 6.0)), [100, 164, 230], 1.8,
                            "Concert let-out at Madison Square Garden"))
    for d in rng.choice(days, size=min(3, days), replace=False):  # airport arrival bank after delays
        t0 = day_list[d] + pd.Timedelta(hours=20)
        shocks.append(Shock("airport_bank", [132], str(t0), str(t0 + pd.Timedelta(hours=2)),
                            float(rng.uniform(2.2, 3.0)), [138], 1.4,
                            "Delayed arrival bank at JFK"))
    for d in rng.choice(days, size=min(2, days), replace=False):  # evening rainstorm, citywide
        t0 = day_list[d] + pd.Timedelta(hours=17)
        shocks.append(Shock("rainstorm", [z.id for z in ZONES], str(t0), str(t0 + pd.Timedelta(hours=3)),
                            1.5, [], 1.0, "Heavy rain during evening rush"))
    d = int(rng.integers(0, days))  # subway outage in the morning rush
    t0 = day_list[d] + pd.Timedelta(hours=8)
    shocks.append(Shock("transit_outage", [164, 186, 100, 161], str(t0), str(t0 + pd.Timedelta(minutes=90)),
                        2.2, [170, 68], 1.5, "Subway signal outage on 7th/8th Ave lines"))
    return shocks


def generate(days: int = 21, start: str = "2024-03-04", seed: int = 7) -> tuple[pd.DataFrame, list[Shock]]:
    rng = np.random.default_rng(seed)
    t_start = pd.Timestamp(start)
    bins = pd.date_range(t_start, periods=days * 96, freq="15min")
    nz = len(ZONES)
    base = np.array([BASE_PER_HOUR[z.venue] * (1.3 if z.id in MIDTOWN else 1.0) for z in ZONES]) / 4.0
    prof = _profile(bins)
    lam_normal = prof[:, None] * base[None, :]                       # (T, Z) expected requests / bin
    # theatre let-out is a *regular* (seasonal) pattern, not an anomaly
    theatre = np.isin([z.id for z in ZONES], [230, 142])
    letout = (bins.hour == 22) & (bins.dayofweek.isin([2, 3, 4, 5, 6]))
    lam_normal[np.ix_(letout, theatre)] *= 1.7

    mult = np.ones_like(lam_normal)
    shocks = plan_shocks(t_start, days, rng)
    for s in shocks:
        m = (bins >= pd.Timestamp(s.start)) & (bins < pd.Timestamp(s.end))
        for zid in s.zones:
            mult[m, ZONE_INDEX[zid]] *= s.multiplier
        for zid in s.spill_zones:
            mult[m, ZONE_INDEX[zid]] *= s.spill_multiplier
    lam = lam_normal * mult * rng.lognormal(0, 0.08, size=lam_normal.shape)
    counts = rng.poisson(lam)

    # wait time grows convexly when demand outruns the supply that normally serves the zone
    ratio = np.maximum(counts / np.maximum(lam_normal, 1e-3), 0.0)
    wait_mean = 3.2 + 3.2 * np.maximum(ratio - 1.0, 0) ** 1.1 +np.where(theatre[None, :] & letout[:, None], 1.0, 0)

    # destination choice: gravity model over zone distance
    dist = np.array(distance_matrix())
    attract = np.array([3.0 if z.venue == "airport" else 1.0 for z in ZONES])
    p_dest = attract[None, :] * np.exp(-dist / 6.0)
    np.fill_diagonal(p_dest, p_dest.diagonal() * 0.3)
    p_dest /= p_dest.sum(1, keepdims=True)

    t_idx, z_idx = np.nonzero(counts)
    reps = counts[t_idx, z_idx]
    t_rep, z_rep = np.repeat(t_idx, reps), np.repeat(z_idx, reps)
    n = len(t_rep)
    req = bins.values[t_rep] + rng.uniform(0, 15 * 60, n).astype("timedelta64[s]")
    wait_min = rng.gamma(2.5, wait_mean[t_rep, z_rep] / 2.5)
    pickup = req + (wait_min * 60).astype("timedelta64[s]")
    u = rng.random(n)
    cdf = p_dest.cumsum(1)
    dest_idx = (u[:, None] > cdf[z_rep]).sum(1).clip(0, nz - 1)
    miles = np.maximum(0.4, dist[z_rep, dest_idx] * 0.62 * rng.uniform(1.1, 1.5, n) + rng.exponential(0.6, n))
    trip_time = (miles / rng.uniform(7, 14, n) * 3600).astype(int)
    dropoff = pickup + trip_time.astype("timedelta64[s]")
    surge = np.clip(1 + 0.25 * np.maximum(ratio[t_rep, z_rep] - 1.2, 0), 1, 2.5)
    fare = np.round((3.0 + 2.1 * miles + 0.75 * trip_time / 60) * surge, 2)
    tlc_min = 1.283 * miles + 0.681 * trip_time / 60  # TLC minimum per-trip pay (2026 rates) is a floor
    zone_ids = np.array([z.id for z in ZONES])
    df = pd.DataFrame({
        "hvfhs_license_num": np.where(rng.random(n) < 0.72, "HV0003", "HV0005"),
        "dispatching_base_num": "B03404",
        "originating_base_num": "B03404",
        "request_datetime": req,
        "on_scene_datetime": pd.NaT,
        "pickup_datetime": pickup,
        "dropoff_datetime": dropoff,
        "PULocationID": zone_ids[z_rep],
        "DOLocationID": zone_ids[dest_idx],
        "trip_miles": np.round(miles, 2),
        "trip_time": trip_time,
        "base_passenger_fare": fare,
        "tolls": 0.0, "bcf": np.round(fare * 0.0275, 2), "sales_tax": np.round(fare * 0.08875, 2),
        "congestion_surcharge": 2.75, "airport_fee": np.where(np.isin(zone_ids[z_rep], [132, 138]), 2.5, 0.0),
        "tips": np.round(np.where(rng.random(n) < 0.3, fare * 0.15, 0), 2),
        "driver_pay": np.round(np.maximum(fare * rng.uniform(0.68, 0.78, n), tlc_min * rng.uniform(1.0, 1.1, n)), 2),
        "shared_request_flag": "N", "shared_match_flag": "N", "access_a_ride_flag": " ",
        "wav_request_flag": np.where(rng.random(n) < 0.01, "Y", "N"), "wav_match_flag": "N",
        "cbd_congestion_fee": 0.0,
    })
    return df.sort_values("request_datetime").reset_index(drop=True), shocks


def save_shocks(shocks: list[Shock], out_path) -> None:
    with open(out_path, "w") as f:
        json.dump([asdict(s) for s in shocks], f, indent=2)
