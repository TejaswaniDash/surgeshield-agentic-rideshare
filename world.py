"""MarketWorld: the agents' view of the marketplace.

Data plane  : the zone x 15-min panel built from public TLC trips (or synthetic twin).
Replay clock: incidents replay *real observed demand* from the panel while supply reacts to the
              agents' incentives in the MarketSim - a closed-loop, counterfactual digital twin.
Signals     : external feeds (weather alerts, permitted events) - treated as untrusted text.
Models      : forecaster (LSTM), wait model (GBM), anomaly detector, RL policy - loaded lazily.
"""
from __future__ import annotations

import json
import logging
import pickle
import threading
import zlib
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from surgeshield.config import get_settings, path
from surgeshield.geo import ZONE_IDS, ZONE_INDEX
from surgeshield.ml.rl.market_sim import MarketSim

log = logging.getLogger(__name__)


@dataclass
class ExternalSignal:
    source: str           # "nws_alerts" | "nyc_events" | "venue_calendar" | "flight_status"
    at: str
    zones: list[int]
    text: str             # untrusted free text
    declared_emergency: bool = False


@dataclass
class LiveIncidentSim:
    sim: MarketSim
    clock: pd.Timestamp
    zones: list[int]
    bonus_idx: np.ndarray
    spend: float = 0.0
    history: list[dict] = field(default_factory=list)


class MarketWorld:
    def __init__(self):
        self.cfg = get_settings()
        proc = path(self.cfg.data.processed_dir)
        self.panel = pd.read_parquet(proc / "panel.parquet")
        self.baseline = pd.read_parquet(proc / "baseline.parquet")
        shocks = proc / "shocks.json"
        self.shocks = json.loads(shocks.read_text()) if shocks.exists() else []
        self.signals: list[ExternalSignal] = []
        self.live: dict[str, LiveIncidentSim] = {}
        self._lock = threading.Lock()
        self._lut = self.panel.set_index(["bin_start", "zone_id"]).sort_index()

    # ------------------------------------------------------------------ models
    @cached_property
    def forecaster(self):
        from surgeshield.ml.forecaster import Forecaster
        return Forecaster.load(path(self.cfg.models.forecaster_path))

    @cached_property
    def wait_model(self):
        return pickle.load(open(path(self.cfg.models.wait_model_path), "rb"))

    @cached_property
    def detector(self):
        return pickle.load(open(path(self.cfg.models.anomaly_path), "rb"))

    @cached_property
    def rl_policy(self):
        from surgeshield.ml.rl.policy import RepositioningPolicy
        return RepositioningPolicy(path(self.cfg.models.rl_policy_path))

    # ------------------------------------------------------------------ data access
    def window(self, start, end, zones: list[int] | None = None) -> pd.DataFrame:
        p = self.panel[(self.panel["bin_start"] >= pd.Timestamp(start)) & (self.panel["bin_start"] < pd.Timestamp(end))]
        return p[p["zone_id"].isin(zones)] if zones else p

    def window_back(self, at, minutes: int, zones: list[int] | None = None) -> pd.DataFrame:
        t = pd.Timestamp(at)
        return self.window(t - pd.Timedelta(minutes=minutes), t + pd.Timedelta(minutes=1), zones)

    def history(self, zone_id: int, at, n: int) -> pd.DataFrame:
        p = self.panel[(self.panel["zone_id"] == zone_id) & (self.panel["bin_start"] <= pd.Timestamp(at))]
        return p.sort_values("bin_start").tail(n)

    def snapshot(self, at) -> pd.DataFrame:
        at = pd.Timestamp(at).floor("15min")
        try:
            return self._lut.loc[at].reset_index()
        except KeyError:
            return pd.DataFrame(columns=self.panel.columns)

    def demand_at(self, at) -> np.ndarray:
        snap = self.snapshot(at)
        d = np.zeros(len(ZONE_IDS))
        for _, r in snap.iterrows():
            d[ZONE_INDEX[int(r["zone_id"])]] = r["requests"]
        return d

    def baseline_demand(self, at) -> np.ndarray:
        at = pd.Timestamp(at)
        slot, weekend = at.hour * 4 + at.minute // 15, int(at.dayofweek >= 5)
        b = self.baseline[(self.baseline["slot"] == slot) & (self.baseline["weekend"] == weekend)]
        d = np.zeros(len(ZONE_IDS))
        for _, r in b.iterrows():
            if int(r["zone_id"]) in ZONE_INDEX:
                d[ZONE_INDEX[int(r["zone_id"])]] = r["base_med"]
        return d

    def surge_history(self, zone_id: int, at, days: int = 60) -> np.ndarray:
        """Proxy for historical surge multipliers: fare-per-mile relative to the zone's median."""
        p = self.panel[(self.panel["zone_id"] == zone_id) & (self.panel["bin_start"] <= pd.Timestamp(at))
                       & (self.panel["bin_start"] > pd.Timestamp(at) - pd.Timedelta(days=days))]
        fpm = p["fare_per_mile"].dropna()
        return (fpm / fpm.median()).to_numpy() if len(fpm) else np.array([1.0])

    def signals_for(self, at, zones: list[int] | None = None, horizon_min: int = 120) -> list[ExternalSignal]:
        t = pd.Timestamp(at)
        out = []
        for s in self.signals:
            st = pd.Timestamp(s.at)
            if t - pd.Timedelta(minutes=horizon_min) <= st <= t + pd.Timedelta(minutes=horizon_min):
                if not zones or not s.zones or set(zones) & set(s.zones):
                    out.append(s)
        return out

    # ------------------------------------------------------------------ live closed-loop sim
    def start_live(self, incident_id: str, at, zones: list[int], seed_key: str | None = None) -> LiveIncidentSim:
        with self._lock:
            if incident_id not in self.live:
                sim = MarketSim(tuple(float(b) for b in self.cfg.rl.bonus_levels_usd), self.cfg.rl.margin_per_trip_usd,
                                self.cfg.rl.unmet_penalty_usd, seed=zlib.crc32((seed_key or incident_id).encode()))
                clock = pd.Timestamp(at).floor("15min")
                sim.reset(self.baseline_demand(clock))
                # supply was sized for *normal* demand -> the shock creates a real shortage
                self.live[incident_id] = LiveIncidentSim(sim, clock, zones, np.zeros(len(ZONE_IDS), int))
            return self.live[incident_id]

    def set_incentives(self, incident_id: str, bonus_by_zone: dict[int, float]) -> None:
        live = self.live[incident_id]
        levels = np.array(self.cfg.rl.bonus_levels_usd, float)
        idx = np.zeros(len(ZONE_IDS), int)
        for zid, usd in bonus_by_zone.items():
            idx[ZONE_INDEX[zid]] = int(np.abs(levels - usd).argmin())
        live.bonus_idx = idx

    def advance(self, incident_id: str, steps: int = 1) -> dict:
        """Advance the twin by `steps` x 15 min replaying observed demand; return zone KPIs."""
        live = self.live[incident_id]
        zi = [ZONE_INDEX[z] for z in live.zones]
        out = {}
        for _ in range(steps):
            observed = self.demand_at(live.clock)
            expected = np.maximum(observed, self.baseline_demand(live.clock))
            res = live.sim.step(expected, live.bonus_idx, demand_override=observed)
            live.spend += res.incentive_cost
            d, s = res.demand[zi].sum(), res.served[zi].sum()
            w = float(np.average(res.wait_min[zi], weights=np.maximum(res.demand[zi], 1e-6)))
            out = {"clock": str(live.clock), "wait_min": round(w, 2), "fulfillment": round(float(s / max(d, 1)), 3),
                   "demand": int(d), "served": int(s), "incentive_spend_usd": round(live.spend, 2),
                   "by_zone": {int(live.zones[k]): {"wait": round(float(res.wait_min[i]), 1),
                                                     "demand": int(res.demand[i]), "idle": round(float(res.idle_before[i]), 1)}
                               for k, i in enumerate(zi)}}
            live.history.append(out)
            live.clock += pd.Timedelta(minutes=15)
        return out


_world: MarketWorld | None = None


def get_world() -> MarketWorld:
    global _world
    if _world is None:
        _world = MarketWorld()
    return _world
