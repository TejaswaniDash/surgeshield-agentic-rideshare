"""Disruption detection = statistical seasonality test + unsupervised ML.

1. Robust z-score of requests vs. the (zone, weekday/weekend, slot) median/MAD baseline.
2. IsolationForest over [z, supply_ratio, wait_med, demand_vs_base] to catch
   multi-signal anomalies (e.g. wait spikes without a demand spike => supply outage).
An alert fires when z >= z_threshold OR the forest flags the point with a high wait.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATS = ["z", "supply_ratio", "wait_med", "demand_vs_base"]


def score_frame(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["weekend"] = (d["dow"] >= 5).astype(int)
    d = d.merge(baseline, on=["zone_id", "weekend", "slot"], how="left")
    d["base_med"] = d["base_med"].fillna(d["requests"].median())
    d["base_mad"] = d["base_mad"].fillna(1.0)
    d["z"] = (d["requests"] - d["base_med"]) / (1.4826 * d["base_mad"])
    d["demand_vs_base"] = d["requests"] / np.maximum(d["base_med"], 1)
    d["supply_ratio"] = d["requests"] / np.maximum(d["dropoffs"], 1)
    return d


@dataclass
class DisruptionDetector:
    forest: IsolationForest
    z_threshold: float = 4.0
    min_requests: int = 12

    def detect(self, scored: pd.DataFrame) -> pd.DataFrame:
        iso = -self.forest.score_samples(scored[FEATS].fillna(0))
        scored = scored.assign(iso_score=iso)
        flag = (scored["requests"] >= self.min_requests) & (
            (scored["z"] >= self.z_threshold) | ((iso > 0.62) & (scored["wait_med"] > 8)))
        return scored[flag]

    @classmethod
    def fit(cls, scored: pd.DataFrame, contamination: float = 0.01) -> "DisruptionDetector":
        f = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
        f.fit(scored[FEATS].fillna(0))
        return cls(f)


def evaluate(detector: DisruptionDetector, scored: pd.DataFrame, shocks: list[dict]) -> dict:
    """Bin-level precision/recall against labelled synthetic shocks (primary zones)."""
    truth = np.zeros(len(scored), bool)
    for s in shocks:
        m = (scored["bin_start"] >= pd.Timestamp(s["start"])) & (scored["bin_start"] < pd.Timestamp(s["end"]))
        truth |= (m & scored["zone_id"].isin(s["zones"])).to_numpy()
    pred = scored.index.isin(detector.detect(scored).index)
    tp = int((pred & truth).sum())
    # event-level recall: did we fire at least once during each shock?
    hit = 0
    for s in shocks:
        m = (scored["bin_start"] >= pd.Timestamp(s["start"])) & (scored["bin_start"] < pd.Timestamp(s["end"])) \
            & scored["zone_id"].isin(s["zones"])
        hit += bool((pred & m.to_numpy()).any())
    return {"bin_precision": round(float(tp / max(pred.sum(), 1)), 3),
            "bin_recall": round(float(tp / max(truth.sum(), 1)), 3),
            "event_recall": round(hit / max(len(shocks), 1), 3), "alerts": int(pred.sum()),
            "note": "bin recall is low by design: citywide 1.5x rain bins are labelled but are not alert-worthy"}
