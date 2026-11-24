"""Classical ML: gradient-boosted model of rider wait time given demand/supply state.

Used by the Forecast agent to translate a demand forecast into the KPI the
business cares about (minutes riders wait), and by the RL simulator for reward shaping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from surgeshield.geo import ZONE_IDS

FEATURES = ["requests", "dropoffs", "supply_ratio", "hour", "weekend", "zone_idx", "base_med", "demand_vs_base"]


def featurize(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["weekend"] = (d["dow"] >= 5).astype(int)
    d = d.merge(baseline[["zone_id", "weekend", "slot", "base_med"]], on=["zone_id", "weekend", "slot"], how="left")
    d["base_med"] = d["base_med"].fillna(d["requests"])
    d["zone_idx"] = d["zone_id"].map({z: i for i, z in enumerate(ZONE_IDS)})
    d["supply_ratio"] = d["requests"] / np.maximum(d["dropoffs"], 1)
    d["demand_vs_base"] = d["requests"] / np.maximum(d["base_med"], 1)
    return d


def train(panel: pd.DataFrame, baseline: pd.DataFrame, val_days: int = 4):
    d = featurize(panel[panel["requests"] > 0], baseline)
    cutoff = d["bin_start"].max() - pd.Timedelta(days=val_days)
    tr, va = d[d["bin_start"] < cutoff], d[d["bin_start"] >= cutoff]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, categorical_features=[5],
                                      loss="absolute_error", random_state=0)
    m.fit(tr[FEATURES], tr["wait_med"])
    pred = m.predict(va[FEATURES])
    hot = (va["demand_vs_base"] > 2).to_numpy()
    metrics = {"wait_mae_min": float(mean_absolute_error(va["wait_med"], pred)),
               "wait_mae_min_disrupted": float(mean_absolute_error(va["wait_med"][hot], pred[hot])) if hot.any() else None,
               "baseline_mae_min_disrupted": float(mean_absolute_error(va["wait_med"][hot], np.full(hot.sum(), tr["wait_med"].median()))) if hot.any() else None,
               "baseline_mae_min": float(mean_absolute_error(va["wait_med"], np.full(len(va), tr["wait_med"].median())))}
    return m, metrics


def predict_wait(model, rows: pd.DataFrame, baseline: pd.DataFrame) -> np.ndarray:
    return model.predict(featurize(rows, baseline)[FEATURES])
