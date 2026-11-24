"""Deep-learning demand forecaster: a global LSTM across zones.

Input : last `lookback` bins of [log1p(requests), log1p(dropoffs), sin/cos(slot), weekend]
        + learned zone embedding
Output: next `horizon` bins of requests (multi-horizon, direct strategy)
Loss  : Poisson NLL on counts (demand is count data)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from surgeshield.geo import ZONE_IDS

log = logging.getLogger(__name__)
N_FEAT = 5


def _features(df: pd.DataFrame) -> np.ndarray:
    slot = df["slot"].to_numpy()
    return np.stack([
        np.log1p(df["requests"].to_numpy()),
        np.log1p(df["dropoffs"].to_numpy()),
        np.sin(2 * math.pi * slot / 96),
        np.cos(2 * math.pi * slot / 96),
        (df["dow"].to_numpy() >= 5).astype(float),
    ], axis=1).astype(np.float32)


class DemandLSTM(nn.Module):
    def __init__(self, n_zones: int, horizon: int, hidden: int = 64, emb: int = 8):
        super().__init__()
        self.zone_emb = nn.Embedding(n_zones, emb)
        self.lstm = nn.LSTM(N_FEAT + emb, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor, zone: torch.Tensor, future_cal: torch.Tensor) -> torch.Tensor:
        e = self.zone_emb(zone)[:, None, :].expand(-1, x.shape[1], -1)
        h, _ = self.lstm(torch.cat([x, e], dim=-1))
        return self.head(torch.cat([h[:, -1], future_cal], dim=-1))  # log-rate


@dataclass
class Forecaster:
    model: DemandLSTM
    lookback: int
    horizon: int

    def predict_zone(self, hist: pd.DataFrame, zone_id: int) -> np.ndarray:
        """hist: last `lookback` panel rows for a zone (ordered). Returns expected requests for next bins."""
        x = torch.tensor(_features(hist.tail(self.lookback)))[None]
        last = hist.iloc[-1]
        nxt = (int(last["slot"]) + 1) % 96
        cal = torch.tensor([[math.sin(2 * math.pi * nxt / 96), math.cos(2 * math.pi * nxt / 96),
                             float(last["dow"] >= 5)]], dtype=torch.float32)
        z = torch.tensor([ZONE_IDS.index(zone_id)])
        self.model.eval()
        with torch.no_grad():
            return torch.exp(self.model(x, z, cal))[0].numpy()

    def save(self, p) -> None:
        torch.save({"state": self.model.state_dict(), "lookback": self.lookback, "horizon": self.horizon,
                    "n_zones": self.model.zone_emb.num_embeddings}, p)

    @classmethod
    def load(cls, p) -> "Forecaster":
        ck = torch.load(p, weights_only=False)
        m = DemandLSTM(ck["n_zones"], ck["horizon"])
        m.load_state_dict(ck["state"])
        return cls(m, ck["lookback"], ck["horizon"])


def make_windows(panel: pd.DataFrame, lookback: int, horizon: int):
    X, Z, C, Y, T = [], [], [], [], []
    for zid, g in panel.sort_values("bin_start").groupby("zone_id"):
        f = _features(g)
        req = g["requests"].to_numpy(np.float32)
        slot, dow, ts = g["slot"].to_numpy(), g["dow"].to_numpy(), g["bin_start"].to_numpy()
        zi = ZONE_IDS.index(zid)
        for i in range(lookback, len(g) - horizon + 1):
            X.append(f[i - lookback:i])
            Z.append(zi)
            C.append([math.sin(2 * math.pi * slot[i] / 96), math.cos(2 * math.pi * slot[i] / 96), float(dow[i] >= 5)])
            Y.append(req[i:i + horizon])
            T.append(ts[i])
    return (np.array(X), np.array(Z), np.array(C, np.float32), np.array(Y), np.array(T))


def train(panel: pd.DataFrame, lookback: int = 16, horizon: int = 4, epochs: int = 6, val_days: int = 4,
          seed: int = 0) -> tuple[Forecaster, dict]:
    torch.manual_seed(seed)
    X, Z, C, Y, T = make_windows(panel, lookback, horizon)
    cutoff = T.max() - np.timedelta64(val_days, "D")
    tr, va = T < cutoff, T >= cutoff
    model = DemandLSTM(len(ZONE_IDS), horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.PoissonNLLLoss(log_input=True)
    Xt, Zt, Ct, Yt = map(torch.tensor, (X[tr], Z[tr], C[tr], Y[tr]))
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx], Zt[idx], Ct[idx]), Yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        log.info("epoch %d poisson_nll=%.4f", ep, tot / n)
    model.eval()
    with torch.no_grad():
        pred = torch.exp(model(*map(torch.tensor, (X[va], Z[va], C[va])))).numpy()
    yv = Y[va]
    # seasonal-naive baseline: same slot one day earlier (96 bins back)
    naive = _seasonal_naive(panel, T[va], Z[va], horizon)
    metrics = {
        "lstm_mae": float(np.abs(pred - yv).mean()),
        "seasonal_naive_mae": float(np.abs(naive - yv).mean()),
        "lstm_mae_h1": float(np.abs(pred[:, 0] - yv[:, 0]).mean()),
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
    }
    return Forecaster(model, lookback, horizon), metrics


def _seasonal_naive(panel: pd.DataFrame, ts: np.ndarray, zidx: np.ndarray, horizon: int) -> np.ndarray:
    lut = panel.set_index(["zone_id", "bin_start"])["requests"]
    out = np.zeros((len(ts), horizon), np.float32)
    for k, (t, zi) in enumerate(zip(ts, zidx)):
        for h in range(horizon):
            key = (ZONE_IDS[zi], pd.Timestamp(t) + pd.Timedelta(minutes=15 * h) - pd.Timedelta(days=1))
            out[k, h] = lut.get(key, 0.0)
    return out
