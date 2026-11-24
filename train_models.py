"""Train every model and write metrics to artifacts/metrics.json.

  python scripts/train_models.py            # all
  python scripts/train_models.py --only rl  # forecaster | wait | anomaly | rl | rag
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from surgeshield.config import get_settings, path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["forecaster", "wait", "anomaly", "rl", "rag"])
    ap.add_argument("--rl-steps", type=int, default=s.rl.train_timesteps)
    args = ap.parse_args()

    proc = path(s.data.processed_dir)
    panel = pd.read_parquet(proc / "panel.parquet")
    baseline = pd.read_parquet(proc / "baseline.parquet")
    art = path("artifacts")
    art.mkdir(exist_ok=True)
    mpath = art / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}

    if "forecaster" in args.only:
        from surgeshield.ml import forecaster
        t = time.time()
        fc, m = forecaster.train(panel, s.models.lookback_bins, s.models.horizon_bins)
        fc.save(path(s.models.forecaster_path))
        metrics["forecaster"] = {**m, "train_sec": round(time.time() - t, 1)}
        log.info("forecaster %s", metrics["forecaster"])

    if "wait" in args.only:
        from surgeshield.ml import wait_model
        m, met = wait_model.train(panel, baseline)
        pickle.dump(m, open(path(s.models.wait_model_path), "wb"))
        metrics["wait_model"] = met
        log.info("wait model %s", met)

    if "anomaly" in args.only:
        from surgeshield.ml import anomaly
        scored = anomaly.score_frame(panel, baseline)
        det = anomaly.DisruptionDetector.fit(scored)
        pickle.dump(det, open(path(s.models.anomaly_path), "wb"))
        shocks_file = proc / "shocks.json"
        if shocks_file.exists():
            metrics["anomaly"] = anomaly.evaluate(det, scored, json.loads(shocks_file.read_text()))
            log.info("anomaly %s", metrics["anomaly"])

    if "rl" in args.only:
        from surgeshield.ml.rl.policy import train_ppo
        t = time.time()
        _, m = train_ppo(args.rl_steps, path(s.models.rl_policy_path), episode_steps=s.rl.episode_steps,
                         bonus_levels=tuple(s.rl.bonus_levels_usd), margin=s.rl.margin_per_trip_usd,
                         unmet_penalty=s.rl.unmet_penalty_usd)
        metrics["rl"] = {**m, "timesteps": args.rl_steps, "train_sec": round(time.time() - t, 1)}
        log.info("rl %s", json.dumps(metrics["rl"], indent=1))

    if "rag" in args.only:
        from surgeshield.rag.index import build_index
        idx = build_index()
        metrics["rag_index"] = {"chunks": len(idx.chunks), "documents": len({c.doc_id for c in idx.chunks})}

    mpath.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", mpath)


if __name__ == "__main__":
    main()
