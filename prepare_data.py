"""Build the zone x 15-min marketplace panel.

  python scripts/prepare_data.py --source synthetic          # offline, seconds
  python scripts/prepare_data.py --source tlc --months 2024-03 2024-04   # real public data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgeshield.config import get_settings, path  # noqa: E402
from surgeshield.data import features, synthetic, tlc  # noqa: E402
from surgeshield.geo import ZONE_IDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    cfg = get_settings().data
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "tlc"], default="synthetic")
    ap.add_argument("--months", nargs="*", default=cfg.months)
    ap.add_argument("--days", type=int, default=cfg.synthetic_days)
    args = ap.parse_args()

    raw, out = path(cfg.raw_dir), path(cfg.processed_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "synthetic":
        trips, shocks = synthetic.generate(days=args.days)
        synthetic.save_shocks(shocks, out / "shocks.json")
        trips = trips[trips["hvfhs_license_num"] == cfg.uber_license]
        logging.info("synthetic Uber trips: %s, labelled shocks: %d", f"{len(trips):,}", len(shocks))
    else:
        files = [tlc.download_month(m) for m in args.months]
        tlc.download_zone_lookup()
        trips = tlc.load_trips(files, zone_ids=ZONE_IDS, uber_only=True)

    panel = features.build_panel(trips, cfg.bin_minutes)
    panel.to_parquet(out / "panel.parquet", index=False)
    features.seasonal_baseline(panel).to_parquet(out / "baseline.parquet", index=False)
    logging.info("panel rows=%s -> %s", f"{len(panel):,}", out / "panel.parquet")


if __name__ == "__main__":
    main()
