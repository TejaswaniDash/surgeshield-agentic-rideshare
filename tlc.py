"""Download + load public NYC TLC High Volume FHV trip records.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Each monthly parquet (fhvhv_tripdata_YYYY-MM.parquet) has ~20M rows across Uber
(HV0003), Lyft (HV0005) and others. We keep Uber trips and the columns we need.
`request_datetime` -> `pickup_datetime` gives a real rider wait time.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from surgeshield.config import get_settings, path

log = logging.getLogger(__name__)

HVFHV_COLUMNS = [
    "hvfhs_license_num", "dispatching_base_num", "originating_base_num",
    "request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_miles", "trip_time", "base_passenger_fare",
    "tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "tips",
    "driver_pay", "shared_request_flag", "shared_match_flag", "access_a_ride_flag",
    "wav_request_flag", "wav_match_flag", "cbd_congestion_fee",
]
NEEDED = [
    "hvfhs_license_num", "request_datetime", "pickup_datetime", "dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_miles", "trip_time", "base_passenger_fare",
    "driver_pay", "wav_request_flag",
]


def download_month(month: str, dest_dir: Path | None = None) -> Path:
    """Stream one month of HVFHV parquet to disk (≈ 450 MB)."""
    import httpx  # local import so the rest of the package works offline

    cfg = get_settings().data
    dest_dir = dest_dir or path(cfg.raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{cfg.tlc_base_url}/fhvhv_tripdata_{month}.parquet"
    out = dest_dir / f"fhvhv_tripdata_{month}.parquet"
    if out.exists():
        log.info("exists: %s", out)
        return out
    log.info("downloading %s", url)
    with httpx.stream("GET", url, timeout=600, follow_redirects=True) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    return out


def download_zone_lookup(dest_dir: Path | None = None) -> Path:
    import httpx

    cfg = get_settings().data
    dest_dir = dest_dir or path(cfg.raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "taxi_zone_lookup.csv"
    if not out.exists():
        out.write_bytes(httpx.get(cfg.tlc_zone_lookup_url, timeout=60, follow_redirects=True).content)
    return out


def load_trips(files: list[Path], zone_ids: list[int] | None = None, uber_only: bool = True) -> pd.DataFrame:
    """Load only needed columns, filter to Uber + zones of interest (pushdown via pyarrow)."""
    import pyarrow.dataset as ds

    cfg = get_settings().data
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    cols = [c for c in NEEDED if c in dataset.schema.names]
    flt = None
    if uber_only:
        flt = ds.field("hvfhs_license_num") == cfg.uber_license
    if zone_ids:
        zf = ds.field("PULocationID").isin(zone_ids) | ds.field("DOLocationID").isin(zone_ids)
        flt = zf if flt is None else (flt & zf)
    df = dataset.to_table(columns=cols, filter=flt).to_pandas()
    for c in ("request_datetime", "pickup_datetime", "dropoff_datetime"):
        df[c] = pd.to_datetime(df[c])
    log.info("loaded %s trips", f"{len(df):,}")
    return df
