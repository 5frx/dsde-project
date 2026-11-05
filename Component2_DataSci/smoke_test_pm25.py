"""Prove the PM2.5 code path runs, before the real air-quality data exists.

Sections 7 and 8 of the analysis only execute when `data/processed/data.csv`
carries the air-quality and weather columns. Until the OpenAQ pull is done,
that code is untested, and "it will work when the data arrives" is a claim
rather than a fact.

This script builds a synthetic `data.csv` in exactly the schema
`DE/api/api.make_df` produces, runs `Component2_DataSci/main.py` against it, and restores
whatever was there before. The synthetic PM2.5 and weather values are random
and independent of the real vehicle counts, which makes the test sharper than a
plain import check: a correlation analysis run on noise must report nulls. If
any section claims a relationship here, that section is manufacturing signal.

    python Component2_DataSci/smoke_test_pm25.py

It prints PASS or FAIL and leaves the repository as it found it. It is a test of
the code, not a source of results: the report it generates mid-run is
overwritten by the next real run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data/processed"
DATA_CSV = PROCESSED / "data.csv"
BACKUP = PROCESSED / "data.csv.smoketest-backup"

WEATHER_RANGES = {
    "temperature_2m": (24, 36), "precipitation": (0, 8),
    "wind_speed_10m": (0, 20), "wind_direction_10m": (0, 360),
    "relative_humidity_2m": (45, 95), "cloud_cover": (0, 100),
    "surface_pressure": (995, 1015),
}
# Fewer stations than cameras, which is the real situation and the thing the
# audit in section 7a is meant to catch.
N_STATIONS = 30
MISSING_PM25_SHARE = 0.07


def build_synthetic(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(PROCESSED / "Carcounts.csv")

    poi_path = PROCESSED / "camera_poi.csv"
    if poi_path.exists():
        poi = pd.read_csv(poi_path)
        cols = ["camera_id"] + [c for c in poi.columns if c.startswith("poi_")]
        df = df.merge(poi[cols], on="camera_id", how="left")

    # Air quality: one value per camera-hour, with station provenance.
    hours = df[["camera_id", "hourly_timestamp"]].drop_duplicates()
    hours["pm25"] = rng.gamma(4, 6, len(hours))
    cameras = df["camera_id"].unique()
    station = dict(zip(cameras, rng.integers(9000, 9000 + N_STATIONS, len(cameras))))
    distance = dict(zip(cameras, rng.uniform(200, 4800, len(cameras))))
    hours["location_id"] = hours["camera_id"].map(station).astype(int)
    hours["sensor_id"] = hours["location_id"] + 100_000
    hours["station_distance_m"] = hours["camera_id"].map(distance)
    hours["is_fallback"] = rng.integers(0, 2, len(hours))

    df = df.merge(hours, on=["camera_id", "hourly_timestamp"], how="left")
    df.loc[rng.random(len(df)) < MISSING_PM25_SHARE, "pm25"] = np.nan

    for col, (lo, hi) in WEATHER_RANGES.items():
        df[col] = rng.uniform(lo, hi, len(df))
    df["coord_key"] = (df["latitude"].round(4).astype(str) + "," +
                       df["longitude"].round(4).astype(str))
    return df


def main() -> int:
    if not (PROCESSED / "Carcounts.csv").exists():
        print("FAIL: data/processed/Carcounts.csv is missing; nothing to build on.")
        return 1

    restore = DATA_CSV.exists()
    if restore:
        shutil.move(DATA_CSV, BACKUP)
    try:
        df = build_synthetic()
        df.to_csv(DATA_CSV, index=False)
        print(f"[smoke] synthetic data.csv: {df.shape[0]:,} rows x {df.shape[1]} cols")

        result = subprocess.run([sys.executable, "Component2_DataSci/main.py"], cwd=ROOT,
                                capture_output=True, text=True)
        print(result.stdout.strip()[-400:])
        if result.returncode != 0:
            print("FAIL: Component2_DataSci/main.py exited non-zero on the PM2.5 path.")
            print(result.stderr.strip()[-2000:])
            return 1

        report = (ROOT / "Component2_DataSci/output/component2_report.md").read_text(encoding="utf-8")
        if "SKIPPED: pm25" in report:
            print("FAIL: section 7 was skipped even though pm25 was present.")
            return 1
        if "pm25 vs traffic decomposition" not in report:
            print("FAIL: the traffic/PM2.5 decomposition table was not produced.")
            return 1
        if "no raw traffic-PM2.5 association" not in report:
            print("FAIL: a relationship was reported on random data. A section is "
                  "manufacturing signal; check the guards in section_pm25.")
            return 1

        print("PASS: the PM2.5 path runs end to end and correctly reports nulls "
              "on random data.")
        return 0
    finally:
        DATA_CSV.unlink(missing_ok=True)
        if restore:
            shutil.move(BACKUP, DATA_CSV)
        print("[smoke] repository restored")


if __name__ == "__main__":
    sys.exit(main())
