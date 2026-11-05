"""Regenerate camera-level POI features without running the whole DE pipeline.

`Component2_DataSci/main.py` can run against `Carcounts.csv` alone, before the air-quality and
weather pull has been done. That file carries no POI columns, so this script
produces them directly from the published POI index, reusing DE's own
`poi_local` / `poi_select` code so the feature definitions cannot drift apart
from the full pipeline's.

    python Component2_DataSci/make_camera_poi.py

Writes data/processed/camera_poi.csv: one row per camera, with the same
poi_* columns `DE/api/api.py` would have merged in. The POI index itself is
downloaded on first use by `poi_local.PoiIndex`.

Once data/processed/data.csv exists this script is unnecessary: the full file
already carries the POI columns and Component2_DataSci/main.py prefers it.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "DE"))

import api.poi_local as poi_local      # noqa: E402  (needs the path above)

CARCOUNTS = ROOT / "data/processed/Carcounts.csv"
INDEX_PATH = ROOT / "data/raw/poi_index.parquet"
OUT_PATH = ROOT / "data/processed/camera_poi.csv"


def main() -> None:
    if not CARCOUNTS.exists():
        raise SystemExit(
            f"{CARCOUNTS} not found. Run DE/main.py, or download the release "
            f"asset it fetches, first.")

    cars = pd.read_csv(CARCOUNTS, usecols=["camera_id", "latitude", "longitude"])
    cameras = cars.drop_duplicates("camera_id").reset_index(drop=True)
    print(f"[poi] {len(cameras)} unique cameras")

    index = poi_local.PoiIndex(str(INDEX_PATH))
    features = poi_local.add_poi(cameras, index)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_PATH, index=False)
    poi_cols = [c for c in features.columns if c.startswith("poi_")]
    print(f"[poi] {len(poi_cols)} POI columns -> {OUT_PATH}")


if __name__ == "__main__":
    main()
