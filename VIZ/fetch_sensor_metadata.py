"""Look up who runs each PM2.5 sensor used in Component 2 (OpenAQ v3 /locations).

Writes VIZ/data/pm25_sensors.csv, which the dashboard uses to split PM2.5 into
government reference monitors (Air4Thai) and low-cost sensors (AirGradient, Clarity...).

Usage (from the repo root):
    set OPENAQ_API_KEY in your environment or in .env, then
    python VIZ/fetch_sensor_metadata.py
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Component2_DataSci" / "output" / "dashboard" / "camera_hours.csv"
OUT = Path(__file__).resolve().parent / "data" / "pm25_sensors.csv"


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("Set OPENAQ_API_KEY first (environment variable or .env).")
    session = requests.Session()
    session.headers.update({"X-API-Key": key})

    pairs = pd.read_csv(SRC, usecols=["location_id", "sensor_id"]).dropna().drop_duplicates()
    rows = []
    for loc, sensor in pairs.itertuples(index=False):
        r = session.get(f"https://api.openaq.org/v3/locations/{int(loc)}", timeout=30)
        r.raise_for_status()
        info = r.json()["results"][0]
        provider = (info.get("provider") or {}).get("name") or "N/A"
        rows.append({
            "sensor_id": int(sensor),
            "location_id": int(loc),
            "station_name": info["name"],
            "provider": provider,
            "owner": (info.get("owner") or {}).get("name"),
            "is_reference_monitor": bool(info.get("isMonitor")),
            "sensor_type": "Air4Thai (government)" if provider == "Air4Thai" else "Low-cost sensor",
            "station_lat": info["coordinates"]["latitude"],
            "station_lon": info["coordinates"]["longitude"],
        })
        time.sleep(1.1)  # free tier: 60 requests per minute
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("sensor_id").to_csv(OUT, index=False)
    print(f"wrote {len(rows)} sensors to {OUT}")


if __name__ == "__main__":
    main()
