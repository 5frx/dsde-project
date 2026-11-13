"""Scrape the extra sources for the insights app (run from the repo root, ~5 minutes).

New sources (all public, no key):
  1. OpenStreetMap Nominatim reverse geocoding -> Bangkok district (khet) of every camera
  2. OpenStreetMap Nominatim search             -> rail transit stations (BTS, MRT, ARL, SRT) in Bangkok
  3. Open-Meteo archive                          -> sunrise / sunset for each day (is the camera image dark?)
Nominatim's policy is max 1 request/second with an identifying User-Agent; this script respects that.
"""
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
UA = {"User-Agent": "dsde-cctv-course-project/1.0 (Chulalongkorn data science course)"}
NOMINATIM = "https://nominatim.openstreetmap.org"


def get(url, **params):
    for attempt in range(4):
        r = requests.get(url, params=params, headers=UA, timeout=60)
        if r.ok:
            time.sleep(1.1)
            return r.json()
        time.sleep(2 ** attempt)
    r.raise_for_status()


def districts(cams):
    rows = []
    for c in cams.itertuples():
        a = get(f"{NOMINATIM}/reverse", format="jsonv2", lat=c.latitude, lon=c.longitude, zoom=14,
                **{"accept-language": "en"}).get("address", {})
        rows.append({"camera_id": c.camera_id, "district": a.get("suburb") or a.get("city_district") or a.get("town"),
                     "subdistrict": a.get("quarter") or a.get("neighbourhood")})
    d = pd.DataFrame(rows)
    d["district"] = d.district.str.replace(" District", "", regex=False)
    return d


def stations():
    seen, rows = [], []
    while True:
        res = get(f"{NOMINATIM}/search", format="jsonv2", q="railway station in Bangkok", limit=50,
                  viewbox="100.30,13.95,100.80,13.55", bounded=1, exclude_place_ids=",".join(seen),
                  **{"accept-language": "en"})
        res = [x for x in res if x["category"] == "railway" and x["type"] in ("station", "halt")]
        if not res:
            break
        for x in res:
            seen.append(str(x["place_id"]))
            rows.append({"station": x["name"], "lat": float(x["lat"]), "lon": float(x["lon"]), "osm_type": x["osm_type"]})
        if len(seen) > 600:  # safety stop
            break
    return pd.DataFrame(rows).drop_duplicates(["station"])


def sun():
    d = get("https://archive-api.open-meteo.com/v1/archive", latitude=13.75, longitude=100.52,
            start_date="2026-08-25", end_date="2026-08-31", daily="sunrise,sunset", timezone="Asia/Bangkok")["daily"]
    return pd.DataFrame({"date": d["time"], "sunrise": d["sunrise"], "sunset": d["sunset"]})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cams = pd.read_csv(ROOT / "Component2_DataSci" / "output" / "dashboard" / "cameras.csv")
    sun().to_csv(OUT / "sun.csv", index=False)
    s = stations()
    s.to_csv(OUT / "rail_stations.csv", index=False)
    print("stations", len(s))
    d = districts(cams)
    d.to_csv(OUT / "camera_districts.csv", index=False)
    print("districts", d.district.nunique(), d.district.isna().sum(), "missing")
