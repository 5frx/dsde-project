"""Helpers for pulling OpenAQ v3 air-quality data for the CCTV locations.

Import and call `build_dataset()`, or run the file directly:
    python openaq_client.py

Nothing runs at import time: the session, API key check and cache are all
created inside functions.

How missing PM2.5 is handled
  1. Every PM2.5 station within SEARCH_RADIUS_M of a coordinate is ranked by distance.
  2. Stations whose reporting period (datetimeFirst/datetimeLast) does not overlap the
     dataset's time range are dropped before any hourly data is downloaded.
  3. For each hour, the reading comes from the nearest remaining station that has one;
     hours it is missing are filled from the next-nearest station, and so on, up to
     CANDIDATE_STATIONS stations. Stations are downloaded lazily, so no more API calls
     are made once every hour for a coordinate is filled.
  4. The output records where each reading came from: `location_id` / `sensor_id` are the
     station that supplied the PM2.5 value, `station_distance_m` is how far away it is, and
     `is_fallback` is 1 when it was not the nearest usable station. Hours no station covers
     are left as NaN.
"""

import json
import math
import os
import time
from typing import Callable, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.openaq.org/v3/"
CCTV_PATH = "data/raw/cctv_info.csv"
CACHE_PATH = "data/cached/cache_openaq.json"
OUTPUT_PATH = "data/processed/cctv_air_quality.csv"

TIMESTAMP_COL = "hourly_timestamp"  # column in the CCTV dataframe
DEFAULT_TZ = "Asia/Bangkok"         # timezone of naive timestamps in that column

PARAMETER = "pm25"
SEARCH_RADIUS_M = 5000
CANDIDATE_STATIONS = 5   # most stations used per coordinate (nearest first) to fill gaps
PAGE_LIMIT = 1000
USER_AGENT = "dsde-project/1.0 (6838075421@student.chula.ac.th)"

# Columns this module adds. They describe the station that supplied each PM2.5 value.
SOURCE_COLS = ["location_id", "sensor_id", "station_distance_m", "is_fallback"]


# --------------------------------------------------------------------------- #
# Session and HTTP
# --------------------------------------------------------------------------- #
def build_session(api_key: Optional[str] = None) -> requests.Session:
    """Create a requests session with the OpenAQ key and User-Agent set."""
    load_dotenv()
    api_key = api_key or os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAQ_API_KEY environment variable first.")

    session = requests.Session()
    session.headers.update({"X-API-Key": api_key, "User-Agent": USER_AGENT})
    return session


def _header_seconds(response: requests.Response, *names: str, default: int) -> int:
    """Return the first of the given headers that parses as an int."""
    for name in names:
        try:
            return int(float(response.headers[name]))
        except (KeyError, ValueError):
            continue
    return default


def api_get(session: requests.Session, path: str, params: Optional[dict] = None,
            max_retries: int = 5) -> dict:
    """GET a path under BASE_URL and return the parsed JSON.

    Waits on HTTP 429 and when the rate-limit window is used up, instead of
    sleeping a fixed time after every call.
    """
    url = f"{BASE_URL}{path.lstrip('/')}"
    for _ in range(max_retries):
        response = session.get(url, params=params, timeout=30)

        if response.status_code == 429:
            wait = _header_seconds(response, "Retry-After", "x-ratelimit-reset", default=60)
            print(f"Rate limit hit, waiting {wait}s")
            time.sleep(wait)
            continue

        # print(response.url)
        response.raise_for_status()

        if response.headers.get("x-ratelimit-remaining") == "0":
            time.sleep(_header_seconds(response, "x-ratelimit-reset", default=1) + 1)
        return response.json()

    raise RuntimeError(f"Gave up on {url} after {max_retries} attempts")


# --------------------------------------------------------------------------- #
# Cache (so a crash or rate-limit stall doesn't lose progress)
# --------------------------------------------------------------------------- #
def load_cache(path: str = CACHE_PATH) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"stations": {}, "hours": {}}


def save_cache(cache: dict, path: str = CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)  # atomic, so a crash can't leave a half-written file


# --------------------------------------------------------------------------- #
# Step 1: coordinates -> ranked list of nearby stations
# --------------------------------------------------------------------------- #
def add_coord_key(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Add a 'coord_key' column (rounded 'lat,lon') used to dedupe lookups."""
    df = df.copy()
    df["coord_key"] = (
        df["latitude"].round(decimals).astype(str) + "," +
        df["longitude"].round(decimals).astype(str)
    )
    return df


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(a))


def pick_sensor_id(location: dict, parameter: str = PARAMETER) -> Optional[int]:
    """Return the id of the sensor measuring `parameter` at a location, if any."""
    for sensor in location.get("sensors", []):
        if sensor["parameter"]["name"] == parameter:
            return sensor["id"]
    return None


def _utc_str(value) -> Optional[str]:
    """OpenAQ datetimes are {'utc': ..., 'local': ...}; return the UTC string or None."""
    if isinstance(value, dict):
        value = value.get("utc")
    return value or None


def find_stations(session: requests.Session, coord_key: str,
                  radius: int = SEARCH_RADIUS_M, parameter: str = PARAMETER) -> list:
    """Every station near a coordinate with a `parameter` sensor, nearest first.

    Each entry keeps the station's reporting period so stations that never reported
    during the dataset's time range can be skipped later without another API call.
    """
    lat, lon = (float(x) for x in coord_key.split(","))
    results = api_get(session, "locations", {
        "coordinates": coord_key,
        "radius": radius,
        "limit": PAGE_LIMIT,
    })["results"]

    found = []
    for location in results:
        sensor_id = pick_sensor_id(location, parameter)
        coords = location.get("coordinates") or {}
        if sensor_id is None or coords.get("latitude") is None:
            continue
        found.append({
            "location_id": location["id"],
            "sensor_id": sensor_id,
            "name": location.get("name"),
            "distance_m": round(haversine_m(lat, lon, coords["latitude"], coords["longitude"])),
            "first": _utc_str(location.get("datetimeFirst")),
            "last": _utc_str(location.get("datetimeLast")),
        })
    return sorted(found, key=lambda c: c["distance_m"])


def resolve_stations(session: requests.Session, coord_keys, cache: dict,
                     cache_path: str = CACHE_PATH) -> None:
    """Look up each unique coordinate once and store its ranked stations in the cache."""
    stations = cache["stations"]
    coord_keys = list(coord_keys)

    for i, key in enumerate(coord_keys, start=1):
        # Older caches stored a single {location_id, sensor_id} dict; redo those.
        if isinstance(stations.get(key), list):
            continue
        stations[key] = find_stations(session, key)
        save_cache(cache, cache_path)
        nearest = stations[key][0]["distance_m"] if stations[key] else None
        print(f"[stations {i}/{len(coord_keys)}] {key} -> {len(stations[key])} "
              f"{PARAMETER} station(s), nearest {nearest} m")


def usable_stations(candidates: list, start: str, end: str,
                    limit: int = CANDIDATE_STATIONS) -> list:
    """Drop stations that never reported during [start, end], then keep the nearest `limit`."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    kept = []
    for cand in candidates:
        if cand["last"] and pd.Timestamp(cand["last"]) < start_ts:
            continue
        if cand["first"] and pd.Timestamp(cand["first"]) > end_ts:
            continue
        kept.append(cand)
    return kept[:limit]


# --------------------------------------------------------------------------- #
# Step 2: sensor -> hourly measurements
# --------------------------------------------------------------------------- #
def to_utc_hour(series: pd.Series, tz: str = DEFAULT_TZ) -> pd.Series:
    """Parse timestamps, convert to UTC and floor to the hour.

    Naive timestamps are treated as local time in `tz` (OpenAQ is in UTC).
    """
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    return ts.dt.tz_convert("UTC").dt.floor("h")


def fetch_sensor_hours(session: requests.Session, sensor_id: int,
                       start: str, end: str) -> list:
    """Download all hourly rows for one sensor between start and end (paginated)."""
    rows, page = [], 1
    while True:
        if (sensor_id == 16312587):  # OpenAQ bug: this sensor returns 500 on page > 1
            print(f"[sensor {sensor_id}] skipping page {page} due to OpenAQ bug")
            break
        results = api_get(session, f"sensors/{sensor_id}/hours", {
            "datetime_from": start,
            "datetime_to": end,
            "limit": PAGE_LIMIT,
            "page": page,
        })["results"]
        rows.extend(results)
        if len(results) < PAGE_LIMIT:
            return rows
        page += 1


def sensor_series(session: requests.Session, sensor_id: int, start: str, end: str,
                  cache: dict, cache_path: str = CACHE_PATH) -> pd.Series:
    """Hourly values for one sensor as a Series indexed by UTC hour (cached on disk)."""
    cache_key = f"{int(sensor_id)}|{start}|{end}"  # range is part of the key
    if (sensor_id == 16312587):  # OpenAQ bug: this sensor returns 500 on page > 1
        print(f"[sensor {sensor_id}] skipping due to OpenAQ bug")
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    if cache_key not in cache["hours"]:
        cache["hours"][cache_key] = fetch_sensor_hours(session, int(sensor_id), start, end)
        save_cache(cache, cache_path)
        print(f"[sensor {sensor_id}] {len(cache['hours'][cache_key])} rows")

    rows = cache["hours"][cache_key]
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    if not rows:
        return empty

    frame = pd.json_normalize(rows)
    if "value" not in frame.columns:
        return empty
    frame["hour"] = to_utc_hour(frame["period.datetimeFrom.utc"])
    frame = frame.dropna(subset=["value"])
    return frame.groupby("hour")["value"].mean()


# --------------------------------------------------------------------------- #
# Step 3: fill each hour from the nearest station that has a reading
# --------------------------------------------------------------------------- #
def fill_coordinate(needed_hours, candidates: list,
                    series_for: Callable[[dict], pd.Series]) -> pd.DataFrame:
    """Fill every needed hour from the nearest candidate that has a value.

    `candidates` must be nearest-first. `series_for(candidate)` returns that station's
    hourly Series (it is only called while hours are still missing, so stations further
    away are never downloaded unless needed). Returns one row per filled hour.
    """
    columns = ["hour", PARAMETER, *SOURCE_COLS]
    remaining = pd.DatetimeIndex(pd.unique(needed_hours))
    pieces = []

    for rank, cand in enumerate(candidates):
        if len(remaining) == 0:
            break
        hit = series_for(cand).reindex(remaining).dropna()
        if hit.empty:
            continue
        pieces.append(pd.DataFrame({
            "hour": hit.index,
            PARAMETER: hit.values,
            "location_id": cand["location_id"],
            "sensor_id": cand["sensor_id"],
            "station_distance_m": cand["distance_m"],
            "is_fallback": int(rank > 0),
        }))
        remaining = remaining.difference(hit.index)

    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True)[columns]


def print_coverage(df: pd.DataFrame) -> None:
    """Summarise how much PM2.5 was found, and how much came from fallback stations."""
    total = len(df)
    have = df[PARAMETER].notna()
    fallback = (df["is_fallback"] == 1).sum()
    print(f"\nPM2.5 found for {have.sum()}/{total} rows ({have.mean():.0%}); "
          f"{fallback} of those came from a fallback station")

    per_coord = df.groupby("coord_key")[PARAMETER].apply(lambda s: s.notna().mean())
    weak = per_coord[per_coord < 0.5]
    if len(weak):
        print(f"{len(weak)} coordinate(s) still have under 50% coverage, e.g.:")
        for key, share in weak.sort_values().head(5).items():
            print(f"  {key}: {share:.0%}")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_dataset(df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL,
                  tz: str = DEFAULT_TZ, cache_path: str = CACHE_PATH,
                  session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Add an hourly PM2.5 column plus the station that supplied each value.

    `df` needs 'latitude', 'longitude' and `timestamp_col`. Naive timestamps
    are treated as local time in `tz`. Added columns: pm25, location_id, sensor_id,
    station_distance_m, is_fallback (plus the helper columns coord_key and hour).
    """
    session = session or build_session()
    cache = load_cache(cache_path)

    # Re-running on an already-built frame must not create duplicate columns.
    df = add_coord_key(df.drop(columns=[c for c in [PARAMETER, *SOURCE_COLS] if c in df.columns]))
    df["hour"] = to_utc_hour(df[timestamp_col], tz)

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start, end = df["hour"].min().strftime(fmt), df["hour"].max().strftime(fmt)

    resolve_stations(session, df["coord_key"].unique(), cache, cache_path)

    filled = []
    for key, hours in df.groupby("coord_key")["hour"]:
        candidates = usable_stations(cache["stations"][key], start, end)
        if not candidates:
            print(f"{key}: no usable {PARAMETER} station within {SEARCH_RADIUS_M} m")
            continue
        result = fill_coordinate(
            hours, candidates,
            lambda c: sensor_series(session, c["sensor_id"], start, end, cache, cache_path),
        )
        filled.append(result.assign(coord_key=key))

    if filled:
        air = pd.concat(filled, ignore_index=True)
    else:
        air = pd.DataFrame(columns=["coord_key", "hour", PARAMETER, *SOURCE_COLS])

    merged = df.merge(air, on=["coord_key", "hour"], how="left")
    for col in ("location_id", "sensor_id", "is_fallback"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")
    merged[PARAMETER] = pd.to_numeric(merged[PARAMETER], errors="coerce")

    print_coverage(merged)
    return merged


def main() -> None:
    """Standalone run from a CSV (the pipeline normally goes through api.py)."""
    result = build_dataset(pd.read_csv(CCTV_PATH))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()