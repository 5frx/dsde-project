"""Helpers for pulling OpenAQ v3 air-quality data for the CCTV locations.

Import and call `build_dataset()`, or run the file directly:
    python openaq_client.py

Nothing runs at import time: the session, API key check and cache are all
created inside functions.
"""

import json
import os
import time
from typing import Optional

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
CANDIDATE_STATIONS = 5   # nearby stations to scan for one with a PM2.5 sensor
PAGE_LIMIT = 1000
USER_AGENT = "dsde-project/1.0 (6838075421@student.chula.ac.th)"


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
# Step 1: coordinates -> station / sensor
# --------------------------------------------------------------------------- #
def add_coord_key(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Add a 'coord_key' column (rounded 'lat,lon') used to dedupe lookups."""
    df = df.copy()
    df["coord_key"] = (
        df["latitude"].round(decimals).astype(str) + "," +
        df["longitude"].round(decimals).astype(str)
    )
    return df


def pick_sensor_id(location: dict, parameter: str = PARAMETER) -> Optional[int]:
    """Return the id of the sensor measuring `parameter` at a location, if any."""
    for sensor in location.get("sensors", []):
        if sensor["parameter"]["name"] == parameter:
            return sensor["id"]
    return None


def find_station(session: requests.Session, coord_key: str,
                 radius: int = SEARCH_RADIUS_M, candidates: int = CANDIDATE_STATIONS,
                 parameter: str = PARAMETER) -> dict:
    """Find the nearest station near a coordinate that has a `parameter` sensor."""
    results = api_get(session, "locations", {
        "coordinates": coord_key,
        "radius": radius,
        "limit": candidates,
    })["results"]

    for location in results:
        sensor_id = pick_sensor_id(location, parameter)
        if sensor_id is not None:
            return {"location_id": location["id"], "sensor_id": sensor_id}
    return {"location_id": None, "sensor_id": None}


def resolve_stations(session: requests.Session, df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Look up each unique coordinate once and add 'location_id'/'sensor_id' columns."""
    stations = cache["stations"]
    unique_keys = df["coord_key"].unique()

    for i, key in enumerate(unique_keys, start=1):
        if key in stations:
            continue
        stations[key] = find_station(session, key)
        save_cache(cache)
        print(f"[stations {i}/{len(unique_keys)}] {key} -> {stations[key]}")

    df = df.copy()
    for col in ("location_id", "sensor_id"):
        values = df["coord_key"].map(lambda k, c=col: stations[k][c])
        df[col] = pd.to_numeric(values, errors="coerce").astype("Int64")
    return df


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


def fetch_all_sensors(session: requests.Session, sensor_ids, start: str, end: str,
                      cache: dict) -> pd.DataFrame:
    """Fetch hourly data for every sensor once and return one tidy DataFrame."""
    frames = []
    sensor_ids = [int(s) for s in sensor_ids]

    for i, sensor_id in enumerate(sensor_ids, start=1):
        cache_key = f"{sensor_id}|{start}|{end}"  # range is part of the key
        if cache_key not in cache["hours"]:
            cache["hours"][cache_key] = fetch_sensor_hours(session, sensor_id, start, end)
            save_cache(cache)
            print(f"[sensors {i}/{len(sensor_ids)}] {sensor_id}: "
                  f"{len(cache['hours'][cache_key])} rows")

        rows = cache["hours"][cache_key]
        if not rows:
            continue
        frame = pd.json_normalize(rows)
        frame["sensor_id"] = sensor_id
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["sensor_id", "hour", "value"])

    air = pd.concat(frames, ignore_index=True)
    air["hour"] = to_utc_hour(air["period.datetimeFrom.utc"])
    return air[["sensor_id", "hour", "value"]]


# --------------------------------------------------------------------------- #
# Step 3: join back onto every original row
# --------------------------------------------------------------------------- #
def merge_air_quality(df: pd.DataFrame, air: pd.DataFrame,
                      timestamp_col: str = TIMESTAMP_COL, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Left-join hourly air-quality values onto the original rows."""
    df = df.copy()
    df["hour"] = to_utc_hour(df[timestamp_col], tz)
    air = air.assign(sensor_id=air["sensor_id"].astype("Int64"))
    air = air.rename(columns={"value": PARAMETER})
    return df.merge(air, on=["sensor_id", "hour"], how="left")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_dataset(df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL,
                  tz: str = DEFAULT_TZ, cache_path: str = CACHE_PATH,
                  session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Add location_id, sensor_id and an hourly PM2.5 column to a dataframe.

    `df` needs 'latitude', 'longitude' and `timestamp_col`. Naive timestamps
    are treated as local time in `tz`.
    """
    session = session or build_session()
    cache = load_cache(cache_path)

    df = add_coord_key(df)
    df = resolve_stations(session, df, cache)

    hours = to_utc_hour(df[timestamp_col], tz)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start, end = hours.min().strftime(fmt), hours.max().strftime(fmt)

    air = fetch_all_sensors(session, df["sensor_id"].dropna().unique(), start, end, cache)
    return merge_air_quality(df, air, timestamp_col, tz)


def main() -> None:
    """Standalone run from a CSV (the pipeline normally goes through api.py)."""
    result = build_dataset(pd.read_csv(CCTV_PATH))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()