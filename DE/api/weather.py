"""Hourly weather from the Open-Meteo archive API for the CCTV locations.

Typical use (from api.py):
    import weather
    df = weather.add_weather(df, timestamp_col="hourly_timestamp")

Nothing runs at import time; the cached client is built inside build_client().
"""

import os
from typing import Optional

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_PATH = "data/cached/weather_cache"

TIMEZONE = "Asia/Bangkok"
TIMESTAMP_COL = "hourly_timestamp"
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "cloud_cover",
    "surface_pressure",
]
COORD_DECIMALS = 2   # ~1 km; Open-Meteo's grid is coarser than this anyway
BATCH_SIZE = 25      # locations per request


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def build_client(cache_path: str = CACHE_PATH) -> openmeteo_requests.Client:
    """Open-Meteo client with a permanent on-disk cache and retries (incl. 429)."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    cache_session = requests_cache.CachedSession(cache_path, expire_after=-1)
    retry_session = retry(
        cache_session,
        retries=5,
        backoff_factor=2,
        status_to_retry=(429, 500, 502, 503, 504),
    )
    return openmeteo_requests.Client(session=retry_session)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def parse_hourly(response) -> pd.DataFrame:
    """Turn one Open-Meteo response into a DataFrame with a naive local 'local_hour'."""
    hourly = response.Hourly()
    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    # Naive local time, so it matches the naive 'hourly_timestamp' from the images
    data = {"local_hour": dates.tz_convert(TIMEZONE).tz_localize(None)}

    # Same list is sent as the request's "hourly", so the order always matches
    for i, name in enumerate(HOURLY_VARS):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def get_weather_batch(client: openmeteo_requests.Client, url: str, coords: list,
                      start_date: str, end_date: str) -> pd.DataFrame:
    """One request for several (lat, lon) pairs; responses come back in the same order."""
    params = {
        "latitude": [lat for lat, _ in coords],
        "longitude": [lon for _, lon in coords],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "timezone": TIMEZONE,
    }
    responses = client.weather_api(url, params=params)
    if len(responses) != len(coords):
        raise RuntimeError(f"Sent {len(coords)} locations, got {len(responses)} responses")

    frames = []
    for (lat, lon), response in zip(coords, responses):
        frame = parse_hourly(response)
        # Use our own rounded coordinates as the key: the response's lat/lon is
        # snapped to the model grid and won't match the input.
        frame["lat_r"], frame["lon_r"] = lat, lon
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def unique_coords(df: pd.DataFrame, decimals: int = COORD_DECIMALS) -> list:
    """Unique rounded (lat, lon) pairs in the dataframe."""
    pairs = df[["latitude", "longitude"]].round(decimals).drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))


def fetch_all_weather_data(client: openmeteo_requests.Client, url: str, df: pd.DataFrame,
                           timestamp_col: str = TIMESTAMP_COL, batch_size: int = BATCH_SIZE,
                           decimals: int = COORD_DECIMALS) -> pd.DataFrame:
    """Fetch hourly weather for every unique location over the dataframe's date range.

    Returns one row per (location, hour), not one per image.
    """
    coords = unique_coords(df, decimals)
    timestamps = pd.to_datetime(df[timestamp_col])
    start_date = timestamps.min().strftime("%Y-%m-%d")
    end_date = timestamps.max().strftime("%Y-%m-%d")

    frames = []
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        frames.append(get_weather_batch(client, url, batch, start_date, end_date))
        print(f"[weather] {min(i + batch_size, len(coords))}/{len(coords)} locations")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Join and pipeline
# --------------------------------------------------------------------------- #
def merge_weather(df: pd.DataFrame, weather: pd.DataFrame,
                  timestamp_col: str = TIMESTAMP_COL,
                  decimals: int = COORD_DECIMALS) -> pd.DataFrame:
    """Left-join the weather onto every original row by location and local hour."""
    df = df.copy()
    df["lat_r"] = df["latitude"].round(decimals)
    df["lon_r"] = df["longitude"].round(decimals)
    df["local_hour"] = pd.to_datetime(df[timestamp_col]).dt.floor("h")

    merged = df.merge(weather, on=["lat_r", "lon_r", "local_hour"], how="left")
    return merged.drop(columns=["lat_r", "lon_r"])


def add_weather(df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL,
                url: str = BASE_URL, client: Optional[openmeteo_requests.Client] = None
                ) -> pd.DataFrame:
    """Add the weather columns (HOURLY_VARS) to the dataframe."""
    client = client or build_client()
    weather = fetch_all_weather_data(client, url, df, timestamp_col)
    return merge_weather(df, weather, timestamp_col)