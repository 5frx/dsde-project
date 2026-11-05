"""Loading and feature engineering for Component 2.

Two things happen here.

1. `load_dataset` reads whichever prepared file exists and reports what it
   actually contains, instead of assuming. The DE pipeline writes
   data/processed/data.csv; when that has not been built, the published
   Carcounts.csv (base + object detection) is used and the air-quality and
   weather blocks are simply reported as absent. Every downstream section
   checks the capability flags before it runs, so the analysis degrades to a
   smaller report rather than crashing.

2. The three analysis levels are built explicitly. The prepared data is one
   row per image, but almost nothing should be correlated at that level:

     image       61k rows   detector noise, ~11 min cadence
     camera-hour 11k rows   the level PM2.5 and weather actually vary at
     camera      100 rows   the level POI actually varies at

   Correlating a camera-constant POI count against 61k image rows is
   pseudo-replication: the coefficient is unchanged but the p-value is wrong by
   a factor of about 25. Each section therefore states its level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:                      # works both as `python Component2_DataSci/main.py` and `python -m Component2_DataSci.main`
    from . import config as C
except ImportError:
    import config as C


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _present(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


def describe_capabilities(df: pd.DataFrame) -> dict:
    """Which blocks of the prepared dataset are usable, and why or why not.

    A column that exists but is entirely null counts as absent: a merge that
    silently produced NaN is the failure mode this is here to catch.
    """
    def usable(cols):
        live = [c for c in _present(df, cols) if df[c].notna().any()]
        return live

    pm25_live = C.TARGET in df.columns and df[C.TARGET].notna().any()
    caps = {
        "vehicles": usable(C.VEHICLE_COLS),
        "weather": usable(C.WEATHER_COLS),
        "poi": usable([c for c in df.columns if c.startswith("poi_")]),
        "pm25": pm25_live,
        "pm25_coverage": float(df[C.TARGET].notna().mean()) if pm25_live else 0.0,
        "pm25_source": usable(C.PM25_SOURCE_COLS),
    }
    caps["has_target"] = caps["pm25"]
    return caps


def load_dataset(path_full=None, path_fallback=None,
                 path_camera_poi=None) -> tuple:
    """Return (df, capabilities, source_label).

    Prefers the full prepared dataset. Falls back to the object-detection
    output, attaching camera-level POI features if they have been generated.
    """
    path_full = C.PATH_FULL if path_full is None else path_full
    path_fallback = C.PATH_CARCOUNTS if path_fallback is None else path_fallback
    path_camera_poi = C.PATH_CAMERA_POI if path_camera_poi is None else path_camera_poi

    if path_full.exists():
        df = pd.read_csv(path_full)
        source = f"{path_full} (full DE pipeline output)"
    elif path_fallback.exists():
        df = pd.read_csv(path_fallback)
        source = (f"{path_fallback} (object detection only; air quality and "
                  f"weather not available)")
        if path_camera_poi.exists():
            poi = pd.read_csv(path_camera_poi)
            poi_cols = [c for c in poi.columns if c.startswith("poi_")]
            df = df.merge(poi[["camera_id"] + poi_cols], on="camera_id", how="left")
            source += " + camera-level POI features"
    else:
        raise FileNotFoundError(
            f"No prepared dataset found. Expected {path_full} (run DE/main.py) "
            f"or {path_fallback}.")

    df = add_time_features(df)
    df = add_vehicle_features(df)
    return df, describe_capabilities(df), source


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar and cyclical time features from the image timestamp.

    hour_sin/hour_cos exist so that hour-of-day can be used as a *linear*
    control in a partial correlation. Using the raw hour number would make
    06:00 and 21:00 maximally different, which is wrong for a daily cycle.
    """
    out = df.copy()
    ts = pd.to_datetime(out["raw_timestamp"])
    out["timestamp"] = ts
    out["hour"] = ts.dt.hour
    out["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    out["dayofweek"] = ts.dt.dayofweek          # Monday = 0
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["date"] = ts.dt.date.astype(str)
    out["day_index"] = (ts.dt.normalize() - ts.dt.normalize().min()).dt.days
    out["day_name"] = ts.dt.day_name()
    # 24-hour cycle, so the period is 24 even though only 06-21 is observed.
    out["hour_sin"] = np.sin(2 * np.pi * out["minute_of_day"] / (24 * 60))
    out["hour_cos"] = np.cos(2 * np.pi * out["minute_of_day"] / (24 * 60))
    return out


def add_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Totals, composition shares and a log transform for the vehicle counts.

    Shares are set to NaN, not 0, on empty frames: "no heavy vehicles out of
    zero vehicles" is a missing proportion, and filling it with 0 would pull
    every quiet frame towards "all light traffic".
    """
    out = df.copy()
    present = _present(out, C.VEHICLE_COLS)
    if not present:
        return out

    out["total_vehicles"] = out[present].sum(axis=1)
    heavy = _present(out, ["bus_count", "truck_count"])
    if heavy:
        out["heavy_vehicles"] = out[heavy].sum(axis=1)

    total = out["total_vehicles"].replace(0, np.nan)
    if "heavy_vehicles" in out:
        out["heavy_share"] = out["heavy_vehicles"] / total
    if "motorcycle_count" in out:
        out["motorcycle_share"] = out["motorcycle_count"] / total
    # Counts are right-skewed and zero-inflated; log1p is what a linear
    # control should see.
    out["log_total_vehicles"] = np.log1p(out["total_vehicles"])
    return out


# --------------------------------------------------------------------------- #
# Analysis levels
# --------------------------------------------------------------------------- #
def _agg_spec(df: pd.DataFrame) -> dict:
    """Mean for anything numeric that is not an identifier."""
    skip = {"camera_id", "location_id", "sensor_id", "day_index", "hour",
            "dayofweek", "is_weekend", "minute_of_day"}
    spec = {}
    for col in df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        spec[col] = "mean"
    return spec


def camera_hour_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per camera per hour: the level PM2.5 and weather vary at.

    Vehicle counts are averaged over the ~5 images in the hour, which is also
    the cheapest available reduction of detector noise.
    """
    spec = _agg_spec(df)
    keys = ["camera_id", "hourly_timestamp"]
    out = df.groupby(keys, as_index=False).agg({**spec, "image_name": "count"})
    out = out.rename(columns={"image_name": "n_images"})

    ts = pd.to_datetime(out["hourly_timestamp"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["date"] = ts.dt.date.astype(str)
    out["day_name"] = ts.dt.day_name()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def camera_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per camera: the level POI varies at.

    This is the only correct level for a POI correlation, and it caps the
    sample at the number of cameras however many images were collected.
    """
    spec = _agg_spec(df)
    out = df.groupby("camera_id", as_index=False).agg({**spec, "image_name": "count"})
    out = out.rename(columns={"image_name": "n_images"})

    peak = df[df["hour"].between(7, 9) | df["hour"].between(17, 19)]
    if len(peak):
        out = out.merge(
            peak.groupby("camera_id", as_index=False)["total_vehicles"]
                .mean().rename(columns={"total_vehicles": "peak_total_vehicles"}),
            on="camera_id", how="left")
    offpeak = df[df["hour"].between(11, 14)]
    if len(offpeak):
        out = out.merge(
            offpeak.groupby("camera_id", as_index=False)["total_vehicles"]
                   .mean().rename(columns={"total_vehicles": "offpeak_total_vehicles"}),
            on="camera_id", how="left")
    if {"peak_total_vehicles", "offpeak_total_vehicles"} <= set(out.columns):
        # How strongly a camera's traffic swings with the rush hour. A camera
        # with a flat profile is probably not watching a commuter route.
        out["peak_ratio"] = (out["peak_total_vehicles"]
                             / out["offpeak_total_vehicles"].replace(0, np.nan))
    return out


def attach_camera_info(df: pd.DataFrame, path=None) -> pd.DataFrame:
    """Add the human-readable camera location names, for reading the tables."""
    path = C.PATH_CCTV_INFO if path is None else path
    if not path.exists():
        return df
    info = pd.read_csv(path)
    keep = [c for c in ["camera_id", "location_en", "location_detail_en", "location"]
            if c in info.columns]
    return df.merge(info[keep].drop_duplicates("camera_id"), on="camera_id", how="left")


def poi_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("poi_")]


def add_lagged_traffic(df: pd.DataFrame, col: str = "total_vehicles",
                       lags=None) -> pd.DataFrame:
    """Lagged traffic per camera on the camera-hour frame.

    Reindexed onto a complete hourly grid first, so a gap in the data cannot
    silently turn a 3-hour-old value into "lag 1".
    """
    lags = C.PM25_LAGS if lags is None else lags
    out = df.copy()
    out["_ts"] = pd.to_datetime(out["hourly_timestamp"])
    out = out.sort_values(["camera_id", "_ts"])

    for lag in lags:
        if lag == 0:
            out[f"{col}_lag0"] = out[col]
            continue
        shifted = []
        for cam, grp in out.groupby("camera_id", sort=False):
            series = grp.set_index("_ts")[col]
            # asfreq inserts the missing hours as NaN; shift then means "exactly
            # `lag` hours earlier" rather than "`lag` rows earlier".
            full = series.asfreq("h") if series.index.is_unique else series
            shifted.append(full.shift(lag).reindex(grp["_ts"]).to_numpy())
        out[f"{col}_lag{lag}"] = np.concatenate(shifted)
    return out.drop(columns="_ts")
