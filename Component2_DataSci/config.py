"""Paths, column groups and analysis constants for Component 2 (Data Science).

Everything the analysis needs to know about the shape of the prepared dataset
lives here, so a change in the DE pipeline is a one-file change.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

# Full prepared dataset from DE/main.py (base + OD + air quality + weather + POI).
PATH_FULL = DATA_DIR / "data.csv"
# Fallback: object-detection output only (base + vehicle counts), published as a
# GitHub release asset. Used when data.csv has not been built yet.
PATH_CARCOUNTS = DATA_DIR / "Carcounts.csv"
# Camera-level POI features, regenerated offline from the POI index.
PATH_CAMERA_POI = DATA_DIR / "camera_poi.csv"
PATH_CCTV_INFO = RAW_DIR / "cctv_info.csv"

OUT_DIR = Path("Component2_DataSci/output")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "component2_report.md"

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
KEY_COLS = ["image_name", "camera_id", "latitude", "longitude",
            "raw_timestamp", "hourly_timestamp"]

VEHICLE_COLS = ["car_count", "motorcycle_count", "bus_count", "truck_count"]

# Target produced by DE/api/airquality.py
TARGET = "pm25"
# Provenance columns that come with the target. station_distance_m and is_fallback
# are data-quality covariates, not predictors: a model must never train on them.
PM25_SOURCE_COLS = ["location_id", "sensor_id", "station_distance_m", "is_fallback"]

# Produced by DE/api/weather.py (HOURLY_VARS, same order)
WEATHER_COLS = ["temperature_2m", "precipitation", "wind_speed_10m",
                "wind_direction_10m", "relative_humidity_2m", "cloud_cover",
                "surface_pressure"]

# Engineered in features.py
TIME_COLS = ["hour", "dayofweek", "is_weekend", "date", "day_index",
             "hour_sin", "hour_cos"]

# Vehicle aggregates engineered in features.py
DERIVED_VEHICLE_COLS = ["total_vehicles", "heavy_vehicles", "heavy_share",
                        "motorcycle_share"]

# --------------------------------------------------------------------------- #
# Analysis constants
# --------------------------------------------------------------------------- #
LOCAL_TZ = "Asia/Bangkok"
ALPHA = 0.05              # significance level before multiplicity correction
FDR_Q = 0.05              # Benjamini-Hochberg false discovery rate
MIN_PAIRS = 30            # fewest complete pairs a correlation is reported for
MORAN_K = 5               # k nearest neighbours for the spatial weight matrix
BOOTSTRAP_N = 2000        # resamples for cluster bootstrap CIs
RANDOM_SEED = 20261103

# Lags (in hours) tested when relating traffic to PM2.5. Emissions need time to
# accumulate, so the strongest association is not necessarily at lag 0.
PM25_LAGS = [0, 1, 2, 3, 6]
