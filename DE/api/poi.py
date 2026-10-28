"""Points of interest (OpenStreetMap, via the Overpass API) around each CCTV location.

POIs don't change with time, so this makes ONE request per unique camera
location (not one per image) and turns the nearby OSM elements into numeric
features: counts within several radii, plus the distance to the nearest few
key features.

Typical use (from api.py):
    import poi
    df = poi.add_poi(df)

Nothing runs at import time; the session and cache are created inside functions.
"""

import hashlib
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_PATH = "data/cache_poi.json"
USER_AGENT = "dsde-project/1.0 (6838075421@student.chula.ac.th)"

RADII_M = (100, 300, 500)   # one request at the largest radius; smaller ones are derived
COORD_DECIMALS = 4          # ~11 m; cameras closer than this share one lookup
QUERY_TIMEOUT_S = 60        # server-side query timeout
PAUSE_S = 1.0               # pause after each live request (public server is shared)

# --------------------------------------------------------------------------- #
# What to extract.  category -> {osm_key: set of values, or None for "any value"}
# Edit this dict to add or remove features; the cache and the query follow it.
# --------------------------------------------------------------------------- #
CATEGORIES = {
    # activity / people attractors
    "food":            {"amenity": {"restaurant", "cafe", "fast_food", "food_court"}},
    "market":          {"amenity": {"marketplace"}},
    "shop":            {"shop": None},
    "office":          {"office": None},
    "education":       {"amenity": {"school", "university", "college", "kindergarten"}},
    "health":          {"amenity": {"hospital", "clinic"}},
    "worship":         {"amenity": {"place_of_worship"}},
    "tourism":         {"tourism": {"hotel", "hostel", "attraction", "museum"}},
    # traffic generators / road furniture
    "parking":         {"amenity": {"parking"}},
    "fuel":            {"amenity": {"fuel"}},
    "bus_stop":        {"highway": {"bus_stop"}},
    "rail_station":    {"railway": {"station", "subway_entrance", "halt"}},
    "traffic_signals": {"highway": {"traffic_signals"}},
    "crossing":        {"highway": {"crossing"}},
    # land use
    "green":           {"leisure": {"park", "garden", "playground"}},
    "industrial":      {"landuse": {"industrial"}},
    "construction":    {"landuse": {"construction"}},
}

# Road classes are counted from ways (road segments), as road_<class> categories.
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary",
                "residential", "service")

# Categories that also get a "distance to nearest" feature.
NEAREST_CATS = ("rail_station", "bus_stop", "traffic_signals", "green")


# --------------------------------------------------------------------------- #
# Query building and classification (pure functions, no network)
# --------------------------------------------------------------------------- #
def all_categories() -> list:
    return list(CATEGORIES) + [f"road_{c}" for c in ROAD_CLASSES]


def schema_id() -> str:
    """Short hash of the category definition; part of the cache key."""
    spec = {
        "categories": {c: {k: (sorted(v) if v else None) for k, v in f.items()}
                       for c, f in CATEGORIES.items()},
        "roads": list(ROAD_CLASSES),
    }
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]


def _tag_filter(key: str, values: Optional[set]) -> str:
    if values is None:
        return f'["{key}"]'
    return f'["{key}"~"^({"|".join(sorted(values))})$"]'


def _values_by_key() -> dict:
    """Merge all categories into one value set per OSM key (fewer query statements)."""
    merged = {}
    for filters in CATEGORIES.values():
        for key, values in filters.items():
            if values is None:
                merged[key] = None
            elif key not in merged:
                merged[key] = set(values)
            elif merged[key] is not None:
                merged[key] |= values
    return merged


def build_query(lat: float, lon: float, radius: int, timeout: int = QUERY_TIMEOUT_S) -> str:
    """Overpass QL that returns every relevant element within `radius` metres."""
    around = f"(around:{radius},{float(lat)},{float(lon)})"
    statements = [f"nwr{around}{_tag_filter(k, v)};" for k, v in _values_by_key().items()]
    roads = "|".join(ROAD_CLASSES)
    statements.append(f'way{around}["highway"~"^({roads})$"];')
    body = "\n  ".join(statements)
    return f"[out:json][timeout:{timeout}];\n(\n  {body}\n);\nout center tags;"


def classify(element: dict) -> set:
    """Which of our categories does this OSM element belong to?"""
    tags = element.get("tags", {})
    cats = set()
    for cat, filters in CATEGORIES.items():
        for key, values in filters.items():
            if key in tags and (values is None or tags[key] in values):
                cats.add(cat)
                break
    if element.get("type") == "way" and tags.get("highway") in ROAD_CLASSES:
        cats.add(f"road_{tags['highway']}")
    return cats


def parse_elements(elements: list) -> list:
    """Compact the raw Overpass elements to [{'cats', 'lat', 'lon'}] (small cache)."""
    points = []
    for el in elements:
        cats = classify(el)
        if not cats:
            continue
        center = el.get("center", {})
        lat, lon = el.get("lat", center.get("lat")), el.get("lon", center.get("lon"))
        if lat is None or lon is None:
            continue
        points.append({"cats": sorted(cats), "lat": lat, "lon": lon})
    return points


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (works on numpy arrays)."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def build_features(points: list, lat: float, lon: float, radii=RADII_M) -> dict:
    """Counts within each radius (+ nearest distance for NEAREST_CATS) for one location."""
    if points:
        dist = haversine_m(lat, lon,
                           np.array([p["lat"] for p in points]),
                           np.array([p["lon"] for p in points]))
    else:
        dist = np.array([])

    row = {}
    for cat in all_categories():
        mask = np.array([cat in p["cats"] for p in points], dtype=bool)
        d = dist[mask]
        for r in radii:
            row[f"poi_{cat}_{r}m"] = int((d <= r).sum())
        if cat in NEAREST_CATS:
            row[f"poi_nearest_{cat}_m"] = float(d.min()) if d.size else np.nan
    return row


# --------------------------------------------------------------------------- #
# Session, HTTP and cache
# --------------------------------------------------------------------------- #
def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})  # Overpass asks clients to identify
    return session


def overpass_query(session: requests.Session, query: str, url: str = OVERPASS_URL,
                   max_retries: int = 6) -> dict:
    """POST a query; back off on 429/504 (no free slot / server busy) and runtime errors."""
    for attempt in range(max_retries):
        wait = min(5 * 2 ** attempt, 120)
        try:
            response = session.post(url, data={"data": query}, timeout=QUERY_TIMEOUT_S + 60)
        except requests.RequestException as exc:
            print(f"Request failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code in (429, 502, 503, 504):
            print(f"Overpass busy (HTTP {response.status_code}); retrying in {wait}s")
            time.sleep(wait)
            continue

        response.raise_for_status()
        data = response.json()
        if "runtime error" in data.get("remark", ""):   # e.g. query timed out, partial result
            print(f"Overpass remark: {data['remark']}; retrying in {wait}s")
            time.sleep(wait)
            continue
        return data

    raise RuntimeError(f"Gave up on Overpass query after {max_retries} attempts")


def load_cache(path: str = CACHE_PATH) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"poi": {}}


def save_cache(cache: dict, path: str = CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Fetching and pipeline
# --------------------------------------------------------------------------- #
def unique_coords(df: pd.DataFrame, decimals: int = COORD_DECIMALS) -> list:
    pairs = df[["latitude", "longitude"]].round(decimals).drop_duplicates()
    return [(float(lat), float(lon)) for lat, lon in pairs.itertuples(index=False, name=None)]


def fetch_points(session: requests.Session, lat: float, lon: float, radius: int,
                 url: str = OVERPASS_URL) -> list:
    data = overpass_query(session, build_query(lat, lon, radius), url)
    return parse_elements(data.get("elements", []))


def fetch_all_poi(session: requests.Session, coords: list, cache: dict,
                  radii=RADII_M, url: str = OVERPASS_URL,
                  cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """One feature row per unique coordinate (one live request each, then cached)."""
    radius = max(radii)
    sid = schema_id()
    store = cache["poi"]
    rows = []

    for i, (lat, lon) in enumerate(coords, start=1):
        key = f"{sid}|{radius}|{lat},{lon}"
        if key not in store:
            store[key] = fetch_points(session, lat, lon, radius, url)
            save_cache(cache, cache_path)
            print(f"[poi {i}/{len(coords)}] {lat},{lon}: {len(store[key])} elements")
            time.sleep(PAUSE_S)

        row = build_features(store[key], lat, lon, radii)
        row["lat_r"], row["lon_r"] = lat, lon
        rows.append(row)
    return pd.DataFrame(rows)


def merge_poi(df: pd.DataFrame, poi_df: pd.DataFrame,
              decimals: int = COORD_DECIMALS) -> pd.DataFrame:
    """Left-join the POI features onto every original row by rounded coordinate."""
    df = df.copy()
    df["lat_r"] = df["latitude"].round(decimals)
    df["lon_r"] = df["longitude"].round(decimals)
    merged = df.merge(poi_df, on=["lat_r", "lon_r"], how="left")
    return merged.drop(columns=["lat_r", "lon_r"])


def add_poi(df: pd.DataFrame, radii=RADII_M, url: str = OVERPASS_URL,
            cache_path: str = CACHE_PATH,
            session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Add the poi_* feature columns to the dataframe (needs latitude/longitude)."""
    session = session or build_session()
    cache = load_cache(cache_path)
    poi_df = fetch_all_poi(session, unique_coords(df), cache, radii, url, cache_path)
    return merge_poi(df, poi_df)