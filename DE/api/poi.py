"""Points of interest (OpenStreetMap, via the Overpass API) around each CCTV location.

Why this version is much faster than one request per camera
-----------------------------------------------------------
POIs don't change with time, and cameras are densely clustered, so instead of
one request per camera we:

1. de-duplicate coordinates (rounded to COORD_DECIMALS),
2. group them into small map TILES (TILE_DEG) and download each occupied tile
   ONCE (tile bbox + a margin of max(radii), so cameras near an edge still see
   every neighbour). Hundreds of cameras share one request,
3. spread the tile requests over several Overpass MIRRORS in parallel; a mirror
   that is busy/failing is put on cool-down and the request moves to another,
4. split a tile into 4 quadrants automatically if the server says it is too
   heavy (timeout / out of memory),
5. compute counts / nearest distances locally with numpy (no network).

Every finished tile is cached on disk (one small file each), so the run is
resumable and re-running costs nothing.

Typical use (from api.py):
    import poi
    print(poi.summary(df))      # unique locations and tiles still to download
    df = poi.add_poi(df)

Nothing runs at import time.
"""

import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Public Overpass instances. Mirrors come and go: check https://wiki.openstreetmap.org/wiki/Overpass_API
# for the current list. Dead ones are cooled down automatically, so extras are harmless.
MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
CACHE_DIR = "data/poi_tiles"
USER_AGENT = "dsde-project/1.0 (6838075421@student.chula.ac.th)"

RADII_M = (100, 300, 500)   # tiles carry a margin of the largest radius; smaller ones are derived
COORD_DECIMALS = 4          # ~11 m; cameras closer than this share one lookup
TILE_DEG = 0.02             # ~2.2 km tile edge. Smaller = lighter queries, more requests
MAX_SPLIT_DEPTH = 2         # a heavy tile may be split into 4 up to this many times
PER_MIRROR = 1              # concurrent requests per mirror (public servers have few slots)
QUERY_TIMEOUT_S = 90        # server-side query timeout
PAUSE_S = 1.0               # per-mirror pause after each successful request
MAX_COOLDOWN_S = 600

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

# Categories that also get a "distance to nearest" feature (capped at max(radii)).
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


def build_query(south: float, west: float, north: float, east: float,
                timeout: int = QUERY_TIMEOUT_S) -> str:
    """Overpass QL that returns every relevant element inside the bounding box."""
    bbox = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    statements = [f"nwr{bbox}{_tag_filter(k, v)};" for k, v in _values_by_key().items()]
    roads = "|".join(ROAD_CLASSES)
    statements.append(f'way{bbox}["highway"~"^({roads})$"];')
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
    """Compact raw Overpass elements to [{'id', 'cats', 'lat', 'lon'}] (small cache)."""
    points = []
    for el in elements:
        cats = classify(el)
        if not cats:
            continue
        center = el.get("center", {})
        lat, lon = el.get("lat", center.get("lat")), el.get("lon", center.get("lon"))
        if lat is None or lon is None:
            continue
        points.append({"id": f"{el.get('type')}/{el.get('id')}",
                       "cats": sorted(cats), "lat": lat, "lon": lon})
    return points


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (works on numpy arrays)."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Local feature computation (vectorised; one tile's points are prepared once)
# --------------------------------------------------------------------------- #
def tile_arrays(points: list) -> tuple:
    """(lat[N], lon[N], membership[N, n_categories]) for one tile's points."""
    cats = all_categories()
    index = {c: i for i, c in enumerate(cats)}
    lat = np.array([p["lat"] for p in points], dtype=float)
    lon = np.array([p["lon"] for p in points], dtype=float)
    member = np.zeros((len(points), len(cats)), dtype=bool)
    for j, p in enumerate(points):
        for c in p["cats"]:
            member[j, index[c]] = True
    return lat, lon, member


def build_features(arrays: tuple, lat: float, lon: float, radii=RADII_M) -> dict:
    """Counts within each radius (+ nearest distance for NEAREST_CATS) for one location."""
    plat, plon, member = arrays
    cats = all_categories()
    index = {c: i for i, c in enumerate(cats)}
    dist = haversine_m(lat, lon, plat, plon) if len(plat) else np.empty(0)

    row = {}
    for r in radii:
        counts = member[dist <= r].sum(axis=0)
        for cat, n in zip(cats, counts):
            row[f"poi_{cat}_{r}m"] = int(n)
    limit = max(radii)
    for cat in NEAREST_CATS:
        d = dist[member[:, index[cat]]]
        row[f"poi_nearest_{cat}_m"] = float(d.min()) if d.size and d.min() <= limit else np.nan
    return row


# --------------------------------------------------------------------------- #
# Mirror pool and HTTP
# --------------------------------------------------------------------------- #
class QueryTooHeavy(Exception):
    """The server could not finish the query (timeout / memory): split the area."""


class MirrorPool:
    """Hands out mirrors that are free and not cooling down; penalises failing ones."""

    def __init__(self, urls, per_mirror: int = PER_MIRROR):
        self.per_mirror = per_mirror
        self._cv = threading.Condition()
        self._state = {u: {"busy": 0, "until": 0.0, "fails": 0, "last": 0.0} for u in urls}

    def acquire(self) -> str:
        with self._cv:
            while True:
                now = time.time()
                ready = [u for u, s in self._state.items()
                         if s["busy"] < self.per_mirror and s["until"] <= now]
                if ready:
                    url = min(ready, key=lambda u: self._state[u]["last"])  # least recently used
                    self._state[url]["busy"] += 1
                    self._state[url]["last"] = now
                    return url
                waits = [s["until"] - now for s in self._state.values() if s["until"] > now]
                self._cv.wait(timeout=max(0.1, min(waits)) if waits else 1.0)

    def release(self, url: str, ok: bool, retry_after: float = 0.0) -> None:
        with self._cv:
            s = self._state[url]
            s["busy"] -= 1
            if ok:
                s["fails"] = 0
                s["until"] = time.time() + PAUSE_S
            else:
                s["fails"] += 1
                backoff = min(10 * 2 ** (s["fails"] - 1), MAX_COOLDOWN_S)
                s["until"] = time.time() + max(backoff, retry_after)
            self._cv.notify_all()


def overpass_query(pool: MirrorPool, query: str, max_attempts: int = 12) -> dict:
    """POST a query to whichever mirror is free; on trouble, cool it down and use another."""
    headers = {"User-Agent": USER_AGENT}   # Overpass asks clients to identify themselves
    last = "no attempt"
    for _ in range(max_attempts):
        url = pool.acquire()
        ok, retry_after = False, 0.0
        try:
            resp = requests.post(url, data={"data": query}, headers=headers,
                                 timeout=QUERY_TIMEOUT_S + 60)
            if resp.status_code == 400:
                raise ValueError(f"Overpass rejected the query: {resp.text[:300]}")
            if resp.status_code in (429, 502, 503, 504):
                last = f"{url}: HTTP {resp.status_code}"
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                continue
            resp.raise_for_status()
            data = resp.json()
            ok = True                     # the mirror itself is healthy from here on
            if "runtime error" in data.get("remark", ""):
                raise QueryTooHeavy(data["remark"])
            return data
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last = f"{url}: {exc}"
        finally:
            pool.release(url, ok, retry_after)
    raise RuntimeError(f"Gave up after {max_attempts} attempts (last: {last})")


# --------------------------------------------------------------------------- #
# Tiling, tile cache and fetching
# --------------------------------------------------------------------------- #
def unique_coords(df: pd.DataFrame, decimals: int = COORD_DECIMALS) -> list:
    pairs = df[["latitude", "longitude"]].dropna().round(decimals).drop_duplicates()
    return [(float(lat), float(lon)) for lat, lon in pairs.itertuples(index=False, name=None)]


def tile_of(lat: float, lon: float, tile_deg: float = TILE_DEG) -> tuple:
    return int(math.floor(lon / tile_deg)), int(math.floor(lat / tile_deg))


def group_by_tile(coords: list, tile_deg: float = TILE_DEG) -> dict:
    tiles = {}
    for lat, lon in coords:
        tiles.setdefault(tile_of(lat, lon, tile_deg), []).append((lat, lon))
    return tiles


def tile_bounds(ix: int, iy: int, tile_deg: float, margin_m: float) -> tuple:
    """(south, west, north, east) of the tile, grown by margin_m on every side."""
    lat_c = (iy + 0.5) * tile_deg
    mlat = margin_m * 1.05 / 111_320
    mlon = margin_m * 1.05 / (111_320 * math.cos(math.radians(lat_c)))
    return (iy * tile_deg - mlat, ix * tile_deg - mlon,
            (iy + 1) * tile_deg + mlat, (ix + 1) * tile_deg + mlon)


def tile_key(ix: int, iy: int, tile_deg: float, margin_m: float) -> str:
    return f"{schema_id()}_t{tile_deg}_m{int(margin_m)}_{ix}_{iy}"


def _tile_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


def load_tile(cache_dir: str, key: str) -> Optional[list]:
    path = _tile_path(cache_dir, key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_tile(cache_dir: str, key: str, points: list) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _tile_path(cache_dir, key)
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(points, f)
    os.replace(tmp, path)


def fetch_bbox(pool: MirrorPool, south: float, west: float, north: float, east: float,
               depth: int = 0) -> list:
    """Fetch one box; if the server finds it too heavy, split into 4 and merge (by element id)."""
    try:
        data = overpass_query(pool, build_query(south, west, north, east))
        return parse_elements(data.get("elements", []))
    except QueryTooHeavy:
        if depth >= MAX_SPLIT_DEPTH:
            raise
        mid_lat, mid_lon = (south + north) / 2, (west + east) / 2
        merged = {}
        for box in ((south, west, mid_lat, mid_lon), (south, mid_lon, mid_lat, east),
                    (mid_lat, west, north, mid_lon), (mid_lat, mid_lon, north, east)):
            for p in fetch_bbox(pool, *box, depth=depth + 1):
                merged[p["id"]] = p
        return list(merged.values())


def fetch_tiles(pool: MirrorPool, tiles: dict, radii=RADII_M, tile_deg: float = TILE_DEG,
                cache_dir: str = CACHE_DIR) -> dict:
    """Make sure every occupied tile is cached. Returns {tile: points}; raises if any failed."""
    margin = max(radii)
    result, todo = {}, []
    for tile in tiles:
        key = tile_key(*tile, tile_deg, margin)
        cached = load_tile(cache_dir, key)
        if cached is not None:
            result[tile] = cached
        else:
            todo.append((tile, key))
    print(f"[poi] {len(tiles)} tiles: {len(result)} cached, {len(todo)} to download")
    if not todo:
        return result

    def work(tile, key):
        points = fetch_bbox(pool, *tile_bounds(*tile, tile_deg, margin))
        save_tile(cache_dir, key, points)
        return tile, points

    failed, done = [], 0
    workers = max(1, len(pool._state) * pool.per_mirror)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, t, k): t for t, k in todo}
        for fut in as_completed(futures):
            tile = futures[fut]
            try:
                _, points = fut.result()
                result[tile] = points
                done += 1
                print(f"[poi {done}/{len(todo)}] tile {tile}: {len(points)} elements")
            except Exception as exc:   # keep going; finished tiles are already cached
                failed.append(tile)
                print(f"[poi] tile {tile} failed: {exc}")
    if failed:
        raise RuntimeError(f"{len(failed)} tile(s) failed; re-run to resume from cache: {failed[:5]}...")
    return result


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def summary(df: pd.DataFrame, radii=RADII_M, tile_deg: float = TILE_DEG,
            cache_dir: str = CACHE_DIR) -> dict:
    """How much work is left: unique locations, occupied tiles, tiles not yet cached."""
    coords = unique_coords(df)
    tiles = group_by_tile(coords, tile_deg)
    margin = max(radii)
    missing = sum(load_tile(cache_dir, tile_key(*t, tile_deg, margin)) is None for t in tiles)
    return {"rows": len(df), "unique_locations": len(coords),
            "tiles": len(tiles), "tiles_to_download": missing}


def compute_poi(tiles: dict, tile_points: dict, radii=RADII_M) -> pd.DataFrame:
    """One feature row per unique coordinate, computed locally from the tile data."""
    rows = []
    for tile, coords in tiles.items():
        arrays = tile_arrays(tile_points[tile])
        for lat, lon in coords:
            row = build_features(arrays, lat, lon, radii)
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


def add_poi(df: pd.DataFrame, radii=RADII_M, mirrors=MIRRORS, cache_dir: str = CACHE_DIR,
            tile_deg: float = TILE_DEG, per_mirror: int = PER_MIRROR) -> pd.DataFrame:
    """Add the poi_* feature columns to the dataframe (needs latitude/longitude)."""
    coords = unique_coords(df)
    tiles = group_by_tile(coords, tile_deg)
    pool = MirrorPool(mirrors, per_mirror)
    tile_points = fetch_tiles(pool, tiles, radii, tile_deg, cache_dir)
    return merge_poi(df, compute_poi(tiles, tile_points, radii))