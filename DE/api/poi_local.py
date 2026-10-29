"""Offline POI features from an OpenStreetMap extract (no Overpass, no rate limits).

Two steps:

1. ONE-TIME build (a few minutes): stream a Geofabrik .osm.pbf, keep only the
   elements poi.CATEGORIES cares about, and save a compact index
   (lat, lon, category bitmask) to data/poi_index.parquet.

       python poi_local.py build thailand-latest.osm.pbf

   Download the extract from https://download.geofabrik.de/asia/thailand.html
   (a few hundred MB; refreshed daily, rebuild whenever you want fresher data).

2. QUERY (milliseconds, in memory): a KD-tree over the index answers
   "what is within 100/300/500 m of this point" for any coordinate, so the API
   can serve brand-new locations live, not just the ones seen at training time.

       import poi_local
       index = poi_local.PoiIndex()            # load once at startup
       index.features(13.7563, 100.5018)       # -> dict of poi_* features
       df = poi_local.add_poi(df, index)       # whole dataframe

Features have exactly the same names and meaning as poi.py (it is reused:
categories, classification, haversine, counting), so models trained on the
Overpass version keep working.

Known difference from Overpass: multipolygon RELATIONS (a few large parks,
campuses, industrial zones mapped as relations) are skipped; nodes and ways,
which cover almost everything, are kept.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from . import poi
from . import poi_select

INDEX_PATH = "data/raw/poi_index.parquet"
# Folder-style URL that serves poi_index.parquet and poi_index.parquet.meta.json, e.g. a GitHub
# release: https://github.com/<user>/<repo>/releases/download/<tag>
# Override without editing code:  export POI_INDEX_URL=...
INDEX_URL = "https://github.com/5frx/dsde-project/releases/download/raw_data/"
EARTH_R_M = 6_371_000.0


def _meta_path(path: str) -> str:
    return f"{path}.meta.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Seamless download: call ensure_index() (PoiIndex does it for you) and the index
# is fetched on first use, then reused from disk.
# --------------------------------------------------------------------------- #
def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = f"{dest}.part"                       # atomic: a half-finished download is never used
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        done, next_report = 0, 0.1
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    print(f"[poi_local] downloading {os.path.basename(dest)}: {done / total:.0%}")
                    next_report += 0.1
    os.replace(tmp, dest)


def ensure_index(path: str = INDEX_PATH, url: str = "") -> str:
    """Return the index path, downloading it first if it isn't on disk yet."""
    url = (url or INDEX_URL).rstrip("/")
    if os.path.exists(path) and os.path.exists(_meta_path(path)):
        return path
    if not url:
        raise FileNotFoundError(
            f"{path} not found and no download URL set. Either set POI_INDEX_URL "
            "(or INDEX_URL in poi_local.py), or build it: python poi_local.py build <extract.osm.pbf>")
    name = os.path.basename(path)
    print(f"[poi_local] {path} missing; downloading from {url}")
    _download(f"{url}/{name}.meta.json", _meta_path(path))
    _download(f"{url}/{name}", path)
    with open(_meta_path(path), encoding="utf-8") as f:
        expected = json.load(f).get("sha256")
    if expected and _sha256(path) != expected:
        os.remove(path)
        raise RuntimeError("Downloaded index failed its checksum; it was deleted, try again")
    print("[poi_local] index ready")
    return path


# --------------------------------------------------------------------------- #
# 1. Build the index from a .osm.pbf
# --------------------------------------------------------------------------- #
def build_index(pbf_path: str, out_path: str = INDEX_PATH) -> int:
    """Stream the extract, keep relevant nodes/ways, write the parquet index."""
    import osmium  # only needed for building

    cats = poi.all_categories()
    bit = {c: 1 << i for i, c in enumerate(cats)}
    keys = list(poi._values_by_key()) + ["highway"]

    proc = (osmium.FileProcessor(pbf_path)
            .with_locations()                       # resolve way node coordinates
            .with_filter(osmium.filter.KeyFilter(*keys)))

    lats, lons, masks = [], [], []
    t0, n_seen = time.time(), 0
    for obj in proc:
        if isinstance(obj, osmium.osm.Node):
            kind = "node"
            if not obj.location.valid():
                continue
            lat, lon = obj.location.lat, obj.location.lon
        elif isinstance(obj, osmium.osm.Way):
            kind = "way"
            try:
                pts = [(n.lat, n.lon) for n in obj.nodes if n.location.valid()]
            except osmium.InvalidLocationError:
                continue
            if not pts:
                continue
            la, lo = zip(*pts)                       # bbox centre, like Overpass `out center`
            lat, lon = (min(la) + max(la)) / 2, (min(lo) + max(lo)) / 2
        else:
            continue                                  # relations skipped

        found = poi.classify({"type": kind, "tags": {t.k: t.v for t in obj.tags}})
        if not found:
            continue
        lats.append(lat)
        lons.append(lon)
        masks.append(sum(bit[c] for c in found))
        n_seen += 1
        if n_seen % 500_000 == 0:
            print(f"[index] kept {n_seen:,} elements ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame({"lat": np.array(lats), "lon": np.array(lons),
                  "mask": np.array(masks, dtype=np.uint32)}).to_parquet(out_path, index=False)
    with open(_meta_path(out_path), "w", encoding="utf-8") as f:
        json.dump({"categories": cats, "source": os.path.basename(pbf_path),
                   "built": time.strftime("%Y-%m-%d %H:%M:%S"), "elements": n_seen,
                   "sha256": _sha256(out_path)}, f)
    print(f"[index] wrote {n_seen:,} elements to {out_path} in {time.time() - t0:.0f}s")
    return n_seen


# --------------------------------------------------------------------------- #
# 2. Query the index
# --------------------------------------------------------------------------- #
def _xyz(lat, lon) -> np.ndarray:
    """Unit-sphere coordinates, so KD-tree distance is exact great-circle (via chord)."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


class PoiIndex:
    def __init__(self, path: str = INDEX_PATH, url: str = ""):
        path = ensure_index(path, url)       # downloads on first run if needed
        with open(_meta_path(path), encoding="utf-8") as f:
            meta = json.load(f)
        if meta["categories"] != poi.all_categories():
            raise RuntimeError("poi.CATEGORIES changed since the index was built; rebuild the index")
        self.meta = meta
        table = pd.read_parquet(path)
        self.lat = table["lat"].to_numpy()
        self.lon = table["lon"].to_numpy()
        self.mask = table["mask"].to_numpy()
        self._bits = np.arange(len(meta["categories"]), dtype=np.uint32)
        self._tree = cKDTree(_xyz(self.lat, self.lon))

    def _row(self, idx, lat: float, lon: float, radii) -> dict:
        idx = np.asarray(idx, dtype=int)
        member = ((self.mask[idx, None] >> self._bits) & 1).astype(bool)
        return poi.build_features((self.lat[idx], self.lon[idx], member), lat, lon, radii)

    def features(self, lat: float, lon: float, radii=poi.RADII_M) -> dict:
        """poi_* features for one location (same keys as poi.py)."""
        chord = 2 * np.sin(max(radii) / (2 * EARTH_R_M))
        idx = self._tree.query_ball_point(_xyz([lat], [lon])[0], chord)
        return self._row(idx, lat, lon, radii)

    def features_many(self, coords: list, radii=poi.RADII_M) -> pd.DataFrame:
        """One row per (lat, lon), with lat_r/lon_r keys for merging."""
        chord = 2 * np.sin(max(radii) / (2 * EARTH_R_M))
        hits = self._tree.query_ball_point(_xyz([c[0] for c in coords], [c[1] for c in coords]),
                                           chord, workers=-1)
        rows = []
        for (lat, lon), idx in zip(coords, hits):
            row = self._row(idx, lat, lon, radii)
            row["lat_r"], row["lon_r"] = lat, lon
            rows.append(row)
        return pd.DataFrame(rows)


def add_poi(df: pd.DataFrame, index: "PoiIndex | None" = None, radii=poi.RADII_M,
            index_path: str = INDEX_PATH) -> pd.DataFrame:
    """Drop-in replacement for poi.add_poi using the local index."""
    index = index or PoiIndex(index_path)
    output =  poi.merge_poi(df, index.features_many(poi.unique_coords(df), radii))
    return poi_select.select_poi(output, prune=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        build_index(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else INDEX_PATH)
    elif len(sys.argv) == 4 and sys.argv[1] == "query":
        print(json.dumps(PoiIndex().features(float(sys.argv[2]), float(sys.argv[3])), indent=1))
    else:
        print("usage: python poi_local.py build <extract.osm.pbf> [out.parquet]\n"
              "       python poi_local.py query <lat> <lon>")