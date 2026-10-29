"""Reduce the 76 raw poi_* columns to a small, PM2.5-relevant set.

    import poi_select
    df = poi_select.select_poi(df)                  # fixed selection + composites
    df = poi_select.select_poi(df, prune=True)      # also drop near-constant / redundant columns

Edit the lists below to change the selection. Non-POI columns are left untouched.
"""

import pandas as pd

RADII = (100, 500)   # 300 m is highly correlated with these two

# Kept as individual counts.
KEEP = ("road_tertiary", "road_residential", "traffic_signals", "crossing",
        "bus_stop", "parking", "green")

# New columns made by summing raw categories at each radius.
COMPOSITES = {
    "commercial":       ("shop", "office", "food", "market"),
    "major_road":       ("road_primary", "road_secondary"),
    "emission_sources": ("industrial", "construction", "fuel", "market"),
}

# Distance-to-nearest features that are kept.
KEEP_NEAREST = ("green", "traffic_signals", "bus_stop")


def _col(cat: str, r: int) -> str:
    return f"poi_{cat}_{r}m"


def select_poi(df: pd.DataFrame, prune: bool = False,
               min_nonzero: float = 0.02, max_corr: float = 0.9) -> pd.DataFrame:
    """Return df with raw poi_* columns replaced by the selected + composite ones.

    prune=True additionally drops, among the new POI columns:
      - columns that are non-zero in fewer than `min_nonzero` of rows (almost constant),
      - columns correlated above `max_corr` with an earlier kept column.
    """
    raw = [c for c in df.columns if c.startswith("poi_")]
    out = df.drop(columns=raw).copy()
    new = {}

    for r in RADII:
        for cat in KEEP:
            if _col(cat, r) in df:
                new[_col(cat, r)] = df[_col(cat, r)]
        for name, parts in COMPOSITES.items():
            present = [_col(p, r) for p in parts if _col(p, r) in df]
            if present:
                new[f"poi_{name}_{r}m"] = df[present].sum(axis=1)
    for cat in KEEP_NEAREST:
        if f"poi_nearest_{cat}_m" in df:
            new[f"poi_nearest_{cat}_m"] = df[f"poi_nearest_{cat}_m"]

    poi_df = pd.DataFrame(new, index=df.index)
    if prune:
        poi_df = _prune(poi_df, min_nonzero, max_corr)
    return pd.concat([out, poi_df], axis=1)


def _prune(poi_df: pd.DataFrame, min_nonzero: float, max_corr: float) -> pd.DataFrame:
    # "non-zero" for distances means "has a value"; NaN (nothing within range) counts as absent
    active = poi_df.fillna(0).ne(0).mean()
    dropped_sparse = active[active < min_nonzero].index.tolist()
    poi_df = poi_df.drop(columns=dropped_sparse)

    corr = poi_df.corr().abs()
    keep = []
    dropped_corr = []
    for c in poi_df.columns:
        twin = next((k for k in keep if corr.loc[c, k] > max_corr), None)
        if twin:
            dropped_corr.append(f"{c} (~{twin})")
        else:
            keep.append(c)
    print(f"[poi_select] dropped near-constant: {dropped_sparse}")
    print(f"[poi_select] dropped redundant: {dropped_corr}")
    return poi_df[keep]


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    data = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    result = select_poi(data, prune="--prune" in sys.argv)
    print(f"{len(data.columns)} -> {len(result.columns)} columns")
    print([c for c in result.columns if c.startswith("poi_")])