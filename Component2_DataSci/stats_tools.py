"""Statistical primitives for Component 2.

Nothing here knows about CCTV or PM2.5: every function takes arrays or a
DataFrame and returns numbers. Keeping the statistics separate from the
domain logic is what makes the results checkable.

Why these particular tools:

* `correlate` reports a confidence interval and the number of complete pairs
  next to every coefficient. A coefficient on its own is not a finding.
* `fdr_bh` is applied whenever more than a handful of correlations are tested
  at once. Testing 23 POI features against traffic at alpha = 0.05 produces
  about one false positive by construction; the correction is not optional.
* `cluster_bootstrap_ci` resamples whole cameras rather than rows. The 61k
  image rows are not 61k independent observations, so a naive interval is far
  too narrow.
* `partial_correlation` is how a confounded association is separated from a
  conditional one. Traffic and PM2.5 both follow a daily cycle, so their raw
  correlation is partly the clock, not the cars.
* `panel_decompose` splits a variable into a between-camera part (a camera's
  own mean) and a within-camera part (deviation from that mean). The two carry
  different information and can even point in opposite directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Correlation with uncertainty
# --------------------------------------------------------------------------- #
def _fisher_ci(r: float, n: int, method: str, conf: float) -> tuple:
    """Confidence interval for a correlation via the Fisher z transform.

    Spearman's standard error is inflated by 1.06, the usual large-sample
    correction (Bonett & Wright, 2000).
    """
    if n <= 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = np.sqrt(1.06 / (n - 3)) if method == "spearman" else np.sqrt(1.0 / (n - 3))
    crit = stats.norm.ppf(0.5 + conf / 2)
    return tuple(np.tanh([z - crit * se, z + crit * se]))


def correlate(x, y, method: str = "spearman", conf: float = 0.95,
              min_pairs: int = 30) -> dict:
    """One correlation, reported honestly.

    Returns r, the two-sided p-value, a confidence interval and `n`, the number
    of complete pairs actually used. Returns NaNs rather than raising when
    there is too little data or no variance, so a sweep over many columns never
    dies on one bad column.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    blank = {"r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "n": n, "method": method}

    if n < min_pairs:
        return blank
    xv, yv = x[ok].to_numpy(float), y[ok].to_numpy(float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return blank

    if method == "spearman":
        r, p = stats.spearmanr(xv, yv)
    elif method == "pearson":
        r, p = stats.pearsonr(xv, yv)
    elif method == "kendall":
        r, p = stats.kendalltau(xv, yv)
    else:
        raise ValueError(f"unknown method {method!r}")

    low, high = _fisher_ci(float(r), n, method, conf)
    return {"r": float(r), "p": float(p), "ci_low": low, "ci_high": high,
            "n": n, "method": method}


def correlate_many(df: pd.DataFrame, target: str, predictors: list,
                   method: str = "spearman", min_pairs: int = 30,
                   fdr_q: float = 0.05) -> pd.DataFrame:
    """Correlate many predictors against one target, then control the FDR.

    Sorted by absolute effect size, not by p-value: with a few thousand rows a
    negligible coefficient is still "significant", and ranking by p-value hides
    that.
    """
    rows = []
    for col in predictors:
        if col not in df.columns or col == target:
            continue
        res = correlate(df[col], df[target], method=method, min_pairs=min_pairs)
        res["predictor"] = col
        rows.append(res)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_r"] = out["r"].abs()
    out["p_fdr"] = np.nan
    testable = out["p"].notna()
    if testable.any():
        out.loc[testable, "p_fdr"] = fdr_bh(out.loc[testable, "p"].to_numpy())
    out["significant"] = out["p_fdr"] < fdr_q
    cols = ["predictor", "r", "ci_low", "ci_high", "p", "p_fdr",
            "significant", "n", "method", "abs_r"]
    return out[cols].sort_values("abs_r", ascending=False).reset_index(drop=True)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# --------------------------------------------------------------------------- #
# Clustered uncertainty
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(df: pd.DataFrame, cluster_col: str, statistic,
                         n_boot: int = 2000, conf: float = 0.95,
                         seed: int = 0) -> dict:
    """Percentile CI for any statistic, resampling whole clusters.

    `statistic` takes a DataFrame and returns a float. Cameras are drawn with
    replacement, which is the right unit here: two images from one camera tell
    you much less than two images from two cameras.
    """
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster_col, sort=False)]
    point = statistic(df)

    draws = []
    k = len(groups)
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        sample = pd.concat([groups[i] for i in pick], ignore_index=True)
        try:
            value = statistic(sample)
        except Exception:
            value = np.nan
        draws.append(value)

    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 50:
        return {"estimate": point, "ci_low": np.nan, "ci_high": np.nan,
                "n_clusters": k, "n_boot_ok": int(draws.size)}
    lo, hi = np.quantile(draws, [(1 - conf) / 2, 0.5 + conf / 2])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "n_clusters": k, "n_boot_ok": int(draws.size)}


def effective_n_warning(n_rows: int, n_clusters: int) -> str:
    """Plain-language note on pseudo-replication, for the report."""
    if n_clusters >= n_rows:
        return ""
    inflation = np.sqrt(n_rows / max(n_clusters, 1))
    return (f"{n_rows:,} rows but only {n_clusters} independent units: a p-value "
            f"computed on the rows is roughly {inflation:.0f}x too confident.")


# --------------------------------------------------------------------------- #
# Conditional association
# --------------------------------------------------------------------------- #
def _residualise(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residuals of y after least-squares removal of the control columns."""
    design = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def partial_correlation(df: pd.DataFrame, x: str, y: str, controls: list,
                        method: str = "spearman") -> dict:
    """Correlation between x and y with the linear effect of `controls` removed.

    Both variables are rank-transformed first when method is "spearman", so the
    result is a Spearman partial correlation rather than a Pearson one on raw
    scales.
    """
    cols = [x, y] + list(controls)
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = len(sub), len(controls)
    blank = {"r": np.nan, "p": np.nan, "n": n, "n_controls": k}
    if n < k + 10:
        return blank

    data = sub.rank() if method == "spearman" else sub
    ctrl = data[list(controls)].to_numpy(float)
    if np.linalg.matrix_rank(np.column_stack([np.ones(n), ctrl])) < k + 1:
        return blank  # collinear controls: the partial is not identified

    rx = _residualise(data[x].to_numpy(float), ctrl)
    ry = _residualise(data[y].to_numpy(float), ctrl)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return blank

    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - k
    if dof <= 0 or abs(r) >= 1:
        return {"r": r, "p": np.nan, "n": n, "n_controls": k}
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = float(2 * stats.t.sf(abs(t), dof))
    return {"r": r, "p": p, "n": n, "n_controls": k}


def vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Variance inflation factor per column, for reading a model's coefficients.

    Above about 10 the individual coefficients are not interpretable, even
    though the model's predictions may be fine.
    """
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    for col in cols:
        others = [c for c in cols if c != col]
        if not others or sub[col].std() == 0:
            rows.append({"feature": col, "vif": np.nan})
            continue
        y = sub[col].to_numpy(float)
        resid = _residualise(y, sub[others].to_numpy(float))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
        rows.append({"feature": col,
                     "vif": np.inf if r2 >= 1 else 1.0 / (1.0 - r2)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Panel structure
# --------------------------------------------------------------------------- #
def panel_decompose(df: pd.DataFrame, cols: list, group: str) -> pd.DataFrame:
    """Add `<col>_between` (group mean) and `<col>_within` (deviation) columns.

    A correlation on the between part answers "do busier places have more
    pollution"; on the within part, "when this place gets busier, does its
    pollution rise". Mixing them is the commonest error in this kind of panel.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        means = values.groupby(out[group]).transform("mean")
        out[f"{col}_between"] = means
        out[f"{col}_within"] = values - means
    return out


def icc(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Share of a variable's variance that sits between groups rather than within.

    High ICC means location explains the variable and time does not, which
    decides whether a predictive model needs spatial or temporal features.
    """
    sub = df[[value_col, group_col]].dropna()
    if sub[group_col].nunique() < 2 or len(sub) < 10:
        return {"icc": np.nan, "var_between": np.nan, "var_within": np.nan,
                "n": len(sub), "n_groups": sub[group_col].nunique()}

    grand = sub[value_col].mean()
    grouped = sub.groupby(group_col)[value_col]
    sizes, means = grouped.size(), grouped.mean()
    k, n = len(sizes), len(sub)

    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(((sub[value_col] - sub[group_col].map(means)) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k) if n > k else np.nan

    # Unbiased between-group variance for unequal group sizes.
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    var_between = max((ms_between - ms_within) / n0, 0.0)
    total = var_between + ms_within
    return {"icc": float(var_between / total) if total > 0 else np.nan,
            "var_between": float(var_between), "var_within": float(ms_within),
            "n": n, "n_groups": k}


# --------------------------------------------------------------------------- #
# Spatial structure
# --------------------------------------------------------------------------- #
EARTH_R_M = 6_371_000.0


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """All pairwise great-circle distances in metres."""
    la = np.radians(np.asarray(lat, float))[:, None]
    lo = np.radians(np.asarray(lon, float))[:, None]
    dlat, dlon = la - la.T, lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_weights(lat, lon, k: int = 5) -> np.ndarray:
    """Row-standardised k-nearest-neighbour spatial weights."""
    dist = haversine_matrix(lat, lon)
    n = dist.shape[0]
    np.fill_diagonal(dist, np.inf)
    w = np.zeros((n, n))
    k = min(k, n - 1)
    for i in range(n):
        w[i, np.argsort(dist[i])[:k]] = 1.0
    rows = w.sum(axis=1, keepdims=True)
    return np.divide(w, rows, out=np.zeros_like(w), where=rows > 0)


def morans_i(values, lat, lon, k: int = 5, n_perm: int = 999,
             seed: int = 0) -> dict:
    """Moran's I with a permutation p-value.

    I near 0 means the variable is spatially random; positive means nearby
    cameras resemble each other. This decides whether a map can be
    interpolated between cameras at all.
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    lat, lon = pd.Series(lat), pd.Series(lon)
    ok = v.notna() & lat.notna() & lon.notna()
    v, lat, lon = v[ok].to_numpy(float), lat[ok].to_numpy(float), lon[ok].to_numpy(float)
    n = v.size
    if n < 10 or np.std(v) == 0:
        return {"I": np.nan, "p": np.nan, "n": n, "k": k}

    w = knn_weights(lat, lon, k)

    def statistic(x):
        z = x - x.mean()
        denom = float(np.sum(z ** 2))
        return float(n * (z @ w @ z) / (w.sum() * denom)) if denom > 0 else np.nan

    obs = statistic(v)
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permutation(v)) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    # Two-sided permutation p-value, with the observed value included.
    p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (null.size + 1)
    return {"I": obs, "p": float(p), "n": n, "k": k,
            "expected_I": -1.0 / (n - 1)}
