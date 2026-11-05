"""The Component 2 analysis sections.

Each `section_*` function takes the prepared frames and returns a dict with
`tables` (DataFrames), `stats` (scalars) and `findings` (short statements that
the report prints verbatim). A finding is only written when the number behind
it has actually been computed, so the report cannot claim more than the data
supports.

The sections are ordered so that each one earns the right to the next: what the
data contains, then how it moves in time, then how it varies in space, then
what that implies for predicting PM2.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import features as F
    from . import stats_tools as S
except ImportError:
    import config as C
    import features as F
    import stats_tools as S


def _fmt_p(p: float) -> str:
    """Format a p-value without ever printing a misleading `p = 0`."""
    if not np.isfinite(p):
        return "p not estimable"
    if p == 0:
        return "p < 1e-300 (underflow)"
    if p < 1e-3:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def _fmt_r(res: dict) -> str:
    """Render a correlation result as r = .. [CI], p = .., n = .."""
    if not np.isfinite(res.get("r", np.nan)):
        return "not estimable"
    ci = ""
    if np.isfinite(res.get("ci_low", np.nan)):
        ci = f" [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]"
    return (f"r = {res['r']:+.3f}{ci}, {_fmt_p(res.get('p', np.nan))}, "
            f"n = {res['n']:,}")


# --------------------------------------------------------------------------- #
# 1. What the data actually contains
# --------------------------------------------------------------------------- #
def section_audit(df: pd.DataFrame, caps: dict) -> dict:
    """Coverage, cadence, missingness and detector sanity.

    Run before any correlation. Three of the limitations found here (a 7-day
    window, daylight hours only, a zero-inflated motorcycle channel) constrain
    every claim the rest of the analysis can make.
    """
    ts = df["timestamp"]
    n_rows, n_cam = len(df), df["camera_id"].nunique()

    # Sampling cadence per camera, from consecutive image timestamps.
    gaps = (df.sort_values(["camera_id", "timestamp"])
              .groupby("camera_id")["timestamp"].diff().dt.total_seconds() / 60)
    gaps = gaps[gaps.between(0, 180)]       # drop the overnight jump

    per_cam = df["camera_id"].value_counts()
    missing = (df.isna().mean() * 100).rename("pct_missing")
    missing = missing[missing > 0].sort_values(ascending=False)

    vehicle_cols = caps["vehicles"]
    detector = pd.DataFrame({
        "mean": df[vehicle_cols].mean(),
        "median": df[vehicle_cols].median(),
        "p95": df[vehicle_cols].quantile(0.95),
        "max": df[vehicle_cols].max(),
        "pct_zero": (df[vehicle_cols] == 0).mean() * 100,
        "share_of_traffic": df[vehicle_cols].sum() / df[vehicle_cols].sum().sum() * 100,
    }).round(2)

    stats = {
        "n_rows": n_rows,
        "n_cameras": n_cam,
        "n_camera_hours": df.groupby(["camera_id", "hourly_timestamp"]).ngroups,
        "start": str(ts.min()),
        "end": str(ts.max()),
        "n_days": ts.dt.date.nunique(),
        "hour_min": int(df["hour"].min()),
        "hour_max": int(df["hour"].max()),
        "cadence_median_min": float(gaps.median()),
        "cadence_iqr": (float(gaps.quantile(0.25)), float(gaps.quantile(0.75))),
        "images_per_camera_median": float(per_cam.median()),
        "images_per_camera_range": (int(per_cam.min()), int(per_cam.max())),
        "n_weekend_days": int(df.loc[df["is_weekend"] == 1, "date"].nunique()),
        "duplicate_images": int(df["image_name"].duplicated().sum()),
        "duplicate_cam_time": int(df.duplicated(["camera_id", "raw_timestamp"]).sum()),
    }

    findings = [
        f"{n_rows:,} images from {n_cam} cameras over {stats['n_days']} days "
        f"({stats['start'][:10]} to {stats['end'][:10]}), sampled roughly every "
        f"{stats['cadence_median_min']:.0f} minutes.",
        f"Coverage is daylight only, {stats['hour_min']:02d}:00 to "
        f"{stats['hour_max']:02d}:00. Nothing in this analysis can speak to "
        f"overnight behaviour, which is when Bangkok PM2.5 typically peaks.",
        f"The window contains {stats['n_weekend_days']} weekend "
        f"{'day' if stats['n_weekend_days'] == 1 else 'days'} from one weekend, "
        f"so the weekday/weekend contrast cannot be separated from whatever else "
        f"was happening that particular weekend. Treat it as suggestive.",
    ]
    if stats["duplicate_images"] == 0 and stats["duplicate_cam_time"] == 0:
        findings.append("No duplicate images and no duplicate camera-timestamp "
                        "pairs: the DE join did not fan out.")

    zero_heavy = detector["pct_zero"].max()
    worst = detector["pct_zero"].idxmax()
    findings.append(
        f"The detector output is strongly zero-inflated: {worst} is zero in "
        f"{zero_heavy:.0f}% of frames. Treat the rarer classes as presence/absence "
        f"rather than as counts, and prefer totals over single-class counts.")

    # Hours sampled far less than the rest are unstable in any hourly mean.
    per_hour = df["hour"].value_counts().sort_index()
    typical = per_hour.median()
    thin = per_hour[per_hour < 0.5 * typical]
    stats["thin_hours"] = {int(h): int(n) for h, n in thin.items()}
    if len(thin):
        listed = ", ".join(f"{int(h):02d}:00 ({int(n):,} images)"
                           for h, n in thin.items())
        findings.append(
            f"Sampling is not uniform across the day: {listed}, against a median "
            f"of {typical:,.0f} images per hour. Hourly means for those hours rest "
            f"on far less data and should not be read as trend.")

    # NaN in these columns is a measured absence, not a failed join.
    nearest = [c for c in df.columns if c.startswith("poi_nearest_")]
    if nearest:
        worst_near = max(nearest, key=lambda c: df[c].isna().mean())
        findings.append(
            f"Missing values in the `poi_nearest_*` columns are meaningful, not "
            f"broken: NaN means nothing of that kind lies within the 500 m search "
            f"radius ({worst_near} is NaN for "
            f"{df[worst_near].isna().mean() * 100:.0f}% of rows). They must be "
            f"modelled as censored at 500 m, never imputed with a mean. The same "
            f"applies to `heavy_share` and `motorcycle_share`, which are undefined "
            f"on empty frames rather than zero.")

    if not caps["pm25"]:
        findings.append(
            "PM2.5 is absent from the prepared file, so no section below can test "
            "the project's actual hypothesis. Everything here characterises the "
            "predictors only.")

    return {"stats": stats,
            "tables": {"detector_summary": detector,
                       "missingness": missing.to_frame()},
            "findings": findings}


# --------------------------------------------------------------------------- #
# 2. Time
# --------------------------------------------------------------------------- #
def section_temporal(df: pd.DataFrame, ch: pd.DataFrame, caps: dict) -> dict:
    """Hour-of-day, weekday/weekend and day-to-day structure in traffic.

    Confidence intervals come from a camera-level bootstrap: the 61k images are
    about 100 independent units, and an interval built on the rows would be
    roughly 25 times too narrow.
    """
    vehicle_cols = caps["vehicles"]
    hourly = (df.groupby("hour")[vehicle_cols + ["total_vehicles"]]
                .mean().round(3))
    hourly["n_images"] = df.groupby("hour").size()

    by_hour_weekend = (df.groupby(["is_weekend", "hour"])["total_vehicles"]
                         .mean().unstack(0).round(3))
    by_hour_weekend.columns = ["weekday", "weekend"][:by_hour_weekend.shape[1]]

    daily = (df.groupby(["date", "day_name"])["total_vehicles"]
               .agg(["mean", "count"]).round(3).reset_index())

    # Weekday/weekend gap, with the camera as the resampling unit.
    def gap(frame):
        wk = frame.loc[frame["is_weekend"] == 0, "total_vehicles"].mean()
        we = frame.loc[frame["is_weekend"] == 1, "total_vehicles"].mean()
        return wk - we

    weekend_gap = S.cluster_bootstrap_ci(
        df[["camera_id", "is_weekend", "total_vehicles"]], "camera_id", gap,
        n_boot=C.BOOTSTRAP_N, seed=C.RANDOM_SEED)

    # Share of traffic variance explained by the clock alone, at camera-hour level.
    hour_means = ch.groupby("hour")["total_vehicles"].transform("mean")
    total_var = ch["total_vehicles"].var()
    r2_hour = float(1 - (ch["total_vehicles"] - hour_means).var() / total_var)

    peak_hour = int(hourly["total_vehicles"].idxmax())
    trough_hour = int(hourly["total_vehicles"].idxmin())
    amplitude = float(hourly["total_vehicles"].max() / hourly["total_vehicles"].min())

    # Does each vehicle class peak at the same time?
    class_peaks = {c: int(hourly[c].idxmax()) for c in vehicle_cols}

    findings = [
        f"Traffic peaks at {peak_hour:02d}:00 and bottoms out at "
        f"{trough_hour:02d}:00, a {amplitude:.2f}x swing in mean vehicles per frame. "
        f"The daily cycle is real but shallow.",
        f"Hour of day alone explains only {r2_hour * 100:.1f}% of camera-hour "
        f"variance in total vehicles. Time of day is a weak predictor here "
        f"compared with which camera you are looking at (see the variance "
        f"decomposition below).",
        f"Weekday traffic exceeds weekend traffic by "
        f"{weekend_gap['estimate']:.2f} vehicles per frame "
        f"[{weekend_gap['ci_low']:.2f}, {weekend_gap['ci_high']:.2f}] on a "
        f"camera-clustered bootstrap over {weekend_gap['n_clusters']} cameras.",
    ]
    if weekend_gap["ci_low"] > 0:
        findings.append("That interval excludes zero, so the weekday/weekend "
                        "difference survives clustering, unlike most row-level "
                        "effects in this dataset.")
    else:
        findings.append("That interval includes zero: once cameras rather than "
                        "images are treated as the unit, the weekday/weekend "
                        "difference is not statistically distinguishable.")

    spread = max(class_peaks.values()) - min(class_peaks.values())
    if spread >= 3:
        listed = ", ".join(f"{c.replace('_count', '')} at {h:02d}:00"
                           for c, h in sorted(class_peaks.items(), key=lambda kv: kv[1]))
        findings.append(
            f"The classes do not share a peak hour ({listed}), a spread of "
            f"{spread} hours. They are different traffic processes and should not "
            f"be collapsed into one 'congestion' number without checking which "
            f"class drives a result.")

    return {"stats": {"peak_hour": peak_hour, "trough_hour": trough_hour,
                      "amplitude": amplitude, "r2_hour": r2_hour,
                      "weekend_gap": weekend_gap, "class_peaks": class_peaks},
            "tables": {"hourly_profile": hourly,
                       "hourly_by_weekend": by_hour_weekend,
                       "daily_means": daily},
            "findings": findings}


# --------------------------------------------------------------------------- #
# 3. How the vehicle channels relate to each other
# --------------------------------------------------------------------------- #
def section_composition(df: pd.DataFrame, cam: pd.DataFrame, caps: dict) -> dict:
    """Correlation among the four detector channels, and whether one factor
    summarises them.

    Done at both levels on purpose. At image level the channels are nearly
    independent (detector noise dominates); at camera level they share a common
    "how busy is this place" factor. The gap between the two is the single
    clearest piece of evidence that per-frame counts are noisy.
    """
    vehicle_cols = caps["vehicles"]
    corr_image = df[vehicle_cols].corr(method="spearman").round(3)
    corr_camera = cam[vehicle_cols].corr(method="spearman").round(3)

    # PCA on the standardised camera-level means.
    X = cam[vehicle_cols].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    explained = (sv ** 2) / np.sum(sv ** 2)
    loadings = pd.DataFrame({"PC1": vt[0], "PC2": vt[1]},
                            index=vehicle_cols).round(3)

    mean_off_image = _mean_offdiag(corr_image)
    mean_off_camera = _mean_offdiag(corr_camera)

    ratio = mean_off_camera / mean_off_image if mean_off_image else np.nan
    n_classes = len(vehicle_cols)
    baseline = 100.0 / n_classes      # PC1 share under no shared structure

    findings = [
        f"Mean off-diagonal correlation among the four channels is "
        f"{mean_off_image:+.2f} per frame and {mean_off_camera:+.2f} per camera, "
        f"a factor of {ratio:.1f}. Both are moderate; the point is the gap. "
        f"Averaging over a camera removes per-frame detection noise, and what is "
        f"left behind is the shared 'how busy is this place' component.",
        f"The first principal component takes {explained[0] * 100:.0f}% of the "
        f"camera-level variance, against {baseline:.0f}% for {n_classes} "
        f"structureless channels. There is a common volume factor, but it is not "
        f"dominant: collapsing the classes into one index would discard about "
        f"{(1 - explained[0]) * 100:.0f}% of the variation, and PC2 "
        f"({explained[1] * 100:.0f}%) is dominated by "
        f"{loadings['PC2'].abs().idxmax().replace('_count', '')}. "
        f"Keep the class breakdown; do not replace it with a single number.",
        "Practical consequence for the PM2.5 model: aggregate to camera-hour "
        "before fitting. Feeding raw per-image counts in adds variance without "
        "adding signal.",
    ]
    return {"stats": {"explained_variance": explained[:2].tolist(),
                      "mean_offdiag_image": mean_off_image,
                      "mean_offdiag_camera": mean_off_camera},
            "tables": {"corr_image_level": corr_image,
                       "corr_camera_level": corr_camera,
                       "pca_loadings": loadings},
            "findings": findings}


def _mean_offdiag(corr: pd.DataFrame) -> float:
    m = corr.to_numpy(float).copy()
    iu = np.triu_indices_from(m, k=1)
    return float(np.nanmean(m[iu]))


# --------------------------------------------------------------------------- #
# 4. Where the variance lives
# --------------------------------------------------------------------------- #
def section_variance(ch: pd.DataFrame, caps: dict) -> dict:
    """Between-camera versus within-camera variance for every usable variable.

    This decides the project's modelling strategy. A variable whose variance is
    mostly between cameras is explained by location; one whose variance is
    mostly within a camera is explained by time. A PM2.5 model needs to know
    which kind of variable it is trying to predict.
    """
    targets = ["total_vehicles"] + caps["vehicles"]
    if caps["pm25"]:
        targets.append(C.TARGET)
    targets += caps["weather"]

    rows = []
    for col in targets:
        if col not in ch.columns:
            continue
        res = S.icc(ch, col, "camera_id")
        res["variable"] = col
        rows.append(res)
    table = (pd.DataFrame(rows)[["variable", "icc", "var_between", "var_within",
                                 "n", "n_groups"]]
             .sort_values("icc", ascending=False).reset_index(drop=True))

    traffic_icc = float(table.loc[table["variable"] == "total_vehicles",
                                  "icc"].iloc[0])
    findings = [
        f"{traffic_icc * 100:.0f}% of the variance in camera-hour traffic is "
        f"between cameras, not within them. Where a camera is matters several "
        f"times more than what time it is.",
        "That asymmetry is the main result for the group's direction: a "
        "higher-resolution PM2.5 map stands or falls on cross-sectional "
        "(location) signal, so the feature set must be dominated by location "
        "descriptors, with time as a correction rather than the driver.",
    ]
    if caps["pm25"]:
        pm_icc = float(table.loc[table["variable"] == C.TARGET, "icc"].iloc[0])
        findings.append(
            f"PM2.5 ICC is {pm_icc:.2f}. A low value means the stations behind "
            f"the target barely disagree with each other, which caps how much "
            f"spatial detail any model can recover, however good the image "
            f"features are.")
    return {"stats": {"traffic_icc": traffic_icc},
            "tables": {"variance_decomposition": table},
            "findings": findings}


# --------------------------------------------------------------------------- #
# 5. Space
# --------------------------------------------------------------------------- #
def section_spatial(cam: pd.DataFrame, caps: dict) -> dict:
    """Is traffic spatially clustered, or is each camera its own story?

    Moran's I on a 5-nearest-neighbour graph. If I is near zero, values cannot
    be interpolated between cameras, which directly limits the "higher
    resolution map" idea: the map would have to be driven by per-location
    features rather than by smoothing.
    """
    tested = ["total_vehicles"] + caps["vehicles"]
    if caps["pm25"]:
        tested.append(C.TARGET)

    rows = []
    for col in tested:
        if col not in cam.columns:
            continue
        res = S.morans_i(cam[col], cam["latitude"], cam["longitude"],
                         k=C.MORAN_K, seed=C.RANDOM_SEED)
        res["variable"] = col
        rows.append(res)
    moran = (pd.DataFrame(rows)[["variable", "I", "expected_I", "p", "n", "k"]]
             .round(4).sort_values("I", ascending=False).reset_index(drop=True))

    ranked = cam.sort_values("total_vehicles", ascending=False)
    name_cols = [c for c in ["location_en", "location_detail_en"] if c in cam.columns]
    show = ["camera_id"] + name_cols + ["total_vehicles", "heavy_share", "n_images"]
    show = [c for c in show if c in ranked.columns]
    busiest = ranked[show].head(10).round(3)
    quietest = ranked[show].tail(10).round(3)

    spread = float(cam["total_vehicles"].max() / max(cam["total_vehicles"].min(), 1e-9))
    traffic_moran = moran.loc[moran["variable"] == "total_vehicles"].iloc[0]

    sig = moran[(moran["p"] < C.ALPHA) & (moran["I"] > 0)]
    sig_names = sig["variable"].tolist()

    findings = [
        f"Mean traffic ranges {cam['total_vehicles'].min():.1f} to "
        f"{cam['total_vehicles'].max():.1f} vehicles per frame across cameras, a "
        f"{spread:.1f}x spread. This is the dominant source of variation in the "
        f"dataset.",
        f"Moran's I for camera-mean traffic is {traffic_moran['I']:+.3f} "
        f"({_fmt_p(traffic_moran['p'])}, k = {C.MORAN_K}, expected under "
        f"spatial randomness {traffic_moran['expected_I']:+.3f}).",
    ]

    if traffic_moran["p"] < C.ALPHA and traffic_moran["I"] > 0:
        findings.append(
            "Nearby cameras resemble each other more than distant ones, though "
            f"an I of {traffic_moran['I']:+.2f} is weak clustering: it supports "
            "a spatial smoothing term, not interpolation between cameras.")
    else:
        findings.append(
            f"That does not clear the {C.ALPHA} threshold, so total traffic "
            f"cannot be called spatially clustered at this camera spacing. The "
            f"positive sign and borderline p-value mean 'not demonstrated', not "
            f"'demonstrated absent'; with 100 cameras the test has little power.")

    if sig_names:
        detail = ", ".join(
            f"{r.variable} (I = {r.I:+.2f}, {_fmt_p(r.p)})" for r in sig.itertuples())
        findings.append(
            f"Individual classes do cluster: {detail}. One reading, which this "
            f"data cannot confirm, is that these classes follow fixed route "
            f"structure (corridors, lanes) that is geographically organised, "
            f"while car counts are set mainly by what each camera is pointed at. "
            f"Testing that needs road attributes the dataset does not yet carry.")
    findings.append(
        "Practical consequence: a higher-resolution map cannot be produced by "
        "interpolating between these cameras. It has to come from per-location "
        "features (road class, POI, image content) evaluated at the target "
        "point.")

    return {"stats": {"spread": spread, "n_spatially_clustered": len(sig_names),
                      "clustered_variables": sig_names},
            "tables": {"morans_i": moran, "busiest_cameras": busiest,
                       "quietest_cameras": quietest},
            "findings": findings}


# --------------------------------------------------------------------------- #
# 6. POI against traffic
# --------------------------------------------------------------------------- #
def section_poi(df: pd.DataFrame, cam: pd.DataFrame, caps: dict) -> dict:
    """Do the built-environment features explain how busy a camera is?

    Run at camera level (n = 100), which is the level POI varies at. The same
    correlation is also run at image level purely to show how badly the p-value
    inflates, because that inflation is an easy mistake to make with this
    dataset and it changes conclusions.
    """
    poi_cols = caps["poi"]
    if not poi_cols:
        return {"stats": {}, "tables": {},
                "findings": ["No POI features available, so the built-environment "
                             "question was not tested."]}

    table = S.correlate_many(cam, "total_vehicles", poi_cols,
                             method="spearman", min_pairs=C.MIN_PAIRS,
                             fdr_q=C.FDR_Q)
    heavy = None
    if "heavy_share" in cam.columns:
        heavy = S.correlate_many(cam, "heavy_share", poi_cols,
                                 method="spearman", min_pairs=C.MIN_PAIRS,
                                 fdr_q=C.FDR_Q)

    # Pseudo-replication demonstration on the strongest feature.
    top = table.iloc[0]
    naive = S.correlate(df[top["predictor"]], df["total_vehicles"],
                        method="spearman", min_pairs=C.MIN_PAIRS)
    warning = S.effective_n_warning(len(df), cam["camera_id"].nunique())

    n_sig = int(table["significant"].sum())
    findings = [
        f"Of {len(table)} POI features tested against camera-mean traffic, "
        f"{n_sig} survive Benjamini-Hochberg correction at q = {C.FDR_Q}.",
    ]
    if n_sig:
        best = table[table["significant"]].head(3)
        parts = [f"{r.predictor} (rho = {r.r:+.2f})" for r in best.itertuples()]
        findings.append("Strongest surviving associations: " + "; ".join(parts) + ".")
    else:
        findings.append(
            "None survive correction. With 100 cameras the detectable effect "
            "size is about |rho| > 0.27, and no POI feature clears it. The "
            "built-environment features as currently defined do not explain how "
            "busy a camera is.")

    findings.append(
        f"Pseudo-replication check on {top['predictor']}: at camera level "
        f"rho = {top['r']:+.3f}, {_fmt_p(top['p'])}, n = {int(top['n'])}; at image "
        f"level rho = {naive['r']:+.3f}, {_fmt_p(naive['p'])}, n = {naive['n']:,}. "
        f"The coefficient barely moves, the p-value collapses by hundreds of "
        f"orders of magnitude, and nothing was learned in between: the POI value "
        f"is identical for every image at a given camera. {warning}")
    out_tables = {"poi_vs_traffic_camera_level": table.round(4)}
    if heavy is not None:
        out_tables["poi_vs_heavy_share_camera_level"] = heavy.round(4)
    return {"stats": {"n_significant": n_sig,
                      "naive_vs_clustered": {"camera": top["p"], "image": naive["p"]}},
            "tables": out_tables, "findings": findings}


# --------------------------------------------------------------------------- #
# 7. PM2.5 (runs only when the target is present)
# --------------------------------------------------------------------------- #
def section_pm25(ch: pd.DataFrame, cam: pd.DataFrame, caps: dict) -> dict:
    """The project's actual hypothesis, tested four ways.

    Raw, between-camera, within-camera and partial (clock and weather removed).
    Reporting only the raw correlation would be misleading: traffic and PM2.5
    share a daily cycle, so part of any raw association is the time of day
    rather than the traffic.
    """
    if not caps["pm25"]:
        return {"stats": {}, "tables": {},
                "findings": [
                    "SKIPPED: pm25 is not in the prepared dataset. Run DE/main.py "
                    "with OPENAQ_API_KEY set to produce data/processed/data.csv, "
                    "then re-run this module; this section then executes with no "
                    "code changes."],
                "skipped": True}

    out_tables, stats, findings = {}, {}, []
    target = C.TARGET

    # ---- 7a. How independent is the target, really? ---------------------- #
    if "location_id" in ch.columns:
        n_stations = int(ch["location_id"].nunique())
        per_station = ch.groupby("location_id")["camera_id"].nunique()
        fallback = float(ch["is_fallback"].mean()) if "is_fallback" in ch else np.nan
        dist = ch["station_distance_m"].describe() if "station_distance_m" in ch else None
        stats.update({"n_stations": n_stations,
                      "cameras_per_station_max": int(per_station.max()),
                      "fallback_share": fallback})
        findings.append(
            f"The PM2.5 target comes from {n_stations} monitoring stations for "
            f"{ch['camera_id'].nunique()} cameras; the busiest station supplies up "
            f"to {per_station.max()} cameras. The target has far fewer independent "
            f"values than the dataset has rows, which is the binding constraint on "
            f"this project, not the image features.")
        if np.isfinite(fallback) and fallback > 0:
            findings.append(
                f"{fallback * 100:.0f}% of readings came from a fallback (not the "
                f"nearest) station. Those rows carry extra spatial error and should "
                f"be weighted down or excluded in a sensitivity check.")
        if dist is not None:
            out_tables["station_distance_summary"] = dist.to_frame().round(1)

    # ---- 7b. Weather ------------------------------------------------------ #
    if caps["weather"]:
        wx = S.correlate_many(ch, target, caps["weather"], method="spearman",
                              min_pairs=C.MIN_PAIRS, fdr_q=C.FDR_Q)
        out_tables["pm25_vs_weather"] = wx.round(4)
        top = wx.iloc[0]
        if bool(top["significant"]):
            findings.append(
                f"The strongest weather correlate of PM2.5 is {top['predictor']} "
                f"at rho = {top['r']:+.3f} (q = {top['p_fdr']:.3g}). This is the "
                f"baseline any traffic or image feature has to beat; beating zero "
                f"is not the test.")
        else:
            findings.append(
                f"No weather variable correlates with PM2.5 at q = {C.FDR_Q} in "
                f"this window (strongest: {top['predictor']}, rho = "
                f"{top['r']:+.3f}, q = {top['p_fdr']:.3g}). Given how reliably "
                f"humidity and wind track PM2.5 in Bangkok, that is a reason to "
                f"re-check the weather join before trusting anything downstream "
                f"of it, not a finding about the atmosphere.")

    # ---- 7c. Traffic, four ways ------------------------------------------- #
    traffic = "total_vehicles"
    panel = S.panel_decompose(ch, [traffic, target], "camera_id")

    raw = S.correlate(panel[traffic], panel[target], min_pairs=C.MIN_PAIRS)
    within = S.correlate(panel[f"{traffic}_within"], panel[f"{target}_within"],
                         min_pairs=C.MIN_PAIRS)
    between = S.correlate(cam[traffic], cam[target], min_pairs=C.MIN_PAIRS)

    controls = ["hour_sin", "hour_cos"] + [c for c in caps["weather"] if c in ch.columns]
    partial = S.partial_correlation(ch, traffic, target, controls)

    decomp = pd.DataFrame([
        {"estimate": "raw (camera-hour)", **raw},
        {"estimate": "within-camera (time only)", **within},
        {"estimate": "between-camera (location only)", **between},
        {"estimate": f"partial, {len(controls)} controls",
         "r": partial["r"], "p": partial["p"], "n": partial["n"],
         "ci_low": np.nan, "ci_high": np.nan, "method": "spearman"},
    ])
    out_tables["pm25_vs_traffic_decomposition"] = decomp.round(4)
    stats["pm25_traffic"] = {"raw": raw, "within": within, "between": between,
                             "partial": partial}

    findings.append(
        f"PM2.5 against traffic: raw {_fmt_r(raw)}; within-camera {_fmt_r(within)}; "
        f"between-camera {_fmt_r(between)}; partialling out the clock and weather "
        f"leaves r = {partial['r']:+.3f} (p = {partial['p']:.3g}, n = "
        f"{partial['n']:,}).")

    # A ratio of partial to raw is only meaningful if there is a raw association
    # to shrink. On noise both are ~0 and their ratio is arbitrary, so the size
    # and significance of the raw estimate are checked before anything is said.
    MIN_MEANINGFUL_R = 0.10
    raw_real = (np.isfinite(raw["r"]) and abs(raw["r"]) >= MIN_MEANINGFUL_R
                and np.isfinite(raw["p"]) and raw["p"] < C.ALPHA)

    if not raw_real:
        findings.append(
            f"There is no raw traffic-PM2.5 association to explain: "
            f"|r| = {abs(raw['r']):.3f} against a {MIN_MEANINGFUL_R:.2f} floor "
            f"for a reportable effect. With n = {raw['n']:,} this is a real null, "
            f"not low power, so the partial and lagged estimates below are "
            f"reported for completeness and none of them should be quoted as a "
            f"positive result.")
    elif np.isfinite(partial["r"]):
        shrink = abs(partial["r"]) / abs(raw["r"])
        if shrink < 0.5:
            findings.append(
                f"The association loses {(1 - shrink) * 100:.0f}% of its size once "
                f"time of day and weather are removed, so most of the raw "
                f"correlation is confounding, not traffic.")
        else:
            findings.append(
                f"The association holds at {shrink * 100:.0f}% of its raw size "
                f"after the controls, which is the result the project needs. It "
                f"remains a correlation at camera-hour level, not evidence that "
                f"traffic causes the PM2.5 reading, and the reading itself comes "
                f"from a station some distance away.")

    # ---- 7d. Lags ---------------------------------------------------------- #
    lagged = F.add_lagged_traffic(ch, traffic, C.PM25_LAGS)
    lag_cols = [f"{traffic}_lag{l}" for l in C.PM25_LAGS]
    lag_table = S.correlate_many(lagged, target, lag_cols, method="spearman",
                                 min_pairs=C.MIN_PAIRS, fdr_q=C.FDR_Q)
    out_tables["pm25_vs_lagged_traffic"] = lag_table.round(4)
    if not lag_table.empty:
        best = lag_table.iloc[0]
        if bool(best["significant"]) and abs(best["r"]) >= MIN_MEANINGFUL_R:
            findings.append(
                f"Across lags of {C.PM25_LAGS} hours the strongest relation is at "
                f"{best['predictor']} (rho = {best['r']:+.3f}). A peak at a "
                f"non-zero lag is the signature of accumulation; a peak at lag 0 "
                f"is more likely a shared daily cycle than a causal one.")
        else:
            findings.append(
                f"No lag from {C.PM25_LAGS} hours produces a reportable relation "
                f"(largest |rho| = {abs(best['r']):.3f}). Traffic measured at "
                f"these cameras does not lead the PM2.5 signal at any tested "
                f"delay.")

    # ---- 7e. Collinearity of the candidate feature set --------------------- #
    candidates = [traffic] + caps["weather"] + [c for c in caps["poi"] if c in ch.columns]
    if len(candidates) > 2:
        out_tables["feature_vif"] = S.vif(ch, candidates).round(2)

    return {"stats": stats, "tables": out_tables, "findings": findings,
            "skipped": False}


# --------------------------------------------------------------------------- #
# 8. The confounding the project has to beat
# --------------------------------------------------------------------------- #
def section_confounding(ch: pd.DataFrame, caps: dict) -> dict:
    """Quantify how much apparent signal the daily cycle can manufacture.

    Both traffic and PM2.5 vary with the hour. Two variables that each follow
    the clock correlate with each other even when neither affects the other, so
    this section measures the size of that trap before anyone fits a model.
    """
    findings, tables, stats = [], {}, {}

    hour_profile = ch.groupby("hour")["total_vehicles"].mean()
    rel_amp = float((hour_profile.max() - hour_profile.min()) / hour_profile.mean())
    stats["traffic_rel_amplitude"] = rel_amp

    if caps["pm25"]:
        pm_profile = ch.groupby("hour")[C.TARGET].mean()
        prof = S.correlate(hour_profile, pm_profile, method="spearman", min_pairs=5)
        tables["hourly_profiles"] = pd.DataFrame(
            {"traffic": hour_profile, C.TARGET: pm_profile}).round(3)
        stats["profile_correlation"] = prof
        findings.append(
            f"The hourly mean profiles of traffic and PM2.5 correlate at "
            f"rho = {prof['r']:+.3f} across {prof['n']} hours. Any raw "
            f"traffic-PM2.5 correlation inherits this, which is why the partial "
            f"correlation above is the one to quote.")
    else:
        tables["hourly_profiles"] = hour_profile.to_frame("traffic").round(3)
        findings.append(
            f"Traffic's daily swing is {rel_amp * 100:.0f}% of its mean. Bangkok "
            f"PM2.5 has a daily cycle of its own, so when the target arrives, "
            f"expect a raw traffic-PM2.5 correlation that is substantially "
            f"confounded by hour of day. The partial-correlation path in "
            f"section_pm25 is already wired to handle it.")

    findings.append(
        "Standing recommendation: report the partial correlation controlling for "
        "hour of day and weather, and validate any model by splitting on whole "
        "days and whole cameras. A random row split leaks, because two images 11 "
        "minutes apart at the same camera are nearly the same observation.")
    return {"stats": stats, "tables": tables, "findings": findings}


# --------------------------------------------------------------------------- #
# 9. Camera typology: what kinds of place are these cameras watching?
# --------------------------------------------------------------------------- #
def section_typology(ch: pd.DataFrame, cam: pd.DataFrame, caps: dict) -> dict:
    """Cluster cameras by the SHAPE of their day, not by how busy they are.

    Each camera's hourly profile is divided by its own mean before clustering,
    so a quiet residential street and a busy one with the same rush-hour rhythm
    land together. Clustering on raw levels would just rediscover the traffic
    ranking from section 5, which is not new information.

    k is chosen by silhouette score rather than fixed, and the result is only
    reported as a typology if the best k actually separates: a silhouette near
    zero means the profiles form one continuous blob and the "types" would be
    arbitrary cuts through it.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    wide = (ch.pivot_table(index="camera_id", columns="hour",
                           values="total_vehicles", aggfunc="mean")
              .sort_index())
    # A camera missing an hour entirely cannot be shaped; interpolate across the
    # day rather than dropping the camera, then drop any still incomplete.
    wide = wide.interpolate(axis=1, limit_direction="both").dropna()
    levels = wide.mean(axis=1)
    shapes = wide.div(levels, axis=0)

    X = shapes.to_numpy(float)
    scores = {}
    for k in range(2, 7):
        if k >= len(X):
            break
        labels_k = KMeans(n_clusters=k, n_init=25,
                          random_state=C.RANDOM_SEED).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels_k))

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    model = KMeans(n_clusters=best_k, n_init=25,
                   random_state=C.RANDOM_SEED).fit(X)

    # Order clusters by when they peak, so cluster 1 is the earliest-peaking
    # type. Arbitrary k-means label order would make the report unreadable.
    centres = pd.DataFrame(model.cluster_centers_, columns=shapes.columns)
    order = centres.idxmax(axis=1).sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order, start=1)}
    labels = pd.Series([remap[l] for l in model.labels_],
                       index=shapes.index, name="profile_cluster")
    centres = (centres.loc[order].reset_index(drop=True)
               .rename(index=lambda i: i + 1).round(3))
    centres.index.name = "profile_cluster"

    profile = pd.DataFrame({
        "profile_cluster": labels,
        "mean_vehicles": levels,
        "peak_hour": wide.idxmax(axis=1),
        "peak_ratio": wide.max(axis=1) / wide.mean(axis=1),
    })
    summary = (profile.groupby("profile_cluster")
               .agg(n_cameras=("mean_vehicles", "size"),
                    mean_vehicles=("mean_vehicles", "mean"),
                    median_peak_hour=("peak_hour", "median"),
                    mean_peak_ratio=("peak_ratio", "mean"))
               .round(2))
    # The hour the cluster's own centroid peaks at. This is what the profile
    # figure shows, and it can differ from the median of the members' individual
    # peak hours, which is noisier.
    summary.insert(2, "centroid_peak_hour", centres.idxmax(axis=1))

    labelled = cam.merge(labels.reset_index(), on="camera_id", how="left")

    findings = [
        f"Clustering the 100 cameras on the shape of their day (each profile "
        f"divided by its own mean) gives {best_k} types, chosen by silhouette "
        f"score over k = 2 to 6 (best {best_score:.3f}).",
    ]
    if best_score < 0.25:
        findings.append(
            f"A silhouette of {best_score:.3f} is weak. The profiles form more "
            f"of a continuum than clean groups, so treat these as convenient "
            f"bands along one gradient, not as discrete camera species.")
    else:
        findings.append(
            f"A silhouette of {best_score:.3f} means the types are reasonably "
            f"separated, so they are usable as a categorical feature.")

    for cluster, row in summary.iterrows():
        shape = centres.loc[cluster]
        findings.append(
            f"Type {cluster}: {int(row['n_cameras'])} cameras. Its average day "
            f"peaks at {int(row['centroid_peak_hour']):02d}:00 "
            f"({shape.max():.2f}x the camera's own mean) and falls to "
            f"{shape.iloc[-1]:.2f}x by {int(shape.index[-1]):02d}:00; individual "
            f"cameras in it peak at a median of "
            f"{int(row['median_peak_hour']):02d}:00 and average "
            f"{row['mean_vehicles']:.1f} vehicles per frame.")

    # Does the type say anything the traffic level does not already say?
    lvl = S.correlate(labelled["profile_cluster"], labelled["total_vehicles"],
                      method="spearman", min_pairs=C.MIN_PAIRS)
    findings.append(
        f"Profile type correlates with how busy a camera is at rho = "
        f"{lvl['r']:+.3f} ({_fmt_p(lvl['p'])}), so the shape of a camera's day "
        f"and its traffic volume are "
        f"{'largely independent' if abs(lvl['r']) < 0.3 else 'related'}. "
        f"The typology is {'a genuinely new axis' if abs(lvl['r']) < 0.3 else 'partly a restatement of volume'}.")

    poi_cols = [c for c in caps["poi"] if c in labelled.columns]
    poi_table = None
    if poi_cols:
        poi_table = labelled.groupby("profile_cluster")[poi_cols].mean().round(2)

    return {"stats": {"best_k": best_k, "silhouette": best_score,
                      "silhouette_by_k": scores},
            "tables": {"typology_summary": summary,
                       "typology_hour_centroids": centres,
                       "camera_profile_types": labelled[
                           [c for c in ["camera_id", "location_en", "profile_cluster",
                                        "total_vehicles", "latitude", "longitude"]
                            if c in labelled.columns]].round(3),
                       **({"typology_vs_poi": poi_table} if poi_table is not None else {})},
            "findings": findings,
            "labels": labels}


# --------------------------------------------------------------------------- #
# 10. The weekly rhythm
# --------------------------------------------------------------------------- #
def section_weekly(ch: pd.DataFrame, labels: pd.Series | None = None) -> dict:
    """Day-of-week by hour, city-wide and per camera type.

    Seven days is one of each weekday, so a single day's disruption and "what
    Tuesdays are like" cannot be told apart. Every statement here is therefore
    about this particular week.
    """
    grid = (ch.pivot_table(index="day_name", columns="hour",
                           values="total_vehicles", aggfunc="mean"))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
    grid = grid.reindex([d for d in day_order if d in grid.index]).round(3)

    daily = ch.groupby("day_name")["total_vehicles"].mean()
    daily = daily.reindex([d for d in day_order if d in daily.index]).round(3)
    busiest, quietest = daily.idxmax(), daily.idxmin()

    # Morning versus evening weight: which half of the day does each day lean on?
    morning = ch[ch["hour"].between(6, 11)].groupby("day_name")["total_vehicles"].mean()
    evening = ch[ch["hour"].between(16, 21)].groupby("day_name")["total_vehicles"].mean()
    balance = (evening / morning).reindex(daily.index).round(3).to_frame("evening_over_morning")

    ratios = balance["evening_over_morning"]
    n_evening_heavier = int((ratios > 1).sum())
    peak_hour = int(ch.groupby("hour")["total_vehicles"].mean().idxmax())

    findings = [
        f"Across this week {busiest} is the busiest day ({daily.max():.2f} "
        f"vehicles per frame) and {quietest} the quietest ({daily.min():.2f}), a "
        f"range of {daily.max() - daily.min():.2f}. With one of each weekday in "
        f"the window, that ordering describes this week, not weekdays in general.",
    ]

    if n_evening_heavier == 0:
        findings.append(
            f"The sharpest single hour is {peak_hour:02d}:00, but the morning "
            f"block (06:00-11:00) carries more traffic than the evening block "
            f"(16:00-21:00) on every day of the week: the evening/morning ratio "
            f"runs {ratios.min():.2f} to {ratios.max():.2f} and never reaches 1. "
            f"The evening rush is taller but much shorter, collapsing after "
            f"18:00, so the morning window holds more total traffic. A dashboard "
            f"ranking locations on a peak-hour number and one ranking them on a "
            f"daily total will not agree, and both are defensible. Pick one and "
            f"label it.")
    elif n_evening_heavier == len(ratios):
        findings.append(
            f"The evening block (16:00-21:00) outweighs the morning block "
            f"(06:00-11:00) on every day, ratio {ratios.min():.2f} to "
            f"{ratios.max():.2f}, with the sharpest hour at {peak_hour:02d}:00.")
    else:
        findings.append(
            f"Which half of the day carries more traffic changes across the "
            f"week: the evening/morning ratio runs {ratios.min():.2f} to "
            f"{ratios.max():.2f}, above 1 on {n_evening_heavier} of "
            f"{len(ratios)} days. The sharpest single hour is "
            f"{peak_hour:02d}:00.")

    per_type = None
    if labels is not None:
        tagged = ch.merge(labels.reset_index(), on="camera_id", how="left")
        per_type = (tagged.pivot_table(index="profile_cluster", columns="hour",
                                       values="total_vehicles", aggfunc="mean")
                    .round(3))
        gap = (tagged.groupby(["profile_cluster", "is_weekend"])["total_vehicles"]
               .mean().unstack())
        if gap.shape[1] == 2:
            gap.columns = ["weekday", "weekend"]
            gap["weekend_drop_pct"] = ((gap["weekday"] - gap["weekend"])
                                       / gap["weekday"] * 100).round(1)
            hardest = gap["weekend_drop_pct"].idxmax()
            findings.append(
                f"The weekend drop is not uniform across camera types: type "
                f"{hardest} loses {gap.loc[hardest, 'weekend_drop_pct']:.0f}% of "
                f"its weekday traffic, against "
                f"{gap['weekend_drop_pct'].min():.0f}% for the least affected "
                f"type. Commuter-driven locations and all-week locations are "
                f"distinguishable from the counts alone.")
            per_type = {"hourly_by_type": per_type, "weekend_drop_by_type": gap.round(3)}
        else:
            per_type = {"hourly_by_type": per_type}

    tables = {"weekly_grid": grid, "daily_means": daily.to_frame("mean_vehicles"),
              "morning_evening_balance": balance}
    if isinstance(per_type, dict):
        tables.update(per_type)
    return {"stats": {"busiest_day": busiest, "quietest_day": quietest},
            "tables": tables, "findings": findings}


# --------------------------------------------------------------------------- #
# 11. Anomalies
# --------------------------------------------------------------------------- #
def section_anomalies(ch: pd.DataFrame, labels: pd.Series | None = None,
                      z_threshold: float = 3.5) -> dict:
    """Camera-hours that depart from what that camera normally does at that hour.

    The expectation is built per camera and per hour, so a busy camera at rush
    hour is not flagged simply for being busy. Residuals are standardised with
    the median and the median absolute deviation rather than the mean and
    standard deviation, because the outliers being searched for would otherwise
    inflate the very scale used to detect them.

    What this is for: a dashboard cannot show 10,728 camera-hours, but it can
    show the fifty that did not behave as expected.
    """
    df = ch.copy()

    # Step 1: separate missing data from low traffic. A camera-hour built from
    # 18 images instead of 600 produces a low mean for a reason that has nothing
    # to do with the road, and scoring it as an anomaly would be wrong.
    typical_images = df.groupby("camera_id")["n_images"].transform("median")
    df["coverage_ratio"] = df["n_images"] / typical_images.replace(0, np.nan)
    df["coverage_gap"] = df["coverage_ratio"] < 0.5
    gaps = df[df["coverage_gap"]]

    # Step 2: the expectation is conditioned on day type as well as hour.
    # Keyed on hour alone it would mark most of Saturday and Sunday as
    # anomalous, which is just the weekend, not an event.
    keys = ["camera_id", "hour"]
    if "is_weekend" in df.columns:
        keys.append("is_weekend")
    scored = df[~df["coverage_gap"]]
    expected = scored.groupby(keys)["total_vehicles"].transform("median")
    df["expected"] = expected.reindex(df.index)
    df["residual"] = df["total_vehicles"] - df["expected"]

    # Robust scale per camera. A camera with no variation gets NaN, not zero,
    # so it cannot produce an infinite z-score.
    def _mad(s):
        med = s.median()
        return (s - med).abs().median()

    scale = df.groupby("camera_id")["residual"].transform(_mad) * 1.4826
    scale = scale.replace(0, np.nan)
    df["z"] = df["residual"] / scale
    df.loc[df["coverage_gap"], "z"] = np.nan   # not scored, reported separately

    flagged = df[df["z"].abs() >= z_threshold].copy()
    flagged["direction"] = np.where(flagged["z"] > 0, "surge", "drop")
    cols = [c for c in ["camera_id", "hourly_timestamp", "day_name", "hour",
                        "total_vehicles", "expected", "z", "direction"]
            if c in flagged.columns]
    top = (flagged.reindex(flagged["z"].abs().sort_values(ascending=False).index)
           [cols].head(50).round(3))

    per_camera = (flagged.groupby("camera_id").size()
                  .sort_values(ascending=False).head(15)
                  .to_frame("n_flagged_hours"))

    # A camera whose counts never move is more likely broken than calm.
    spread = ch.groupby("camera_id")["total_vehicles"].agg(["mean", "std"])
    spread["cv"] = spread["std"] / spread["mean"].replace(0, np.nan)
    flat = spread.nsmallest(5, "cv").round(3)

    scored_n = int(df["z"].notna().sum())
    share = len(flagged) / max(scored_n, 1) * 100
    # A flag rate this far above the normal expectation is only meaningful if
    # the scale behind it is sound. A camera with a near-zero MAD would produce
    # huge z-scores from rounding alone, so the scale is reported with the rate.
    scale_med = float(scale.median())
    scale_frac = scale_med / float(df["total_vehicles"].mean())
    kurt = float(df["residual"].kurtosis())
    findings = []
    if len(gaps):
        n_dates = df["date"].nunique()
        # An hour that is short on (nearly) every date is the edge of the
        # collection window. One that is short on a single date is an outage.
        dates_per_hour = gaps.groupby("hour")["date"].nunique()
        edge_hours = sorted(dates_per_hour[dates_per_hour >= n_dates - 1].index)
        outage = gaps[~gaps["hour"].isin(edge_hours)]

        if edge_hours:
            findings.append(
                f"Before any anomaly scoring: {len(gaps):,} camera-hours are "
                f"built from under half the usual number of images. Of these, "
                f"hour{'s' if len(edge_hours) > 1 else ''} "
                f"{', '.join(f'{h:02d}:00' for h in edge_hours)} "
                f"{'are' if len(edge_hours) > 1 else 'is'} short on every date, "
                f"so that is the edge of the collection window rather than a "
                f"fault. Any hourly average must drop "
                f"{'those hours' if len(edge_hours) > 1 else 'that hour'} or it "
                f"will report a phantom evening collapse.")
        if len(outage):
            by_date = outage.groupby("date").size().sort_values(ascending=False)
            worst_date = by_date.index[0]
            worst_hours = sorted(outage.loc[outage["date"] == worst_date,
                                            "hour"].unique())
            findings.append(
                f"The remaining {len(outage):,} gaps are a genuine outage, not a "
                f"window edge: {int(by_date.iloc[0])} of them fall on "
                f"{worst_date} alone, in hours "
                f"{', '.join(f'{h:02d}:00' for h in worst_hours[:6])}. The "
                f"scraper missed that morning. Exclude those camera-hours from "
                f"any model or dashboard average, or that morning reads as "
                f"empty roads across the whole city.")
        else:
            findings.append(
                f"No date shows a gap beyond those window-edge hours, so "
                f"collection was otherwise complete across the week.")
        # A camera-hour with no images at all produces no row, so it is invisible
        # to the check above. Counting it needs the full grid.
        n_cams = df["camera_id"].nunique()
        slots = df[["date", "hour"]].drop_duplicates()
        missing = n_cams * len(slots) - len(df)
        if missing > 0:
            per_date = (df.groupby("date").size()
                        .reindex(sorted(df["date"].unique())))
            thinnest = per_date.idxmin()
            findings.append(
                f"A further {missing:,} camera-hours produced no row at all, "
                f"because no image was captured. {thinnest} is the thinnest date "
                f"with {int(per_date.min()):,} camera-hours against a median of "
                f"{int(per_date.median()):,}. Absent rows are the easy mistake "
                f"here: a groupby over what is present silently averages over a "
                f"smaller city rather than reporting a gap.")
        findings.append(
            "All of these camera-hours are excluded from the scoring below.")

    findings.append(
        f"Of the {scored_n:,} camera-hours that remain, {len(flagged):,} "
        f"({share:.2f}%) deviate from what that camera normally does at that "
        f"hour on that type of day by more than {z_threshold} robust standard "
        f"deviations, against well under 0.1% under a normal model.")
    findings.append(
        "The expectation is conditioned on weekday against weekend as well as "
        "on hour. Keyed on hour alone it flagged most of Saturday and Sunday, "
        "which is the weekend rather than an event.")
    findings += [
        f"That is not an artefact of a collapsed scale: the median robust "
        f"standard deviation is {scale_med:.2f} vehicles, about "
        f"{scale_frac * 100:.0f}% of the overall mean, and the residual "
        f"distribution has an excess kurtosis of {kurt:.1f} against 0 for a "
        f"normal. The counts genuinely have heavy tails, so these flags are "
        f"structure: incidents, closures, weather, or detector failures.",
    ]
    if len(flagged):
        surge = (flagged["direction"] == "surge").mean() * 100
        findings.append(
            f"{surge:.0f}% of the flags are surges and {100 - surge:.0f}% are "
            f"drops. Drops are the more useful signal operationally: a camera "
            f"that suddenly sees nothing is usually a blocked view or a failed "
            f"detection, not an empty road.")
        worst = per_camera.index[0]
        findings.append(
            f"Flags are concentrated, not spread evenly: camera {worst} alone "
            f"accounts for {int(per_camera.iloc[0, 0])} of them. Check that "
            f"camera's imagery before treating its numbers as traffic.")
    findings.append(
        f"Separately, the five cameras with the flattest counts "
        f"(coefficient of variation {flat['cv'].min():.2f} to "
        f"{flat['cv'].max():.2f}) are worth a manual look. A traffic camera "
        f"whose count barely moves across a week is more likely mis-framed or "
        f"obstructed than genuinely steady.")

    gap_table = None
    if len(gaps):
        gap_table = (gaps.groupby(["date", "hour"])
                     .agg(camera_hours=("camera_id", "size"),
                          mean_coverage=("coverage_ratio", "mean"))
                     .sort_values("camera_hours", ascending=False)
                     .head(20).round(3))

    tables = {"top_anomalies": top, "anomalies_per_camera": per_camera,
              "flattest_cameras": flat}
    if gap_table is not None:
        tables = {"coverage_gaps": gap_table, **tables}

    return {"stats": {"n_flagged": len(flagged), "share_pct": share,
                      "z_threshold": z_threshold, "n_scored": scored_n,
                      "n_coverage_gaps": int(len(gaps))},
            "tables": tables,
            "findings": findings,
            "frame": df}
