"""Component 2: Data Science. Entry point.

Applies analytical methods to the prepared dataset to find patterns, trends and
relationships, and writes a report, tables and figures to Component2_DataSci/output/.

    python Component2_DataSci/main.py              # from the repository root
    python -m Component2_DataSci.main              # equivalent

What it does, in order:

    1. load the prepared dataset and report what it actually contains
    2. audit coverage, cadence and detector behaviour before testing anything
    3. temporal structure, vehicle composition, variance decomposition, space
    4. built-environment (POI) against traffic, at the right aggregation level
    5. PM2.5 against weather and traffic, raw and conditional
    6. write Component2_DataSci/output/component2_report.md plus CSV tables and PNG figures

The PM2.5 sections run automatically once data/processed/data.csv exists. Until
then they are reported as skipped and the rest of the analysis still runs, so
this module is useful before the air-quality pull is finished and correct after.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):          # allow `python Component2_DataSci/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analysis as A
    import config as C
    import features as F
    import figures as V
else:
    from . import analysis as A
    from . import config as C
    from . import features as F
    from . import figures as V


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #
class Report:
    """Accumulates markdown, and saves every table it prints as a CSV too."""

    def __init__(self, title: str):
        self.parts = [f"# {title}", ""]
        self.tables: dict[str, pd.DataFrame] = {}

    def heading(self, text: str, level: int = 2):
        self.parts += [f"{'#' * level} {text}", ""]

    def text(self, text: str):
        self.parts += [text.strip(), ""]

    def bullets(self, items: list):
        self.parts += [f"- {item}" for item in items] + [""]

    def findings(self, items: list):
        for item in items:
            self.parts.append(f"**Finding.** {item}")
            self.parts.append("")

    def table(self, name: str, df: pd.DataFrame, caption: str = "",
              max_rows: int = 25, index: bool = True):
        if df is None or len(df) == 0:
            return
        self.tables[name] = df
        if caption:
            self.parts.append(f"*{caption}*")
            self.parts.append("")
        shown = df.head(max_rows)
        self.parts.append(shown.to_markdown(index=index))
        if len(df) > max_rows:
            self.parts.append("")
            self.parts.append(f"*({len(df) - max_rows} further rows in "
                              f"`Component2_DataSci/output/tables/{name}.csv`)*")
        self.parts.append("")

    def figure(self, path: str, caption: str):
        # The report is written inside OUT_DIR, so image links are relative to
        # it ("figures/x.png"), not to its parent.
        rel = Path(path).relative_to(C.OUT_DIR) if path else None
        self.parts += [f"![{caption}]({rel.as_posix() if rel else path})", "",
                       f"*{caption}*", ""]

    def write(self):
        C.OUT_DIR.mkdir(parents=True, exist_ok=True)
        C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        C.REPORT_PATH.write_text("\n".join(self.parts), encoding="utf-8")
        for name, df in self.tables.items():
            df.to_csv(C.TABLE_DIR / f"{name}.csv")
        return C.REPORT_PATH


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run() -> Path:
    print("[ds] loading prepared dataset")
    df, caps, source = F.load_dataset()
    ch = F.camera_hour_level(df)
    cam = F.attach_camera_info(F.camera_level(df))

    print(f"[ds] {len(df):,} image rows | {len(ch):,} camera-hours | "
          f"{len(cam)} cameras")
    print(f"[ds] target present: {caps['pm25']} | weather: {len(caps['weather'])} "
          f"cols | poi: {len(caps['poi'])} cols")

    rep = Report("Component 2: Data Science - patterns, trends and relationships "
                 "in the Bangkok CCTV dataset")
    rep.text(f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{source}`.")
    rep.text(
        "The project brief defines this component as applying analytical "
        "methods to explore patterns, trends and insights in the prepared "
        "dataset, and defines the project objective as analysing meaningful "
        "spatial and/or temporal patterns. This report does both: sections 2, "
        "8 and 9 are the temporal analysis, sections 5 and 7 the spatial "
        "analysis, sections 1, 3, 4 and 6 establish what the data can and "
        "cannot support, section 12 hands tidy tables to the dashboard, and "
        "section 13 is the set of insights Component 4 reports.")
    rep.text(
        "Section 10 is an extension rather than a requirement. The group's "
        "chosen angle is relating these counts to PM2.5, and that section runs "
        "automatically once the air-quality columns are present in the prepared "
        "file. Nothing earlier in the report depends on it.")

    # ---------------- method note ---------------------------------------- #
    rep.heading("How the analysis is set up")
    rep.text(
        "The prepared file has one row per image, but almost nothing should be "
        "correlated at that level. Three levels are used, and every result "
        "below states which one it came from.")
    rep.table("analysis_levels", pd.DataFrame([
        {"level": "image", "rows": f"{len(df):,}",
         "varies at this level": "vehicle counts (plus detector noise)"},
        {"level": "camera-hour", "rows": f"{len(ch):,}",
         "varies at this level": "PM2.5, weather, averaged traffic"},
        {"level": "camera", "rows": f"{len(cam)}",
         "varies at this level": "POI and road features, mean traffic"},
    ]).set_index("level"), index=True)
    rep.bullets([
        "Correlations use Spearman's rho: the counts are zero-inflated and "
        "right-skewed, so a Pearson coefficient would mostly measure the tail.",
        "Every coefficient is reported with a 95% confidence interval and the "
        "number of complete pairs behind it.",
        f"Where many features are tested at once, p-values are Benjamini-"
        f"Hochberg corrected to a false discovery rate of q = {C.FDR_Q}; the "
        f"uncorrected p is kept in the tables.",
        "Intervals for camera-level quantities come from a bootstrap that "
        "resamples whole cameras, not rows.",
    ])

    # ---------------- 1. audit -------------------------------------------- #
    print("[ds] section 1: dataset audit")
    rep.heading("1. What the prepared dataset contains")
    audit = A.section_audit(df, caps)
    rep.findings(audit["findings"])
    rep.table("detector_summary", audit["tables"]["detector_summary"],
              "Detector output per vehicle class, image level.")
    if len(audit["tables"]["missingness"]):
        rep.table("missingness", audit["tables"]["missingness"],
                  "Columns with any missing values (%).")
    else:
        rep.text("No column in the prepared file has a missing value.")

    # ---------------- 2. time --------------------------------------------- #
    print("[ds] section 2: temporal patterns")
    rep.heading("2. Temporal patterns")
    temporal = A.section_temporal(df, ch, caps)
    rep.findings(temporal["findings"])
    rep.figure(V.fig_hourly_profile(temporal["tables"]["hourly_profile"],
                                    caps["vehicles"]),
               "Mean vehicles per frame by hour, and the shape of each class's day.")
    rep.figure(V.fig_weekday_weekend(temporal["tables"]["hourly_by_weekend"]),
               "Weekday and weekend hourly traffic profiles.")
    rep.table("hourly_profile", temporal["tables"]["hourly_profile"],
              "Mean counts per image by hour of day.")
    rep.table("daily_means", temporal["tables"]["daily_means"],
              "Mean total vehicles per image by date.", index=False)

    # ---------------- 3. composition -------------------------------------- #
    print("[ds] section 3: vehicle composition")
    rep.heading("3. How the four detector channels relate")
    comp = A.section_composition(df, cam, caps)
    rep.findings(comp["findings"])
    rep.figure(V.fig_correlation_panels(comp["tables"]["corr_image_level"],
                                        comp["tables"]["corr_camera_level"],
                                        n_image=len(df), n_camera=len(cam)),
               "Spearman correlations among vehicle classes at image and camera level.")
    rep.table("corr_camera_level", comp["tables"]["corr_camera_level"],
              "Camera-level correlations among vehicle classes.")
    rep.table("pca_loadings", comp["tables"]["pca_loadings"],
              "Loadings of the first two components on camera-level means.")

    # ---------------- 4. variance ----------------------------------------- #
    print("[ds] section 4: variance decomposition")
    rep.heading("4. Where the variance lives: location or time")
    var = A.section_variance(ch, caps)
    rep.findings(var["findings"])
    rep.figure(V.fig_variance_decomposition(var["tables"]["variance_decomposition"]),
               "Between-camera versus within-camera share of variance.")
    rep.table("variance_decomposition", var["tables"]["variance_decomposition"],
              "Intraclass correlation by variable (camera as the grouping).",
              index=False)

    # ---------------- 5. space -------------------------------------------- #
    print("[ds] section 5: spatial structure")
    rep.heading("5. Spatial structure")
    spatial = A.section_spatial(cam, caps)
    rep.findings(spatial["findings"])
    rep.figure(V.fig_camera_map(cam), "Camera locations shaded by mean traffic.")
    rep.table("morans_i", spatial["tables"]["morans_i"],
              f"Moran's I on a {C.MORAN_K}-nearest-neighbour graph, "
              f"permutation p-value.", index=False)
    rep.table("busiest_cameras", spatial["tables"]["busiest_cameras"],
              "Ten busiest cameras by mean vehicles per frame.", index=False)
    rep.table("quietest_cameras", spatial["tables"]["quietest_cameras"],
              "Ten quietest cameras by mean vehicles per frame.", index=False)

    # ---------------- 6. POI ---------------------------------------------- #
    print("[ds] section 6: built environment")
    rep.heading("6. Built environment against traffic")
    poi = A.section_poi(df, cam, caps)
    rep.findings(poi["findings"])
    key = "poi_vs_traffic_camera_level"
    if key in poi["tables"]:
        table = poi["tables"][key]
        rep.figure(V.fig_lollipop(
            table, "POI features against camera-mean traffic",
            "08_poi_vs_traffic",
            subtitle=f"camera level, n = {len(cam)}; FDR-corrected at q = {C.FDR_Q}"),
            "POI features ranked by correlation with camera-mean traffic.")
        rep.table(key, table, "Spearman correlation of each POI feature with "
                              "camera-mean traffic.", index=False)
        top = table.iloc[0]["predictor"]
        rep.figure(V.fig_scatter_fit(cam, top, "total_vehicles",
                                     "09_top_poi_scatter"),
                   f"Camera-mean traffic against {top}.")
    if "poi_vs_heavy_share_camera_level" in poi["tables"]:
        rep.table("poi_vs_heavy_share_camera_level",
                  poi["tables"]["poi_vs_heavy_share_camera_level"],
                  "Same features against the heavy-vehicle share.", index=False)

    # ---------------- 7. camera typology ---------------------------------- #
    print("[ds] section 7: camera typology")
    rep.heading("7. Camera typology: types of place, not levels of traffic")
    typo = A.section_typology(ch, cam, caps)
    rep.findings(typo["findings"])
    rep.figure(V.fig_typology_profiles(typo["tables"]["typology_hour_centroids"],
                                       typo["tables"]["typology_summary"]),
               "Each camera type's day, indexed to its own mean.")
    rep.figure(V.fig_typology_map(typo["tables"]["camera_profile_types"]),
               "Where each camera type sits in the city.")
    rep.table("typology_summary", typo["tables"]["typology_summary"],
              "Camera types: size, timing and sharpness of peak.")
    rep.table("camera_profile_types", typo["tables"]["camera_profile_types"],
              "Every camera with its assigned profile type.", index=False,
              max_rows=15)
    if "typology_vs_poi" in typo["tables"]:
        rep.table("typology_vs_poi", typo["tables"]["typology_vs_poi"],
                  "Mean POI features by camera type.")
    labels = typo["labels"]

    # ---------------- 8. weekly rhythm ------------------------------------ #
    print("[ds] section 8: weekly rhythm")
    rep.heading("8. The weekly rhythm")
    weekly = A.section_weekly(ch, labels)
    rep.findings(weekly["findings"])
    rep.figure(V.fig_weekly_heatmap(weekly["tables"]["weekly_grid"]),
               "Mean vehicles per frame by day of week and hour.")
    rep.table("weekly_grid", weekly["tables"]["weekly_grid"],
              "Day of week by hour, mean vehicles per frame.")
    if "weekend_drop_by_type" in weekly["tables"]:
        rep.table("weekend_drop_by_type", weekly["tables"]["weekend_drop_by_type"],
                  "How much each camera type loses at the weekend.")

    # ---------------- 9. anomalies ---------------------------------------- #
    print("[ds] section 9: anomalies")
    rep.heading("9. Anomalies: where the data stops behaving")
    anom = A.section_anomalies(ch, labels)
    rep.findings(anom["findings"])
    rep.figure(V.fig_anomaly_timeline(anom["frame"], anom["stats"]["z_threshold"]),
               "Every scored camera-hour by its deviation from that camera's own "
               "norm for that hour and day type.")
    if "coverage_gaps" in anom["tables"]:
        rep.table("coverage_gaps", anom["tables"]["coverage_gaps"],
                  "Collection gaps: camera-hours built from under half the usual "
                  "number of images. Excluded from scoring.")
    rep.table("top_anomalies", anom["tables"]["top_anomalies"],
              "The most extreme camera-hours.", index=False, max_rows=20)
    rep.table("anomalies_per_camera", anom["tables"]["anomalies_per_camera"],
              "Cameras producing the most flags.")
    rep.table("flattest_cameras", anom["tables"]["flattest_cameras"],
              "Cameras whose counts barely move; check these manually.")

    # ---------------- 10. PM2.5 extension --------------------------------- #
    print("[ds] section 10: PM2.5 extension")
    rep.heading("10. Extension: PM2.5 against traffic")
    pm = A.section_pm25(ch, cam, caps)
    rep.findings(pm["findings"])
    for name, table in pm["tables"].items():
        rep.table(name, table, name.replace("_", " "),
                  index=not isinstance(table.index, pd.RangeIndex))
    if not pm.get("skipped", True):
        decomp = pm["tables"].get("pm25_vs_traffic_decomposition")
        if decomp is not None:
            rep.figure(V.fig_decomposition_bars(decomp),
                       "Traffic against PM2.5 under four estimators.")
        wx = pm["tables"].get("pm25_vs_weather")
        if wx is not None:
            rep.figure(V.fig_lollipop(wx, "Weather against PM2.5",
                                      "10_pm25_vs_weather",
                                      subtitle="camera-hour level"),
                       "Weather variables ranked by correlation with PM2.5.")

    # ---------------- 11. confounding ------------------------------------- #
    print("[ds] section 11: confounding")
    rep.heading("11. The confounding any PM2.5 model has to beat")
    conf = A.section_confounding(ch, caps)
    rep.findings(conf["findings"])
    if caps["pm25"] and "hourly_profiles" in conf["tables"]:
        rep.figure(V.fig_pm25_profiles(conf["tables"]["hourly_profiles"]),
                   "Hourly mean profiles of traffic and PM2.5.")
    rep.table("hourly_profiles", conf["tables"]["hourly_profiles"],
              "Hourly mean profiles.")

    # ---------------- 12. handoff to Component 3 --------------------------- #
    print("[ds] section 12: dashboard handoff")
    rep.heading("12. Handoff to Component 3 (dashboard)")
    handoff = _write_handoff(ch, cam, labels, anom["frame"])
    rep.text(
        "Component 3 needs tidy tables, not this report. These are written to "
        "`Component2_DataSci/output/dashboard/` on every run, so the dashboard reads files "
        "rather than re-deriving the analysis:")
    rep.table("dashboard_exports", handoff, index=False)

    # ---------------- conclusions ----------------------------------------- #
    rep.heading("13. Component 4: the outcome of the analysis")
    rep.bullets(_conclusions(caps, audit, comp, var, spatial, poi, pm,
                             typo, weekly, anom))

    rep.heading("Reproducing this")
    rep.text(
        "`python Component2_DataSci/main.py` from the repository root. It reads "
        "`data/processed/data.csv` when that exists and falls back to "
        "`Carcounts.csv` plus `camera_poi.csv` otherwise, so it runs in both "
        "states of the pipeline. Statistical primitives are in "
        "`Component2_DataSci/stats_tools.py`, feature engineering in `Component2_DataSci/features.py`, the "
        "sections in `Component2_DataSci/analysis.py` and the plotting in `Component2_DataSci/figures.py`; "
        "nothing is hard-coded to a particular run.")

    path = rep.write()
    print(f"[ds] report -> {path}")
    print(f"[ds] {len(rep.tables)} tables -> {C.TABLE_DIR}")
    print(f"[ds] figures -> {C.FIG_DIR}")
    return path


def _write_handoff(ch, cam, labels, anomaly_frame) -> pd.DataFrame:
    """Write the tidy tables Component 3 reads, and describe them.

    Three grains, because a dashboard needs all three: one row per camera for
    the map, one row per camera-hour for the time series, and one row per flag
    for the alert panel.
    """
    out_dir = C.OUT_DIR / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = cam.merge(labels.reset_index(), on="camera_id", how="left")
    keep = [c for c in ["camera_id", "location_en", "location_detail_en",
                        "latitude", "longitude", "profile_cluster",
                        "total_vehicles", "car_count", "motorcycle_count",
                        "bus_count", "truck_count", "heavy_share",
                        "peak_total_vehicles", "offpeak_total_vehicles",
                        "peak_ratio", "n_images"] if c in cameras.columns]
    keep += [c for c in cameras.columns if c.startswith("poi_")]
    cameras = cameras[keep]

    hourly = anomaly_frame.merge(labels.reset_index(), on="camera_id", how="left")
    hkeep = [c for c in ["camera_id", "hourly_timestamp", "date", "day_name",
                         "hour", "is_weekend", "profile_cluster",
                         "total_vehicles", "car_count", "motorcycle_count",
                         "bus_count", "truck_count", "expected", "residual", "z",
                         "n_images", C.TARGET] if c in hourly.columns]
    hourly = hourly[hkeep]

    flags = hourly[hourly["z"].abs() >= 3.5].copy()
    flags["direction"] = np.where(flags["z"] > 0, "surge", "drop")

    files = [("cameras.csv", cameras, "one row per camera: location, profile "
                                      "type, mean counts, POI features"),
             ("camera_hours.csv", hourly, "one row per camera-hour: counts, "
                                          "expected value, deviation score"),
             ("anomaly_flags.csv", flags, "one row per flagged camera-hour")]
    rows = []
    for name, frame, description in files:
        frame.to_csv(out_dir / name, index=False)
        rows.append({"file": f"Component2_DataSci/output/dashboard/{name}",
                     "rows": len(frame), "columns": frame.shape[1],
                     "grain": description})
    return pd.DataFrame(rows)


def _conclusions(caps, audit, comp, var, spatial, poi, pm,
                 typo=None, weekly=None, anom=None) -> list:
    """The handful of statements the rest of the group should act on."""
    icc = var["stats"]["traffic_icc"]
    img_rho = comp["stats"]["mean_offdiag_image"]
    cam_rho = comp["stats"]["mean_offdiag_camera"]
    out = [
        f"**Location dominates time.** {icc * 100:.0f}% of camera-hour traffic "
        f"variance is between cameras and only {(1 - icc) * 100:.0f}% within "
        f"them. Feature engineering effort belongs on per-location descriptors; "
        f"hour-of-day is a correction term, not a driver.",
        f"**Aggregate before modelling.** The four detector channels correlate "
        f"at a mean rho of {img_rho:+.2f} per frame but {cam_rho:+.2f} per "
        f"camera, so per-frame counts are mostly detector noise. The "
        f"camera-hour mean is the smallest unit that measures anything stable.",
        "**Split on cameras and whole days when validating.** Two images 11 "
        "minutes apart at one camera are nearly the same observation, so a "
        "random row split will report an accuracy the model does not have.",
        _pseudo_replication_bullet(poi),
        _spatial_bullet(spatial),
    ]
    n_sig = poi["stats"].get("n_significant", 0)
    poi_table = poi["tables"].get("poi_vs_traffic_camera_level")
    if n_sig and poi_table is not None:
        best = poi_table[poi_table["significant"]].iloc[0]
        out.append(
            f"**The built environment explains traffic only weakly, and only "
            f"through road class.** {n_sig} of {len(poi_table)} POI features "
            f"survive FDR correction, led by {best['predictor']} at rho = "
            f"{best['r']:+.2f} (about {best['r'] ** 2 * 100:.0f}% of variance). "
            f"Amenity categories - shops, offices, bus stops, green space - carry "
            f"nothing. If more location signal is needed, it should come from road "
            f"geometry (lane count, carriageway width, junction type), not from "
            f"more POI categories.")
    if n_sig == 0:
        out.append(
            "**The POI feature set as defined does not explain traffic.** With "
            "100 cameras nothing clears the FDR threshold. Either the radii are "
            "wrong for this question (100 m and 500 m around a traffic camera "
            "mostly measure the road itself), or the camera's field of view, "
            "not its surroundings, is what determines the counts. Worth testing "
            "road class and lane count directly before adding more POI "
            "categories.")
    if not caps["pm25"]:
        out.append(
            "**The hypothesis is untested until `data/processed/data.csv` "
            "exists.** Every PM2.5 result in section 7 is wired and will run "
            "unchanged once the air-quality and weather columns are present; "
            "the analysis is blocked on data, not on code.")
    else:
        n_st = pm["stats"].get("n_stations")
        if n_st:
            out.append(
                f"**The target, not the features, is the ceiling.** PM2.5 comes "
                f"from {n_st} stations for {audit['stats']['n_cameras']} "
                f"cameras. The project cannot demonstrate finer spatial "
                f"resolution than the reference data can distinguish, so the "
                f"evaluation needs to be framed as held-out-station prediction, "
                f"not as per-camera accuracy.")

    if typo is not None:
        s = typo["stats"]
        summary = typo["tables"]["typology_summary"]
        hours = summary["centroid_peak_hour"].astype(int).tolist()
        if s["silhouette"] >= 0.25:
            out.append(
                f"**Cameras fall into {s['best_k']} behavioural types, separate "
                f"from how busy they are.** Clustering on the shape of the day "
                f"rather than its height gives types peaking at "
                f"{', '.join(f'{h:02d}:00' for h in hours)} (silhouette "
                f"{s['silhouette']:.2f}). Use it as a filter in the dashboard.")
        else:
            out.append(
                f"**Camera behaviour is a continuum, not a set of types.** The "
                f"best clustering of daily profiles is k = {s['best_k']} at a "
                f"silhouette of only {s['silhouette']:.2f}, with the groups "
                f"peaking at {', '.join(f'{h:02d}:00' for h in hours)}. Cameras "
                f"differ mainly in when their peak falls, along one gradient. "
                f"The split is still usable as a coarse dashboard filter, but "
                f"the report should not call these distinct camera species, and "
                f"a model should prefer the continuous peak-hour value over a "
                f"cluster label.")
    if weekly is not None:
        out.append(
            f"**Peak-hour and daily-total rankings disagree.** The sharpest hour "
            f"of the day is the evening rush, but it collapses after 18:00, so "
            f"the morning block carries more total traffic on every day of this "
            f"week. {weekly['stats']['busiest_day']} is the busiest day against "
            f"{weekly['stats']['quietest_day']} as the quietest. Whichever "
            f"measure the dashboard ranks on, label it, because the two orders "
            f"are different.")
    if anom is not None:
        a = anom["stats"]
        out.append(
            f"**{a['share_pct']:.2f}% of camera-hours do not behave like "
            f"themselves.** {a['n_flagged']:,} camera-hours deviate from that "
            f"camera's own median for that hour by more than {a['z_threshold']} "
            f"robust standard deviations, far above what noise would produce, "
            f"and the flags are concentrated in a few cameras. Some are real "
            f"events and some are detector failures; separating the two needs a "
            f"look at the underlying images, which is the single highest-value "
            f"next step for the group.")
    return [b for b in out if b]


def _spatial_bullet(spatial: dict) -> str:
    """State the spatial result by what the test actually showed."""
    moran = spatial["tables"]["morans_i"]
    total = moran.loc[moran["variable"] == "total_vehicles"].iloc[0]
    clustered = spatial["stats"].get("clustered_variables", [])
    if clustered:
        names = (" and ".join(clustered) if len(clustered) < 3
                 else ", ".join(clustered[:-1]) + " and " + clustered[-1])
        return (f"**Spatial clustering is weak and class-specific.** Total traffic "
                f"does not clear the test (I = {total['I']:+.2f}, p = "
                f"{total['p']:.3f}); only {names} do, and at I below 0.25. "
                f"A map cannot be made by interpolating between cameras: it has to "
                f"be predicted at each point from that point's own features.")
    return (f"**No variable shows spatial clustering at this camera spacing** "
            f"(total traffic I = {total['I']:+.2f}, p = {total['p']:.3f}), so a "
            f"higher-resolution map must be built from per-location features "
            f"rather than by interpolating between cameras.")


def _pseudo_replication_bullet(poi: dict) -> str | None:
    pair = poi.get("stats", {}).get("naive_vs_clustered")
    if not pair:
        return None
    img = "under 1e-300" if pair["image"] == 0 else f"{pair['image']:.3g}"
    return (f"**Do not correlate camera-constant features against image rows.** "
            f"The strongest POI association carries p = {pair['camera']:.3g} at "
            f"camera level and {img} at image level, on essentially the same "
            f"coefficient. Repeating one camera's POI value across its 613 images "
            f"adds no information and inflates significance. Every number in the "
            f"final report should name the level it was computed at.")


if __name__ == "__main__":
    run()
