"""Figures for the Component 2 report.

Design rules applied throughout, so the plots read as one set:

* Categorical hues are assigned from a fixed, colour-vision-safe order and are
  never cycled. A fifth series folds into "other" instead of inventing a hue.
* Correlations carry a sign, so they get a diverging blue/red scale with a
  neutral grey midpoint. Magnitudes get a single-hue blue ramp. No rainbows.
* One y-axis per panel. Two measures on different scales become two panels,
  never two axes on one plot.
* Every chart with two or more series has a legend and, at four or fewer,
  direct labels as well, so identity never depends on colour alone.
* Grid and axis lines are recessive hairlines; the data is the darkest thing in
  the frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from . import config as C
except ImportError:
    import config as C

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical order. Index 0 first, always.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
POS, NEG = "#2a78d6", "#e34948"          # diverging poles
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#0d366b", "#256abf", "#86b6ef", "#f0efec",
                      "#ec835a", "#d03b3b", "#8f1f1f"])

BASE_RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11.5, "axes.labelsize": 9.5, "axes.titlepad": 10,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "lines.markersize": 5.0,
    "font.size": 9.5, "figure.dpi": 110,
}
plt.rcParams.update(BASE_RC)


def _clean(ax, grid_axis: str = "y"):
    """Recessive chrome: no box, hairline grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis=grid_axis, linewidth=0.8, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    return ax


def _save(fig, name: str) -> str:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _label_ends(ax, x, items, dx=0.18, min_sep_frac=0.062):
    """Direct labels at the right-hand end of several lines, de-collided.

    Labels are nudged apart vertically until they clear each other, so a chart
    with four series does not print two names on top of one another. The line
    end itself is not moved, only the text.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    min_sep = (hi - lo) * min_sep_frac
    placed = sorted(items, key=lambda it: it[0])

    positions = [it[0] for it in placed]
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_sep:
            positions[i] = positions[i - 1] + min_sep
    # Re-centre the block if pushing upwards overflowed the axis.
    overflow = positions[-1] - (hi - min_sep * 0.3)
    if overflow > 0:
        positions = [p - overflow for p in positions]

    for (y_true, text, colour), y_text in zip(placed, positions):
        ax.annotate(text, xy=(x, y_true), xytext=(x + dx, y_text), color=colour,
                    fontsize=8.5, va="center", ha="left", fontweight="bold",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.8,
                                    alpha=0.55, shrinkA=0, shrinkB=2)
                    if abs(y_text - y_true) > min_sep * 0.35 else None)


# --------------------------------------------------------------------------- #
# 1. Daily cycle
# --------------------------------------------------------------------------- #
def fig_hourly_profile(hourly: pd.DataFrame, vehicle_cols: list) -> str:
    """Total traffic by hour, and each class indexed to its own mean.

    Two panels rather than two y-axes: the classes differ by an order of
    magnitude in level, and indexing is the only honest way to compare their
    *shapes* on one scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = _clean(axes[0])
    h = hourly.index.to_numpy()
    ax.plot(h, hourly["total_vehicles"], color=SERIES[0], solid_capstyle="round")
    ax.scatter(h, hourly["total_vehicles"], s=28, color=SERIES[0],
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
    peak = int(hourly["total_vehicles"].idxmax())
    ax.annotate(f"peak {peak:02d}:00", xy=(peak, hourly["total_vehicles"].max()),
                xytext=(0, 10), textcoords="offset points", ha="center",
                color=INK_2, fontsize=8.5, fontweight="bold")
    ax.set_title("Total vehicles per frame, by hour")
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("mean vehicles per image")
    ax.set_xticks(h[::2])

    ax = _clean(axes[1])
    ends = []
    for i, col in enumerate(vehicle_cols[:4]):
        series = hourly[col] / hourly[col].mean()
        ax.plot(h, series, color=SERIES[i], solid_capstyle="round",
                label=col.replace("_count", ""))
        ends.append((series.iloc[-1], col.replace("_count", ""), SERIES[i]))
    ax.axhline(1.0, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_title("Shape of each class's day, indexed to its own mean")
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("ratio to class mean")
    ax.set_xticks(h[::2])
    ax.set_xlim(h[0] - 0.3, h[-1] + 2.8)
    ax.legend(loc="upper left", ncol=2)
    _label_ends(ax, h[-1], ends)

    fig.suptitle("Traffic follows a shallow daily cycle, and the classes do not share it",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "01_hourly_profile")


def fig_weekday_weekend(by_hour: pd.DataFrame) -> str:
    """Weekday against weekend, same axis, two labelled series."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _clean(ax)
    h = by_hour.index.to_numpy()
    ends = []
    for i, col in enumerate(by_hour.columns):
        ax.plot(h, by_hour[col], color=SERIES[i], solid_capstyle="round", label=col)
        ax.scatter(h, by_hour[col], s=24, color=SERIES[i], edgecolor=SURFACE,
                   linewidth=1.2, zorder=3)
        ends.append((by_hour[col].iloc[-1], col, SERIES[i]))
    ax.set_title("Weekday and weekend traffic by hour")
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("mean vehicles per image")
    ax.set_xticks(h[::2])
    ax.set_xlim(h[0] - 0.3, h[-1] + 2.6)
    ax.legend(loc="upper left")
    _label_ends(ax, h[-1], ends)
    fig.tight_layout()
    return _save(fig, "02_weekday_weekend")


# --------------------------------------------------------------------------- #
# 2. Correlation structure
# --------------------------------------------------------------------------- #
def fig_correlation_panels(corr_image: pd.DataFrame, corr_camera: pd.DataFrame,
                           n_image: int = 0, n_camera: int = 0) -> str:
    """The same four channels at two aggregation levels, one shared colour scale.

    A shared scale is the point of the figure: the left panel is nearly blank
    and the right panel is not.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    titles = [f"Image level ({n_image:,} frames)" if n_image else "Image level",
              f"Camera level ({n_camera} cameras)" if n_camera else "Camera level"]

    for ax, corr, title in zip(axes, [corr_image, corr_camera], titles):
        labels = [c.replace("_count", "") for c in corr.columns]
        im = ax.imshow(corr.to_numpy(float), cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = corr.to_numpy(float)[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if abs(v) > 0.55 else INK,
                        fontweight="bold")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("Spearman rho", color=INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=MUTED, labelcolor=MUTED, labelsize=8)
    fig.suptitle("Correlations among vehicle classes strengthen when frames are "
                 "averaged over a camera",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.01, y=1.02,
                 ha="left")
    return _save(fig, "03_correlation_levels")


def fig_lollipop(table: pd.DataFrame, title: str, name: str,
                 subtitle: str = "", max_rows: int = 20) -> str:
    """Ranked correlations as a lollipop chart.

    A lollipop rather than bars: the marks are thin, the dot gives the rounded
    data-end, and the zero line stays visible through the whole stack. Sign is
    carried by the diverging pair, and significance by a filled versus hollow
    dot, so colour is never the only channel.
    """
    df = table.head(max_rows).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, max(3.2, 0.33 * len(df) + 1.6)))
    _clean(ax, grid_axis="x")

    y = np.arange(len(df))
    for yi, row in zip(y, df.itertuples()):
        colour = POS if row.r >= 0 else NEG
        ax.plot([0, row.r], [yi, yi], color=colour, linewidth=2.0,
                solid_capstyle="round", zorder=2)
        sig = bool(getattr(row, "significant", False))
        ax.scatter([row.r], [yi], s=58, zorder=3,
                   color=colour if sig else SURFACE,
                   edgecolor=colour, linewidth=1.8)
        if np.isfinite(getattr(row, "ci_low", np.nan)):
            ax.plot([row.ci_low, row.ci_high], [yi, yi], color=colour,
                    linewidth=0.9, alpha=0.45, zorder=1, solid_capstyle="round")

    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yticks(y, [p.replace("poi_", "").replace("_", " ") for p in df["predictor"]])
    ax.set_xlabel("Spearman rho (thin line = 95% CI)")
    ax.set_ylim(-0.7, len(df) - 0.3)
    # Title above subtitle, both left-aligned, with enough pad that neither the
    # subtitle nor the top row of marks can collide with them.
    ax.set_title(title, loc="left", pad=26 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.006), xycoords="axes fraction",
                    color=INK_2, fontsize=9, va="bottom", ha="left")

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7.5,
                          markerfacecolor=POS, markeredgecolor=POS,
                          label=f"survives FDR q = {C.FDR_Q}"),
               plt.Line2D([], [], marker="o", linestyle="none", markersize=7.5,
                          markerfacecolor=SURFACE, markeredgecolor=MUTED,
                          label="does not survive")]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# 3. Variance and space
# --------------------------------------------------------------------------- #
def fig_variance_decomposition(table: pd.DataFrame) -> str:
    """Between-camera versus within-camera share, as stacked proportion bars.

    A 2px surface gap separates the two segments so the boundary is a real edge
    rather than a colour change.
    """
    df = table.dropna(subset=["icc"]).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, max(3.0, 0.4 * len(df) + 1.4)))
    _clean(ax, grid_axis="x")

    y = np.arange(len(df))
    between = df["icc"].to_numpy(float)
    gap = 0.006
    ax.barh(y, between, color=SERIES[0], height=0.58, zorder=2)
    ax.barh(y, 1 - between - gap, left=between + gap, color=SEQ[1],
            height=0.58, zorder=2)

    for yi, b in zip(y, between):
        ax.text(min(b + 0.015, 0.9), yi, f"{b * 100:.0f}%", va="center",
                ha="left", fontsize=8.5, color=INK, fontweight="bold")

    ax.set_yticks(y, [v.replace("_", " ") for v in df["variable"]])
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("share of variance")
    ax.set_title("Where each variable's variance lives")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label="between cameras (location)"),
               plt.Rectangle((0, 0), 1, 1, color=SEQ[1], label="within camera (time)")]
    # Below the axis: inside the plot the legend would sit on top of the bars.
    ax.legend(handles=handles, loc="upper right", ncol=2,
              bbox_to_anchor=(1.0, -0.16), borderaxespad=0)
    fig.tight_layout()
    return _save(fig, "04_variance_decomposition")


def fig_camera_map(cam: pd.DataFrame, value: str = "total_vehicles") -> str:
    """Cameras in space, shaded by magnitude on a single-hue ramp.

    Sequential, not categorical: the variable is a magnitude. Size repeats the
    encoding so the lightest steps stay findable.
    """
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    _clean(ax, grid_axis="both")
    v = cam[value].to_numpy(float)
    norm = (v - v.min()) / max(float(np.ptp(v)), 1e-9)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    sc = ax.scatter(cam["longitude"], cam["latitude"], c=v, cmap=cmap,
                    s=40 + 150 * norm, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    # A degree of longitude is shorter than a degree of latitude; without this
    # the city is stretched east-west and distances are misread.
    ax.set_aspect(1.0 / np.cos(np.radians(float(cam["latitude"].mean()))))
    ax.set_title("Camera locations, shaded by mean traffic")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("mean vehicles per image", color=INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=MUTED, labelcolor=MUTED, labelsize=8)
    fig.tight_layout()
    return _save(fig, "05_camera_map")


def fig_scatter_fit(cam: pd.DataFrame, x: str, y: str, name: str,
                    title: str = "") -> str:
    """One scatter with a least-squares line, for a single camera-level pair."""
    sub = cam[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _clean(ax, grid_axis="both")
    ax.scatter(sub[x], sub[y], s=42, color=SERIES[0], alpha=0.75,
               edgecolor=SURFACE, linewidth=1.0, zorder=3)
    if len(sub) > 3 and sub[x].std() > 0:
        b, a = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        ax.plot(xs, a + b * xs, color=SERIES[1], linewidth=2.0,
                solid_capstyle="round", zorder=4,
                label=f"OLS fit (slope {b:+.3f})")
        ax.legend(loc="best")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(title or f"{y.replace('_', ' ')} against {x.replace('_', ' ')} "
                          f"(camera level, n = {len(sub)})")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# 4. PM2.5 figures (only drawn when the target exists)
# --------------------------------------------------------------------------- #
def fig_pm25_profiles(profiles: pd.DataFrame) -> str:
    """Traffic and PM2.5 hourly profiles in separate panels on one row.

    Separate panels because the two have different units. Overlaying them on a
    twin axis would let the scales be chosen to manufacture agreement, which is
    exactly the illusion this section exists to test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharex=True)
    h = profiles.index.to_numpy()
    for ax, col, colour in zip(axes, profiles.columns, [SERIES[0], SERIES[1]]):
        _clean(ax)
        ax.plot(h, profiles[col], color=colour, solid_capstyle="round")
        ax.scatter(h, profiles[col], s=26, color=colour, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3)
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("hour of day (local)")
        ax.set_xticks(h[::2])
    axes[0].set_ylabel("mean per hour")
    fig.suptitle("Both series carry a daily cycle, which is why the raw correlation is confounded",
                 fontsize=12.0, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "06_pm25_daily_cycles")


def fig_decomposition_bars(decomp: pd.DataFrame) -> str:
    """The same traffic-PM2.5 relation under four estimators."""
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    _clean(ax, grid_axis="y")
    labels = decomp["estimate"].tolist()
    vals = decomp["r"].to_numpy(float)
    x = np.arange(len(labels))
    colours = [POS if v >= 0 else NEG for v in vals]
    ax.bar(x, vals, color=colours, width=0.55, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.01 if v >= 0 else -0.01), f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9,
                color=INK, fontweight="bold")
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(x, [l.replace(" (", "\n(") for l in labels], fontsize=8.5)
    ax.set_ylabel("correlation with PM2.5")
    ax.set_title("Traffic against PM2.5 under four estimators")
    fig.tight_layout()
    return _save(fig, "07_pm25_traffic_estimators")


# --------------------------------------------------------------------------- #
# 5. Typology, weekly rhythm and anomalies
# --------------------------------------------------------------------------- #
def fig_typology_profiles(centroids: pd.DataFrame, summary: pd.DataFrame) -> str:
    """One line per camera type, each indexed to its own daily mean.

    Indexed, because the point of the typology is shape rather than level: two
    types can sit on top of each other in absolute vehicles and still run on
    completely different schedules.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    _clean(ax)
    h = centroids.columns.to_numpy()
    ends = []
    for i, (cluster, row) in enumerate(centroids.iterrows()):
        colour = SERIES[i % len(SERIES)]
        n = int(summary.loc[cluster, "n_cameras"])
        ax.plot(h, row.to_numpy(float), color=colour, solid_capstyle="round",
                label=f"Type {cluster} ({n} cameras)")
        ends.append((float(row.iloc[-1]), f"Type {cluster}", colour))
    ax.axhline(1.0, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("ratio to the camera's own daily mean")
    ax.set_title("Camera types by the shape of their day")
    ax.set_xticks(h[::2])
    ax.set_xlim(h[0] - 0.3, h[-1] + 2.6)
    ax.legend(loc="upper left")
    _label_ends(ax, h[-1], ends)
    fig.tight_layout()
    return _save(fig, "11_camera_typology")


def fig_typology_map(labelled: pd.DataFrame) -> str:
    """Where each camera type sits. Categorical, so a fixed hue order."""
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    _clean(ax, grid_axis="both")
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (cluster, grp) in enumerate(labelled.groupby("profile_cluster")):
        ax.scatter(grp["longitude"], grp["latitude"], s=70,
                   color=SERIES[i % len(SERIES)], marker=markers[i % len(markers)],
                   edgecolor=SURFACE, linewidth=1.1, zorder=3,
                   label=f"Type {int(cluster)}")
    ax.set_aspect(1.0 / np.cos(np.radians(float(labelled["latitude"].mean()))))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Camera types in space")
    # Shape as well as colour, so the types survive a greyscale print.
    ax.legend(loc="best", title="profile type", title_fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "12_typology_map")


def fig_weekly_heatmap(grid: pd.DataFrame) -> str:
    """Day of week by hour. Sequential single hue: this is a magnitude."""
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    data = grid.to_numpy(float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(grid.columns)), [f"{h:02d}" for h in grid.columns],
                  fontsize=8)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=9)
    ax.set_xlabel("hour of day (local)")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(grid.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.tick_params(which="minor", length=0)

    hi = np.nanmax(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v) and v >= 0.92 * hi:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("mean vehicles per frame", color=INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=MUTED, labelcolor=MUTED, labelsize=8)
    ax.set_title("The week, hour by hour (busiest cells labelled)")
    fig.tight_layout()
    return _save(fig, "13_weekly_heatmap")


def fig_anomaly_timeline(frame: pd.DataFrame, z_threshold: float) -> str:
    """Every camera-hour by its deviation from that camera-hour's own norm.

    Drawn as a scatter over time so the flagged points can be seen in context
    rather than only as a table: a vertical stripe of flags is a city-wide
    event, a horizontal one is a single broken camera.
    """
    df = frame.dropna(subset=["z"]).copy()
    df["_ts"] = pd.to_datetime(df["hourly_timestamp"])
    normal = df[df["z"].abs() < z_threshold]
    flagged = df[df["z"].abs() >= z_threshold]

    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    _clean(ax)
    ax.scatter(normal["_ts"], normal["z"], s=5, color=MUTED, alpha=0.25,
               linewidth=0, zorder=2, label="within normal range")
    surge = flagged[flagged["z"] > 0]
    drop = flagged[flagged["z"] <= 0]
    ax.scatter(surge["_ts"], surge["z"], s=26, color=POS, alpha=0.85,
               edgecolor=SURFACE, linewidth=0.6, zorder=3, label="surge")
    ax.scatter(drop["_ts"], drop["z"], s=26, color=NEG, alpha=0.85,
               edgecolor=SURFACE, linewidth=0.6, zorder=3, label="drop")
    for y in (z_threshold, -z_threshold):
        ax.axhline(y, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)

    ax.set_ylabel("robust z against the camera's own norm")
    ax.set_xlabel("")
    ax.set_title(f"Camera-hours departing from their own norm (|z| >= {z_threshold})")
    ax.legend(loc="upper right", ncol=3)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return _save(fig, "14_anomaly_timeline")
