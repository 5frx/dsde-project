"""Component 3: Bangkok CCTV traffic and PM2.5 dashboard (Streamlit).

Run from the repo root:
    pip install -r VIZ/requirements.txt
    streamlit run VIZ/main.py

Reads the visualization-ready CSVs that Component 2 writes to
Component2_DataSci/output/dashboard/, plus VIZ/data/pm25_sensors.csv (who runs each
PM2.5 sensor, from VIZ/fetch_sensor_metadata.py) and data/raw/cctv_info.csv (camera names).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "Component2_DataSci" / "output" / "dashboard"
SENSORS = Path(__file__).resolve().parent / "data" / "pm25_sensors.csv"
CCTV_INFO = ROOT / "data" / "raw" / "cctv_info.csv"

VEHICLES = {"car_count": "Car", "motorcycle_count": "Motorcycle", "bus_count": "Bus", "truck_count": "Truck"}
WEATHER = {
    "temperature_2m": "Temperature (°C)",
    "precipitation": "Rain (mm)",
    "wind_speed_10m": "Wind speed (km/h)",
    "relative_humidity_2m": "Humidity (%)",
    "cloud_cover": "Cloud cover (%)",
    "surface_pressure": "Surface pressure (hPa)",
}
# Validated categorical slots (light mode); sequential = one blue hue; diverging = blue / gray / red.
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
CLASS_COLORS = {"Car": C_BLUE, "Motorcycle": C_ORANGE, "Bus": C_AQUA, "Truck": C_VIOLET}
SOURCE_COLORS = {"Air4Thai (government)": C_BLUE, "Low-cost sensor": C_ORANGE, "No reading": "#9b9a94"}
SEQ_BLUE = ["#dbe9f9", "#9cc3ef", "#5b9be3", "#2a78d6", "#1a539a", "#0f3463"]
GRID = "#ebe9e3"
INK, MUTED = "#1b1b19", "#5f5d56"
FONT = "IBM Plex Sans, IBM Plex Sans Thai, sans-serif"

# Small layer on top of .streamlit/config.toml: data numerals, selection colour, compact header and tabs.
CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1360px;}
[data-testid="stMetricValue"], [data-testid="stDataFrame"], .js-plotly-plot text {font-variant-numeric: tabular-nums;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #5f5d56;}
[data-testid="stMetric"] {padding: 0.15rem 0;}
::selection {background: #cfe0f5; color: #1b1b19;}
[role="tablist"] {gap: 1.75rem;}
[data-testid="stTab"] {padding-top: 0.55rem; padding-bottom: 0.55rem;}
[data-testid="stTab"] p {font-size: 0.97rem; font-weight: 500;}
[role="tabpanel"] {padding-top: 1.2rem;}
h2, h3 {margin-top: 0.4rem;}
.lede {color: #3d3c37; font-size: 1rem; line-height: 1.55; max-width: 72ch; margin: -0.35rem 0 0.9rem;}
.dash-sub {color: #5f5d56; font-size: 0.95rem; margin: -0.6rem 0 1.1rem;}
[data-testid="stSidebar"] h2 {font-size: 1.05rem;}
:focus-visible {outline: 2px solid #163a63; outline-offset: 2px;}
@media (max-width: 640px) {
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {flex-flow: row wrap; gap: 0.75rem 1rem;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {flex: 1 1 calc(50% - 0.5rem) !important; min-width: calc(50% - 0.5rem) !important;}
}
</style>
"""


def lede(text):
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)

st.set_page_config(page_title="Bangkok CCTV traffic & PM2.5", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data
def load():
    hours = pd.read_csv(DASH / "camera_hours.csv", parse_dates=["hourly_timestamp"])
    cams = pd.read_csv(DASH / "cameras.csv")
    corr = pd.read_csv(DASH / "correlations.csv")
    stations = pd.read_csv(DASH / "station_hours.csv", parse_dates=["hourly_timestamp"])

    if SENSORS.exists():
        sensors = pd.read_csv(SENSORS)
    else:  # dashboard still works, just without the provider split
        sensors = pd.DataFrame({"sensor_id": hours.sensor_id.dropna().unique()})
        sensors["sensor_type"] = "Unknown"
        sensors["station_name"] = sensors.sensor_id.astype("Int64").astype(str)
        sensors["provider"] = "Unknown"
    hours = hours.merge(sensors[["sensor_id", "sensor_type", "station_name", "provider"]], on="sensor_id", how="left")
    hours["sensor_type"] = hours["sensor_type"].fillna("No reading")
    hours["date"] = hours.hourly_timestamp.dt.date
    hours["weekday"] = hours.hourly_timestamp.dt.day_name().str[:3]
    stations = stations.merge(sensors, on="sensor_id", how="left")

    names = {}
    if CCTV_INFO.exists():
        info = pd.read_csv(CCTV_INFO)
        for r in info.itertuples():
            label = r.location_en if isinstance(r.location_en, str) and "BMAcctv" not in r.location_en else r.location
            names[r.camera_id] = str(label)
    cams["name"] = cams.camera_id.map(lambda c: names.get(c, f"Camera {c}"))
    main_source = hours.dropna(subset=["pm25"]).groupby("camera_id").sensor_type.agg(lambda s: s.mode().iat[0])
    cams["pm25_source"] = cams.camera_id.map(main_source).fillna("No reading")
    hours = hours.merge(cams[["camera_id", "name"]], on="camera_id", how="left")
    return hours, cams, corr, stations, sensors


def style(fig, height=340, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=70 if legend else 40, b=10),
        title_y=0.97, title_yanchor="top",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None) if legend else None,
        showlegend=legend, hoverlabel=dict(namelength=-1, font_family=FONT, bgcolor="#ffffff", bordercolor=GRID),
        font=dict(family=FONT, size=13, color=MUTED),
        title=dict(x=0, xanchor="left", font=dict(family=FONT, size=14, color=INK, weight=600)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font_size=12)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_font_size=12)
    return fig


def spearman(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() < 30:
        return np.nan
    return a[ok].rank().corr(b[ok].rank())


hours_all, cams_all, corr_c2, stations_all, sensors = load()

# --------------------------------------------------------------------------- filters
st.sidebar.header("Filters")
dates = sorted(hours_all.date.unique())
d0, d1 = st.sidebar.select_slider("Dates", options=dates, value=(dates[0], dates[-1]),
                                  format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b"))
h0, h1 = st.sidebar.slider("Hours of the day", 6, 21, (6, 21))
source_opts = ["Air4Thai (government)", "Low-cost sensor"]
sources = st.sidebar.multiselect(
    "PM2.5 source", source_opts, default=source_opts,
    help="Component 2 matched each camera to the nearest OpenAQ PM2.5 sensor. 17 of the 31 are Air4Thai "
         "reference monitors; 14 are low-cost sensors (AirGradient, Clarity), which read about half as high.")
cam_labels = {r.camera_id: f"{r.camera_id} · {r.name}" for r in cams_all.itertuples()}
picked = st.sidebar.multiselect("Cameras (empty = all)", list(cam_labels), format_func=cam_labels.get)

mask = (hours_all.date.between(d0, d1) & hours_all.hourly_timestamp.dt.hour.between(h0, h1))
if picked:
    mask &= hours_all.camera_id.isin(picked)
hours = hours_all[mask].copy()
# PM2.5 is kept only for the chosen sources; traffic rows stay so traffic views are unaffected.
pm_ok = hours.sensor_type.isin(sources)
hours["pm25_f"] = hours.pm25.where(pm_ok)
st.sidebar.caption(f"{len(hours):,} camera-hours · {hours.camera_id.nunique()} cameras · "
                   f"{hours.pm25_f.notna().sum():,} with PM2.5")
st.sidebar.divider()
st.sidebar.caption("Vehicle counts are YOLOv8 detections per CCTV frame (a snapshot every 10 min), averaged per hour. "
                   "PM2.5 is the matched station's hourly reading (µg/m³), shared by nearby cameras.")

if hours.empty:
    st.warning("No data for these filters.")
    st.stop()

# --------------------------------------------------------------------------- header
st.title("Bangkok CCTV traffic and PM2.5")
st.markdown('<p class="dash-sub">100 BMA traffic cameras, 25–31 August 2026, 06:00–21:00. '
            'Vehicle counts from YOLOv8, PM2.5 from the nearest OpenAQ sensor.</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.container(border=True).columns(5)
k1.metric("Cameras", hours.camera_id.nunique())
k2.metric("Camera-hours", f"{len(hours):,}")
k3.metric("Vehicles per frame", f"{hours.total_vehicles.mean():.1f}")
k4.metric("Mean PM2.5", f"{hours.pm25_f.mean():.1f} µg/m³" if hours.pm25_f.notna().any() else "–")
rho_all = spearman(hours.total_vehicles, hours.pm25_f)
k5.metric("Traffic vs PM2.5 (ρ)", "–" if np.isnan(rho_all) else f"{rho_all:+.2f}",
          help="Spearman rank correlation across the filtered camera-hours. Descriptive only.")

tab_over, tab_map, tab_cam, tab_rel, tab_q = st.tabs(
    ["Daily patterns", "Map", "Camera explorer", "Traffic vs PM2.5", "Data quality"])

# --------------------------------------------------------------------------- daily patterns
with tab_over:
    st.subheader("When is traffic heaviest, and when is the air worst?")
    lede("Traffic builds through the day and drops after 18:00, while PM2.5 climbs in the evening. "
         "That opposite daily rhythm is what produces the small negative raw correlation.")
    by_h = hours.groupby(hours.hourly_timestamp.dt.hour).agg(
        traffic=("total_vehicles", "mean"), pm25=("pm25_f", "mean"),
        **{v: (v, "mean") for v in VEHICLES}).reset_index(names="hour")
    c1, c2 = st.columns(2)
    long = by_h.melt("hour", list(VEHICLES), "cls", "count")
    long["cls"] = long.cls.map(VEHICLES)
    f = px.bar(long, x="hour", y="count", color="cls", color_discrete_map=CLASS_COLORS,
               labels={"hour": "Hour of day", "count": "Vehicles per frame", "cls": "Class"},
               title="Vehicles per frame by hour")
    f.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.9)",
                    hovertemplate="%{x}:00 · %{y:.1f}<extra>%{fullData.name}</extra>")
    c1.plotly_chart(style(f), width="stretch")
    f = px.line(by_h, x="hour", y="pm25", markers=True, title="PM2.5 by hour",
                labels={"hour": "Hour of day", "pm25": "PM2.5 (µg/m³)"})
    f.update_traces(line_color=C_BLUE, line_width=2, marker_size=8,
                    hovertemplate="%{x}:00 · %{y:.1f} µg/m³<extra></extra>")
    c2.plotly_chart(style(f, legend=False), width="stretch")

    st.subheader("Day by hour")
    c1, c2 = st.columns(2)
    for col, metric, title, unit in [(c1, "total_vehicles", "Vehicles per frame", ""),
                                     (c2, "pm25_f", "PM2.5", " µg/m³")]:
        grid = hours.pivot_table(index=hours.hourly_timestamp.dt.strftime("%a %d"),
                                 columns=hours.hourly_timestamp.dt.hour, values=metric, aggfunc="mean")
        grid = grid.reindex(sorted(grid.index, key=lambda s: s[-2:]))
        f = go.Figure(go.Heatmap(z=grid.values, x=grid.columns, y=grid.index, colorscale=SEQ_BLUE,
                                 xgap=2, ygap=2, colorbar=dict(thickness=10),
                                 hovertemplate="%{y} %{x}:00 · %{z:.1f}" + unit + "<extra></extra>"))
        f.update_layout(title=title, xaxis_title="Hour of day")
        col.plotly_chart(style(f, 300, legend=False), width="stretch")
    st.caption("Blank cells are hours with no frames (for example the camera outage on the morning of 27 Aug).")

# --------------------------------------------------------------------------- map
with tab_map:
    metric_names = {"total_vehicles": "Vehicles per frame", "pm25_f": "PM2.5 (µg/m³)",
                    "truck_share": "Truck share (%)", "motorcycle_share": "Motorcycle share (%)"}
    st.subheader("Where traffic and PM2.5 concentrate")
    lede("Each dot is a camera, sized by vehicles per frame. Pick what the colour shows; hover a dot for details.")
    metric = st.radio("Colour cameras by", list(metric_names), format_func=metric_names.get, horizontal=True)
    per = hours.groupby("camera_id").agg(total_vehicles=("total_vehicles", "mean"), pm25_f=("pm25_f", "mean"),
                                         truck=("truck_count", "mean"), moto=("motorcycle_count", "mean")).reset_index()
    per["truck_share"] = 100 * per.truck / per.total_vehicles.replace(0, np.nan)
    per["motorcycle_share"] = 100 * per.moto / per.total_vehicles.replace(0, np.nan)
    per = per.merge(cams_all[["camera_id", "name", "latitude", "longitude", "pm25_source"]], on="camera_id")
    per = per.dropna(subset=[metric])
    if per.empty:
        st.info("No values for this metric with the current filters.")
    else:
        # pydeck renders reliably inside Streamlit tabs (plotly map tiles stay blank in a hidden tab).
        lo, hi = per[metric].quantile([0.02, 0.98])
        t = ((per[metric] - lo) / (hi - lo if hi > lo else 1)).clip(0, 1)
        rgb = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in SEQ_BLUE]
        per["color"] = [list(rgb[int(round(v * (len(rgb) - 1)))]) + [230] for v in t]
        per["radius"] = 90 + 14 * per.total_vehicles
        per["label"] = per[metric].round(1)
        per["metric_name"] = metric_names[metric]
        layers = [pdk.Layer("ScatterplotLayer", per, get_position=["longitude", "latitude"], get_fill_color="color",
                            get_radius="radius", stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1,
                            pickable=True)]
        if metric == "pm25_f" and "station_lat" in sensors:
            used = sensors[sensors.sensor_type.isin(sources)].assign(
                name=lambda x: "Sensor: " + x.station_name + " (" + x.provider + ")",
                camera_id="", label="", metric_name="", pm25_source=lambda x: x.sensor_type)
            layers.append(pdk.Layer("ScatterplotLayer", used, get_position=["station_lon", "station_lat"],
                                    get_fill_color=[11, 11, 11, 230], get_radius=70, radius_min_pixels=4, pickable=True))
        st.pydeck_chart(pdk.Deck(
            layers=layers, map_style=None,
            initial_view_state=pdk.ViewState(latitude=per.latitude.mean(), longitude=per.longitude.mean(), zoom=11.2),
            tooltip={"html": "<b>{name}</b><br/>Camera {camera_id}<br/>{metric_name}: {label}<br/>PM2.5 source: {pm25_source}"},
        ), height=540)
        st.caption(f"Colour: {metric_names[metric]}, light = {lo:.1f} to dark = {hi:.1f}. Marker size = vehicles per frame. "
                   "With PM2.5 selected, black dots are the matched sensors.")

        top_cols = list(dict.fromkeys(["camera_id", "name", metric, "total_vehicles", "pm25_source"]))
        top = per.sort_values(metric, ascending=False)[top_cols]
        st.dataframe(top.rename(columns={**metric_names, "camera_id": "Camera", "name": "Location",
                                         "pm25_source": "PM2.5 source"}).round(1),
                     hide_index=True, width="stretch", height=260)

# --------------------------------------------------------------------------- camera explorer
with tab_cam:
    st.subheader("One camera, hour by hour")
    cam = st.selectbox("Camera", list(cam_labels), format_func=cam_labels.get,
                       index=list(cam_labels).index(picked[0]) if picked else 0)
    ch = hours_all[(hours_all.camera_id == cam) & hours_all.date.between(d0, d1)].sort_values("hourly_timestamp")
    grid_ts = pd.date_range(ch.hourly_timestamp.min(), ch.hourly_timestamp.max(), freq="h", name="hourly_timestamp")
    ch_plot = ch.set_index("hourly_timestamp").reindex(grid_ts).reset_index()  # empty rows break lines overnight
    a, b, c, d = st.container(border=True).columns(4)
    a.metric("Vehicles per frame", f"{ch.total_vehicles.mean():.1f}")
    b.metric("Mean PM2.5", f"{ch.pm25.mean():.1f} µg/m³")
    c.metric("Sensor distance", f"{ch.station_distance_m.mean() / 1000:.1f} km")
    d.metric("Fallback sensor", f"{100 * ch.is_fallback.mean():.0f}% of hours")
    sens = ch.dropna(subset=["station_name"]).groupby(["station_name", "provider"]).size().reset_index(name="hours")
    st.caption("PM2.5 from: " + "; ".join(f"{r.station_name} ({r.provider}, {r.hours} h)" for r in sens.itertuples()))

    long = ch_plot.melt("hourly_timestamp", list(VEHICLES), "cls", "count")
    long["cls"] = long.cls.map(VEHICLES)
    f = px.bar(long, x="hourly_timestamp", y="count", color="cls", color_discrete_map=CLASS_COLORS,
               labels={"hourly_timestamp": "", "count": "Vehicles per frame", "cls": "Class"},
               title="Vehicles per frame (hourly mean)")
    f.update_traces(marker_line_width=0, hovertemplate="%{y:.1f}<extra>%{fullData.name}</extra>")
    f.update_layout(hovermode="x unified", bargap=0.15)
    st.plotly_chart(style(f, 300), width="stretch")

    f = px.line(ch_plot, x="hourly_timestamp", y="pm25", markers=True, title="PM2.5 at the matched sensor",
                labels={"hourly_timestamp": "", "pm25": "PM2.5 (µg/m³)"})
    f.update_traces(line_color=C_BLUE, line_width=2, marker_size=6, connectgaps=False,
                    hovertemplate="%{x|%a %d %H:00} · %{y:.1f} µg/m³<extra></extra>")
    st.plotly_chart(style(f, 260, legend=False), width="stretch")

    w = st.selectbox("Weather variable", list(WEATHER), format_func=WEATHER.get)
    f = px.line(ch_plot, x="hourly_timestamp", y=w, title=WEATHER[w], labels={"hourly_timestamp": "", w: WEATHER[w]})
    f.update_traces(line_color=C_AQUA if w != "precipitation" else C_BLUE, line_width=2,
                    hovertemplate="%{x|%a %d %H:00} · %{y:.1f}<extra></extra>")
    st.plotly_chart(style(f, 220, legend=False), width="stretch")
    st.caption("Gaps are overnight (no frames 21:00–06:00) or hours without a reading.")

# --------------------------------------------------------------------------- relationships
with tab_rel:
    st.subheader("What moves with PM2.5?")
    lede("Wind clears PM2.5 out. Traffic stays weak and slightly negative however the data is sliced.")
    view = st.radio("Correlation view", ["Raw (all camera-hours)", "Within each camera, net of hour of day"],
                    horizontal=True,
                    help="The second view subtracts each camera's average and the hour-of-day average first, "
                         "so it compares a camera with itself at the same time of day.")
    cols = {"total_vehicles": "Total vehicles", **{k: v for k, v in VEHICLES.items()}, **WEATHER}
    d = hours.dropna(subset=["pm25_f"]).copy()
    if view.startswith("Within"):
        for c in ["pm25_f", *cols]:
            d[c] = d[c] - d.groupby("camera_id")[c].transform("mean")
            d[c] = d[c] - d.groupby(d.hourly_timestamp.dt.hour)[c].transform("mean")
    rho = pd.DataFrame({"variable": list(cols.values()),
                        "rho": [spearman(d[c], d.pm25_f) for c in cols]}).dropna().sort_values("rho")
    f = go.Figure(go.Bar(x=rho.rho, y=rho.variable, orientation="h",
                         marker_color=np.where(rho.rho < 0, C_ORANGE, C_BLUE),
                         hovertemplate="%{y}: ρ = %{x:+.2f}<extra></extra>"))
    f.add_vline(x=0, line_color="#8a8984", line_width=1)
    f.update_layout(title="Correlation with PM2.5 (orange = lower PM2.5, blue = higher)",
                    xaxis_title="Spearman ρ", xaxis_range=[-0.65, 0.65])
    st.plotly_chart(style(f, 380, legend=False), width="stretch")
    st.caption(f"Spearman ρ over {len(d):,} camera-hours. p-values are not shown because cameras share sensors, which makes them look far stronger than they are.")

    c1, c2 = st.columns(2)
    f = go.Figure(go.Histogram2d(x=hours.total_vehicles, y=hours.pm25_f, nbinsx=40, nbinsy=40,
                                 colorscale=[[0, "rgba(0,0,0,0)"], [0.001, SEQ_BLUE[0]], [1, SEQ_BLUE[-1]]],
                                 colorbar=dict(title=dict(text="Camera-hours", font_size=12), thickness=10),
                                 hovertemplate="%{x:.0f} vehicles · %{y:.0f} µg/m³ · %{z} camera-hours<extra></extra>"))
    f.update_layout(title="Traffic vs PM2.5 (density)", xaxis_title="Vehicles per frame",
                    yaxis_title="PM2.5 (µg/m³)")
    c1.plotly_chart(style(f, 360, legend=False), width="stretch")
    wind = hours.dropna(subset=["pm25_f"]).copy()
    wind["wind_bin"] = pd.cut(wind.wind_speed_10m, [0, 4, 8, 12, 16, 40],
                              labels=["0–4", "4–8", "8–12", "12–16", "16+"])
    wb = wind.groupby("wind_bin", observed=True).agg(pm25=("pm25_f", "mean"), n=("pm25_f", "size")).reset_index()
    f = px.bar(wb, x="wind_bin", y="pm25", title="PM2.5 by wind speed", text_auto=".1f",
               labels={"wind_bin": "Wind speed (km/h)", "pm25": "Mean PM2.5 (µg/m³)"}, custom_data=["n"])
    f.update_traces(marker_color=C_BLUE, textposition="outside",
                    hovertemplate="%{x} km/h · %{y:.1f} µg/m³ · %{customdata[0]:,} camera-hours<extra></extra>")
    c2.plotly_chart(style(f, 360, legend=False), width="stretch")

    st.subheader("Component 2 regression result")
    st.markdown(
        "Component 2 fitted *PM2.5 ~ log(1 + vehicles) + camera, hour and date fixed effects + weather*, "
        "with errors clustered by sensor. Traffic coefficient **−0.16** (95% CI −0.35 to +0.02, p = 0.08): "
        "**no reliable traffic effect** in this week of data.")
    with st.expander("Component 2 correlation table (all sources, raw)"):
        st.dataframe(corr_c2.round(4), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- data quality
with tab_q:
    st.subheader("Where does each camera's PM2.5 come from?")
    lede("Low-cost sensors read about half as high as Air4Thai monitors and produce every zero reading. "
         "Use the PM2.5 source filter to check that a finding holds on Air4Thai data alone.")
    q = hours.dropna(subset=["pm25"])
    c1, c2, c3 = st.container(border=True).columns(3)
    c1.metric("From low-cost sensors", f"{100 * (q.sensor_type == 'Low-cost sensor').mean():.0f}% of hours")
    c2.metric("From a fallback sensor", f"{100 * q.is_fallback.mean():.0f}% of hours")
    c3.metric("Readings of exactly 0", f"{(q.pm25 == 0).sum():,}")

    c1, c2 = st.columns(2)
    f = go.Figure()
    for name in ["Air4Thai (government)", "Low-cost sensor"]:
        f.add_trace(go.Box(y=q.pm25[q.sensor_type == name], name=name, marker_color=SOURCE_COLORS[name],
                           boxpoints=False, line_width=1.5, fillcolor=SOURCE_COLORS[name] + "33"))
    f.update_layout(title="PM2.5 by sensor type", yaxis_title="PM2.5 (µg/m³)")
    c1.plotly_chart(style(f, 340, legend=False), width="stretch")
    f = px.histogram(q.drop_duplicates(["camera_id", "sensor_id"]), x="station_distance_m", color="sensor_type",
                     nbins=20, color_discrete_map=SOURCE_COLORS, title="Camera-to-sensor distance",
                     labels={"station_distance_m": "Distance (m)", "sensor_type": "Sensor"})
    f.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.9)")
    f.update_layout(yaxis_title="Camera–sensor pairs", barmode="stack")
    c2.plotly_chart(style(f, 340), width="stretch")

    st.subheader("Sensors")
    sens = q.groupby(["sensor_id", "station_name", "provider", "sensor_type"]).agg(
        cameras=("camera_id", "nunique"), camera_hours=("pm25", "size"), mean_pm25=("pm25", "mean"),
        zero_readings=("pm25", lambda s: int((s == 0).sum())), mean_distance_m=("station_distance_m", "mean"),
    ).reset_index().sort_values("camera_hours", ascending=False)
    sens["sensor_id"] = sens.sensor_id.astype("int64").astype(str)
    st.dataframe(sens.round(1).rename(columns={
        "sensor_id": "Sensor", "station_name": "Station", "provider": "Provider", "sensor_type": "Type",
        "cameras": "Cameras", "camera_hours": "Camera-hours", "mean_pm25": "Mean PM2.5",
        "zero_readings": "Zero readings", "mean_distance_m": "Mean distance (m)"}),
        hide_index=True, width="stretch")

    st.subheader("Camera coverage")
    cov = hours.groupby("camera_id").agg(camera_hours=("hourly_timestamp", "size"),
                                         pm25_hours=("pm25", "count"), fallback=("is_fallback", "mean"),
                                         vehicles=("total_vehicles", "mean")).reset_index()
    cov = cov.merge(cams_all[["camera_id", "name", "pm25_source"]], on="camera_id")[
        ["camera_id", "name", "pm25_source", "camera_hours", "pm25_hours", "fallback", "vehicles"]]
    cov["fallback"] = (100 * cov.fallback).round(0)
    st.dataframe(cov.rename(columns={"camera_id": "Camera", "name": "Location", "camera_hours": "Camera-hours",
                                     "pm25_hours": "Hours with PM2.5", "fallback": "Fallback %",
                                     "vehicles": "Vehicles per frame", "pm25_source": "PM2.5 source"}).round(1),
                 hide_index=True, width="stretch", height=300)
