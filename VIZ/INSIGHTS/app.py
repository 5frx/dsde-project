"""Bangkok traffic insights: one week of city CCTV read as a map sheet.

    cd INSIGHTS
    pip install -r requirements.txt
    streamlit run app.py

Reads only INSIGHTS/data/derived (built by build_insights.py) and INSIGHTS/assets/frames.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

HERE = Path(__file__).resolve().parent
D = HERE / "data" / "derived"
FRAMES = HERE / "assets" / "frames"

INK, MUTED, GRID = "#1d1d1b", "#5c5d58", "#e6e7e2"
ROAD, WATER, GREEN, BROWN = "#c2412d", "#1f6fb2", "#3f8f4a", "#8a5a2e"
# validated categorical order (adjacent CVD ΔE >= 19.9 light): car, motorcycle, bus, truck
CLASS = {"car_count": ("Car", "#c2412d"), "motorcycle_count": ("Motorcycle", "#1f6fb2"),
         "bus_count": ("Bus", "#d98a12"), "truck_count": ("Truck", "#7a5a9e")}
SHAPE_COLORS = {"Daytime plateau": "#c2412d", "Busy into the night": "#1f6fb2",
                "Evening spike": "#d98a12", "Two rush hours": "#7a5a9e"}
FONT = "Sarabun, sans-serif"

st.set_page_config(page_title="Bangkok traffic, one week", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
.block-container {padding-top: 4.2rem; max-width: 1320px;}
@media (max-width: 640px) {[data-testid="stHorizontalBlock"]:has([data-testid="stHorizontalBlock"]) {flex-direction: column !important;} [data-testid="stHorizontalBlock"]:has([data-testid="stHorizontalBlock"]) > [data-testid="stColumn"] {width: 100% !important; flex: 1 1 100% !important;}}
[data-testid="stMetricValue"], [data-testid="stDataFrame"], .js-plotly-plot text {font-variant-numeric: tabular-nums;}
[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #5c5d58;}
::selection {background: #f3d3cd; color: #1d1d1b;}
:focus-visible {outline: 2px solid #b3382a; outline-offset: 2px;}
.sheet {background: #1d1d1b; color: #fff; padding: 1.1rem 1.4rem 1rem; border-radius: 0.25rem; margin-bottom: 1.4rem;
        display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap;}
.sheet h1 {color: #fff; margin: 0; padding: 0; font-size: 2rem; line-height: 1.15;}
.sheet .cov {color: #d6d6d0; font-size: 0.95rem; margin-top: 0.3rem;}
.sheet .idx {font-variant-numeric: tabular-nums; font-size: 0.85rem; color: #d6d6d0; text-align: right; line-height: 1.5;}
.lede {font-size: 1.12rem; line-height: 1.55; max-width: 68ch; color: #2b2b28; margin: 0 0 1rem;}
.finding {border-top: 2px solid #1d1d1b; padding-top: 0.55rem; margin-bottom: 1.1rem;}
.finding b {display: block; font-size: 1.05rem; margin-bottom: 0.15rem;}
.finding span {color: #5c5d58; font-size: 0.95rem; line-height: 1.45;}
.key {display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 50%; margin-right: 0.45rem; vertical-align: -1px;}
.key.sq {border-radius: 2px;}
.margin-h {font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; color: #5c5d58; margin: 1.1rem 0 0.35rem;}
.margin-p {font-size: 0.88rem; color: #3d3d39; line-height: 1.5; margin: 0;}
.note {font-size: 0.9rem; color: #5c5d58; max-width: 80ch;}
@media (max-width: 640px) {
  .sheet h1 {font-size: 1.5rem;} .sheet .idx {text-align: left;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {flex-flow: row wrap; gap: 0.75rem 1rem;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {flex: 1 1 calc(50% - 0.5rem) !important; min-width: calc(50% - 0.5rem) !important;}
}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    r = lambda n, **kw: pd.read_csv(D / n, **kw)  # noqa: E731
    return dict(k=json.loads((D / "kpis.json").read_text()), cams=r("cameras.csv"), pulse=r("pulse.csv"),
                shapes=r("shapes.csv"), daily=r("daily.csv"), dusk=r("dusk.csv"), rain=r("rain.csv"),
                rain_h=r("rain_by_hour.csv"), air=r("air_24h.csv"), lag=r("lag.csv"), wind=r("wind.csv"),
                wind_s=r("wind_speed.csv"), bias=r("sensor_bias.csv"), pairs=r("sensor_pairs.csv"),
                districts=r("districts.csv"), roads=r("road_class.csv"), rail=r("rail.csv"),
                unusual=r("unusual.csv", parse_dates=["ts"]), stations=pd.read_csv(HERE / "data" / "rail_stations.csv"))


X = load()
K = X["k"]


def style(fig, h=320, legend=True, title=None):
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=64 if (legend and title) else 36 if (legend or title) else 12, b=8),
        title=dict(text=title, x=0, xanchor="left", font=dict(family=FONT, size=16, color=INK, weight=600)) if title else None,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=13, color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font_size=13) if legend else None,
        showlegend=legend, hoverlabel=dict(font_family=FONT, bgcolor="#fff", bordercolor=GRID),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#b9bab4", title_font_size=13)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_font_size=13)
    return fig


def show(fig, **kw):
    st.plotly_chart(style(fig, **kw), width="stretch", config={"displayModeBar": False})


def lede(t):
    st.markdown(f'<p class="lede">{t}</p>', unsafe_allow_html=True)


def margin(title, body):
    st.sidebar.markdown(f'<p class="margin-h">{title}</p><p class="margin-p">{body}</p>', unsafe_allow_html=True)


def sheet(title, coverage, index):
    st.markdown(f'<div class="sheet"><div><h1>{title}</h1><div class="cov">{coverage}</div></div>'
                f'<div class="idx">{index}</div></div>', unsafe_allow_html=True)


def keys(items, square=False):
    return " &nbsp; ".join(f'<span class="key{" sq" if square else ""}" style="background:{c}"></span>{n}'
                           for n, c in items)


def slot_ticks(fig):
    fig.update_xaxes(tickmode="array", tickvals=[f"{h:02d}:00" for h in range(6, 22, 2)],
                     ticktext=[f"{h:02d}:00" for h in range(6, 22, 2)])


def basemap(cams, color_col, colors, extra_layers=(), height=520, tooltip=None):
    """Camera map drawn as a survey sheet: 0.01° graticule (about 1.1 km), rail stations, cameras."""
    lat0, lat1, lon0, lon1 = 13.66, 13.86, 100.40, 100.66
    grid = [{"path": [[lo, lat0], [lo, lat1]]} for lo in np.arange(lon0, lon1 + 1e-9, 0.01)]
    grid += [{"path": [[lon0, la], [lon1, la]]} for la in np.arange(lat0, lat1 + 1e-9, 0.01)]
    labels = [{"pos": [lo, lat1 - 0.004], "t": f"{lo:.2f}°E"} for lo in np.arange(lon0 + 0.04, lon1, 0.04)]
    labels += [{"pos": [lon0 + 0.013, la], "t": f"{la:.2f}°N"} for la in np.arange(lat0 + 0.02, lat1, 0.04)]
    c = cams.copy()
    c["rgb"] = c[color_col].map(lambda v: [int(colors[v][i:i + 2], 16) for i in (1, 3, 5)] + [235])
    c["r"] = 120 + 16 * c.vehicles
    st_ = X["stations"]
    layers = [
        pdk.Layer("PathLayer", grid, get_path="path", get_color=[29, 29, 27, 28], width_min_pixels=1),
        pdk.Layer("TextLayer", labels, get_position="pos", get_text="t", get_size=11, get_color=[92, 93, 88, 220],
                  font_family="Sarabun, sans-serif"),
        pdk.Layer("ScatterplotLayer", st_, get_position=["lon", "lat"], get_radius=55, radius_min_pixels=2.5,
                  get_fill_color=[255, 255, 255, 255], stroked=True, get_line_color=[29, 29, 27], line_width_min_pixels=1.5),
        pdk.Layer("ScatterplotLayer", c, get_position=["lon", "lat"], get_fill_color="rgb", get_radius="r",
                  radius_min_pixels=3, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True),
        *extra_layers,
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, map_style=None, tooltip=tooltip or {
        "html": "<b>{name}</b><br/>{district} · {road_class} road<br/>{vehicles_r} vehicles per frame<br/>{shape}"},
        initial_view_state=pdk.ViewState(latitude=13.765, longitude=100.535, zoom=11.1)), height=height)


cams = X["cams"].assign(vehicles_r=lambda d: d.vehicles.round(1))


# ------------------------------------------------------------------ pages
def page_sheet():
    sheet("Bangkok traffic, one week",
          "100 BMA traffic cameras · 25–31 August 2026 · 06:00–21:00, one frame every 10 minutes",
          f"{K['frames']:,} frames<br/>{K['vehicles_counted']:,} vehicle detections<br/>{K['districts']} districts")
    left, right = st.columns([0.92, 1.25], gap="large")
    with left:
        st.header("The city peaks at 16:50")
        lede(f"On weekdays the busiest ten minutes are {K['pm_peak_slot']}, with {K['pm_peak_vehicles']} vehicles in "
             f"view per camera. Weekends carry {abs(K['weekend_change_pct']):.0f}% less traffic. Rain barely moves "
             "the numbers; darkness does.")
        a, b = st.columns(2)
        a.metric("Evening peak", K["pm_peak_slot"], help="Weekday 10-minute slot with the most vehicles per frame")
        b.metric("Weekend vs weekday", f"{K['weekend_change_pct']:+.0f}%")
        a.metric("Heavy rain effect", f"{K['heavy_rain_change_pct']:+.0f}%", help="Same camera, hour and day type, vs dry")
        b.metric("PM2.5, official monitors", f"{K['pm25_a4t_mean']} µg/m³")
    with right:
        basemap(cams, "shape", SHAPE_COLORS, height=470)
        st.markdown(keys([(n, c) for n, c in SHAPE_COLORS.items()]) + ' &nbsp; <span class="key" style="background:#fff;'
                    'border:1.5px solid #1d1d1b"></span>Rail station', unsafe_allow_html=True)
        st.caption("Cameras coloured by the shape of their weekday traffic, sized by vehicles per frame. Grid every 0.01°.")

    st.subheader("Six findings")
    f = [
        ("Rhythm", f"Weekdays build from 06:00 and peak at {K['pm_peak_slot']}. Sunday is the quietest day.",
         "page_rhythm"),
        ("Places", "Old-town Phra Nakhon has the fewest vehicles and the most motorcycles; outer districts carry the "
                   "most trucks.", "page_places"),
        ("Rain", f"Rain falls in about a third of camera-hours, and traffic barely changes: {K['heavy_rain_change_pct']:+.0f}% "
                 "in the heaviest rain.", "page_weather"),
        ("Dusk", f"Detections drop {abs(K['dusk_drop_pct']):.0f}% in the half hour around sunset. Most of that is the "
                 "camera, not the road.", "page_weather"),
        ("Air", f"Calm air, not traffic, sets PM2.5. Traffic and later PM2.5 stay unrelated (|ρ| ≤ {K['lag_rho_max_abs']}).",
         "page_air"),
        ("Sensors", f"Low-cost PM2.5 sensors read {K['lowcost_ratio']:.0%} of the official monitor beside them.",
         "page_air"),
    ]
    for i, (t, body, pg) in enumerate(f):
        if i % 3 == 0:
            cols = st.columns(3, gap="large")  # one row per three findings keeps the rules aligned
        with cols[i % 3]:
            st.markdown(f'<div class="finding"><b>{t}</b><span>{body}</span></div>', unsafe_allow_html=True)
            st.page_link(PAGES[pg], label=f"See the evidence: {PAGES[pg].title}")
    margin("Sheet", "Bangkok inner city, 13.66–13.86°N, 100.40–100.66°E.")
    margin("Symbols", keys([(n, c) for n, c in SHAPE_COLORS.items()]).replace(" &nbsp; ", "<br/>"))


def page_rhythm():
    sheet("Rhythm", "When the city moves, at 10-minute resolution", f"Weekday peak {K['pm_peak_slot']}<br/>"
          f"Morning peak {K['am_peak_slot']}")
    lede(f"Traffic climbs from 06:00, holds a long plateau through the day and peaks at {K['pm_peak_slot']} on weekdays. "
         f"After sunset (about {K['sunset'].split('–')[0]}) counts fall fast, partly because the cameras see less "
         "in the dark.")
    p = X["pulse"]
    fig = go.Figure()
    for t, c, dash in [("Weekday", ROAD, "solid"), ("Weekend", INK, "dot")]:
        s = p[p.daytype == t]
        fig.add_scatter(x=s.slot, y=s.total, name=t, line=dict(color=c, width=2.2, dash=dash),
                        hovertemplate="%{x} · %{y:.1f} vehicles<extra>" + t + "</extra>")
    fig.add_vrect(x0="18:30", x1="21:00", fillcolor="#1d1d1b", opacity=0.05, line_width=0,
                  annotation_text="after sunset", annotation_position="top left", annotation_font_color=MUTED)
    fig.update_layout(hovermode="x unified", yaxis_title="Vehicles per frame")
    slot_ticks(fig)
    show(fig, h=360, title="Vehicles in view, average of all cameras")

    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        d = X["daily"]
        fig = go.Figure(go.Bar(x=d.weekday.str[:3] + " " + d.date.str[-2:], y=d.vehicles, text=d.vehicles.round(1),
                               textposition="outside", marker_color=[INK if w in ("Saturday", "Sunday") else ROAD
                                                                     for w in d.weekday],
                               hovertemplate="%{x} · %{y:.1f}<extra></extra>"))
        fig.update_layout(yaxis_title="Vehicles per frame", yaxis_range=[0, d.vehicles.max() * 1.18])
        show(fig, h=320, legend=False, title="By day: the weekend dip")
    with c2:
        s = p[p.daytype == "Weekday"]
        tot = s[list(CLASS)].sum(axis=1)
        fig = go.Figure()
        for col, (n, c) in CLASS.items():
            fig.add_scatter(x=s.slot, y=100 * s[col] / tot, name=n, stackgroup="one", line=dict(width=0.5, color="#fff"),
                            fillcolor=c, hovertemplate="%{x} · %{y:.0f}%<extra>" + n + "</extra>")
        fig.update_layout(yaxis_title="Share of vehicles (%)", yaxis_range=[0, 100], hovermode="x unified")
        slot_ticks(fig)
        show(fig, h=320, title="What is on the road, weekdays")
    st.caption("Trucks take their biggest share early in the morning; motorcycles rise after dark, when cars "
               "become harder to detect.")

    st.subheader("Four kinds of road")
    lede("Grouping cameras by the shape of their weekday (each camera divided by its own average) gives four "
         "rhythms. Most roads are busy all day; a handful have a sharp evening spike or two clear rush hours.")
    sh = X["shapes"]
    cols = st.columns(4)
    for i, (n, c) in enumerate(SHAPE_COLORS.items()):
        s = sh[sh["shape"] == n]
        if s.empty:
            continue
        fig = go.Figure(go.Scatter(x=s.hour, y=s["index"], line=dict(color=c, width=2.4), fill="tozeroy",
                                   fillcolor=c + "1f", hovertemplate="%{x}:00 · %{y:.2f}× average<extra></extra>"))
        fig.add_hline(y=1, line_color="#b9bab4", line_width=1, line_dash="dot")
        fig.update_layout(yaxis_range=[0, 2.4], xaxis_title=None, yaxis_title=None)
        with cols[i]:
            st.markdown(f"**{n}** · {int(s.cameras.iloc[0])} cameras")
            show(fig, h=190, legend=False)
    margin("Source", "Vehicle counts: team YOLOv8 detections on every BMA frame (Component 1).")
    margin("Shapes", "k-means (k = 4, 20 starts) on each camera's weekday hourly profile.")


def page_places():
    sheet("Places", "Where traffic and its mix differ", f"{K['districts']} districts<br/>{K['rail_stations']} rail stations")
    lede("Busy central districts like Ratchathewi and Bang Rak see three times the traffic of old-town Phra Nakhon, "
         "but Phra Nakhon and Dusit carry the highest motorcycle shares. Trucks concentrate on the outer edge.")
    metric = st.segmented_control("Map colour", ["Vehicles per frame", "Motorcycle share", "Truck share", "PM2.5 (official)"],
                                  default="Vehicles per frame")
    col = {"Vehicles per frame": "vehicles", "Motorcycle share": "moto_share", "Truck share": "truck_share",
           "PM2.5 (official)": "pm25_a4t"}[metric or "Vehicles per frame"]
    c = cams.dropna(subset=[col]).copy()
    q = pd.qcut(c[col], 4, labels=["Lowest quarter", "Second quarter", "Third quarter", "Highest quarter"])
    c["band"] = q.astype(str)
    ramp = {"Lowest quarter": "#f0c9c1", "Second quarter": "#de8b7b", "Third quarter": "#c2412d", "Highest quarter": "#7c2318"}
    if col == "pm25_a4t":
        ramp = {k_: v for k_, v in zip(ramp, ["#cfe3d1", "#8fc097", "#3f8f4a", "#1f5428"])}
    if col == "moto_share":
        ramp = {k_: v for k_, v in zip(ramp, ["#cddff0", "#86b3dc", "#1f6fb2", "#0f3e66"])}
    edges = c.groupby("band")[col].agg(["min", "max"])
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        basemap(c, "band", ramp, height=500)
        st.markdown(keys([(f"{b}: {edges.loc[b, 'min']:.1f}–{edges.loc[b, 'max']:.1f}", ramp[b]) for b in ramp]),
                    unsafe_allow_html=True)
    with c2:
        d = X["districts"].sort_values("vehicles")
        fig = go.Figure(go.Bar(x=d.vehicles, y=d.district + "  (" + d.cameras.astype(str) + ")", orientation="h",
                               marker_color=ROAD, customdata=np.c_[d.moto_share, d.truck_share],
                               hovertemplate="%{y}<br>%{x:.1f} vehicles per frame<br>motorcycles %{customdata[0]:.0f}%"
                                             " · trucks %{customdata[1]:.0f}%<extra></extra>"))
        fig.update_layout(xaxis_title="Vehicles per frame", yaxis_tickfont_size=12)
        show(fig, h=560, legend=False, title="Districts, by traffic (cameras)")

    st.subheader("Rail stations and road types")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        r = X["rail"]
        fig = go.Figure()
        for colname, n, colr in [("moto_share", "Motorcycles", WATER), ("truck_share", "Trucks", "#7a5a9e")]:
            fig.add_bar(x=r.rail_band, y=r[colname], name=n, marker_color=colr, text=r[colname].round(0),
                        textposition="outside", hovertemplate="%{x} · %{y:.1f}%<extra>" + n + "</extra>")
        fig.update_layout(barmode="group", yaxis_title="Share of vehicles (%)", xaxis_title="Distance to nearest rail station",
                          yaxis_range=[0, r[["moto_share", "truck_share"]].max().max() * 1.2])
        show(fig, h=340, title="Far from rail, more trucks")
        st.caption(f"{K['near_rail_cams']} of 100 cameras are within 500 m of a BTS, MRT, ARL or SRT station. "
                   "Only 5 cameras are more than 2 km away, so read that bar with care.")
    with c2:
        t = X["roads"][["road_class", "cameras", "vehicles", "moto_share", "truck_share"]].rename(columns={
            "road_class": "Road (OSM)", "cameras": "Cams", "vehicles": "Per frame", "moto_share": "Moto %", "truck_share": "Truck %"})
        st.markdown("**Road type of the nearest street**")
        st.dataframe(t.round(1), hide_index=True, width="stretch")
        st.caption("Secondary roads carry as much traffic as primary roads and twice the motorcycle share.")
    margin("Sources", "Districts: OpenStreetMap Nominatim reverse geocoding.<br/>Rail stations: OpenStreetMap "
                      "(railway=station).<br/>Road type: OpenStreetMap ways within 60 m.")


def page_weather():
    sheet("Rain and dusk", "What the weather and the light do to the counts",
          f"Rain in {K['rain_hours_share']:.0f}% of camera-hours<br/>Sunset {K['sunset']}")
    lede("This was the rainy season: a third of camera-hours had at least 0.5 mm of rain. Traffic hardly reacts. "
         "Compared with the same camera, hour and day type when dry, even heavy rain changes counts by about 1%.")
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        r = X["rain"]
        fig = go.Figure(go.Bar(x=r.change_pct, y=r.rain_band, orientation="h", text=[f"{v:+.1f}%" for v in r.change_pct],
                               textposition="outside", customdata=r.camera_hours,
                               marker_color=[WATER if "METAR" not in b else INK for b in r.rain_band],
                               hovertemplate="%{y}: %{x:+.1f}% · %{customdata:,} camera-hours<extra></extra>"))
        fig.add_vline(x=0, line_color="#8a8b86", line_width=1)
        fig.update_layout(xaxis_range=[-10, 10], xaxis_title="Vehicles vs the dry baseline (%)", yaxis_autorange="reversed")
        show(fig, h=320, legend=False, title="Traffic change by rain")
        st.caption("Rain from the Open-Meteo model at each camera; the black bar uses independent rain reports from the "
                   "Don Mueang and Suvarnabhumi airport weather stations (METAR). Heavy rain covers only 36 camera-hours.")
    with c2:
        h = X["rain_h"]
        fig = go.Figure()
        for t, c, dash in [("Weekday", WATER, "solid"), ("Weekend", INK, "dot")]:
            s = h[h.daytype == t]
            fig.add_scatter(x=s.hour, y=s.rain_pct, name=t, line=dict(color=c, width=2.2, dash=dash),
                            hovertemplate="%{x}:00 · %{y:.0f}%<extra>" + t + "</extra>")
        fig.update_layout(yaxis_title="Camera-hours with rain (%)", xaxis_title="Hour", hovermode="x unified")
        show(fig, h=320, title="When it rained")

    st.subheader("Dusk: the camera goes blind faster than the road empties")
    lede(f"In the 30 minutes after sunset, detections fall {abs(K['dusk_drop_pct']):.0f}% compared with the half hour "
         f"before, while image brightness falls only {abs(K['dusk_brightness_drop_pct']):.0f}%. Weekday traffic at 18:30 "
         "does not halve in half an hour; headlight glare and dark frames make the detector miss vehicles. Treat "
         "counts after 18:30 as a lower bound.")
    d = X["dusk"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure(go.Scatter(x=d.bin, y=d.vehicles, line=dict(color=ROAD, width=2.4), mode="lines+markers",
                                   marker_size=6, hovertemplate="%{x:+} min · %{y:.1f} vehicles<extra></extra>"))
        fig.add_vline(x=0, line_color=INK, line_width=1, annotation_text="sunset", annotation_position="top right")
        fig.update_layout(xaxis_title="Minutes from sunset", yaxis_title="Vehicles per frame", yaxis_range=[0, 15])
        show(fig, h=300, legend=False, title="Detections")
    with c2:
        fig = go.Figure(go.Scatter(x=d.bin, y=d.brightness, line=dict(color=INK, width=2.4), mode="lines+markers",
                                   marker_size=6, hovertemplate="%{x:+} min · brightness %{y:.0f}<extra></extra>"))
        fig.add_vline(x=0, line_color=INK, line_width=1, annotation_text="sunset", annotation_position="top right")
        fig.update_layout(xaxis_title="Minutes from sunset", yaxis_title="Mean pixel brightness (0–255)", yaxis_range=[0, 130])
        show(fig, h=300, legend=False, title="Image brightness")
    dusk_imgs = sorted(FRAMES.glob("dusk_*.jpg"))
    if dusk_imgs:
        st.markdown("**The same camera, every 10 minutes across sunset**")
        cols = st.columns(len(dusk_imgs))
        for col, p in zip(cols, dusk_imgs):
            col.image(str(p), caption=p.stem.split("_", 2)[2].replace("-", ":"), width="stretch")
    margin("Sources", "Rain: Open-Meteo hourly model at each camera; METAR reports from VTBD and VTBS (Iowa "
                      "Environmental Mesonet).<br/>Sunset: Open-Meteo archive.<br/>Brightness: mean pixel value of "
                      "each frame.")


def page_air():
    sheet("Air", "PM2.5 from the official Air4Thai network", f"{K['a4t_stations_used']} monitors matched<br/>"
          f"median {K['a4t_median_m']:,} m from a camera")
    lede("PM2.5 follows the weather, not the traffic. It is highest overnight and in calm air, lowest in the "
         "early afternoon when the air mixes, and shows no link to traffic in the same or the previous three hours.")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        a = X["air"]
        fig = go.Figure()
        for t, c, dash in [("Weekday", GREEN, "solid"), ("Weekend", INK, "dot")]:
            s = a[a.daytype == t]
            fig.add_scatter(x=s.hour, y=s["mean"], name=t, line=dict(color=c, width=2.4, dash=dash),
                            hovertemplate="%{x}:00 · %{y:.1f} µg/m³<extra>" + t + "</extra>")
        fig.add_vrect(x0=6, x1=21, fillcolor=ROAD, opacity=0.05, line_width=0, annotation_text="cameras recording",
                      annotation_position="top left", annotation_font_color=MUTED)
        fig.update_layout(xaxis_title="Hour", yaxis_title="PM2.5 (µg/m³)", hovermode="x unified", yaxis_rangemode="tozero")
        show(fig, h=330, title="PM2.5 over 24 hours, all Bangkok monitors")
    with c2:
        w = X["wind_s"]
        fig = go.Figure(go.Bar(x=w.wind_band, y=w.pm25_a4t, marker_color=GREEN, text=w.pm25_a4t.round(1),
                               textposition="outside", hovertemplate="%{x} km/h · %{y:.1f} µg/m³<extra></extra>"))
        fig.update_layout(xaxis_title="Wind speed (km/h)", yaxis_title="PM2.5 (µg/m³)",
                          yaxis_range=[0, w.pm25_a4t.max() * 1.2])
        show(fig, h=330, legend=False, title="Calm air holds PM2.5")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        l_ = X["lag"]
        fig = go.Figure(go.Bar(x=[f"{h} h" for h in l_.lag_h], y=l_.rho, marker_color=INK, text=l_.rho.round(2),
                               textposition="outside", hovertemplate="traffic %{x} earlier · ρ = %{y:+.3f}<extra></extra>"))
        fig.add_hline(y=0, line_color="#8a8b86", line_width=1)
        fig.update_layout(yaxis_range=[-0.5, 0.5], xaxis_title="Traffic measured this long before the PM2.5 reading",
                          yaxis_title="Spearman ρ")
        show(fig, h=310, legend=False, title="Traffic then, PM2.5 now")
        st.caption("Within each camera and net of hour of day, so the daily cycle cannot fake a link.")
    with c2:
        wd = X["wind"].set_index("wind_sector").reindex(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]).reset_index()
        fig = go.Figure(go.Barpolar(r=wd.pm25, theta=wd.wind_sector, marker_color=GREEN, marker_line_color="#fff",
                                    customdata=wd.hours,
                                    hovertemplate="wind from %{theta} · %{r:.1f} µg/m³ · %{customdata} h<extra></extra>"))
        fig.update_layout(polar=dict(radialaxis=dict(showticklabels=False, ticks="", gridcolor=GRID),
                                     angularaxis=dict(direction="clockwise", gridcolor=GRID), bgcolor="rgba(0,0,0,0)"))
        show(fig, h=310, legend=False, title="PM2.5 by wind direction")
        st.caption("Most hours had a south-westerly monsoon wind and cleaner air; rare north-easterly hours were dirtier.")

    st.subheader("Low-cost sensors read half")
    lede(f"The team's PM2.5 came from the nearest OpenAQ sensor. For the 43% of camera-hours served by low-cost "
         f"sensors (AirGradient, Clarity), the reading is {K['lowcost_ratio']:.0%} of the official Air4Thai monitor "
         "nearest the same camera in the same hour. Air4Thai sensors seen through OpenAQ agree with Air4Thai directly.")
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        pr = X["pairs"]
        fig = go.Figure()
        for t, c in [("Air4Thai (government)", GREEN), ("Low-cost sensor", ROAD)]:
            s = pr[pr.sensor_type == t]
            fig.add_scatter(x=s.pm25_a4t, y=s.pm25, mode="markers", name=t, marker=dict(color=c, size=5, opacity=0.45),
                            hovertemplate="Air4Thai %{x:.1f} · sensor %{y:.1f}<extra>" + t + "</extra>")
        fig.add_scatter(x=[0, 45], y=[0, 45], mode="lines", line=dict(color=INK, dash="dot", width=1), name="Equal readings",
                        hoverinfo="skip")
        fig.update_layout(xaxis_title="Official Air4Thai monitor nearest the camera (µg/m³)",
                          yaxis_title="Sensor the team used (µg/m³)", xaxis_range=[0, 45], yaxis_range=[0, 45])
        show(fig, h=380, title="Sensor reading vs official monitor, same hour")
    with c2:
        b = X["bias"][["sensor_type", "openaq", "a4t", "r"]].rename(columns={
            "sensor_type": "Sensor", "openaq": "Sensor mean", "a4t": "Official", "r": "Corr."})
        st.markdown("**Means over the matched camera-hours (µg/m³)**")
        st.dataframe(b.round(2), hide_index=True, width="stretch")
        st.caption("Correlations are modest even for Air4Thai sensors because the team's sensor and the nearest official "
                   "monitor are often different stations.")
    margin("Sources", "Official PM2.5: Air4Thai (Pollution Control Department), 79 Bangkok stations, hourly.<br/>"
                      "Team PM2.5: OpenAQ v3, provider looked up per sensor.<br/>Wind: Open-Meteo.")


def page_unusual():
    sheet("Unusual moments", "Frames far above their camera's normal level", f"{K['unusual_frames']:,} frames over 4σ")
    lede("Each frame is compared with the median of the same camera, same 10-minute slot and same day type. These are "
         "the strongest jumps, one per camera and day. Most are queues at signals or a bus or truck platoon caught at "
         "the right moment.")
    u = X["unusual"]
    have = [p for p in u.file if (FRAMES / p).exists()]
    if have:
        g = u[u.file.isin(have)].head(9)
        cols = st.columns(3, gap="medium")
        for i, r in enumerate(g.itertuples()):
            with cols[i % 3]:
                st.image(str(FRAMES / r.file), width="stretch")
                st.markdown(f"**{r.total} vehicles** where {r.base:.0f} is normal · {r.ts:%a %d %b %H:%M}  \n"
                            f"<span class='note'>{r.name} · {r.district}</span>", unsafe_allow_html=True)
    t = u.assign(when=u.ts.dt.strftime("%a %d %b %H:%M"), z=u.z.round(1), rain=u.precipitation)[
        ["when", "name", "district", "total", "base", "z", "rain"]]
    st.markdown("**All top events**")
    st.dataframe(t.rename(columns={"when": "When", "camera_id": "Camera", "name": "Location", "district": "District",
                                   "total": "Vehicles", "base": "Normal", "z": "Robust z", "rain": "Rain (mm)"}),
                 hide_index=True, width="stretch", height=380)
    margin("Method", "Robust z = (count − median) / (1.4826 × MAD), per camera, slot and day type; MAD floored at 1.")


def page_sources():
    sheet("Sources and method", "Everything this app draws on", "8 sources<br/>3 scraped for this app")
    src = pd.DataFrame([
        ("BMA traffic CCTV", "61,319 frames, 100 cameras", "Course dataset", "Base"),
        ("YOLOv8m (COCO)", "Car, motorcycle, bus, truck per frame", "Team pipeline, Component 1", "Base"),
        ("OpenAQ v3", "Hourly PM2.5 at the nearest sensor, provider per sensor", "api.openaq.org", "Team"),
        ("Open-Meteo", "Hourly weather at each camera; sunrise and sunset", "open-meteo.com", "Team + new"),
        ("Air4Thai (PCD)", "Official hourly PM2.5, 79 Bangkok stations", "air4thai.com", "Reused"),
        ("Iowa Environmental Mesonet", "METAR reports, Don Mueang and Suvarnabhumi", "mesonet.agron.iastate.edu", "Reused"),
        ("OpenStreetMap Nominatim", "District of every camera; 153 rail stations", "nominatim.openstreetmap.org", "New"),
        ("OpenStreetMap API", "Road type within 60 m of each camera", "api.openstreetmap.org", "Reused"),
    ], columns=["Source", "What it gives", "Where", "In this app"])
    st.dataframe(src, hide_index=True, width="stretch")
    st.subheader("Read the findings with these limits")
    for t in [
        "One week in the rainy season. Patterns may differ in the dry, smoggy months (November to March).",
        "Counts are vehicles visible in one frame every 10 minutes, a proxy for traffic, not a flow count.",
        "The detector misses vehicles at dusk and at night; evening counts are lower bounds.",
        "PM2.5 is measured at monitors a median of about 1 km from the cameras, not at the roadside.",
        "Associations only. Nothing here identifies a cause.",
        "Truck rules: Bangkok's heavy-truck restrictions vary by season and pollution level; we could not verify "
        "the exact hours in force for this week, so no truck-ban finding is drawn.",
    ]:
        st.markdown(f"- {t}")
    st.caption("Rebuild: python scrape.py && python build_insights.py from the repo root, then streamlit run app.py here.")


PAGES = {
    "page_sheet": st.Page(page_sheet, title="Overview", default=True, url_path="overview"),
    "page_rhythm": st.Page(page_rhythm, title="Rhythm", url_path="rhythm"),
    "page_places": st.Page(page_places, title="Places", url_path="places"),
    "page_weather": st.Page(page_weather, title="Rain and dusk", url_path="rain-and-dusk"),
    "page_air": st.Page(page_air, title="Air", url_path="air"),
    "page_unusual": st.Page(page_unusual, title="Unusual moments", url_path="unusual"),
    "page_sources": st.Page(page_sources, title="Sources", url_path="sources"),
}
st.sidebar.markdown("### Map sheet margin")
margin("Coverage", "25–31 August 2026 · 06:00–21:00<br/>100 cameras · 1 frame per 10 minutes")
st.navigation(list(PAGES.values()), position="top").run()
