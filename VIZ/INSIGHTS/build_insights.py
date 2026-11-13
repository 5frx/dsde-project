"""Build every table the insights app reads (run from the repo root after scrape.py; ~30 s).

Inputs
  Component2_DataSci/data/processed/data.csv   frames: YOLOv8 counts, OpenAQ PM2.5, Open-Meteo weather, OSM POI
  VIZ/data/pm25_sensors.csv                    who runs each OpenAQ sensor (read only)
  INSIGHTS/data/external/                       Air4Thai PM2.5 (79 stations), METAR, OSM roads, frame brightness
  INSIGHTS/data/                                scrape.py output: districts, rail stations, sunrise/sunset
Output
  INSIGHTS/data/derived/*.csv + kpis.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
EXT, OUT = HERE / "data" / "external", HERE / "data" / "derived"
V = ["car_count", "motorcycle_count", "bus_count", "truck_count"]
WEEKEND = {"2026-08-29", "2026-08-30"}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def nearest(lat, lon, plat, plon):
    """Index of and distance to the nearest point for every (lat, lon)."""
    d = haversine(np.asarray(lat)[:, None], np.asarray(lon)[:, None], np.asarray(plat)[None], np.asarray(plon)[None])
    return d.argmin(1), d.min(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(ROOT / "Component2_DataSci" / "data" / "processed" / "data.csv")
    f["ts"] = pd.to_datetime(f.raw_timestamp)
    f["total"] = f[V].sum(axis=1)
    f["date"] = f.ts.dt.strftime("%Y-%m-%d")
    f["daytype"] = np.where(f.date.isin(WEEKEND), "Weekend", "Weekday")
    f["slot"] = f.ts.dt.floor("10min").dt.strftime("%H:%M")
    f["hour"] = f.ts.dt.hour
    bright = pd.read_csv(EXT / "frame_brightness.csv", parse_dates=["ts"])
    f = f.merge(bright, on=["camera_id", "ts"], how="left")
    k = {}

    # ---------------------------------------------------------------- cameras + places
    cams = f.groupby("camera_id").agg(lat=("latitude", "first"), lon=("longitude", "first"), vehicles=("total", "mean"),
                                      **{c: (c, "mean") for c in V}).reset_index()
    cams["moto_share"] = 100 * cams.motorcycle_count / cams.vehicles
    cams["truck_share"] = 100 * cams.truck_count / cams.vehicles
    cams["bus_share"] = 100 * cams.bus_count / cams.vehicles
    info = pd.read_csv(ROOT / "data" / "raw" / "cctv_info.csv")
    info["name"] = np.where(info.location_en.fillna("").str.contains("BMAcctv") | info.location_en.isna(),
                            info.location, info.location_en)
    cams = cams.merge(info[["camera_id", "name"]], on="camera_id", how="left")
    cams = cams.merge(pd.read_csv(HERE / "data" / "camera_districts.csv"), on="camera_id", how="left")
    rail = pd.read_csv(HERE / "data" / "rail_stations.csv")
    i, d = nearest(cams.lat, cams.lon, rail.lat, rail.lon)
    cams["rail_station"], cams["rail_m"] = rail.station.values[i], d.round()
    roads = pd.read_csv(EXT / "osm_roads_near_cameras.csv")
    roads = roads[~roads.highway.isin(["service", "footway", "steps", "path", "pedestrian"])]
    roads["highway"] = roads.highway.str.replace("_link", "", regex=False)
    road = roads.sort_values("dist_m").groupby("camera_id").first()
    cams["road_class"] = cams.camera_id.map(road.highway).fillna("other")
    cams["road_name"] = cams.camera_id.map(road.name_en.fillna(road.name))

    # daily shape clusters: each camera's hourly weekday profile divided by its own mean
    prof = f[f.daytype == "Weekday"].pivot_table(index="camera_id", columns="hour", values="total", aggfunc="mean")
    prof = prof.div(prof.mean(axis=1), axis=0).astype(float).fillna(1.0)
    rng = np.random.default_rng(7)
    best = None
    for _ in range(20):  # kmeans2 has no n_init; keep the tightest of 20 starts
        cen, lab = kmeans2(prof.values, 4, seed=rng, minit="++")
        sse = ((prof.values - cen[lab]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, cen, lab)
    _, cen, lab = best
    names = {}
    for c in range(4):
        row = pd.Series(cen[c], index=prof.columns)
        am, pm, night = row.loc[6:9].max(), row.loc[15:19].max(), row.loc[19:21].mean()
        if am > 1.5 and pm > 1.3:
            names[c] = "Two rush hours"
        elif pm > 1.8:
            names[c] = "Evening spike"
        elif night > 0.6:
            names[c] = "Busy into the night"
        else:
            names[c] = "Daytime plateau"
    for c in range(4):  # two centroids can earn the same name; number them
        dup = [x for x in names if names[x] == names[c]]
        if len(dup) > 1:
            for j, x in enumerate(sorted(dup)):
                names[x] = f"{names[c].split(' (')[0]} ({j + 1})"
    cams = cams.merge(pd.DataFrame({"camera_id": prof.index, "shape": [names[x] for x in lab]}), on="camera_id")
    shapes = pd.DataFrame(cen, columns=prof.columns).assign(shape=[names[c] for c in range(4)])
    shapes = shapes.melt("shape", var_name="hour", value_name="index")
    shapes["cameras"] = shapes["shape"].map(cams["shape"].value_counts())
    shapes.to_csv(OUT / "shapes.csv", index=False)

    # ---------------------------------------------------------------- the pulse (10-minute resolution)
    per_day = f.groupby(["daytype", "date", "slot"]).total.mean().reset_index()
    pulse = per_day.groupby(["daytype", "slot"]).total.mean().reset_index()
    cls = f.groupby(["daytype", "slot"])[V].mean().reset_index()
    pulse = pulse.merge(cls, on=["daytype", "slot"])
    pulse.to_csv(OUT / "pulse.csv", index=False)
    wd, we = (f[f.daytype == t].groupby(["camera_id", "slot"]).total.mean() for t in ("Weekday", "Weekend"))
    both = pd.concat([wd, we], axis=1, keys=["wd", "we"]).dropna()
    k["weekend_change_pct"] = round(100 * (both.we.sum() / both.wd.sum() - 1), 1)
    wdp = pulse[pulse.daytype == "Weekday"].set_index("slot").total
    k["am_peak_slot"] = wdp.loc["06:00":"10:50"].idxmax()
    k["pm_peak_slot"] = wdp.loc["15:00":"19:50"].idxmax()
    k["pm_peak_vehicles"] = round(wdp.max(), 1)
    k["frames"] = int(len(f))
    k["cameras"] = int(f.camera_id.nunique())
    k["vehicles_counted"] = int(f.total.sum())
    daily = f.groupby("date").total.mean().reset_index().rename(columns={"total": "vehicles"})
    daily["weekday"] = pd.to_datetime(daily.date).dt.day_name()
    daily.to_csv(OUT / "daily.csv", index=False)

    # ---------------------------------------------------------------- dusk: is the evening drop real?
    sun = pd.read_csv(HERE / "data" / "sun.csv")
    f = f.merge(sun[["date", "sunset"]], on="date", how="left")
    f["min_from_sunset"] = (f.ts - pd.to_datetime(f.sunset)).dt.total_seconds() / 60
    dusk = f[f.min_from_sunset.between(-120, 150)].copy()
    dusk["bin"] = (dusk.min_from_sunset // 10 * 10).astype(int)
    dusk = dusk.groupby("bin").agg(vehicles=("total", "mean"), brightness=("brightness", "mean"),
                                   frames=("total", "size")).reset_index()
    dusk.to_csv(OUT / "dusk.csv", index=False)
    k["sunset"] = sun.sunset.str[-5:].iloc[0] + "–" + sun.sunset.str[-5:].iloc[-1]
    # the 30 minutes before vs after sunset: real traffic does not fall this fast at 18:30 on a weekday
    pre, post = dusk[dusk.bin.between(-40, -11)], dusk[dusk.bin.between(10, 39)]
    k["dusk_drop_pct"] = round(100 * (post.vehicles.mean() / pre.vehicles.mean() - 1), 1)
    k["dusk_brightness_drop_pct"] = round(100 * (post.brightness.mean() / pre.brightness.mean() - 1), 1)

    # ---------------------------------------------------------------- rain
    ch = f.groupby(["camera_id", "hourly_timestamp"]).agg(
        vehicles=("total", "mean"), rain=("precipitation", "first"), daytype=("daytype", "first"),
        hour=("hour", "first"), pm25=("pm25", "first"), sensor_id=("sensor_id", "first"),
        wind=("wind_speed_10m", "first"), wind_dir=("wind_direction_10m", "first"),
        **{c: (c, "mean") for c in V}).reset_index()
    ch["t"] = pd.to_datetime(ch.hourly_timestamp)
    dry = ch[ch.rain == 0].groupby(["camera_id", "hour", "daytype"]).vehicles.median().rename("dry_base")
    ch = ch.join(dry, on=["camera_id", "hour", "daytype"])
    ch["rain_band"] = pd.cut(ch.rain, [-0.01, 0, 1, 5, 100], labels=["Dry", "Light (<1 mm)", "Moderate (1–5 mm)",
                                                                     "Heavy (>5 mm)"])
    ok = ch[ch.dry_base > 0]
    rain = ok.groupby("rain_band", observed=True).apply(
        lambda g: pd.Series({"change_pct": 100 * (g.vehicles.sum() / g.dry_base.sum() - 1), "camera_hours": len(g),
                             "moto_share": 100 * g.motorcycle_count.sum() / g.vehicles.sum()}),
        include_groups=False).reset_index()
    # METAR: did either airport report rain that hour? (independent observation, not a model)
    m = pd.read_csv(EXT / "metar_bangkok_airports.csv", usecols=["station", "valid", "wxcodes"])
    m["t"] = (pd.to_datetime(m.valid) + pd.Timedelta(hours=7)).dt.floor("h")
    wet = set(m[m.wxcodes.fillna("").str.contains("RA")].t)
    ok = ok.assign(metar_rain=ok.t.isin(wet))
    mr = ok.groupby("metar_rain").apply(lambda g: 100 * (g.vehicles.sum() / g.dry_base.sum() - 1), include_groups=False)
    rain["metar_change_pct"] = np.nan
    rain = pd.concat([rain, pd.DataFrame([{"rain_band": "Airport METAR reports rain", "change_pct": mr.get(True),
                                           "camera_hours": int(ok.metar_rain.sum()), "moto_share": 100 *
                                           ok[ok.metar_rain].motorcycle_count.sum() / ok[ok.metar_rain].vehicles.sum()}])])
    rain.drop(columns="metar_change_pct").to_csv(OUT / "rain.csv", index=False)
    k["heavy_rain_change_pct"] = round(float(rain.set_index("rain_band").loc["Heavy (>5 mm)", "change_pct"]), 1)
    k["rain_hours_share"] = round(100 * (ch.rain >= 0.5).mean(), 1)  # >= 0.5 mm: model drizzle below that is noise
    rh = ch.groupby(["daytype", "hour"]).rain.apply(lambda s: 100 * (s >= 0.5).mean()).reset_index(name="rain_pct")
    rh.to_csv(OUT / "rain_by_hour.csv", index=False)

    # ---------------------------------------------------------------- air: official Air4Thai vs traffic
    st_ = pd.read_csv(EXT / "air4thai_stations.csv")
    aq = pd.read_csv(EXT / "air4thai_pm_hourly.csv", parse_dates=["DATETIMEDATA"])
    aq = aq[aq.stationID.isin(st_.stationID)]
    i, d = nearest(cams.lat, cams.lon, st_.lat, st_["long"])
    cams["a4t_station"], cams["a4t_name"], cams["a4t_m"] = st_.stationID.values[i], st_.nameEN.values[i], d.round()
    ch = ch.merge(cams[["camera_id", "a4t_station"]], on="camera_id")
    ch = ch.merge(aq.rename(columns={"stationID": "a4t_station", "DATETIMEDATA": "t", "PM25": "pm25_a4t"})
                  [["a4t_station", "t", "pm25_a4t"]], on=["a4t_station", "t"], how="left")
    sensors = pd.read_csv(ROOT / "VIZ" / "data" / "pm25_sensors.csv")
    ch = ch.merge(sensors[["sensor_id", "sensor_type"]], on="sensor_id", how="left")
    k["a4t_stations_used"] = int(cams.a4t_station.nunique())
    k["a4t_median_m"] = int(cams.a4t_m.median())
    k["pm25_a4t_mean"] = round(ch.pm25_a4t.mean(), 1)
    # citywide diurnal from all Bangkok Air4Thai stations, full 24 h (cameras only cover 06–21)
    aq["hour"] = aq.DATETIMEDATA.dt.hour
    aq["daytype"] = np.where(aq.DATETIMEDATA.dt.strftime("%Y-%m-%d").isin(WEEKEND), "Weekend", "Weekday")
    aq.groupby(["daytype", "hour"]).PM25.agg(["mean", "median"]).reset_index().to_csv(OUT / "air_24h.csv", index=False)
    # traffic now vs PM2.5 later, within camera and net of hour of day
    ch = ch.sort_values(["camera_id", "t"])
    lag_rows = []
    for lag in range(0, 4):
        x = ch.groupby("camera_id").vehicles.shift(lag)
        same_day = ch.t.dt.date == (ch.t - pd.Timedelta(hours=lag)).dt.date
        g = ch.assign(x=x).loc[same_day].dropna(subset=["x", "pm25_a4t"])
        for c in ("x", "pm25_a4t"):
            g[c] = g[c] - g.groupby("camera_id")[c].transform("mean")
            g[c] = g[c] - g.groupby("hour")[c].transform("mean")
        rho, p = spearmanr(g.x, g.pm25_a4t)
        lag_rows.append({"lag_h": lag, "rho": rho, "n": len(g)})
    pd.DataFrame(lag_rows).to_csv(OUT / "lag.csv", index=False)
    k["lag_rho_max_abs"] = round(max(abs(r["rho"]) for r in lag_rows), 3)
    # wind direction rose for PM2.5 (official)
    ch["wind_sector"] = ((ch.wind_dir + 22.5) // 45 % 8).map(dict(enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])))
    ch.groupby("wind_sector").agg(pm25=("pm25_a4t", "mean"), wind=("wind", "mean"),
                                  hours=("pm25_a4t", "size")).reset_index().to_csv(OUT / "wind.csv", index=False)
    wb = ch.assign(wind_band=pd.cut(ch.wind, [0, 5, 10, 15, 40], labels=["0–5", "5–10", "10–15", "15+"]))
    wb.groupby("wind_band", observed=True).pm25_a4t.mean().reset_index().to_csv(OUT / "wind_speed.csv", index=False)
    # low-cost sensors vs the official monitor nearest the same camera, same hour
    pair = ch.dropna(subset=["pm25", "pm25_a4t"])
    bias = pair.groupby("sensor_type").agg(openaq=("pm25", "mean"), a4t=("pm25_a4t", "mean"),
                                            hours=("pm25", "size"),
                                            r=("pm25", lambda s: s.corr(pair.loc[s.index, "pm25_a4t"]))).reset_index()
    bias.to_csv(OUT / "sensor_bias.csv", index=False)
    lc = bias.set_index("sensor_type").loc["Low-cost sensor"]
    k["lowcost_ratio"] = round(lc.openaq / lc.a4t, 2)
    pair.sample(min(4000, len(pair)), random_state=1)[["pm25", "pm25_a4t", "sensor_type"]].to_csv(
        OUT / "sensor_pairs.csv", index=False)
    cams = cams.merge(ch.groupby("camera_id").pm25_a4t.mean().rename("pm25_a4t"), on="camera_id", how="left")

    # ---------------------------------------------------------------- places
    def summarise(by):
        g = cams.groupby(by).agg(cameras=("camera_id", "size"), vehicles=("vehicles", "mean"),
                                 moto_share=("moto_share", "mean"), truck_share=("truck_share", "mean"),
                                 bus_share=("bus_share", "mean"), pm25_a4t=("pm25_a4t", "mean")).reset_index()
        return g.sort_values("vehicles", ascending=False)
    summarise("district").to_csv(OUT / "districts.csv", index=False)
    summarise("road_class").to_csv(OUT / "road_class.csv", index=False)
    cams["rail_band"] = pd.cut(cams.rail_m, [0, 500, 1000, 2000, 1e6],
                               labels=["< 500 m", "500 m–1 km", "1–2 km", "> 2 km"])
    summarise("rail_band").sort_values("rail_band").to_csv(OUT / "rail.csv", index=False)
    k["districts"] = int(cams.district.nunique())
    k["rail_stations"] = int(len(rail))
    k["near_rail_cams"] = int((cams.rail_m < 500).sum())

    # ---------------------------------------------------------------- unusual moments
    base = f.groupby(["camera_id", "daytype", "slot"]).total.agg(["median", lambda s: (s - s.median()).abs().median()])
    base.columns = ["base", "mad"]
    a = f.join(base, on=["camera_id", "daytype", "slot"])
    a["z"] = (a.total - a.base) / (1.4826 * a.mad.clip(lower=1))
    top = a[a.z > 4].sort_values("z", ascending=False)
    top = top.drop_duplicates(["camera_id", "date"]).head(40)
    top = top.merge(cams[["camera_id", "name", "district"]], on="camera_id", how="left")
    top["file"] = top.camera_id.astype(str) + "_" + top.camera_id.astype(str) + "_" + top.ts.dt.strftime("%Y%m%d_%H%M%S") + ".jpg"
    top[["camera_id", "name", "district", "ts", "total", "base", "z", "precipitation", "file", *V]].to_csv(
        OUT / "unusual.csv", index=False)
    k["unusual_frames"] = int((a.z > 4).sum())

    cams.to_csv(OUT / "cameras.csv", index=False)
    ch[["camera_id", "t", "vehicles", "pm25", "pm25_a4t", "rain", "wind", "sensor_type", *V]].to_csv(
        OUT / "camera_hours.csv", index=False)
    (OUT / "kpis.json").write_text(json.dumps(k, indent=2, default=str))
    print(json.dumps(k, indent=2, default=str))

    # self-checks
    assert cams.camera_id.is_unique and len(cams) == 100
    assert cams["shape"].notna().all() and cams.district.notna().all()
    assert -60 < k["weekend_change_pct"] < 20 and 0 < k["lowcost_ratio"] < 2


if __name__ == "__main__":
    main()
