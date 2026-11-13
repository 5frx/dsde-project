# Component 3: Visualization (Streamlit)

Interactive dashboard on top of Component 2's outputs.

```bash
pip install -r VIZ/requirements.txt
streamlit run VIZ/main.py          # run from the repo root
```

## What it reads
- `Component2_DataSci/output/dashboard/*.csv` (camera-hours, cameras, station-hours, correlations)
- `data/raw/cctv_info.csv` for camera names
- `VIZ/data/pm25_sensors.csv`: who runs each of the 31 PM2.5 sensors (from the OpenAQ v3 `/locations` endpoint).
  Regenerate with `python VIZ/fetch_sensor_metadata.py` after putting `OPENAQ_API_KEY=...` in `.env`
  (`.env` is git-ignored, so never commit the key).

## Tabs
| Tab | Shows |
|---|---|
| Daily patterns | vehicles per frame by class and PM2.5 by hour; day x hour heatmaps |
| Map | cameras coloured by traffic, PM2.5, truck share or motorcycle share; matched PM2.5 sensors |
| Camera explorer | one camera's hourly vehicle classes, PM2.5 and weather, plus which sensor fed it |
| Traffic vs PM2.5 | Spearman correlations (raw, or within camera and net of hour of day), density plot, PM2.5 by wind speed, Component 2 regression result |
| Data quality | Air4Thai vs low-cost sensor split, fallback share, zero readings, sensor table, camera coverage |

Sidebar filters (dates, hours, PM2.5 source, cameras) apply to every tab. The theme (colours, IBM Plex Sans) lives in `.streamlit/config.toml` at the repo root, so run Streamlit from the repo root.

## Note on PM2.5 sources
17 of the 31 sensors are Air4Thai government monitors; 14 are low-cost sensors (AirGradient, Clarity, one unnamed).
Low-cost sensors supply 43% of camera-hours, read about half as high (median 4.3 vs 12.0 µg/m³) and produce all 299 zero
readings. Use the "PM2.5 source" filter to check that a finding holds on Air4Thai data alone.
