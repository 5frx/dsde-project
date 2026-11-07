# Component 2 â€” Data Science: traffic and PM2.5

This folder contains the dataset, code and analysis deliverables for **Component 2: identifying patterns, correlations and insights**, plus tidy CSV exports for **Component 3: visualization**.

## Contents

- `Component2_DataSci/data/processed/data.csv` â€” prepared CCTV, PM2.5, weather and location data used for the analysis.
- `Component2_DataSci/analyze_pm25.py` â€” reads the CSV, validates coverage, computes statistical analyses and writes outputs.
- `Component2_DataSci/make_report.py` â€” regenerates the Word findings report from outputs.
- `Component2_DataSci/Component2_Findings_REVISED.docx` â€” findings, correlations, visualizations, limitations and visualization handoff.
- `Component2_DataSci/output/` â€” generated figures, audit, correlation tables, regression and sensitivity outputs.
- `Component2_DataSci/output/dashboard/` â€” clean visualization-ready CSVs detailed below.
- `Component2_DataSci/requirements_analysis.txt` â€” Python dependencies for reproducibility.

## Reproduce Component 2

Run from this package's root directory:

```bash
python -m pip install -r Component2_DataSci/requirements_analysis.txt
python Component2_DataSci/analyze_pm25.py --csv Component2_DataSci/data/processed/data.csv --out Component2_DataSci/output
python Component2_DataSci/make_report.py
```

## Component 3: visualization-ready files

All paths below are relative to `Component2_DataSci/output/dashboard/`.

| File | Grain | Visualization use |
| --- | --- | --- |
| `cameras.csv` | One row per CCTV camera | Map markers, camera rankings, pollutant/traffic summaries, PM2.5 source-quality filters |
| `camera_hours.csv` | One row per camera and local hour | Traffic/PM2.5 timelines, vehicle class charts, camera/date/hour filters |
| `camera_hourly_profiles.csv` | One row per camera and hour of day | Diurnal profile comparisons |
| `station_hours.csv` | One row per sensor and hour | Sensor-level PM2.5 plots, independent-source awareness |
| `correlations.csv` | One row per tested variable | Ranked association chart, correlation table |

**Suggested Streamlit flow:** Load these CSVs with `pd.read_csv()`. Use `cameras.csv` for map charts (latitude/longitude), `camera_hours.csv` for filters and time-series views, and the provided `correlations.csv` for the statistical panel. Use `camera_id`, `hourly_timestamp` and `sensor_id` as linkage keys where available. Do not count multiple camera records sharing one station-hour as independent pollution readings. Label observations as measured station-attributed PM2.5 and camera-derived vehicle counts, not direct emissions.

**Dataset interpretation:** The PM2.5 values are taken from the supplied prepared dataset, not a newly fetched or independently verified API response. This does not prevent descriptive analysis, but the values' provenance and sensor-hour matching should be checked before stronger environmental claims.

