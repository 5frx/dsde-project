# Component 2: Data Science

Applies analytical methods to the dataset Component 1 prepares, to explore the
patterns, trends and insights in it. The brief defines the project objective as
analysing meaningful spatial and/or temporal patterns; this module does both and
hands tidy tables to Component 3.

| Report section | Kind |
|---|---|
| 1 What the data contains, 3 detector channels, 4 variance, 6 built environment | what the data can support |
| 2 Daily cycle, 8 weekly rhythm, 9 anomalies | temporal |
| 5 Spatial structure, 7 camera typology | spatial |
| 10, 11 PM2.5 | extension, runs when the target exists |
| 12 Dashboard handoff, 13 Insights | Components 3 and 4 |

## Running it

```bash
python Component2_DataSci/main.py          # from the repository root
```

`Component2_Findings.docx` in this folder is the written report: the patterns,
correlations, trends and insights, with the figures embedded. Read that first.

Running the analysis writes `Component2_DataSci/output/component2_report.md`
(the same findings in full technical detail), plus every table as a CSV in
`Component2_DataSci/output/tables/`, every figure as a PNG in `Component2_DataSci/output/figures/`, and three
tidy tables for the dashboard in `Component2_DataSci/output/dashboard/`:

| File | Grain |
|---|---|
| `cameras.csv` | one row per camera: location, profile type, mean counts, POI features |
| `camera_hours.csv` | one row per camera-hour: counts, expected value, deviation score |
| `anomaly_flags.csv` | one row per flagged camera-hour |

Component 3 should read those rather than re-deriving the analysis.

## What it reads

It prefers `data/processed/data.csv`, the full output of `DE/main.py`. If that
has not been built yet it falls back to `data/processed/Carcounts.csv` (base +
object detection) joined to camera-level POI features, and reports the
air-quality and weather sections as skipped rather than failing.

To generate the POI features for that fallback path without running the whole
DE pipeline:

```bash
python Component2_DataSci/make_camera_poi.py     # -> data/processed/camera_poi.csv
```

It reuses `DE/api/poi_local.py` and `DE/api/poi_select.py`, so the feature
definitions cannot drift away from the full pipeline's.

## Checking that the PM2.5 path works

Sections 7 and 8 only run when the target column exists, so until the OpenAQ
pull is done that code is untested. This builds a synthetic `data.csv` in the
exact schema `DE/api/api.make_df` emits, runs the analysis against it, and
restores the repository afterwards:

```bash
python Component2_DataSci/smoke_test_pm25.py     # prints PASS or FAIL
```

The synthetic PM2.5 and weather values are random and independent of the real
vehicle counts, so the test also checks that the analysis reports nulls on noise
instead of inventing a relationship.

## Files

| File | Contains |
|---|---|
| `main.py` | Orchestration and report generation |
| `config.py` | Paths, column groups, thresholds |
| `features.py` | Loading, feature engineering, the three aggregation levels |
| `stats_tools.py` | Correlation with CIs, FDR, cluster bootstrap, partial correlation, ICC, Moran's I, VIF |
| `analysis.py` | The eight analysis sections |
| `figures.py` | Plotting |
| `make_camera_poi.py` | POI features for the fallback path |
| `smoke_test_pm25.py` | Test for the not-yet-reachable PM2.5 code |

## The one thing to know before reading the numbers

The prepared file has one row per image, but the variables live at three
different levels:

| Level | Rows | What varies here |
|---|---|---|
| image | 61,319 | vehicle counts, plus detector noise |
| camera-hour | 10,728 | PM2.5, weather, averaged traffic |
| camera | 100 | POI and road features, mean traffic |

Correlating a camera-constant feature (any `poi_*` column) against the 61,319
image rows repeats each camera's single value 613 times. The coefficient barely
moves but the p-value collapses by hundreds of orders of magnitude, because the
test believes it has 61,319 independent observations when it has 100. Every
result in the report states the level it was computed at, and the POI section
shows the same correlation both ways so the size of the trap is visible.

## Dependencies

Beyond the repository's `requirements.txt`: `scipy`, `matplotlib` and
`tabulate` (the last only for writing markdown tables).
