# Bangkok traffic, one week (insights app)

A separate Streamlit app with the project's deeper findings. It does not change VIZ/.

## Run

    pip install -r INSIGHTS/requirements.txt
    cd INSIGHTS && streamlit run app.py

## Rebuild the data (optional, from the repo root)

    python INSIGHTS/scrape.py          # Nominatim districts + rail stations, Open-Meteo sunset (about 3 min, 1 request/s)
    python INSIGHTS/build_insights.py  # writes INSIGHTS/data/derived/ and runs self-checks

Inputs: Component2_DataSci/data/processed/data.csv, VIZ/data/pm25_sensors.csv and INSIGHTS/data/external/
(Air4Thai hourly PM2.5, airport METAR, OSM road types, frame brightness).

## Pages

Overview, Rhythm, Places, Rain and dusk, Air, Unusual moments, Sources.
