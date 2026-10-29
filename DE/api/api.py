"""Combine the CCTV image dataframe with other data sources and save it.

Weather (Open-Meteo), POIs (Overpass) and congestion could be added here
the same way as air quality: one function per source, called from make_df().
"""

from pathlib import Path
import sys
import os
# Add the parent directory (DE/) to the system path so Python can find airquality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import airquality as aq

import base_and_object_detection.base as base
from . import weather as weather
from . import poi as poi
from . import poi_local as poi_local

OUTPUT_PATH = Path("data/processed/test.csv")

index = poi_local.PoiIndex(poi_local.INDEX_PATH)

def make_df(base_df):
    df = aq.build_dataset(base_df, timestamp_col="hourly_timestamp")
    df = weather.add_weather(df, timestamp_col="hourly_timestamp")
    df = poi_local.add_poi(df)
    return df

def main():
    base_df = base.create_df()
    df = make_df(base_df)
    print(df.head())
    print(df.info())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)



if __name__ == "__main__":
    main()