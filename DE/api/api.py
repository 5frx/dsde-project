"""Combine the CCTV image dataframe with other data sources and save it.

Weather (Open-Meteo), POIs (Overpass) and congestion could be added here
the same way as air quality: one function per source, called from make_df().
"""

import sys
from pathlib import Path

# Add the parent directory (DE) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import base_and_object_detection.base as base
import airquality as aq
import weather as weather
import poi as poi

OUTPUT_PATH = Path("data/processed/test.csv")


def make_df():
    df_base = base.create_df()
    df = aq.build_dataset(df_base, timestamp_col="hourly_timestamp")
    df = weather.add_weather(df, timestamp_col="hourly_timestamp")
    df = poi.add_poi(df)
    return df

def main():
    df = make_df()
    print(df.head())
    print(df.info())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)



if __name__ == "__main__":
    main()