# runs everything and puts it all into a .csv file
import requests
import os
import pandas as pd

import api.api as api
import api.airquality as aq
import base_and_object_detection.base as base
import base_and_object_detection.OD as OD
# check if Carcounts.csv exists, if not, run the code to create it

PATH_CARCOUNTS = "data/processed/Carcounts.csv"

def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = f"{dest}.part"                       # atomic: a half-finished download is never used
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        done, next_report = 0, 0.1
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    print(f"[de] downloading {os.path.basename(dest)}: {done / total:.0%}")
                    next_report += 0.1
    os.replace(tmp, dest)

def check_and_create_carcsv(base_df: pd.DataFrame, run_detection: bool = False):
    if os.path.exists(PATH_CARCOUNTS):
        print("Already exists")
        return  # file already exists, no need to create it
    if run_detection:
        print("Running car detection to create Carcounts.csv...")
        cars = OD.CarsDetection_df(base_df)
        cars.to_csv(PATH_CARCOUNTS, index=False)
    # OR JUST DOWNLOAD THE FILE FROM THE REPO, SINCE IT'S ALREADY PROCESSED
    _download("https://github.com/5frx/dsde-project/releases/download/processed_data/Carcounts.csv", PATH_CARCOUNTS)


def main():
    base_df = base.create_df()
    check_and_create_carcsv(base_df, run_detection=False)
    df = api.make_df(base_df)
    od_df = pd.read_csv(PATH_CARCOUNTS)
    merged = df.merge(od_df, on=["image_name","camera_id","latitude", "longitude", "raw_timestamp", "hourly_timestamp"],how="left")
    merged.to_csv("data/processed/data.csv", index=False)

if __name__ == "__main__":
    main()