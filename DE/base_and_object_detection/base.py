from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Dict matching camera id with latitude/longitude
cm = pd.read_csv("data/raw/cctv_info.csv")
cm = cm[['camera_id', 'latitude', 'longitude']]
LocDict = cm.set_index('camera_id').to_dict(orient='index')

IMAGE_ROOT = Path("data/image/cctv_images")


# Step 1: build the base df (matches each image with location and time)
def create_base():
    listdict = []

    for cam in IMAGE_ROOT.iterdir():
        cam_id = int(cam.stem)
        for img in cam.glob("*.*"):
            name = img.name
            parts = img.stem.split("_", 3)
            datetime_str = parts[3]

            raw_dt = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
            hourly_dt = raw_dt.replace(minute=0, second=0)

            cam_info = LocDict.get(cam_id)

            listdict.append({
                "image_name": name,
                "camera_id": cam_id,
                "latitude": cam_info["latitude"],
                "longitude": cam_info["longitude"],
                "raw_timestamp": raw_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "hourly_timestamp": hourly_dt.strftime("%Y-%m-%d %H:%M:%S"),
            })

    return pd.DataFrame(listdict)


# Step 2: remove pure white images
def is_white(path, white_mean=250, max_std=3):
    data = np.fromfile(path, dtype=np.uint8)   # avoids Windows unicode path issues
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:        # unreadable file, leave it alone
        return False
    return img.mean() > white_mean and img.std() < max_std


def remove_white(df):
    mask = []
    for cam_id, name in tqdm(zip(df["camera_id"], df["image_name"]),
                             total=len(df), desc="Checking for white images"):
        path = IMAGE_ROOT / str(cam_id) / name
        try:
            mask.append(is_white(path))
        except FileNotFoundError:
            mask.append(False)

    mask = pd.Series(mask, index=df.index)
    print(f"Removing {mask.sum()} white images out of {len(df)}")

    flagged = df[mask]
    clean = df[~mask].reset_index(drop=True)
    return clean, flagged


def create_df():
    df = create_base()
    # Sort so row order is deterministic (glob order isn't guaranteed)
    df = df.sort_values(["camera_id", "raw_timestamp"]).reset_index(drop=True)

    clean, flagged = remove_white(df)
    return clean

