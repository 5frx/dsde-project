#Dict matching camera id with latitude longitude
from pathlib import Path
import pandas as pd
from datetime import datetime

cm = pd.read_csv("data/raw/cctv_info.csv")
cm = cm[['camera_id', 'latitude', 'longitude']]
LocDict = cm.set_index('camera_id').to_dict(orient='index')

#Function that creates the base csv (matches each image with location, and time)
def create_csv():
    folder = Path("data/image/cctv_images")
    listdict = []

    for cam in folder.iterdir():
        cam_id = int(cam.stem)
        for img in cam.glob("*.*"):
            name = img.name
            name_without_ext = img.stem  
            parts = name_without_ext.split("_", 3) 

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
    df = pd.DataFrame(listdict)
    df.to_csv("data/processed/base.csv", index=False)

def create_df():
    folder = Path("data/image/cctv_images")
    listdict = []

    for cam in folder.iterdir():
        cam_id = int(cam.stem)
        for img in cam.glob("*.*"):
            name = img.name
            name_without_ext = img.stem  
            parts = name_without_ext.split("_", 3) 

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
    df = pd.DataFrame(listdict)
    
    return df