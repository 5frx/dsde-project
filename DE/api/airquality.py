import os
import sys 
from dotenv import load_dotenv
import requests
import pandas as pd
import time

load_dotenv()
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

URL = "https://api.openaq.org/v3/"

CCTV_PATH = "data/raw/cctv_info.csv"

if not OPENAQ_API_KEY:
    sys.exit("Set the OPENAQ_API_KEY environment variable first (see the top of this file).")

session = requests.Session()
session.headers.update({
    "X-API-Key": OPENAQ_API_KEY,
    "User-Agent": "dsde-project/1.0 (6838075421@student.chula.ac.th)",
})

def get_location_data(coordinates):
    lat, lon = coordinates
    url = f"{URL}locations?coordinates={lat},{lon}&radius=5000&limit=1"
    response = session.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            time.sleep(1.1)  # Sleep for 1.1 seconds to avoid hitting the rate limit
            return data['results'][0]
        else:
            return None
    elif response.status_code == 429:
        print("Rate limit exceeded. Please try again later.")
        time.sleep(60)  # Wait for 1 minute before retrying
        return get_location_data(coordinates)  # Retry the request
    else:
        print(f"Error: {response.status_code}")
        return None

def get_locations_json():
    df = pd.read_csv(CCTV_PATH)
    locations_data = []
    for index, row in df.iterrows():
        coordinates = (row['latitude'], row['longitude'])
        location_data = get_location_data(coordinates)
        if location_data:
            locations_data.append(location_data)
    return locations_data

print(get_locations_json())

