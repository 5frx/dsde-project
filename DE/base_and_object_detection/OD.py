import os
import shutil
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import cv2

def CarsDetection():
    # 1. Setup Data and Model
    df = pd.read_csv("data/processed/base.csv") 
    model = YOLO('yolov8m.pt') 

    TARGET_CLASSES = {
        2: 'car_count', 
        3: 'motorcycle_count', 
        5: 'bus_count', 
        7: 'truck_count'
    }

    image_paths = (
        "data/image/cctv_images/" + 
        df['camera_id'].astype(str) + "/" + 
        df['image_name'].astype(str)
    ).tolist()

    # 2. Prepare the Staging Directory
    STAGING_DIR = "data/image/temp_cctv_staging"
    os.makedirs(STAGING_DIR, exist_ok=True)

    print("Step 1: Creating safe staging directory (Bypassing Windows encoding errors)...")
    staged_paths = []
    
    # We copy the files and rename them based on their row number (e.g., 000000.jpg, 000001.jpg)
    # The 6-digit zero padding ensures YOLO reads them in the exact alphabetical order of your DataFrame
    for i, original_path in enumerate(tqdm(image_paths, desc="Staging Images")):
        safe_name = f"{i:06d}.jpg"
        safe_path = os.path.join(STAGING_DIR, safe_name)
        
        try:
            # Python's shutil handles Windows Unicode paths perfectly natively
            shutil.copy2(original_path, safe_path)
        except FileNotFoundError:
            # Failsafe: If an image is missing from your drive, create a tiny blank image 
            # in the staging folder so the DataFrame row counts stay perfectly aligned
            blank_image = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(safe_path, blank_image)
            
        staged_paths.append(safe_path)

    # 3. Run Inference on the Staging Directory
    print("Step 2: Running highly optimized YOLO inference...")
    
    # By passing a folder path, YOLO defaults to its internal streaming data loader. 
    # Zero RAM crashes, zero encoding errors.
    results = model.predict(source=STAGING_DIR,conf=0.12, stream=True, verbose=False)

    all_counts = []
    
    for result in tqdm(results, total=len(staged_paths), desc="Processing YOLO Detections"):
        detected_classes = result.boxes.cls.cpu().numpy()
        
        image_counts = {}
        for class_id, col_name in TARGET_CLASSES.items():
            image_counts[col_name] = (detected_classes == class_id).sum()
            
        all_counts.append(image_counts)

    # 4. Merge and Cleanup
    print("Step 3: Saving data and cleaning up...")
    counts_df = pd.DataFrame(all_counts)
    
    # Because we padded the filenames, they merge side-by-side flawlessly
    final_df = pd.concat([df, counts_df], axis=1)
    final_df.to_csv("data/processed/Carcounts.csv", index=False)

    # Automatically delete the staging folder and all 60,000 temporary images
    shutil.rmtree(STAGING_DIR)
    
    print("Processing complete! CSV saved and temporary files deleted.")

def CarsDetection_df(df: pd.DataFrame):
    # 1. Setup Data and Model
    model = YOLO('yolov8m.pt') 

    TARGET_CLASSES = {
        2: 'car_count', 
        3: 'motorcycle_count', 
        5: 'bus_count', 
        7: 'truck_count'
    }

    image_paths = (
        "data/image/cctv_images/" + 
        df['camera_id'].astype(str) + "/" + 
        df['image_name'].astype(str)
    ).tolist()

    # 2. Prepare the Staging Directory
    STAGING_DIR = "data/image/temp_cctv_staging"
    os.makedirs(STAGING_DIR, exist_ok=True)

    print("Step 1: Creating safe staging directory (Bypassing Windows encoding errors)...")
    staged_paths = []
    
    # We copy the files and rename them based on their row number (e.g., 000000.jpg, 000001.jpg)
    # The 6-digit zero padding ensures YOLO reads them in the exact alphabetical order of your DataFrame
    for i, original_path in enumerate(tqdm(image_paths, desc="Staging Images")):
        safe_name = f"{i:06d}.jpg"
        safe_path = os.path.join(STAGING_DIR, safe_name)
        
        try:
            # Python's shutil handles Windows Unicode paths perfectly natively
            shutil.copy2(original_path, safe_path)
        except FileNotFoundError:
            # Failsafe: If an image is missing from your drive, create a tiny blank image 
            # in the staging folder so the DataFrame row counts stay perfectly aligned
            blank_image = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(safe_path, blank_image)
            
        staged_paths.append(safe_path)

    # 3. Run Inference on the Staging Directory
    print("Step 2: Running highly optimized YOLO inference...")
    
    # By passing a folder path, YOLO defaults to its internal streaming data loader. 
    # Zero RAM crashes, zero encoding errors.
    results = model.predict(source=STAGING_DIR,conf=0.12, stream=True, verbose=False)

    all_counts = []
    
    for result in tqdm(results, total=len(staged_paths), desc="Processing YOLO Detections"):
        detected_classes = result.boxes.cls.cpu().numpy()
        
        image_counts = {}
        for class_id, col_name in TARGET_CLASSES.items():
            image_counts[col_name] = (detected_classes == class_id).sum()
            
        all_counts.append(image_counts)

    # 4. Merge and Cleanup
    print("Step 3: Saving data and cleaning up...")
    counts_df = pd.DataFrame(all_counts)
    
    # Because we padded the filenames, they merge side-by-side flawlessly
    final_df = pd.concat([df, counts_df], axis=1)
    shutil.rmtree(STAGING_DIR)
    return final_df
