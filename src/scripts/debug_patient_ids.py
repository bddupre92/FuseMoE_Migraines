# %%
# % ## Start of debug_patient_ids.py

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Clear existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# --- Configuration ---
# Assuming script is run from the project root or this path is adjusted
# Use absolute path for robustness
BASE_DIR = "/Users/blair.dupre/Downloads/FuseMoE_Migraines" # Adjust if your workspace root is different
DATA_DIR = os.path.join(BASE_DIR, "data", "migraine")
EEG_RAW_PATH = os.path.join(DATA_DIR, "eeg", "all_eeg_data.csv")

# --- Simulate inputs similar to create_multimodal_dataset ---
# Use a representative time window and range from the dev run
start_time = pd.Timestamp("2023-01-01 09:00:00")
end_time = pd.Timestamp("2023-01-10 00:00:00")
time_window = '1H'

# --- Main Debugging Logic ---
def debug_patient_id_join():
    """Isolates and debugs the patient ID joining process."""
    logging.info(f"--- Starting Patient ID Join Debug --- ")
    logging.info(f"Using Base Dir: {BASE_DIR}")
    logging.info(f"Using EEG Raw Path: {EEG_RAW_PATH}")

    # --- 1. Load Raw EEG Data ---
    try:
        eeg_raw_df = pd.read_csv(EEG_RAW_PATH, parse_dates=['timestamp'])
        if eeg_raw_df['timestamp'].dt.tz is not None:
             eeg_raw_df['timestamp'] = eeg_raw_df['timestamp'].dt.tz_localize(None) # Ensure TZ naive
        logging.info(f"Loaded eeg_raw_df: Shape={eeg_raw_df.shape}")
        raw_patient_counts = eeg_raw_df['patient_id'].value_counts()
        logging.info(f"Raw EEG Patient Counts ({len(raw_patient_counts)} unique):\\n{raw_patient_counts.head()}")
        logging.info(f"Raw EEG timestamp duplicates: {eeg_raw_df['timestamp'].duplicated().sum()}")
        print("\n--- Raw EEG DataFrame Head ---")
        print(eeg_raw_df.head())
        print("-" * 30)
    except FileNotFoundError:
        logging.error(f"EEG Raw file not found at {EEG_RAW_PATH}. Cannot proceed.")
        return
    except Exception as e:
        logging.error(f"Error loading raw EEG data: {e}")
        return

    # --- 2. Create Patient ID Map ---
    try:
        logging.info("Creating unique patient ID map (groupby timestamp -> first patient)... ")
        patient_id_map = eeg_raw_df.groupby('timestamp')['patient_id'].first()
        patient_id_map = patient_id_map.sort_index()
        logging.info(f"Created patient_id_map: Length={len(patient_id_map)}")
        map_patient_counts = patient_id_map.value_counts()
        logging.info(f"Patient ID Map Counts ({len(map_patient_counts)} unique):\\n{map_patient_counts.head()}")
        if patient_id_map.index.duplicated().any():
            logging.warning(f"**Duplicates found in patient_id_map index! Count: {patient_id_map.index.duplicated().sum()}**")
        else:
            logging.info("patient_id_map index is unique.")
        print("\n--- Patient ID Map Head ---")
        print(patient_id_map.head())
        print("-" * 30)
    except Exception as e:
        logging.error(f"Error creating patient ID map: {e}")
        return

    # --- 3. Create Sample Multimodal DataFrame ---
    logging.info(f"Creating sample multimodal_df index: Freq='{time_window}', Start='{start_time}', End='{end_time}'")
    multimodal_index = pd.date_range(start=start_time, end=end_time, freq=time_window)
    multimodal_df = pd.DataFrame(index=multimodal_index)
    multimodal_df.index.name = 'timestamp'
    logging.info(f"Created sample multimodal_df: Shape={multimodal_df.shape}")
    if multimodal_df.index.duplicated().any():
        logging.warning(f"**Duplicates found in initial multimodal_df index! Count: {multimodal_df.index.duplicated().sum()}**")
    else:
        logging.info("Initial multimodal_df index is unique.")
    print("\n--- Initial Multimodal DataFrame Index Head ---")
    print(multimodal_df.head().index)
    print("-" * 30)

    # --- 4. Aggregate Patient ID by Hour ---
    try:
        logging.info("Aggregating patient_id by hour using groupby().first()...")
        # Create a DataFrame from the map
        patient_id_map_df = patient_id_map.reset_index()
        
        # Create an 'hour' column by flooring the timestamp
        patient_id_map_df['hour'] = patient_id_map_df['timestamp'].dt.floor('H')
        
        # Group by the floored hour and get the first patient_id in each group
        # Sort by original timestamp within group first to ensure stability if needed, though .first() usually does this
        # patient_id_map_df.sort_values('timestamp', inplace=True)
        hourly_patient_map = patient_id_map_df.groupby('hour')['patient_id'].first()
        
        logging.info(f"Shape of hourly_patient_map: {hourly_patient_map.shape}")
        logging.info(f"NaNs in hourly_patient_map: {hourly_patient_map.isnull().sum()}")
        agg_counts = hourly_patient_map.dropna().value_counts()
        logging.info(f"Patient Counts after hourly aggregation ({len(agg_counts)} unique non-NaN):\n{agg_counts.head()}")

        # Map the aggregated hourly IDs onto the multimodal_df index
        # Use .map() which aligns based on the index values
        multimodal_df['patient_id'] = multimodal_df.index.map(hourly_patient_map)

        logging.info(f"Shape after mapping aggregated IDs: {multimodal_df.shape}")
        logging.info(f"NaNs in patient_id after mapping: {multimodal_df['patient_id'].isnull().sum()}")
        mapped_counts = multimodal_df['patient_id'].dropna().value_counts()
        logging.info(f"Patient Counts after mapping ({len(mapped_counts)} unique non-NaN):\n{mapped_counts.head()}")
        
        # Check index again
        if multimodal_df.index.duplicated().any():
             logging.warning(f"**Duplicates found in multimodal_df index AFTER mapping! Count: {multimodal_df.index.duplicated().sum()}**")
        print("\n--- Multimodal DataFrame Head After Hourly Aggregation Mapping ---")
        print(multimodal_df.head())
        print("-" * 30)

    except Exception as e:
        logging.error(f"Error during hourly aggregation/mapping: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 5. Apply Backfill (if needed for hours with no data) ---
    if multimodal_df['patient_id'].isnull().any():
        missing_count = multimodal_df['patient_id'].isnull().sum()
        logging.info(f"Applying standard backfill for {missing_count} remaining NaNs (hours with no patient data)...")
        multimodal_df['patient_id'] = multimodal_df['patient_id'].bfill()
        if multimodal_df['patient_id'].isnull().any():
            logging.error("Patient ID still missing after aggregation and bfill! Filling with UNKNOWN_PATIENT.")
            multimodal_df['patient_id'].fillna('UNKNOWN_PATIENT', inplace=True)
        logging.info(f"NaNs remaining after bfill: {multimodal_df['patient_id'].isnull().sum()}")
        print("\n--- Multimodal DataFrame Head After Backfill ---")
        print(multimodal_df.head())
        print("-" * 30)
    else:
        logging.info("No NaNs remaining after hourly aggregation mapping, skipping backfill.")

    # --- 6. Final Check ---
    logging.info("--- Final Patient ID Status --- ")
    final_patient_counts = multimodal_df['patient_id'].value_counts()
    logging.info(f"Final Patient Counts ({len(final_patient_counts)} unique):\\n{final_patient_counts.head()}")
    logging.info(f"Total rows in final df: {len(multimodal_df)}")
    logging.info(f"Total non-NaN patient IDs: {multimodal_df['patient_id'].notnull().sum()}")
    print("\n--- Final Multimodal DataFrame Head ---")
    print(multimodal_df.head())
    print("-" * 30)


if __name__ == "__main__":
    debug_patient_id_join()

# % ## End of debug_patient_ids.py
# %%
