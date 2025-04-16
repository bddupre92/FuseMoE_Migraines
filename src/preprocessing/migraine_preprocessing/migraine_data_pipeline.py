import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import json
from datetime import datetime, timedelta
import glob
import logging
import re
from tqdm.auto import tqdm # Import tqdm

# Import processors
from .eeg_processor import EEGProcessor
from .weather_connector import WeatherConnector
from .sleep_processor import SleepProcessor
from .stress_processor import StressProcessor
from ..advanced_imputation import BaseImputer, KNNImputer, IterativeImputerWrapper, AutoencoderImputer

# Add logging configuration if not already present globally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper function for imputation (can be expanded later) ---
def get_imputer(method: Optional[str], config: Optional[Dict[str, Any]] = None) -> Optional[BaseImputer]:
    """
    Get an imputer instance based on the method name.
    Currently only returns advanced imputers if requested,
    otherwise returns None (indicating basic ffill should be used).
    """
    config = config or {}
    if method is None or method.lower() == 'none' or method.lower() == 'ffill':
        return None # Signal to use basic ffill/fillna
    elif method.lower() == 'knn':
        # We'll re-enable this later once the structure is stable
        # return KNNImputer(**config.get('knn', {}))
        logging.warning("Advanced imputation (KNN) is temporarily disabled. Using basic ffill/fillna.")
        return None
    elif method.lower() == 'iterative':
        # return IterativeImputerWrapper(**config.get('iterative', {}))
        logging.warning("Advanced imputation (Iterative) is temporarily disabled. Using basic ffill/fillna.")
        return None
    elif method.lower() == 'autoencoder':
        # return AutoencoderImputer(**config.get('autoencoder', {}))
        logging.warning("Advanced imputation (Autoencoder) is temporarily disabled. Using basic ffill/fillna.")
        return None
    else:
        logging.warning(f"Unsupported imputation method: {method}. Falling back to basic ffill/fillna.")
        return None

class MigraineDataPipeline:
    """
    Main class for integrating and aligning all data modalities for migraine prediction.
    
    This pipeline processes and aligns data from:
    - EEG data
    - Weather data
    - Sleep data
    - Stress/physiological data
    
    And aligns it with migraine events for prediction.
    """
    
    def __init__(self, 
                data_dir: str,
                cache_dir: str = "./migraine_data_cache",
                weather_api_key: Optional[str] = None):
        """
        Initialize the migraine data pipeline.
        
        Args:
            data_dir: Directory containing raw data
            cache_dir: Directory to cache processed data
            weather_api_key: API key for weather data service
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize processors
        self.eeg_processor = EEGProcessor()
        self.weather_connector = WeatherConnector(api_key=weather_api_key, 
                                                cache_dir=os.path.join(cache_dir, "weather"))
        self.sleep_processor = SleepProcessor()
        self.stress_processor = StressProcessor()
        
        # Store processed data
        self.eeg_data = None
        self.weather_data = None
        self.sleep_data = None
        self.stress_data = None
        self.migraine_events = None
        self.aligned_data = None
        self.global_start_time = None # Add attributes to store global time range
        self.global_end_time = None
    
    def _load_raw_data(self, modality: str) -> pd.DataFrame:
        """
        Loads the raw combined CSV for a given modality.
        Handles potential errors during loading and basic timestamp parsing.
        """
        file_path = os.path.join(self.data_dir, modality, f"all_{modality}_data.csv")
        if not os.path.exists(file_path):
            logging.warning(f"Raw data file not found for {modality}: {file_path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Loaded raw {modality} data from {file_path} ({len(df)} rows)")

            # --- Standardize Timestamp Column Name --- #
            timestamp_col = None
            
            # --- Special handling for sleep data --- #
            if modality == 'sleep' and 'sleep_start' in df.columns:
                logging.debug(f"Found 'sleep_start' column in {modality} data. Renaming to 'timestamp'.")
                df = df.rename(columns={'sleep_start': 'timestamp'})
            # --- End special handling --- #
                
            if 'timestamp' in df.columns:
                timestamp_col = 'timestamp'
            elif 'start_time' in df.columns: # Handle different naming conventions
                timestamp_col = 'start_time'
                df = df.rename(columns={'start_time': 'timestamp'})
            elif 'date' in df.columns and 'hour' in df.columns:
                # Synthesize timestamp if only date/hour exist
                df['timestamp'] = pd.to_datetime(df['date'], errors='coerce') + pd.to_timedelta(df['hour'], unit='h', errors='coerce')
                timestamp_col = 'timestamp'
            else:
                logging.warning(f"Could not find a standard timestamp column ('timestamp', 'start_time', or 'date'/'hour') in raw {modality} data.")
                return pd.DataFrame() # Cannot proceed without time

            # Convert the identified timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']) # Drop rows where timestamp parsing failed

            # Ensure patient_id is of a consistent type (string) if it exists
            if 'patient_id' in df.columns:
                df['patient_id'] = df['patient_id'].astype(str)
            elif modality not in ['weather']: # Weather is expected not to have patient_id
                 logging.warning(f"'patient_id' column missing in non-weather modality: {modality}")

            return df

        except Exception as e:
            logging.error(f"Error loading or performing initial processing on raw {modality} data from {file_path}: {e}")
            return pd.DataFrame()
    
    def load_migraine_events(self, file_path: Optional[str] = None) -> List[Dict]:
        """
        Load migraine events from file or use data_dir/migraine_events.json.
        
        Args:
            file_path: Path to migraine events JSON file (optional)
            
        Returns:
            List of migraine event dictionaries
        """
        if file_path is None:
            # Look in migraines subfolder for individual patient files
            migraines_dir = os.path.join(self.data_dir, "migraines")
            file_pattern = os.path.join(migraines_dir, "P*_migraines.csv") # Look for CSV
        else:
            # If a specific file path is given, use it directly
            migraines_dir = os.path.dirname(file_path)
            file_pattern = file_path

        event_files = glob.glob(file_pattern)

        if not event_files:
            print(f"Warning: No Migraine events files found matching pattern {file_pattern}")
            self.migraine_events = []
            return []
        
        all_events = []
        print(f"Found {len(event_files)} migraine event files to load...")
        for file in event_files:
            try:
                # Load from CSV
                events_df = pd.read_csv(file)
                # Add patient_id from filename if not present
                if 'patient_id' not in events_df.columns:
                    patient_id_match = re.search(r"P(\d+)", os.path.basename(file)) # Adjust regex if needed
                    if patient_id_match:
                        events_df['patient_id'] = f"P{patient_id_match.group(1)}"
                    else:
                        logging.warning(f"Could not extract patient_id from filename {file}. Migraine events from this file might not be correctly mapped.")
                # Convert DataFrame rows to list of dictionaries
                events = events_df.to_dict('records')
            
                # Ensure proper format (Now correctly indented within try)
                for event in events:
                        # Check for start_time (actual column) and severity
                        if 'start_time' in event and 'severity' in event:
                            # Parse the start_time field
                            if isinstance(event['start_time'], str):
                                try:
                                    event['start_time'] = pd.Timestamp(event['start_time'])
                                except Exception as time_e:
                                    print(f"Warning: Could not parse start_time '{event['start_time']}' in {file}: {time_e}. Skipping event.")
                                    continue # Skip this event if timestamp is invalid
                            
                            # Ensure start_time is a valid timestamp after potential parsing
                            if isinstance(event['start_time'], pd.Timestamp) and pd.notna(event['start_time']):
                                # Ensure patient_id is str
                                if 'patient_id' in event and event['patient_id'] is not None:
                                    event['patient_id'] = str(event['patient_id'])
                                    all_events.append(event) # Append valid event
                                else:
                                    print(f"Warning: Skipping event with missing or invalid patient_id in {file}: {event}")
                            else:
                                print(f"Warning: Skipping event with invalid start_time type or NaT after parsing in {file}: {event}")
                        else:
                            # Print warning if essential keys are missing
                            print(f"Warning: Skipping invalid event entry (missing start_time or severity) in {file}: {event}")
            except Exception as e:
                print(f"Error loading or processing migraine events file {file}: {e}")

        # Sort events by start_time just in case
        try:
            # Filter out events without a valid Timestamp before sorting
            valid_events_for_sort = [e for e in all_events if isinstance(e.get('start_time'), pd.Timestamp)]
            valid_events_for_sort.sort(key=lambda x: x['start_time'])
            all_events = valid_events_for_sort # Replace original list with sorted valid ones
        except TypeError as sort_e:
            print(f"Warning: Could not sort events by start_time due to type error: {sort_e}. Events may be out of order.")

        # --- Deduplicate events based on start_time AND patient_id --- #
        deduplicated_events = []
        seen_patient_start_times = set()
        print("Deduplicating events based on patient_id and start_time...")
        for event in all_events: # Iterate through already validated and sorted events
            start_time = event.get('start_time')
            patient_id = event.get('patient_id') # Already ensured it exists and is str
            
            # We know start_time is a valid Timestamp here due to filtering above
            event_key = (patient_id, start_time)
            if event_key not in seen_patient_start_times:
                deduplicated_events.append(event)
                seen_patient_start_times.add(event_key)
            # else: # Implicitly skip duplicates
                # print(f"DEBUG: Skipping duplicate event with key {event_key}")
        
        num_removed = len(all_events) - len(deduplicated_events)
        if num_removed > 0:
            print(f"Removed {num_removed} duplicate events based on patient_id and start_time.")
            
        self.migraine_events = deduplicated_events
        print(f"Loaded a total of {len(self.migraine_events)} unique patient-specific migraine events.")
        return self.migraine_events
    
    def process_eeg_data(self, file_pattern: str = "eeg_*.npy") -> pd.DataFrame:
        """
        Process EEG data files.
        
        Args:
            file_pattern: Pattern to match EEG files
            
        Returns:
            DataFrame with processed EEG features
        """
        eeg_dir = os.path.join(self.data_dir, "eeg")
        eeg_csv_path = os.path.join(eeg_dir, "all_eeg_data.csv") # Path to combined CSV
        
        if not os.path.exists(eeg_csv_path):
            print(f"Warning: Combined EEG CSV file not found at {eeg_csv_path}")
            return pd.DataFrame()
        
        # Load the combined CSV
        try:
            eeg_df_raw = pd.read_csv(eeg_csv_path)
            print(f"Loaded combined EEG data from {eeg_csv_path}")
        except Exception as e:
            print(f"Error loading combined EEG CSV: {e}")
            return pd.DataFrame()
        
        # Process EEG data (Corrected try/except structure)
        try:
            print("Processing combined EEG DataFrame...")
            eeg_df = self.eeg_processor.process_dataset(eeg_df_raw) # Pass DataFrame
            self.eeg_data = eeg_df # Assign inside try
            print(f"Processed EEG data: {len(self.eeg_data)} records") # Print inside try
            return eeg_df # Return inside try
        except Exception as e: # Correctly associated except
             print(f"Error processing EEG data: {e}")
             print("Please check if EEGProcessor.process_dataset can handle a DataFrame.")
             return pd.DataFrame() # Return empty df on error
    
    def process_weather_data(self, location: Tuple[float, float],
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Process weather data for specified location and date range.
        
        Args:
            location: Location coordinates (latitude, longitude)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with processed weather features
        """
        # Fetch historical weather data
        weather_df = self.weather_connector.fetch_historical_weather(
            location, start_date, end_date)
        
        # Process weather data to extract migraine-relevant features
        processed_df = self.weather_connector.process_weather_data(weather_df)
        
        self.weather_data = processed_df
        return processed_df
    
    def process_sleep_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process sleep tracking data.
        
        Args:
            file_path: Path to sleep data JSON file (optional)
            
        Returns:
            DataFrame with processed sleep features
        """
        if file_path is None:
            # Look in sleep subfolder for the combined file
            file_path = os.path.join(self.data_dir, "sleep", "all_sleep_data.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: Sleep data file not found at {file_path}")
            return pd.DataFrame()
        
        try:
            # Load from CSV instead of JSON
            sleep_df_raw = pd.read_csv(file_path)
            # Assuming process_sleep_dataset can handle a DataFrame
            sleep_df = self.sleep_processor.process_sleep_dataset(sleep_df_raw)
            self.sleep_data = sleep_df
            
            return sleep_df
            
        except Exception as e:
            print(f"Error processing sleep data: {e}")
            return pd.DataFrame()
    
    def process_stress_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process stress/physiological data.
        
        Args:
            file_path: Path to stress data JSON file (optional)
            
        Returns:
            DataFrame with processed stress features
        """
        if file_path is None:
            # Look in stress subfolder for the combined file
            file_path = os.path.join(self.data_dir, "stress", "all_stress_data.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: Stress data file not found at {file_path}")
            return pd.DataFrame()
        
        try:
            # Load from CSV instead of JSON
            stress_df_raw = pd.read_csv(file_path)
            # Assuming process_hrv_dataset can handle a DataFrame
            stress_df = self.stress_processor.process_hrv_dataset(stress_df_raw)
            self.stress_data = stress_df
            
            return stress_df
            
        except Exception as e:
            print(f"Error processing stress data: {e}")
            return pd.DataFrame()
    
    def calculate_migraine_risk_scores(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate migraine risk scores for each data modality.
        
        Returns:
            Dictionary of DataFrames with risk scores for each modality
        """
        risk_dfs = {}
        
        # Calculate weather-based migraine risk
        if self.weather_data is not None and not self.weather_data.empty:
            weather_risk_df = self.weather_connector.get_migraine_risk_from_weather(self.weather_data)
            risk_dfs['weather'] = weather_risk_df
        
        # Calculate sleep-based migraine risk
        if self.sleep_data is not None and not self.sleep_data.empty:
            sleep_risk_df = self.sleep_processor.get_migraine_risk_from_sleep(self.sleep_data)
            risk_dfs['sleep'] = sleep_risk_df
        
        # Calculate stress-based migraine risk
        if self.stress_data is not None and not self.stress_data.empty:
            stress_risk_df = self.stress_processor.get_migraine_risk_from_stress(self.stress_data)
            risk_dfs['stress'] = stress_risk_df
        
        return risk_dfs
    
    def align_data_with_migraine_events(self) -> Dict[str, pd.DataFrame]:
        """
        Aligns processed data modalities with migraine events.
        
        Note: This method might be less relevant with the new 'create_multimodal_dataset'
              approach which handles alignment internally. Keeping for potential legacy use.
        """
        if self.migraine_events is None:
            self.load_migraine_events()

        if not self.migraine_events:
            print("No migraine events loaded, cannot align data.")
            return {}
        
        # Convert migraine events list to DataFrame for easier merging
        events_df = pd.DataFrame(self.migraine_events)
        if 'start_time' not in events_df.columns:
             print("Warning: Migraine events list does not contain 'start_time'. Cannot align.")
             return {}
        events_df['start_time'] = pd.to_datetime(events_df['start_time'])
        events_df = events_df.sort_values('start_time')
        
        aligned_modalities = {}
        
        # Example: Align EEG data
        # Assuming self.eeg_data is a DataFrame with a 'timestamp' column
        if self.eeg_data is not None and 'timestamp' in self.eeg_data.columns:
            eeg_data_sorted = self.eeg_data.sort_values('timestamp')
            # Use merge_asof to find the nearest EEG reading before each migraine event
            aligned_modalities['eeg'] = pd.merge_asof(
                events_df,
                eeg_data_sorted,
                left_on='start_time',
                right_on='timestamp',
                direction='backward', # Find last reading *before* or at the event time
                tolerance=pd.Timedelta('1 hour') # Example tolerance
            )
            # Rename merged timestamp to avoid confusion
            aligned_modalities['eeg'] = aligned_modalities['eeg'].rename(columns={'timestamp': 'eeg_timestamp'})
        
        # ... similar alignment for other modalities like sleep, stress ...
        # Ensure they also have a 'timestamp' column for alignment

        self.aligned_data = aligned_modalities
        return aligned_modalities

    def _regularize_impute_modality(self,
                                   raw_df: pd.DataFrame,
                                   modality_name: str,
                                   patient_ids: List[str],
                                   global_start_time: pd.Timestamp,
                                   global_end_time: pd.Timestamp,
                                   feature_cols: List[str]) -> pd.DataFrame:
        """
        Regularizes and imputes data for a single modality across all patients.
        Uses BASIC imputation (ffill then fillna(0)) for this version.

        Args:
            raw_df: Raw DataFrame for the modality (must contain 'patient_id' and 'timestamp').
            modality_name: Name of the modality (e.g., 'eeg').
            patient_ids: List of unique patient IDs to process.
            global_start_time: The earliest timestamp across all data.
            global_end_time: The latest timestamp across all data.
            feature_cols: List of feature column names for this modality.

        Returns:
            A single DataFrame containing regularized and imputed data for this modality
            for all specified patients, with columns ['patient_id', 'timestamp'] + feature_cols.
        """
        processed_patient_dfs = []

        # Ensure essential columns exist
        if 'timestamp' not in raw_df.columns or 'patient_id' not in raw_df.columns:
             logging.error(f"Essential columns ('timestamp', 'patient_id') missing in {modality_name} raw data. Cannot process.")
             return pd.DataFrame()

        logging.info(f"Regularizing and imputing modality: {modality_name} using basic ffill/fillna(0)...")

        for patient_id in patient_ids:
            # Filter data for the current patient
            patient_df = raw_df[raw_df['patient_id'] == patient_id].copy()

            if patient_df.empty:
                # logging.debug(f"No {modality_name} data found for patient {patient_id}.") # Can be noisy
                continue

            # Define the GLOBAL hourly index for this patient
            hourly_index = pd.date_range(start=global_start_time, end=global_end_time, freq='H', name='timestamp')

            # Prepare for reindexing: Set timestamp as index, drop duplicates within the hour for this patient
            # Take the *first* record if multiple exist within the same hour for this patient
            patient_df = patient_df.sort_values('timestamp') # Sort before dropping
            # Use drop_duplicates on timestamp, keeping the first entry for that hour
            patient_df_unique_time = patient_df.drop_duplicates(subset=['timestamp'], keep='first').set_index('timestamp')

            # Reindex to the GLOBAL hourly grid - introduces NaNs
            regular_df = patient_df_unique_time.reindex(hourly_index)

            # --- Basic Imputation --- #
            # Select only the feature columns intended for this modality
            current_feature_cols = [col for col in feature_cols if col in regular_df.columns]
            if current_feature_cols:
                # Forward fill first, then fill remaining NaNs (usually at the beginning) with 0
                regular_df[current_feature_cols] = regular_df[current_feature_cols].ffill().fillna(0)
            else:
                logging.warning(f"No defined feature columns found in regularized data for patient {patient_id}, modality {modality_name}. Imputation skipped for this patient/modality.")

            # Add patient_id back as a column (it was lost during reindex if not in feature_cols)
            regular_df['patient_id'] = patient_id

            # Reset index to make timestamp a column again for later merging
            processed_patient_dfs.append(regular_df.reset_index())

        if not processed_patient_dfs:
             logging.warning(f"No data processed for modality {modality_name} across all patients.")
             return pd.DataFrame()

        # Concatenate all processed patient data for this modality
        modality_df_processed = pd.concat(processed_patient_dfs, ignore_index=True)

        logging.info(f"Finished processing {modality_name}. Result shape: {modality_df_processed.shape}")
        return modality_df_processed

    def create_multimodal_dataset(self,
                                prediction_horizon: int = 6,
                                imputation_method: Optional[str] = 'ffill', # Default to ffill, advanced optional later
                                imputer_config: Optional[Dict[str, Any]] = None,
                                required_modalities: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Creates a single, aligned multimodal dataset using the 'regularize-then-combine' strategy.
        Applies BASIC imputation (ffill -> fillna(0)) during regularization by default.
        Advanced imputation can be enabled via imputation_method argument (currently disabled).

        Steps:
        1. Load raw data for all relevant modalities (EEG, Sleep, Stress, Weather).
        2. Determine unique patient IDs and the global time range.
        3. For each patient-specific modality (EEG, Sleep, Stress):
           - Call _regularize_impute_modality to process data per patient (reindex hourly, basic impute)
             and combine results for that modality across all patients.
        4. Process global modalities (Weather): Reindex to global hourly grid and basic impute.
        5. Create a base patient-time grid.
        6. Merge all processed modality DataFrames onto the base grid using left joins.
        7. Calculate the target variable ('migraine_within_horizon').
        8. Perform final feature engineering (time features).
        9. Return the final DataFrame indexed by timestamp.

        Args:
            prediction_horizon: Hours ahead to predict migraine onset.
            imputation_method: Method for imputation ('ffill', or advanced like 'knn' - currently disabled).
            imputer_config: Configuration dictionary for advanced imputers.
            required_modalities: List of modalities that MUST be present. If None, uses all available.

        Returns:
            A single Pandas DataFrame with aligned multimodal data, indexed by hourly timestamp.
            Includes 'patient_id' and the target 'migraine_within_horizon' columns.
        """
        logging.info("Starting multimodal dataset creation using 'regularize-then-combine' strategy...")

        # --- 1. Load Raw Data --- #
        raw_eeg = self._load_raw_data("eeg")
        raw_sleep = self._load_raw_data("sleep")
        raw_stress = self._load_raw_data("stress")
        raw_weather = self._load_raw_data("weather") # Assumes weather has 'timestamp'

        # --- 2. Determine Unique Patients and Global Time Range --- #
        all_dfs = {"eeg": raw_eeg, "sleep": raw_sleep, "stress": raw_stress, "weather": raw_weather}
        patient_ids_per_modality = {}
        all_timestamps = []
        all_patient_ids = set()

        for name, df in all_dfs.items():
            if not df.empty:
                if 'timestamp' in df.columns:
                    all_timestamps.append(df['timestamp'])
                if 'patient_id' in df.columns:
                    p_ids = set(df['patient_id'].unique())
                    patient_ids_per_modality[name] = p_ids
                    all_patient_ids.update(p_ids)

        if not all_patient_ids:
            logging.error("No patient IDs found across any patient-specific modality. Cannot proceed.")
            return pd.DataFrame()
        all_patient_ids = sorted(list(all_patient_ids))
        logging.info(f"Found {len(all_patient_ids)} unique patient IDs across all modalities.")

        if not all_timestamps:
            logging.error("No valid timestamps found across any modality. Cannot determine time range.")
            return pd.DataFrame()

        global_start_time = pd.concat(all_timestamps).min().floor('H')
        global_end_time = pd.concat(all_timestamps).max().ceil('H')
        self.global_start_time = global_start_time # Store for potential use elsewhere
        self.global_end_time = global_end_time
        logging.info(f"Global time range: {self.global_start_time} to {self.global_end_time}")

        processed_modalities = {}
        modality_feature_cols = {} # Store identified feature columns per modality

        # --- 3. Process Patient-Specific Modalities --- #
        modalities_to_process = {
            'eeg': raw_eeg,
            'sleep': raw_sleep,
            'stress': raw_stress
        }
        # Define FEATURE columns expected for each modality (excluding IDs, timestamps, etc.)
        for name, df in modalities_to_process.items():
             if not df.empty:
                 # Identify potential features robustly
                 potential_feature_cols = [col for col in df.columns if col not in ['patient_id', 'timestamp', 'date', 'hour', 'start_time', 'end_time', 'source_file']]
                 # Filter out any remaining non-numeric types just in case
                 numeric_feature_cols = df[potential_feature_cols].select_dtypes(include=np.number).columns.tolist()
                 modality_feature_cols[name] = numeric_feature_cols
                 logging.info(f"Identified numeric feature columns for {name}: {numeric_feature_cols}")

        for name, df in modalities_to_process.items():
            if df.empty or name not in modality_feature_cols or not modality_feature_cols[name]:
                 logging.warning(f"Skipping empty or featureless raw data for modality: {name}")
                 continue

            processed_df = self._regularize_impute_modality(
                raw_df=df,
                modality_name=name,
                patient_ids=all_patient_ids,
                global_start_time=self.global_start_time,
                global_end_time=self.global_end_time,
                feature_cols=modality_feature_cols[name]
            )
            if not processed_df.empty:
                # Add prefix to avoid column name collisions during merge
                feature_cols_renamed = modality_feature_cols[name]
                processed_df = processed_df.rename(columns={col: f"{name}_{col}" for col in feature_cols_renamed if col in processed_df.columns})
                processed_modalities[name] = processed_df

        # --- 4. Process Global Modalities (Weather) --- #
        if not raw_weather.empty and 'timestamp' in raw_weather.columns:
            logging.info("Processing global modality: weather using basic ffill/fillna(0)...")
            potential_weather_features = [col for col in raw_weather.columns if col not in ['timestamp', 'latitude', 'longitude']] # Adjust non-feature cols
            weather_feature_cols = raw_weather[potential_weather_features].select_dtypes(include=np.number).columns.tolist()
            modality_feature_cols['weather'] = weather_feature_cols # Store for reference
            logging.info(f"Identified numeric feature columns for weather: {weather_feature_cols}")

            hourly_index = pd.date_range(start=self.global_start_time, end=self.global_end_time, freq='H', name='timestamp')
            # Resample taking the first value within each hour, then reindex
            # Use drop_duplicates before set_index
            weather_unique_time = raw_weather.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first').set_index('timestamp')
            weather_regular = weather_unique_time.reindex(hourly_index)

            # Basic Impute weather features
            current_weather_features = [col for col in weather_feature_cols if col in weather_regular.columns]
            if current_weather_features:
                 weather_regular[current_weather_features] = weather_regular[current_weather_features].ffill().fillna(0)

            weather_regular = weather_regular.reset_index() # Keep timestamp as column for merge
            # Add prefix
            weather_regular = weather_regular.rename(columns={col: f"weather_{col}" for col in weather_feature_cols if col in weather_regular.columns})
            processed_modalities['weather'] = weather_regular
            logging.info(f"Finished processing weather. Result shape: {weather_regular.shape}")
        else:
            logging.warning("Weather data is empty or missing 'timestamp'. Skipping weather processing.")

        # --- 5. Create Base Grid --- #
        if not all_patient_ids:
             logging.error("Cannot create base grid without patient IDs.")
             return pd.DataFrame()
        logging.info("Creating base patient-time grid...")
        all_hourly_timestamps = pd.date_range(start=self.global_start_time, end=self.global_end_time, freq='H')
        base_index = pd.MultiIndex.from_product(
             [all_patient_ids, all_hourly_timestamps],
             names=['patient_id', 'timestamp']
        )
        final_df = pd.DataFrame(index=base_index).reset_index()
        logging.info(f"Base grid created with shape: {final_df.shape}")

        # --- 6. Merge All Processed Data Onto Base Grid --- #
        # Ensure base types are correct for merging
        final_df['patient_id'] = final_df['patient_id'].astype(str)
        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])

        modalities_to_merge = ['eeg', 'sleep', 'stress', 'weather'] # Order might matter
        for name in modalities_to_merge:
             if name in processed_modalities:
                 logging.info(f"Merging {name} data onto base grid...")
                 mod_df = processed_modalities[name]
                 merge_keys = ['patient_id', 'timestamp'] if 'patient_id' in mod_df.columns else ['timestamp']

                 # Ensure key types match base grid
                 if 'patient_id' in merge_keys:
                     mod_df['patient_id'] = mod_df['patient_id'].astype(str)
                 mod_df['timestamp'] = pd.to_datetime(mod_df['timestamp'])

                 # Select only key columns and feature columns for merging
                 cols_to_merge = merge_keys + [col for col in mod_df.columns if col not in merge_keys]
                 # Ensure keys are unique before merge to avoid duplication issues
                 mod_df_subset = mod_df[cols_to_merge].drop_duplicates(subset=merge_keys, keep='first')

                 final_df = pd.merge(final_df, mod_df_subset,
                                     on=merge_keys,
                                     how='left') # Use left merge to keep all patient-time slots
                 logging.info(f"Shape after merging {name}: {final_df.shape}")
             else:
                  logging.warning(f"Modality '{name}' was not processed successfully or empty. Skipping merge.")

        # --- 7. Calculate Target Variable --- #
        logging.info("Calculating target variable 'migraine_within_horizon'...")
        if self.migraine_events is None:
            self.load_migraine_events()

        if not self.migraine_events:
            logging.warning("No migraine events loaded, cannot calculate target variable. Column will be all False.")
            final_df['migraine_within_horizon'] = False
        else:
            # Create a DataFrame from migraine events
            events_df = pd.DataFrame(self.migraine_events)
            if 'start_time' not in events_df.columns or 'patient_id' not in events_df.columns:
                 logging.error("'start_time' or 'patient_id' column missing in migraine events. Cannot calculate patient-specific target.")
                 final_df['migraine_within_horizon'] = False
            else:
                events_df['start_time'] = pd.to_datetime(events_df['start_time'])
                events_df['patient_id'] = events_df['patient_id'].astype(str)

                # Create a helper structure: dictionary mapping patient_id to sorted list of their migraine start times
                patient_migraine_times = events_df.dropna(subset=['start_time', 'patient_id'])\
                                                .groupby('patient_id')['start_time']\
                                                .apply(lambda x: sorted(x.unique()))\
                                                .to_dict()

                # Define horizon delta outside the function for efficiency
                horizon_delta = pd.Timedelta(hours=prediction_horizon)

                def check_migraine_horizon_patient(row):
                    patient_id = row['patient_id']
                    ts = row['timestamp']
                    if patient_id in patient_migraine_times:
                        patient_times = patient_migraine_times[patient_id]
                        # Check if any event falls within (ts, ts + horizon_delta]
                        for event_time in patient_times:
                            if ts < event_time <= ts + horizon_delta:
                                return True
                            if event_time > ts + horizon_delta:
                                 break # Since times are sorted
                    return False

                logging.info("Applying target calculation function...")
                final_df['migraine_within_horizon'] = final_df.apply(check_migraine_horizon_patient, axis=1)
                logging.info("Finished applying target calculation function.")

            target_counts = final_df['migraine_within_horizon'].value_counts()
            logging.info(f"Target variable calculated. Distribution: {target_counts.to_dict()}")
            if True not in target_counts or target_counts[True] == 0:
                logging.warning("No positive migraine events found within the horizon for any timestamp.")

        # --- 8. Final Feature Engineering (Time Features) --- #
        logging.info("Performing final feature engineering...")
        final_df['hour_of_day'] = final_df['timestamp'].dt.hour
        final_df['day_of_week'] = final_df['timestamp'].dt.dayofweek
        final_df['month_of_year'] = final_df['timestamp'].dt.month
        # Add more features if needed (e.g., rolling averages across modalities)

        # --- 9. Set Index and Sort --- #
        final_df = final_df.set_index('timestamp').sort_index()

        # Optional: Drop rows where all features are NaN (unlikely with ffill but possible)
        # feature_columns_final = [col for col in final_df.columns if col not in ['patient_id', 'migraine_within_horizon', 'hour_of_day', 'day_of_week', 'month_of_year']]
        # final_df = final_df.dropna(subset=feature_columns_final, how='all')

        logging.info(f"Multimodal dataset creation complete. Final shape: {final_df.shape}")
        # Check for patient ID loss again
        final_patients = final_df['patient_id'].nunique()
        if final_patients < len(all_patient_ids):
             logging.warning(f"Potential patient ID loss: Started with {len(all_patient_ids)}, final DF has {final_patients} unique IDs.")
        else:
            logging.info(f"Patient ID count consistent: {final_patients} unique IDs found in final DataFrame.")

        self.aligned_data = final_df # Store the final result
        return final_df
    
    def prepare_data_for_fusemoe(self, multimodal_df: pd.DataFrame,
                               window_size: int = 24,
                               step_size: int = 1,
                               prediction_horizon: int = 6) -> Dict[str, Any]:
        """
        Prepare data in the format required by the FuseMOE model.

        Args:
            multimodal_df: DataFrame containing aligned multimodal data including 'patient_id'.
            window_size: Lookback window size in hours.
            step_size: Step size for sliding window.
            prediction_horizon: Hours ahead to predict.

        Returns:
            Dictionary containing prepared data:
                'X': Dictionary of feature tensors per modality [Batch, Window, Features]
                'y': Target labels array [Batch]
                'groups': Patient IDs array for grouping [Batch] or None
                'modalities': List of modality names included
                'features_per_modality': Dictionary mapping modality to feature count
        """
        if multimodal_df is None or multimodal_df.empty:
            logging.error("Input multimodal_df is None or empty. Cannot prepare data.")
            return {'X': {}, 'y': np.array([]), 'groups': None, 'modalities': [], 'features_per_modality': {}}

        logging.info(f"Preparing data for FuseMoE with window_size={window_size}, step_size={step_size}, horizon={prediction_horizon}")

        X_list = []
        y_list = []
        group_ids = [] # Initialize list to store group IDs for each window

        # --- Check for patient_id column BEFORE looping --- #
        has_patient_id = 'patient_id' in multimodal_df.columns
        if not has_patient_id:
            logging.error("'patient_id' column missing from multimodal_df. Cannot create groups for CV.")
            groups_np = None # Set groups to None if column is missing
        else:
            logging.info("Found 'patient_id' column. Proceeding to extract group IDs.")
        # --- ------------------------------------------- --- #

        # Ensure target column exists
        if 'migraine_within_horizon' not in multimodal_df.columns:
             logging.error("'migraine_within_horizon' target column missing. Cannot prepare data.")
             return {'X': {}, 'y': np.array([]), 'groups': None, 'modalities': [], 'features_per_modality': {}}

        # --- Determine Feature Columns Rigorously --- #
        # Start with all columns
        all_cols = multimodal_df.columns.tolist()
        # Define non-feature columns explicitly
        non_feature_cols = ['migraine_within_horizon']
        if has_patient_id:
            non_feature_cols.append('patient_id')
        # Explicitly define columns NOT belonging to a modality's features
        # These are often metadata or engineered time features added later.
        # They should not be part of the windowed input for the model's core layers.
        excluded_cols_from_modality_features = set([
            'latitude', 
            'longitude', 
            'hour_of_day', 
            'day_of_week', 
            'month_of_year'
        ])
        # Select only columns that are NOT in non_feature_cols AND are numeric
        numeric_df = multimodal_df.select_dtypes(include=np.number)
        # Further filter out the explicitly excluded columns
        feature_columns = [col for col in numeric_df.columns \
                           if col not in non_feature_cols \
                           and col not in excluded_cols_from_modality_features]
        
        logging.info(f"Identified {len(feature_columns)} numeric feature columns for windowing: {feature_columns}")
        if not feature_columns:
            logging.error("No numeric feature columns found after filtering. Cannot prepare data.")
            return {'X': {}, 'y': np.array([]), 'groups': None, 'modalities': [], 'features_per_modality': {}}
        # --------------------------------------------- #

        # Iterate through windows and create feature/target/group triplets
        num_possible_windows = len(multimodal_df) - window_size - prediction_horizon + 1
        logging.info(f"Total rows: {len(multimodal_df)}, Creating {num_possible_windows} windows...")

        # Wrap the range with tqdm for progress bar
        for i in tqdm(range(0, num_possible_windows, step_size), desc="Creating windows", unit="window"):
            window_start_idx = i
            window_end_idx = i + window_size
            target_idx = window_end_idx + prediction_horizon - 1 # Correct target index

            # Ensure target index is within bounds
            if target_idx >= len(multimodal_df):
                logging.warning(f"Target index {target_idx} out of bounds for window starting at {window_start_idx}. Skipping.")
                continue

            # --- Extract ONLY the identified numeric features --- #
            features_df_window = multimodal_df.iloc[window_start_idx:window_end_idx][feature_columns]
            # Check for unexpected dtypes AFTER slicing (paranoid check)
            if features_df_window.select_dtypes(exclude=np.number).shape[1] > 0:
                 logging.error(f"Non-numeric dtypes found in window {i}! Columns: {features_df_window.select_dtypes(exclude=np.number).columns.tolist()}")
                 # Handle error: skip window, try coercing, etc.
                 # For now, let's try coercing again, though it shouldn't be needed
                 features_df_window = features_df_window.apply(pd.to_numeric, errors='coerce').fillna(0) # Coerce and fill any new NaNs
                 logging.warning(f"Coerced non-numeric types found in window {i}.")
                 
            X_list.append(features_df_window.values) # Append numpy array
            # --------------------------------------------------- #

            # Extract target label
            y_list.append(multimodal_df.iloc[target_idx]['migraine_within_horizon'])

            # Extract patient ID for this window IF the column exists
            if has_patient_id:
                try:
                    # Get patient ID corresponding to the start of the window
                    # Use iloc for integer position based indexing, which is safer within the loop
                    window_patient_id = multimodal_df.iloc[window_start_idx]['patient_id']

                    # Handle cases where patient ID might be NaN even after ffill
                    if pd.isna(window_patient_id):
                         logging.debug(f"Found NaN patient_id for window starting at index {window_start_idx}. Using placeholder.") # Use debug
                         group_ids.append("UNKNOWN_PATIENT") # Use a placeholder
                    else:
                         group_ids.append(window_patient_id)

                except IndexError:
                     logging.error(f"IndexError accessing patient_id at index {window_start_idx}. This should not happen.")
                     group_ids.append(None) # Indicate failure
                except Exception as e: # Corrected indentation
                     logging.error(f"Error extracting patient_id for window at index {i}: {e}")
                     group_ids.append(None)

        # Convert lists to numpy arrays
        y_np = np.array(y_list).astype(int)

        # Convert group_ids to numpy array if available and consistent
        groups_np = None # Initialize to None
        if has_patient_id and group_ids:
            if len(group_ids) == len(X_list):
                groups_np = np.array(group_ids)
                logging.info(f"Successfully created groups array with shape: {groups_np.shape}")
                # Log unique groups found
                unique_groups, group_counts = np.unique(groups_np, return_counts=True)
                logging.info(f"Found {len(unique_groups)} unique patient groups in the prepared data.")
                logging.debug(f"Group counts: {dict(zip(unique_groups, group_counts))}")
            else:
                logging.error(f"Mismatch between number of extracted groups ({len(group_ids)}) and feature samples ({len(X_list)}). Cannot use groups.")
                # groups_np remains None

        # --- Prepare final data structure ---
        X_dict = {}
        features_per_modality = {}
        ordered_modalities = []

        # Determine feature columns and order from the identified numeric feature_columns list
        if X_list and feature_columns:
            # Extract modality prefixes and counts FROM THE RIGOROUSLY DEFINED feature_columns
            # Ensure columns used for prefix splitting are indeed the numeric ones selected earlier
            modality_prefixes = sorted(list(set([col.split('_')[0] for col in feature_columns if '_' in col])))

            current_col_idx = 0
            temp_features_per_modality = {} # Use temporary dict to build indices
            for mod_prefix in modality_prefixes:
                # Find columns in our *selected* feature_columns list that match the prefix
                mod_cols = [col for col in feature_columns if col.startswith(mod_prefix + '_')]
                if mod_cols:
                    count = len(mod_cols)
                    temp_features_per_modality[mod_prefix] = count
                    ordered_modalities.append(mod_prefix)
                    # Store indices range for slicing later
                    temp_features_per_modality[f"{mod_prefix}_indices"] = (current_col_idx, current_col_idx + count)
                    current_col_idx += count
                else:
                     # This case should ideally not happen if prefixes are derived from feature_columns
                     logging.warning(f"No columns found for prefix '{mod_prefix}' within the selected numeric feature columns. This might indicate an issue.")
                     
            # Check if the total count matches
            total_features_counted = sum(temp_features_per_modality[mod] for mod in ordered_modalities)
            if total_features_counted != len(feature_columns):
                 logging.error(f"Mismatch in feature count: Selected {len(feature_columns)} features, but counted {total_features_counted} via prefixes.")
                 # Handle error - perhaps don't proceed with splitting?
            else:
                 features_per_modality = temp_features_per_modality # Assign if counts match
                 logging.info(f"Successfully mapped {total_features_counted} features to {len(ordered_modalities)} modalities based on prefixes.")

            # Now split the stacked numpy arrays in X_list back into modalities
            if ordered_modalities and features_per_modality:
                try:
                    # Convert X_list (list of [window, features]) into one large array [batch, window, features]
                    X_stacked_np = np.stack(X_list, axis=0) # Shape: [Batch, Window, TotalNumericFeatures]

                    # Ensure the number of features in stacked array matches expected count
                    if X_stacked_np.shape[2] != len(feature_columns):
                         logging.error(f"Stacked array feature dimension ({X_stacked_np.shape[2]}) does not match expected numeric feature count ({len(feature_columns)}). Aborting split.")
                         X_dict = {} # Clear dict to indicate failure
                         ordered_modalities = []
                         features_per_modality = {}
                    else:
                        for modality in ordered_modalities:
                            if f"{modality}_indices" in features_per_modality:
                                start_idx, end_idx = features_per_modality[f"{modality}_indices"]
                                X_dict[modality] = X_stacked_np[:, :, start_idx:end_idx] # Slice the stacked array
                            else:
                                 logging.warning(f"Indices not found for modality '{modality}' during splitting. Skipping.")

                        # Validate shapes
                        for modality, data in X_dict.items():
                            expected_feature_count = features_per_modality.get(modality)
                            if expected_feature_count is None:
                                 logging.warning(f"Could not find feature count for modality '{modality}' during shape validation.")
                                 continue
                            expected_shape = (len(X_list), window_size, expected_feature_count)
                            if data.shape != expected_shape:
                                logging.warning(f"Shape mismatch for modality '{modality}'. Expected {expected_shape}, got {data.shape}")

                except ValueError as e:
                     logging.error(f"Error stacking feature list into numpy array: {e}. Check for consistent shapes in X_list.")
                     # Clear dicts to indicate failure
                     X_dict = {}
                     ordered_modalities = []
                     features_per_modality = {}
                except Exception as e:
                     logging.error(f"Unexpected error during feature splitting: {e}", exc_info=True)
                     X_dict = {}
                     ordered_modalities = []
                     features_per_modality = {}
            else:
                 logging.warning("Could not determine modality order or features properly. X_dict will be empty.")
                 X_dict = {}
                 ordered_modalities = []
                 features_per_modality = {}
        else:
             logging.warning("X_list or feature_columns is empty. Cannot prepare X_dict.")

        return {
            'X': X_dict,
            'y': y_np,
            'groups': groups_np, # Included groups array
            'modalities': list(X_dict.keys()),
            'features_per_modality': {mod: count for mod, count in features_per_modality.items() if not mod.endswith('_indices')}
        }
    
    def run_full_pipeline(self, 
                        location: Tuple[float, float],
                        start_date: Union[str, datetime],
                        end_date: Union[str, datetime],
                        prediction_horizon: int = 6,
                        window_size: int = 24,
                        imputation_method: Optional[str] = 'ffill', # Match default in create_multimodal
                        imputer_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline including imputation.
        This method orchestrates the loading, processing, alignment, and preparation.
        
        Args:
            location: Location coordinates (latitude, longitude) for weather.
            start_date: Start date for data processing range.
            end_date: End date for data processing range.
            prediction_horizon: Hours ahead to predict migraines.
            window_size: Number of hours to use as input window for FuseMOE.
            imputation_method: Method for imputation ('ffill', or advanced like 'knn' - currently disabled).
            imputer_config: Configuration dictionary for advanced imputers.
            
        Returns:
            Dictionary with processed data ready for the FuseMOE model
        """
        logging.info("Running full data pipeline...")
        # Load migraine events first - needed for target calculation
        self.load_migraine_events()

        # Process individual modalities (Load raw data implicitly within create_multimodal_dataset)
        # We no longer need to call process_eeg_data, etc. here if create_multimodal handles it.
        # However, processing weather might still need explicit dates if not loaded from file
        # Let's assume _load_raw_data handles weather loading for now.
        # If weather needs fetching based on start/end date, that logic needs integration.

        # Create multimodal dataset using the refactored method
        multimodal_df = self.create_multimodal_dataset(
            prediction_horizon=prediction_horizon,
            imputation_method=imputation_method, # Pass along the method
            imputer_config=imputer_config
        )
        
        if multimodal_df is None or multimodal_df.empty:
            logging.error("Multimodal dataset is empty after creation/imputation. Cannot proceed.")
            return {'X': {}, 'y': np.array([]), 'groups': None, 'modalities': [], 'features_per_modality': {}}
        
        logging.info(f"Multimodal dataset successfully created. Shape: {multimodal_df.shape}")
        
        # Prepare data for FuseMOE model
        fusemoe_input_data = self.prepare_data_for_fusemoe(
            multimodal_df=multimodal_df,
            window_size=window_size,
            # step_size is handled by default in prepare_data_for_fusemoe
            prediction_horizon=prediction_horizon # Pass horizon if needed
        )
        
        logging.info("Full data pipeline finished.")
        return fusemoe_input_data