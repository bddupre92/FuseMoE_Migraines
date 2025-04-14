import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import json
from datetime import datetime, timedelta
import glob
import logging

# Import processors
from .eeg_processor import EEGProcessor
from .weather_connector import WeatherConnector
from .sleep_processor import SleepProcessor
from .stress_processor import StressProcessor
from ..advanced_imputation import BaseImputer, KNNImputer, IterativeImputerWrapper, AutoencoderImputer


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
                                all_events.append(event) # Append valid event
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

        # --- Deduplicate events based on start_time (Refactored) --- #
        deduplicated_events = []
        seen_start_times = set()
        print("Deduplicating events based on start_time...") 
        for event in all_events: # Iterate through already validated and sorted events
            start_time = event.get('start_time') 
            # We know start_time is a valid Timestamp here due to filtering above
            if start_time not in seen_start_times:
                deduplicated_events.append(event)
                seen_start_times.add(start_time)
            # else: # Implicitly skip duplicates
                # print(f"DEBUG: Skipping duplicate event with start_time {start_time}")
        # No need to handle invalid start_times here anymore
        
        num_removed = len(all_events) - len(deduplicated_events)
        if num_removed > 0:
            print(f"Removed {num_removed} duplicate events based on start_time.")
            
        self.migraine_events = deduplicated_events
        print(f"Loaded a total of {len(self.migraine_events)} unique migraine events.")
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
        Align all modalities with migraine events.
        
        Returns:
            Dictionary of aligned DataFrames for each modality
        """
        if self.migraine_events is None or len(self.migraine_events) == 0:
            print("Warning: No migraine events available for alignment")
            return {}
        
        aligned_dfs = {}
        
        # Align weather data
        if self.weather_data is not None and not self.weather_data.empty:
            aligned_weather = self.weather_connector.align_weather_with_migraine_events(
                self.weather_data, self.migraine_events)
            aligned_dfs['weather'] = aligned_weather
        
        # Align sleep data
        if self.sleep_data is not None and not self.sleep_data.empty:
            aligned_sleep = self.sleep_processor.align_sleep_with_migraine_events(
                self.sleep_data, self.migraine_events)
            aligned_dfs['sleep'] = aligned_sleep
        
        # Align stress data
        if self.stress_data is not None and not self.stress_data.empty:
            aligned_stress = self.stress_processor.align_stress_with_migraine_events(
                self.stress_data, self.migraine_events)
            aligned_dfs['stress'] = aligned_stress
        
        # Align EEG data
        if self.eeg_data is not None and not self.eeg_data.empty:
            # Convert EEG timestamp to datetime if needed
            if 'start_time' in self.eeg_data.columns:
                # Create hours to next migraine column
                self.eeg_data['hours_to_next_migraine'] = np.inf
                self.eeg_data['next_migraine_severity'] = None
                
                for idx, row in self.eeg_data.iterrows():
                    eeg_time = row['start_time']
                    
                    # Find the next migraine event after this EEG record
                    next_events = [e for e in self.migraine_events if e['start_time'] > eeg_time]
                    
                    if next_events:
                        next_event = next_events[0]
                        time_diff = next_event['start_time'] - eeg_time
                        self.eeg_data.at[idx, 'hours_to_next_migraine'] = time_diff.total_seconds() / 3600
                        self.eeg_data.at[idx, 'next_migraine_severity'] = next_event['severity']
                
                # Create binary label for prediction (migraine within next 24 hours)
                self.eeg_data['migraine_within_24h'] = self.eeg_data['hours_to_next_migraine'] <= 24
                aligned_dfs['eeg'] = self.eeg_data
        
        self.aligned_data = aligned_dfs
        return aligned_dfs
    
    def create_multimodal_dataset(self, time_window: str = '1H', 
                                prediction_horizon: int = 6,
                                imputation_method: Optional[str] = 'knn',
                                imputer_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Merge all processed modalities into a single time-aligned DataFrame,
        optionally impute missing values, and prepare for target creation.
        
        Args:
            time_window: Resampling frequency (e.g., '1H', '30T').
            prediction_horizon: How many hours ahead to predict migraine events.
            imputation_method: Method to use for imputation ('knn', 'iterative', 'autoencoder', 'none').
            imputer_config: Dictionary of parameters for the chosen imputer.
            
        Returns:
            A time-indexed DataFrame with all features, patient_id, and a target column.
        """
        # Check and collect dataframes with valid DatetimeIndex
        logging.info("Checking processed modality dataframes for merging...")
        modality_sources = {
            'eeg': self.eeg_data,
            'weather': self.weather_data,
            'sleep': self.sleep_data,
            'stress': self.stress_data
        }
        dfs_to_merge = []
        for name, df in modality_sources.items():
            if df is not None and not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    logging.info(f"  - Found valid DatetimeIndex for '{name}' data.")
                    # IMPORTANT: Ensure timezone consistency or remove timezone before merging
                    if df.index.tz is not None:
                        df = df.tz_localize(None) # Make timezone naive
                    dfs_to_merge.append(df.add_prefix(f"{name}_")) # Add prefix now
                else:
                    logging.warning(f"Warning: '{name}' data found but index is not DatetimeIndex (type: {type(df.index)}). Skipping merge.")

        if not dfs_to_merge:
             logging.error("Error: No modality dataframes available with DatetimeIndex for merging.")
             return pd.DataFrame()

        # --- Determine Overlap Range ---
        min_times = [df.index.min() for df in dfs_to_merge]
        max_times = [df.index.max() for df in dfs_to_merge]
        start_time = max(min_times)
        end_time = min(max_times)
        logging.debug(f"Calculated overlap range: Start={start_time}, End={end_time}")

        if start_time >= end_time:
            logging.warning(f"No time overlap found between data sources. Start: {start_time}, End: {end_time}")
            return pd.DataFrame()
            
        # --- Load Patient ID Mapping ---
        patient_id_map = None
        try:
            eeg_raw_path = os.path.join(self.data_dir, "eeg", "all_eeg_data.csv")
            if os.path.exists(eeg_raw_path):
                eeg_raw_df = pd.read_csv(eeg_raw_path, parse_dates=['timestamp'])
                if eeg_raw_df['timestamp'].dt.tz is not None:
                     eeg_raw_df['timestamp'] = eeg_raw_df['timestamp'].dt.tz_localize(None) # Ensure TZ naive
                
                # Log patient distribution from raw file before any modification
                patient_counts_raw = eeg_raw_df['patient_id'].value_counts()
                logging.info(f"Found {len(patient_counts_raw)} unique patients in the raw EEG data file.")
                logging.debug(f"Patient ID counts (raw): {dict(list(patient_counts_raw.items())[:10])}{'...' if len(patient_counts_raw) > 10 else ''}")

                # --- Create a map with UNIQUE timestamp index --- # 
                # Group by timestamp and take the first patient_id for each unique timestamp
                logging.info("Creating unique patient ID map by taking the first patient per timestamp...")
                patient_id_map = eeg_raw_df.groupby('timestamp')['patient_id'].first()
                # --- ----------------------------------------- --- #
                
                # Ensure the map index is sorted for reliable reindexing
                patient_id_map = patient_id_map.sort_index()
                
                logging.info("Successfully loaded unique patient ID mapping from EEG data.")
                
            else:
                 logging.warning("Could not load patient ID map: Raw EEG CSV not found.")
        except Exception as e:
            logging.warning(f"Error loading patient ID map from EEG data: {e}")
            import traceback
            traceback.print_exc()
        # -----------------------------

        # --- Create Base DataFrame and Merge ---
        logging.info(f"Creating multimodal index from {start_time} to {end_time} with frequency {time_window}")
        multimodal_index = pd.date_range(start=start_time, end=end_time, freq=time_window)
        multimodal_df = pd.DataFrame(index=multimodal_index)
        multimodal_df.index.name = 'timestamp'

        logging.info("Resampling and joining numeric modalities...")
        for df_prefixed in dfs_to_merge:
            # Select only numeric columns BEFORE resampling
            numeric_cols = df_prefixed.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                # Resample numeric data using mean aggregation
                df_resampled = df_prefixed[numeric_cols].resample(time_window).mean()
                # Join with the main DataFrame
                multimodal_df = multimodal_df.join(df_resampled, how='left')
                logging.info(f"  - Joined data with columns: {list(df_resampled.columns)}")
            else:
                mod_name = df_prefixed.columns[0].split('_')[0] if df_prefixed.columns else 'Unknown'
                logging.warning(f"No numeric columns found for modality '{mod_name}'. Skipping join.")

        logging.info(f"Shape after joining numeric modalities: {multimodal_df.shape}")

        # --- Join and Fill Patient ID using Reindex ---
        if patient_id_map is not None:
            # Reindex WITHOUT forward fill first to preserve alignment
            patient_ids_reindexed = patient_id_map.reindex(multimodal_df.index) 
            multimodal_df['patient_id'] = patient_ids_reindexed
            
            # --- Custom Forward Fill Logic --- #
            logging.info("Applying custom forward fill for patient IDs...")
            # Identify rows with NaN patient IDs that need filling
            nan_indices = multimodal_df.index[multimodal_df['patient_id'].isnull()]
            if not nan_indices.empty:
                logging.debug(f"Found {len(nan_indices)} timestamps with NaN patient IDs initially.")
                # Create a temporary series with original IDs (non-NaN)
                original_id_series = multimodal_df['patient_id'].dropna()
                
                if not original_id_series.empty:
                    # Find the index of the last known ID for each NaN timestamp
                    # Use searchsorted on the index of known IDs
                    last_known_id_indices = original_id_series.index.searchsorted(nan_indices, side='right') - 1
                    
                    # Map these indices back to the actual timestamps with known IDs
                    valid_indices_mask = last_known_id_indices >= 0
                    fill_values_indices = original_id_series.index[last_known_id_indices[valid_indices_mask]]
                    
                    # Get the patient IDs corresponding to these timestamps
                    fill_values = original_id_series.loc[fill_values_indices]
                    
                    # Assign the found IDs to the NaN rows
                    # Ensure index alignment for assignment
                    fill_series = pd.Series(fill_values.values, index=nan_indices[valid_indices_mask])
                    multimodal_df['patient_id'].fillna(fill_series, inplace=True)
                    logging.info(f"Filled {len(fill_series)} NaN patient IDs using custom forward fill.")
                else:
                    logging.warning("No non-NaN patient IDs found to perform forward fill.")
            # --- End Custom Forward Fill --- #
            
            # Check for remaining NaNs and backfill if necessary
            if multimodal_df['patient_id'].isnull().any():
                missing_count = multimodal_df['patient_id'].isnull().sum()
                logging.warning(f"Patient ID is still missing for {missing_count} records after custom ffill. Applying standard backfill.")
                multimodal_df['patient_id'] = multimodal_df['patient_id'].bfill()
                if multimodal_df['patient_id'].isnull().any():
                    logging.error("Patient ID still missing after custom ffill and bfill. Cannot proceed reliably with patient-aware CV.")
                    multimodal_df['patient_id'].fillna('UNKNOWN_PATIENT', inplace=True) # Fill remaining with placeholder
            
            logging.info("Successfully joined and filled patient IDs to multimodal dataframe.")
            # Log final patient distribution in the multimodal dataframe
            final_patient_counts = multimodal_df['patient_id'].value_counts()
            logging.info(f"Found {len(final_patient_counts)} unique patients in the final multimodal dataframe.")
            logging.debug(f"Patient ID counts (final): {dict(list(final_patient_counts.items())[:20])}{'...' if len(final_patient_counts) > 20 else ''}")

        else:
             logging.warning("Could not join patient IDs as the mapping was not loaded.")
        # --- ------------------------------------- ---

        # --- Basic Imputation (ffill + fillna(0)) ---
        logging.info("Forward filling missing values...")
        multimodal_df_filled = multimodal_df.ffill() # Apply ffill first
        logging.info("Filling remaining NaNs with 0...")
        multimodal_df_filled.fillna(0, inplace=True) # Apply fillna(0) after ffill
        logging.info(f"Shape after basic ffill/fillna(0): {multimodal_df_filled.shape}")

        # Check for NaNs before advanced imputation
        numeric_cols_for_impute = multimodal_df_filled.select_dtypes(include=np.number).columns
        if multimodal_df_filled[numeric_cols_for_impute].isnull().any().any():
             logging.warning("Warning: NaNs still present before advanced imputation block.")
             logging.debug(multimodal_df_filled.isnull().sum())
        else:
             logging.info("No NaNs detected in numeric columns before advanced imputation block.")

        # --- Advanced Imputation (applied to the ffill/fillna'd data) ---
        multimodal_df_imputed = multimodal_df_filled.copy() # Start with the basically filled data
        if imputation_method and imputation_method.lower() != 'none':
             logging.info(f"Attempting advanced imputation using method: {imputation_method}")
             if not numeric_cols_for_impute.empty:
                 X_numeric = multimodal_df_filled[numeric_cols_for_impute].values
                 imputer: Optional[BaseImputer] = None
                 imputer_params = {}
                 if imputer_config:
                    try:
                        imputer_params = json.loads(imputer_config)
                    except json.JSONDecodeError: pass # Ignore errors, use defaults

                 # Reshape for imputer wrappers if needed (most expect 2D or 3D)
                 if X_numeric.ndim == 2:
                     X_for_impute = X_numeric.reshape(1, X_numeric.shape[0], X_numeric.shape[1]) # Treat as 1 sample
                 else:
                     logging.error("Unexpected shape for numeric data before imputation.")
                     X_for_impute = None

                 if X_for_impute is not None:
                     # Initialize imputer
                     if imputation_method.lower() == 'knn':
                         imputer = KNNImputer(**imputer_params)
                     elif imputation_method.lower() == 'iterative':
                         imputer = IterativeImputerWrapper(**imputer_params)
                     elif imputation_method.lower() == 'autoencoder':
                         imputer_params.setdefault('random_state', imputer_params.get('seed'))
                         imputer = AutoencoderImputer(**imputer_params)
                     else:
                          logging.warning(f"Unknown advanced imputation method '{imputation_method}'.")

                     # Fit and transform
                     if imputer:
                          try:
                              # Note: Mask might not be needed if data is already filled, but some imputers might use it
                              mask = ~np.isnan(X_for_impute)
                              X_imputed_3d = imputer.fit_transform(X_for_impute, mask)
                              X_imputed_2d = X_imputed_3d.reshape(X_imputed_3d.shape[1], X_imputed_3d.shape[2])
                              # Update the numeric columns in our dataframe
                              multimodal_df_imputed[numeric_cols_for_impute] = X_imputed_2d
                              logging.info(f"Advanced imputation ({imputation_method}) applied.")
                          except Exception as e:
                              logging.error(f"ERROR during {imputation_method} imputation: {e}")
                              import traceback
                              traceback.print_exc()
                              logging.warning("Skipping advanced imputation due to error.")
             else:
                  logging.warning("No numeric columns found for advanced imputation.")
        else:
             logging.info("Skipping advanced imputation as method is 'none' or not specified.")
        # --- End Advanced Imputation ---

        # --- Calculate Migraine Target Label ---
        logging.info("Calculating migraine target label...")
        multimodal_df_imputed['migraine_within_horizon'] = False # Initialize
        if self.migraine_events:
            valid_event_times = []
            for event in self.migraine_events:
                if 'start_time' in event and pd.notna(event['start_time']):
                    event_ts = pd.Timestamp(event['start_time'])
                    if event_ts.tz is not None: # Ensure timezone naive like index
                         event_ts = event_ts.tz_localize(None)
                    valid_event_times.append(event_ts)
            
            if valid_event_times:
                valid_event_times.sort()
                horizon_delta = pd.Timedelta(hours=prediction_horizon)
                
                # MODIFIED: Use a shorter effective horizon for labeling to create more balanced classes
                # This reduces the window where a migraine is considered "positive", making positive labels less common
                effective_horizon = min(prediction_horizon, 3)  # Use at most 3 hours for positive labeling
                effective_delta = pd.Timedelta(hours=effective_horizon)
                
                # MODIFIED: Skip some timestamps to balance classes (controlled downsampling of time points)
                # Calculate how many indices to skip based on event density
                time_points = multimodal_df_imputed.index
                estimated_positive_rate = len(valid_event_times) * effective_horizon / (len(time_points) * 1.0)
                target_positive_rate = 0.4  # Target ~40% positive rate
                
                # Skip factor calculation - skip more time points if estimated positive rate is too high
                skip_factor = 1
                if estimated_positive_rate > target_positive_rate and estimated_positive_rate > 0:
                    skip_factor = min(3, max(1, int(estimated_positive_rate / target_positive_rate)))
                
                logging.info(f"Using effective horizon of {effective_horizon}h for positive labeling")
                if skip_factor > 1:
                    logging.info(f"Applying time point sampling with skip factor {skip_factor} for better class balance")
                
                num_positive = 0
                for i, current_ts in enumerate(time_points):
                    # Skip some indices for better balance if needed
                    if skip_factor > 1 and i % skip_factor != 0 and num_positive > 0:
                        continue
                        
                    # Use effective horizon for labeling
                    window_end = current_ts + effective_delta
                    
                    # Optimized check using searchsorted (requires sorted event times)
                    start_idx = np.searchsorted(valid_event_times, current_ts, side='left')
                    end_idx = np.searchsorted(valid_event_times, window_end, side='left')
                    
                    if start_idx != end_idx: # If events exist between start and end index
                        multimodal_df_imputed.loc[current_ts, 'migraine_within_horizon'] = True
                        num_positive += 1
                
                # Calculate actual positive rate after labeling
                positive_rate = num_positive / len(multimodal_df_imputed)
                logging.info(f"Calculated target labels. Found {num_positive} positive samples (migraine within {effective_horizon}h horizon), {positive_rate:.2%} positive rate.")
            else:
                logging.warning("No valid migraine event times found to create target labels.")
        else:
            logging.warning("No migraine events loaded. Target labels will all be False.")
        # --- End Target Calculation ---

        logging.info(f"Final shape of multimodal dataset: {multimodal_df_imputed.shape}")
        # Log final NaN check
        if multimodal_df_imputed.isnull().any().any():
             logging.warning("Warning: NaNs detected in final multimodal DataFrame.")
             logging.debug(multimodal_df_imputed.isnull().sum())

        return multimodal_df_imputed
    
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

        # Determine columns to drop for feature extraction
        feature_cols_to_drop = ['migraine_within_horizon']
        if has_patient_id:
            feature_cols_to_drop.append('patient_id')

        # Iterate through windows and create feature/target/group triplets
        num_possible_windows = len(multimodal_df) - window_size - prediction_horizon + 1
        logging.debug(f"Total rows: {len(multimodal_df)}, Num possible windows: {num_possible_windows}")

        for i in range(0, num_possible_windows, step_size):
            window_start_idx = i
            window_end_idx = i + window_size
            target_idx = window_end_idx + prediction_horizon - 1 # Correct target index

            # Ensure target index is within bounds
            if target_idx >= len(multimodal_df):
                logging.warning(f"Target index {target_idx} out of bounds for window starting at {window_start_idx}. Skipping.")
                continue

            # Extract features for the window, dropping non-feature columns
            features_df = multimodal_df.iloc[window_start_idx:window_end_idx].drop(columns=feature_cols_to_drop, errors='ignore')
            X_list.append(features_df.values) # Append numpy array

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

        # Determine feature columns and order from the first sample's DataFrame structure
        if X_list:
            # Get column names from the DataFrame used to create the first sample
            # Need the actual column names preserved before .values was called
            # Let's re-extract the first feature DF to get columns reliably
            first_features_df = multimodal_df.iloc[0:window_size].drop(columns=feature_cols_to_drop, errors='ignore')
            ordered_feature_columns = first_features_df.columns.tolist()

            # Extract modality prefixes and counts
            modality_prefixes = sorted(list(set([col.split('_')[0] for col in ordered_feature_columns if '_' in col])))

            current_col_idx = 0
            for mod_prefix in modality_prefixes:
                mod_cols = [col for col in ordered_feature_columns if col.startswith(mod_prefix + '_')]
                if mod_cols:
                    count = len(mod_cols)
                    features_per_modality[mod_prefix] = count
                    ordered_modalities.append(mod_prefix)
                    # Store indices range for slicing later
                    features_per_modality[f"{mod_prefix}_indices"] = (current_col_idx, current_col_idx + count)
                    current_col_idx += count

            # Now split the stacked numpy arrays in X_list back into modalities
            if ordered_modalities:
                # Convert X_list (list of [window, features]) into one large array [batch, window, features]
                X_stacked_np = np.stack(X_list, axis=0) # Shape: [Batch, Window, TotalFeatures]

                for modality in ordered_modalities:
                    start_idx, end_idx = features_per_modality[f"{modality}_indices"]
                    X_dict[modality] = X_stacked_np[:, :, start_idx:end_idx] # Slice the stacked array

                # Validate shapes
                for modality, data in X_dict.items():
                    expected_shape = (len(X_list), window_size, features_per_modality[modality])
                    if data.shape != expected_shape:
                        logging.warning(f"Warning: Shape mismatch for modality '{modality}'. Expected {expected_shape}, got {data.shape}")
            else:
                 logging.warning("Could not determine modality order or features. X_dict will be empty.")

        return {
            'X': X_dict,
            'y': y_np,
            'groups': groups_np, # Included groups array
            'modalities': list(X_dict.keys()),
            'features_per_modality': {mod: count for mod, count in features_per_modality.items() if not mod.endswith('_indices')}
        }
    
    def create_simple_multimodal_dataset(self, time_window: str = '24H',
                               prediction_horizon: int = 6) -> pd.DataFrame:
        """
        Create a simplified multimodal dataset from all available data without complex alignments.
        
        Args:
            time_window: Time window for feature extraction (e.g., '24H', '12H')
            prediction_horizon: Prediction horizon in hours
            
        Returns:
            DataFrame with multimodal features and target labels
        """
        print(f"Creating simplified multimodal dataset with {time_window} window and {prediction_horizon}h horizon...")
        
        # Create date range from the start of data to the end
        date_range = []
        
        # Use migraine dates for reference
        migraine_dates = []
        for event in self.migraine_events:
            if isinstance(event.get('start_time'), str):
                date = pd.Timestamp(event.get('start_time')).date()
            else:
                date = pd.Timestamp(event.get('start_time')).date()
            migraine_dates.append(date)
        
        # Use sleep data dates as reference for normal days
        sleep_dates = []
        if not self.sleep_data.empty:
            for _, row in self.sleep_data.iterrows():
                if isinstance(row.get('date'), str):
                    date = pd.Timestamp(row.get('date')).date()
                else:
                    date = row.get('date').date()
                sleep_dates.append(date)
        
        # Create dataset rows
        multimodal_rows = []
        
        # Use all unique dates
        unique_dates = sorted(set(migraine_dates + sleep_dates))
        
        # For each date
        for date in unique_dates:
            timestamp = pd.Timestamp(date)
            
            # Check if this date had a migraine
            had_migraine = date in migraine_dates
            
            # Features dictionary
            features = {}
            
            # Add EEG features if available
            if not self.eeg_data.empty:
                eeg_on_date = self.eeg_data[
                    self.eeg_data['start_time'].dt.date == date
                ]
                if not eeg_on_date.empty:
                    # Use mean of EEG features for the day
                    for col in eeg_on_date.columns:
                        if col not in ['start_time', 'patient_id', 'file', 'channel']:
                            features[f'eeg_{col}'] = eeg_on_date[col].mean()
            
            # Add weather features if available
            if not self.weather_data.empty:
                weather_on_date = self.weather_data[
                    self.weather_data['start_time'].dt.date == date
                ]
                if not weather_on_date.empty:
                    # Use mean of weather features for the day
                    for col in weather_on_date.columns:
                        if col not in ['start_time', 'latitude', 'longitude']:
                            features[f'weather_{col}'] = weather_on_date[col].mean()
            
            # Add sleep features if available
            if not self.sleep_data.empty:
                sleep_on_date = self.sleep_data[
                    self.sleep_data['date'].dt.date == date
                ]
                if not sleep_on_date.empty:
                    # Use all sleep features
                    for col in sleep_on_date.columns:
                        if col not in ['date', 'patient_id', 'sleep_start', 'sleep_end']:
                            features[f'sleep_{col}'] = sleep_on_date[col].values[0]
            
            # Add stress features if available
            if not self.stress_data.empty:
                stress_on_date = self.stress_data[
                    self.stress_data['start_time'].dt.date == date
                ]
                if not stress_on_date.empty:
                    # Use mean of stress features for the day
                    for col in stress_on_date.columns:
                        if col not in ['start_time', 'patient_id', 'source']:
                            features[f'stress_{col}'] = stress_on_date[col].mean()
            
            # Only add the row if we have features
            if features:
                # Add timestamp and target
                features['start_time'] = timestamp
                features['had_migraine'] = int(had_migraine)
                
                multimodal_rows.append(features)
        
        # Create DataFrame
        if multimodal_rows:
            multimodal_df = pd.DataFrame(multimodal_rows)
            print(f"Created multimodal dataset with {len(multimodal_df)} rows and {len(multimodal_df.columns)} columns")
            return multimodal_df
        else:
            print("Warning: No multimodal rows created")
            return pd.DataFrame()
            
            
    def run_full_pipeline(self, 
                        location: Tuple[float, float],
                        start_date: Union[str, datetime],
                        end_date: Union[str, datetime],
                        prediction_horizon: int = 6,
                        window_size: int = 24,
                        imputation_method: Optional[str] = 'knn',
                        imputer_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline including imputation.
        
        Args:
            location: Location coordinates (latitude, longitude)
            start_date: Start date for data
            end_date: End date for data
            prediction_horizon: Hours ahead to predict migraines
            window_size: Number of hours to use as input window
            imputation_method: Method for imputation ('knn', 'iterative', 'none').
            imputer_config: Configuration dictionary for the imputer.
            
        Returns:
            Dictionary with processed data ready for the FuseMOE model
        """

        # Load migraine events
        self.load_migraine_events()

        # Process each modality
        self.process_eeg_data()
        self.process_weather_data(location, start_date, end_date)
        self.process_sleep_data()
        self.process_stress_data()

        # Align data (this method might need adjustment depending on requirements)
        self.align_data_with_migraine_events()

        # Create multimodal dataset
        multimodal_df = self.create_multimodal_dataset(
            time_window='1H', # Example time window, adjust as needed
            prediction_horizon=6, # Reduced from 24 to 6
            imputation_method=imputation_method,
            imputer_config=imputer_config
        )
        
        if multimodal_df.empty:
            print("Warning: Multimodal dataset is empty after creation/imputation.")
            return {'X': [], 'y': [], 'modalities': [], 'features_per_modality': {}}
        
        # Prepare data for FuseMOE model
        fusemoe_input_data = self.prepare_data_for_fusemoe(
            multimodal_df=multimodal_df,
            window_size=window_size
            # Removed explicit step_size to use the function default (step_size=1)
            # step_size=window_size // 2  # Example overlap
        )
        
        return fusemoe_input_data