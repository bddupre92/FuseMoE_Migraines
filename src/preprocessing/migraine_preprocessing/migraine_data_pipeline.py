import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import json
from datetime import datetime, timedelta
import glob

# Import processors
from .eeg_processor import EEGProcessor
from .weather_connector import WeatherConnector
from .sleep_processor import SleepProcessor
from .stress_processor import StressProcessor
from ..advanced_imputation import BaseImputer, KNNImputer, IterativeImputer


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

                # Ensure proper format
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
                        # Add other checks/parsing if necessary (e.g., for severity, triggers)
                        
                        # Append valid event
                        all_events.append(event)
                    else:
                        # Print warning if essential keys are missing
                        print(f"Warning: Skipping invalid event entry (missing start_time or severity) in {file}: {event}")
            except Exception as e:
                print(f"Error loading or processing migraine events file {file}: {e}")

        # Sort events by start_time just in case
        # Use a try-except block in case start_time is not always present or valid after parsing
        try:
            all_events.sort(key=lambda x: x.get('start_time', pd.Timestamp.min))
        except TypeError as sort_e:
            print(f"Warning: Could not sort events by start_time due to type error: {sort_e}. Events may be out of order.")

        self.migraine_events = all_events
        print(f"Loaded a total of {len(self.migraine_events)} migraine events.")
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

        # --- Process EEG data --- #
        # NOTE: self.eeg_processor.process_dataset might expect a list of .npy file paths.
        # It needs to be adapted to handle the pre-loaded DataFrame (eeg_df_raw)
        # or the path to the CSV file.
        # For now, let's assume it can handle the DataFrame directly or adjust it later.
        try:
            print("Processing combined EEG DataFrame...")
            eeg_df = self.eeg_processor.process_dataset(eeg_df_raw) # Pass DataFrame
            self.eeg_data = eeg_df
            print(f"Processed EEG data: {len(self.eeg_data)} records")
            return eeg_df
        except Exception as e:
             print(f"Error processing EEG data: {e}")
             print("Please check if EEGProcessor.process_dataset can handle a DataFrame.")
             return pd.DataFrame()
    
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
                                prediction_horizon: int = 24,
                                imputation_method: Optional[str] = 'knn',
                                imputer_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Merge all processed modalities into a single time-aligned DataFrame,
        optionally impute missing values, and prepare for target creation.

        Args:
            time_window: Resampling frequency (e.g., '1H', '30T').
            prediction_horizon: How many hours ahead to predict migraine events.
            imputation_method: Method to use for imputation ('knn', 'iterative', or None).
            imputer_config: Dictionary of parameters for the chosen imputer.

        Returns:
            A time-indexed DataFrame with all features and a target column.
        """
        if self.aligned_data is None or len(self.aligned_data) == 0:
            print("Warning: No aligned data available")
            return pd.DataFrame()
        
        dfs_to_merge = []
        modality_prefixes = {} # To avoid column name collisions

        # Check and collect dataframes with valid DatetimeIndex
        print("Checking processed modality dataframes for merging...")
        modality_sources = {
            'eeg': self.eeg_data,
            'weather': self.weather_data,
            'sleep': self.sleep_data,
            'stress': self.stress_data
        }

        for name, df in modality_sources.items():
            if df is not None and not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    print(f"  - Found valid DatetimeIndex for '{name}' data.")
                    dfs_to_merge.append(df)
                    modality_prefixes[name] = f"{name}_" # Use name as prefix base
                else:
                    print(f"Warning: '{name}' data found but index is not DatetimeIndex (type: {type(df.index)}). Skipping merge.")
            # Optionally print if data is None or empty
            # else:
            #     print(f"  - '{name}' data is None or empty.")

        if not dfs_to_merge:
             print("Error: No modality dataframes available with DatetimeIndex for merging.")
             return pd.DataFrame()

        # --- REVISED MERGING LOGIC --- 
        # Rename columns with prefixes first
        prefixed_dfs = []
        for name, df in modality_sources.items():
            if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
                prefix = f"{name}_"
                df_renamed = df.copy().add_prefix(prefix)
                prefixed_dfs.append(df_renamed)
        
        if not prefixed_dfs:
            print("Error: No valid dataframes left after prefixing.")
            return pd.DataFrame()
            
        # --- Debug: Print time range for each modality --- 
        print("Individual modality time ranges:")
        for df in prefixed_dfs:
            col_prefix = df.columns[0].split('_')[0] # Get modality name from first column prefix
            min_ts = df.index.min()
            max_ts = df.index.max()
            print(f"  - {col_prefix}: Min={min_ts}, Max={max_ts}")
        # --- End Debug --- 
        
        # Determine common time range from the *actual* data to be merged
        min_times = [df.index.min() for df in prefixed_dfs]
        max_times = [df.index.max() for df in prefixed_dfs]
        
        if not min_times or not max_times:
            print("Warning: Could not determine time range from data sources.")
            return pd.DataFrame()
        
        # Use the latest start time and earliest end time for overlap
        start_time = max(min_times) 
        end_time = min(max_times)
        
        if start_time >= end_time:
            print(f"Warning: No time overlap found between data sources. Start: {start_time}, End: {end_time}")
            return pd.DataFrame()

        # Create a datetime index with regular intervals covering the overlap
        print(f"Creating multimodal index from {start_time} to {end_time} with frequency {time_window}")
        multimodal_df = pd.DataFrame(index=pd.date_range(start=start_time, end=end_time, freq=time_window))
        multimodal_df.index.name = 'timestamp' # Match index name for joining

        # Resample and join each modality
        print("Resampling and joining modalities...")
        for df in prefixed_dfs:
            # Resample each modality to the common time window
            # Use mean aggregation by default, can be customized if needed
            resampled_df = df.resample(time_window).mean() 
            
            # Join onto the main dataframe
            multimodal_df = multimodal_df.join(resampled_df, how='left')
            print(f"  - Joined data with columns: {list(resampled_df.columns)}")
        
        print(f"Shape after joining all modalities: {multimodal_df.shape}")
        
        # --- TODO: Add Migraine Target Label --- #
        # This part needs to be implemented by aligning self.migraine_events 
        # to the multimodal_df.index and calculating the target
        # For now, initialize the column
        multimodal_df['migraine_within_horizon'] = False
        print("Placeholder for migraine target label added.")
        # Example (needs refinement based on migraine_events structure):
        # if self.migraine_events:
        #     migraine_times = pd.to_datetime([e['start_time'] for e in self.migraine_events if 'start_time' in e])
        #     migraine_series = pd.Series(True, index=migraine_times)
        #     # Align and check horizon...
        

        # Forward fill missing values (carry forward last observation)
        print("Forward filling missing values...")
        multimodal_df = multimodal_df.ffill()
        print(f"Shape after ffill: {multimodal_df.shape}")
        
        # Fill remaining NaNs with 0 (or consider median/mean)
        print("Filling remaining NaNs with 0...")
        multimodal_df = multimodal_df.fillna(0)
        print(f"Shape after fillna(0): {multimodal_df.shape}")
        
        # Print info about the dataframe before imputation
        print("--- DataFrame before Imputation ---")
        print(multimodal_df.info())
        if multimodal_df.isnull().any().any():
             print("Warning: NaNs still present before imputation block.")
             print(multimodal_df.isnull().sum())
        else:
             print("No NaNs detected before imputation block.")
        print("-----------------------------------")

        # Reset index to make timestamp a column if needed downstream
        # Keep it as index for now, imputation expects it
        # multimodal_df = multimodal_df.reset_index()
        
        # >>> START IMPUTATION BLOCK <<<
        if imputation_method is not None:
            print(f"Attempting imputation using method: {imputation_method}")
            # Convert to NumPy, preserving original index and columns
            original_index = multimodal_df.index
            original_columns = multimodal_df.columns
            X = multimodal_df.values

            # Ensure NaNs are used for missing values
            if not np.issubdtype(X.dtype, np.floating):
                 X = X.astype(float) # Convert to float if not already
            # Assume missing values might be represented differently, ensure they are NaN
            # This part might need adjustment based on how missingness is truly represented
            # For now, we assume pd.concat introduced NaNs where expected.

            if np.isnan(X).any():
                print("Missing values (NaNs) detected, proceeding with imputation.")
                mask = ~np.isnan(X)
                imputer: Optional[BaseImputer] = None
                imputer_params = imputer_config if imputer_config is not None else {}

                # Reshape 2D DataFrame (time, features) to 3D (samples, time, features)
                # Assumption: Treating the entire merged DataFrame as one sample sequence.
                # This aligns with the current design of the BaseImputer wrappers.
                if X.ndim == 2:
                    print("Reshaping 2D data (time, features) to 3D (1, time, features) for imputation.")
                    X_3d = X.reshape(1, X.shape[0], X.shape[1])
                    mask_3d = mask.reshape(1, mask.shape[0], mask.shape[1])
                else:
                    # If data is already 3D+, handle appropriately or raise error
                    # For now, assume pipeline provides 2D at this stage.
                    raise ValueError(f"Expected 2D data after merging, but got shape {X.shape}")

                if imputation_method == 'knn':
                    print(f"Applying KNN Imputation with config: {imputer_params}")
                    imputer = KNNImputer(**imputer_params)
                elif imputation_method == 'iterative':
                    print(f"Applying Iterative Imputation with config: {imputer_params}")
                    imputer = IterativeImputer(**imputer_params)
                # Add elif for other methods like 'pso', 'autoencoder' if implemented
                else:
                    print(f"Warning: Unknown imputation method '{imputation_method}'. Supported methods: 'knn', 'iterative'. Skipping imputation.")

                if imputer:
                    try:
                        print(f"Shape before imputation (3D): {X_3d.shape}")
                        X_imputed_3d = imputer.fit_transform(X_3d, mask_3d)
                        print(f"Shape after imputation (3D): {X_imputed_3d.shape}")

                        # Reshape back to 2D
                        X_imputed_2d = X_imputed_3d.reshape(X_imputed_3d.shape[1], X_imputed_3d.shape[2])

                        # Convert back to DataFrame
                        multimodal_df = pd.DataFrame(X_imputed_2d, index=original_index, columns=original_columns)
                        print("Imputation successful.")
                    except Exception as e:
                        print(f"ERROR during {imputation_method} imputation: {e}")
                        import traceback
                        traceback.print_exc()
                        print("Skipping imputation due to error.")
            else:
                print("No missing values (NaNs) found. Skipping imputation.")
        else:
            print("No imputation method specified. Skipping imputation.")
        # >>> END IMPUTATION BLOCK <<<

        # Create target variable (e.g., binary indicator for migraine within prediction_horizon)
        # ... rest of the method ...

        return multimodal_df
    
    def prepare_data_for_fusemoe(self, multimodal_df: pd.DataFrame,
                               window_size: int = 24,
                               step_size: int = 12) -> Dict[str, Any]:
        """
        Prepare data for FuseMoE model in the format expected by the model.
        
        Args:
            multimodal_df: Multimodal DataFrame from create_multimodal_dataset
            window_size: Size of time windows in rows
            step_size: Step size for sliding windows
            
        Returns:
            Dictionary with data prepared for FuseMoE model
        """
        # Group columns by modality
        eeg_cols = [col for col in multimodal_df.columns if col.startswith('eeg_')]
        weather_cols = [col for col in multimodal_df.columns if col.startswith('weather_')]
        sleep_cols = [col for col in multimodal_df.columns if col.startswith('sleep_')]
        stress_cols = [col for col in multimodal_df.columns if col.startswith('stress_')]
        
        # Create sliding windows
        X_windows = []
        y_windows = []
        
        for i in range(0, len(multimodal_df) - window_size + 1, step_size):
            window = multimodal_df.iloc[i:i+window_size]
            
            # Extract features by modality
            eeg_features = window[eeg_cols].values if eeg_cols else None
            weather_features = window[weather_cols].values if weather_cols else None
            sleep_features = window[sleep_cols].values if sleep_cols else None
            stress_features = window[stress_cols].values if stress_cols else None
            
            # Create a dictionary of modality features
            x_dict = {}
            if eeg_features is not None:
                x_dict['eeg'] = eeg_features
            if weather_features is not None:
                x_dict['weather'] = weather_features
            if sleep_features is not None:
                x_dict['sleep'] = sleep_features
            if stress_features is not None:
                x_dict['stress'] = stress_features
            
            # Target is whether there's a migraine in the next prediction_horizon hours after the window
            target_idx = i + window_size
            if target_idx < len(multimodal_df):
                y = multimodal_df.iloc[target_idx]['migraine_within_horizon']
            else:
                y = False
            
            X_windows.append(x_dict)
            y_windows.append(int(y))
        
        # Convert to format expected by FuseMoE
        fusemoe_data = {
            'X': X_windows,
            'y': np.array(y_windows),
            'modalities': list(X_windows[0].keys()) if X_windows else [],
            'features_per_modality': {
                'eeg': len(eeg_cols) if eeg_cols else 0,
                'weather': len(weather_cols) if weather_cols else 0,
                'sleep': len(sleep_cols) if sleep_cols else 0,
                'stress': len(stress_cols) if stress_cols else 0
            }
        }
        
        return fusemoe_data
    
    def create_simple_multimodal_dataset(self, time_window: str = '24H',
                               prediction_horizon: int = 24) -> pd.DataFrame:
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
                        prediction_horizon: int = 24,
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
            prediction_horizon=prediction_horizon,
            imputation_method=imputation_method,
            imputer_config=imputer_config
        )

        if multimodal_df.empty:
            print("Warning: Multimodal dataset is empty after creation/imputation.")
            return {'X': [], 'y': [], 'modalities': [], 'features_per_modality': {}}

        # Prepare data for FuseMOE model
        fusemoe_input_data = self.prepare_data_for_fusemoe(
            multimodal_df=multimodal_df,
            window_size=window_size,
            step_size=window_size // 2  # Example overlap
        )

        return fusemoe_input_data 