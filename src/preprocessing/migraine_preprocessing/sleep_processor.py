import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import json
import os


def extract_sleep_duration(sleep_data: Dict) -> float:
    """
    Extract total sleep duration in hours from sleep tracking data.
    
    Args:
        sleep_data: Raw sleep tracking data
        
    Returns:
        Sleep duration in hours
    """
    if 'duration' in sleep_data:
        # If duration is already provided in seconds
        return sleep_data['duration'] / 3600
    
    # Calculate from start and end times
    if 'start_time' in sleep_data and 'end_time' in sleep_data:
        start = pd.Timestamp(sleep_data['start_time'])
        end = pd.Timestamp(sleep_data['end_time'])
        duration = (end - start).total_seconds() / 3600
        return duration
    
    # If there are multiple sleep stages, sum them
    if 'stages' in sleep_data:
        return sum(stage.get('duration', 0) for stage in sleep_data['stages']) / 3600
    
    return 0.0


def extract_sleep_efficiency(sleep_data: Dict) -> float:
    """
    Extract sleep efficiency (percentage of time in bed actually sleeping).
    
    Args:
        sleep_data: Raw sleep tracking data
        
    Returns:
        Sleep efficiency (0-100)
    """
    if 'efficiency' in sleep_data:
        return sleep_data['efficiency']
    
    if 'sleep_duration' in sleep_data and 'time_in_bed' in sleep_data:
        if sleep_data['time_in_bed'] > 0:
            return (sleep_data['sleep_duration'] / sleep_data['time_in_bed']) * 100
    
    if 'awake' in sleep_data and 'duration' in sleep_data:
        if sleep_data['duration'] > 0:
            return ((sleep_data['duration'] - sleep_data['awake']) / sleep_data['duration']) * 100
    
    return 0.0


def extract_sleep_stages(sleep_data: Dict) -> Dict[str, float]:
    """
    Extract sleep stages percentages.
    
    Args:
        sleep_data: Raw sleep tracking data
        
    Returns:
        Dictionary with percentages of different sleep stages
    """
    stages = {
        'deep_sleep': 0.0,
        'light_sleep': 0.0,
        'rem_sleep': 0.0,
        'awake': 0.0
    }
    
    if 'stages' in sleep_data:
        total_duration = sum(stage.get('duration', 0) for stage in sleep_data['stages'])
        if total_duration > 0:
            for stage in sleep_data['stages']:
                stage_type = stage.get('stage_type', '').lower()
                duration = stage.get('duration', 0)
                
                if 'deep' in stage_type:
                    stages['deep_sleep'] += duration
                elif 'light' in stage_type:
                    stages['light_sleep'] += duration
                elif 'rem' in stage_type:
                    stages['rem_sleep'] += duration
                elif 'awake' in stage_type:
                    stages['awake'] += duration
            
            # Convert to percentages
            for key in stages:
                stages[key] = (stages[key] / total_duration) * 100
    else:
        # Try individual stage keys
        if 'deep_sleep' in sleep_data and 'duration' in sleep_data:
            stages['deep_sleep'] = (sleep_data['deep_sleep'] / sleep_data['duration']) * 100
        if 'light_sleep' in sleep_data and 'duration' in sleep_data:
            stages['light_sleep'] = (sleep_data['light_sleep'] / sleep_data['duration']) * 100
        if 'rem_sleep' in sleep_data and 'duration' in sleep_data:
            stages['rem_sleep'] = (sleep_data['rem_sleep'] / sleep_data['duration']) * 100
        if 'awake' in sleep_data and 'duration' in sleep_data:
            stages['awake'] = (sleep_data['awake'] / sleep_data['duration']) * 100
    
    return stages


def extract_awakenings(sleep_data: Dict) -> int:
    """
    Extract number of awakenings during sleep.
    
    Args:
        sleep_data: Raw sleep tracking data
        
    Returns:
        Number of awakenings
    """
    if 'awakenings' in sleep_data:
        return sleep_data['awakenings']
    
    if 'stages' in sleep_data:
        awakenings = 0
        previous_stage = None
        
        for stage in sleep_data['stages']:
            current_stage = stage.get('stage_type', '').lower()
            if previous_stage and previous_stage != 'awake' and current_stage == 'awake':
                awakenings += 1
            previous_stage = current_stage
        
        return awakenings
    
    return 0


def extract_sleep_onset_time(sleep_data: Dict) -> datetime:
    """
    Extract sleep onset time (when the person fell asleep).
    
    Args:
        sleep_data: Raw sleep tracking data
        
    Returns:
        Datetime of sleep onset, or pd.NaT if not found.
    """
    if 'onset_time' in sleep_data:
        try:
            return pd.Timestamp(sleep_data['onset_time'])
        except Exception as e:
            print(f"Warning: Could not parse onset_time: {sleep_data['onset_time']} - {e}")
    
    # Check for 'sleep_start' (common in synthetic data)
    if 'sleep_start' in sleep_data: 
        try:
            return pd.Timestamp(sleep_data['sleep_start'])
        except Exception as e:
            print(f"Warning: Could not parse sleep_start: {sleep_data['sleep_start']} - {e}")
            
    # Check for 'start_time' (alternative key)
    if 'start_time' in sleep_data:
        try:
            return pd.Timestamp(sleep_data['start_time'])
        except Exception as e:
            print(f"Warning: Could not parse start_time: {sleep_data['start_time']} - {e}")
    
    # Fallback: try to use the 'date' field if available
    if 'date' in sleep_data:
        try:
            # Combine date with midnight as a reasonable default onset
            date_only = pd.to_datetime(sleep_data['date']).date()
            return pd.Timestamp.combine(date_only, datetime.min.time())
        except Exception as e:
            print(f"Warning: Could not parse date for fallback onset: {sleep_data['date']} - {e}")

    # Default to NaT if no valid time information found
    print(f"Warning: Could not determine sleep onset time from record: {sleep_data}. Returning NaT.")
    return pd.NaT


def calculate_sleep_consistency(sleep_records: List[Dict]) -> float:
    """
    Calculate sleep consistency (variation in sleep timing).
    
    Args:
        sleep_records: List of sleep records
        
    Returns:
        Sleep consistency score (0-1)
    """
    if not sleep_records:
        return 0.0
    
    # Extract sleep onset times
    onset_times = []
    for record in sleep_records:
        onset_time = extract_sleep_onset_time(record)
        # Skip NaT values for consistency calculation
        if pd.notna(onset_time):
             onset_times.append(onset_time.hour + onset_time.minute / 60)
    
    # Calculate standard deviation of onset times
    if len(onset_times) > 1:
        std_dev = np.std(onset_times)
        # Convert to consistency score (lower std_dev means higher consistency)
        # Scale to 0-1 range (4 hours std dev -> 0, 0 hours std dev -> 1)
        consistency = max(0, 1 - (std_dev / 4))
        return consistency
    
    return 0.5  # Default value with only one record


class SleepProcessor:
    """
    Class for processing sleep tracking data for migraine prediction.
    Sleep disturbances and changes in sleep patterns can be migraine triggers.
    """
    
    def __init__(self, lookback_window: int = 7):
        """
        Initialize sleep processor.
        
        Args:
            lookback_window: Number of days to look back for sleep consistency calculation
        """
        self.lookback_window = lookback_window
    
    def process_sleep_record(self, sleep_record: Dict) -> Dict[str, float]:
        """
        Process a single sleep record to extract relevant features.
        
        Args:
            sleep_record: Raw sleep tracking data
            
        Returns:
            Dictionary of extracted sleep features
        """
        # Basic sleep metrics
        features = {}
        
        features['sleep_duration'] = extract_sleep_duration(sleep_record)
        features['sleep_efficiency'] = extract_sleep_efficiency(sleep_record)
        features['awakenings'] = extract_awakenings(sleep_record)
        
        # Sleep stages
        sleep_stages = extract_sleep_stages(sleep_record)
        features.update(sleep_stages)
        
        # Sleep onset time
        onset_time = extract_sleep_onset_time(sleep_record)
        # Handle NaT case for derived features
        if pd.notna(onset_time):
             features['sleep_onset_hour'] = onset_time.hour + onset_time.minute / 60
             features['is_delayed_sleep'] = 1 if features['sleep_onset_hour'] > 23.5 else 0
             features['timestamp'] = onset_time # Add the full timestamp
        else:
             features['sleep_onset_hour'] = np.nan
             features['is_delayed_sleep'] = np.nan
             features['timestamp'] = pd.NaT # Ensure timestamp is NaT
        
        # Extract date (use onset_time's date if valid, otherwise fallback)
        if pd.notna(onset_time):
             features['date'] = onset_time.date()
        elif 'date' in sleep_record:
             try:
                  features['date'] = pd.Timestamp(sleep_record['date']).date()
             except:
                  features['date'] = pd.NaT
        else:
            features['date'] = pd.NaT
        
        return features
    
    def process_sleep_dataset(self, sleep_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataset of sleep records provided as a DataFrame.

        Args:
            sleep_data_df: DataFrame containing sleep data (e.g., loaded from CSV).

        Returns:
            DataFrame with processed sleep features.
        """
        all_features = []
        if sleep_data_df.empty:
            return pd.DataFrame(all_features)

        print(f"Processing {len(sleep_data_df)} sleep records from DataFrame...")
        # Convert DataFrame rows to dictionaries for process_sleep_record
        sleep_records_list = sleep_data_df.to_dict('records')

        for record_dict in sleep_records_list:
            try:
                # Ensure timestamps are handled correctly if they are strings
                if 'start_time' in record_dict and isinstance(record_dict['start_time'], str):
                    record_dict['start_time'] = pd.Timestamp(record_dict['start_time'])
                if 'end_time' in record_dict and isinstance(record_dict['end_time'], str):
                    record_dict['end_time'] = pd.Timestamp(record_dict['end_time'])
                if 'date' in record_dict and isinstance(record_dict['date'], str):
                     record_dict['date'] = pd.Timestamp(record_dict['date']).date()
                # Add similar checks for other potential date/time fields if needed

                processed_features = self.process_sleep_record(record_dict)
                all_features.append(processed_features)
            except Exception as e:
                print(f"Warning: Skipping sleep record due to processing error: {e}. Record: {record_dict}")
                continue

        # Convert list of feature dictionaries to DataFrame
        if not all_features:
             return pd.DataFrame(all_features)

        sleep_df = pd.DataFrame(all_features)
        # Convert timestamp column to datetime and set as index
        if 'timestamp' in sleep_df.columns:
             try:
                  sleep_df['timestamp'] = pd.to_datetime(sleep_df['timestamp'])
                  sleep_df = sleep_df.set_index('timestamp')
                  sleep_df = sleep_df.sort_index() # Ensure index is sorted
             except Exception as e:
                  print(f"Warning: Could not set DatetimeIndex for sleep data: {e}")
        else:
             print("Warning: 'timestamp' column missing after sleep processing.")

        # --- Add check to drop non-numeric columns like 'date' --- 
        cols_to_drop = []
        numeric_cols = []
        for col in sleep_df.columns:
            # Keep the index
            if col == sleep_df.index.name:
                continue 
            # Try converting to numeric, drop if it fails or if it's explicitly non-numeric like date
            try:
                pd.to_numeric(sleep_df[col])
                numeric_cols.append(col) # Keep track of numeric columns
            except (ValueError, TypeError):
                 # Also explicitly drop the 'date' column if it exists
                 if col != 'date': 
                      print(f"Warning: Column '{col}' is non-numeric in SleepProcessor. Dropping.")
                 cols_to_drop.append(col)
        
        # Always drop the 'date' column if present
        if 'date' in sleep_df.columns and 'date' not in cols_to_drop:
            cols_to_drop.append('date')
            
        # Drop identified non-numeric columns
        if cols_to_drop:
             sleep_df = sleep_df.drop(columns=cols_to_drop)
             print(f"Dropped non-numeric columns from Sleep data: {cols_to_drop}")
        # --- End check --- 

        return sleep_df
    
    def get_migraine_risk_from_sleep(self, sleep_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate migraine risk score based on sleep patterns.
        
        Args:
            sleep_data: DataFrame with processed sleep data
            
        Returns:
            DataFrame with original data plus migraine risk scores
        """
        if sleep_data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        risk_df = sleep_data.copy()
        
        # Calculate risk score components
        # These weights should be optimized based on patient data
        
        # Sleep deprivation risk
        if 'sleep_debt_3day' in risk_df.columns:
            deprivation_risk = np.clip(risk_df['sleep_debt_3day'] / 8, 0, 1) * 4
        else:
            deprivation_risk = np.clip((8 - risk_df['sleep_duration']) / 3, 0, 1) * 3
        
        # Sleep quality risk
        if 'sleep_quality_index' in risk_df.columns:
            quality_risk = np.clip((10 - risk_df['sleep_quality_index']) / 5, 0, 1) * 3
        else:
            quality_risk = np.clip((100 - risk_df['sleep_efficiency']) / 30, 0, 1) * 3
        
        # Sleep pattern disruption risk
        if 'sleep_disruption' in risk_df.columns:
            disruption_risk = risk_df['sleep_disruption'] * 3
        else:
            disruption_risk = 0
        
        # REM sleep risk (low REM has been associated with migraines)
        if 'rem_sleep' in risk_df.columns:
            rem_risk = np.clip((20 - risk_df['rem_sleep']) / 10, 0, 1) * 2
        else:
            rem_risk = 0
        
        # Combine into overall sleep-based migraine risk score (0-10 scale)
        risk_df['sleep_migraine_risk'] = np.clip(
            deprivation_risk + quality_risk + disruption_risk + rem_risk,
            0, 10
        )
        
        return risk_df
    
    def align_sleep_with_migraine_events(self, 
                                        sleep_data: pd.DataFrame,
                                        migraine_events: List[Dict]) -> pd.DataFrame:
        """
        Align sleep data features with migraine events.
        
        Args:
            sleep_data: DataFrame with processed sleep data
            migraine_events: List of dictionaries with migraine events
                             (must contain 'start_time' and 'severity' keys)
            
        Returns:
            DataFrame with sleep data aligned to migraine events
        """
        if sleep_data.empty or not migraine_events:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        aligned_df = sleep_data.copy()
        
        # Add a 'hours_to_next_migraine' column
        aligned_df['hours_to_next_migraine'] = np.inf
        aligned_df['next_migraine_severity'] = None

        # Ensure sleep_data index is datetime
        if not isinstance(aligned_df.index, pd.DatetimeIndex):
            print("Warning: Sleep data index is not DatetimeIndex. Cannot align with events.")
            return aligned_df # Return unmodified

        # Sort events by time for efficiency
        valid_events = []
        for event in migraine_events:
             event_time = event.get('start_time') # Use start_time
             if event_time:
                  try:
                       valid_events.append({
                            'start_time': pd.Timestamp(event_time),
                            'severity': event.get('severity')
                       })
                  except Exception:
                       print(f"Warning: Could not parse start_time for event {event}. Skipping.")
        valid_events.sort(key=lambda x: x['start_time'])

        # Iterate through sleep data (assuming index is the date/time of the sleep record)
        for sleep_idx_time in aligned_df.index:
            # Find the next migraine event *after* this sleep record time
            next_events = [e for e in valid_events if e['start_time'] > sleep_idx_time]

            if next_events:
                next_event = next_events[0] # The first event after the sleep record
                time_diff = next_event['start_time'] - sleep_idx_time
                hours_diff = time_diff.total_seconds() / 3600

                # Update the row in sleep_data
                aligned_df.loc[sleep_idx_time, 'hours_to_next_migraine'] = hours_diff
                aligned_df.loc[sleep_idx_time, 'next_migraine_severity'] = next_event.get('severity')

        return aligned_df 