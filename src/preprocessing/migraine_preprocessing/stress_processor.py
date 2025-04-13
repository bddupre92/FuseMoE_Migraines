import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import json
import os
from scipy import signal


def calculate_hrv_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Calculate HRV (Heart Rate Variability) features from RR intervals.
    
    Args:
        rr_intervals: Array of RR intervals in milliseconds
        
    Returns:
        Dictionary of HRV features
    """
    if len(rr_intervals) < 10:
        # Not enough data points
        return {
            'rmssd': 0.0,
            'sdnn': 0.0,
            'pnn50': 0.0,
            'lf_power': 0.0,
            'hf_power': 0.0,
            'lf_hf_ratio': 1.0
        }
    
    # Time domain features
    # RMSSD: Root Mean Square of Successive Differences
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # SDNN: Standard Deviation of NN intervals
    sdnn = np.std(rr_intervals)
    
    # pNN50: Percentage of successive RR intervals that differ by more than 50ms
    pnn50 = 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
    
    # Frequency domain features (simplified)
    # Convert RR intervals to evenly sampled time series
    rr_x = np.cumsum(rr_intervals) / 1000  # Convert to seconds
    rr_y = rr_intervals
    
    # Interpolate to get evenly sampled data
    fs = 4.0  # 4 Hz sampling rate
    interpolation_time = np.arange(rr_x[0], rr_x[-1], 1/fs)
    interpolated_rr = np.interp(interpolation_time, rr_x, rr_y)
    
    # Calculate Power Spectral Density using Welch method
    try:
        frequencies, psd = signal.welch(interpolated_rr, fs=fs, nperseg=256)
        
        # Define frequency bands
        lf_band = (0.04, 0.15)  # Low frequency band
        hf_band = (0.15, 0.4)   # High frequency band
        
        # Calculate power in each band
        lf_indices = np.logical_and(frequencies >= lf_band[0], frequencies <= lf_band[1])
        hf_indices = np.logical_and(frequencies >= hf_band[0], frequencies <= hf_band[1])
        
        lf_power = np.trapz(psd[lf_indices], frequencies[lf_indices])
        hf_power = np.trapz(psd[hf_indices], frequencies[hf_indices])
        
        # Calculate LF/HF ratio
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 1.0
    except:
        # Fallback values if spectral analysis fails
        lf_power = 0.0
        hf_power = 0.0
        lf_hf_ratio = 1.0
    
    return {
        'rmssd': rmssd,
        'sdnn': sdnn,
        'pnn50': pnn50,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_hf_ratio
    }


def extract_scl_features(scl_data: np.ndarray) -> Dict[str, float]:
    """
    Extract Skin Conductance Level features.
    
    Args:
        scl_data: Array of skin conductance levels
        
    Returns:
        Dictionary of SCL features
    """
    if len(scl_data) < 10:
        return {
            'scl_mean': 0.0,
            'scl_std': 0.0,
            'scl_min': 0.0,
            'scl_max': 0.0,
            'scl_range': 0.0,
            'scl_slope': 0.0
        }
    
    # Basic statistics
    scl_mean = np.mean(scl_data)
    scl_std = np.std(scl_data)
    scl_min = np.min(scl_data)
    scl_max = np.max(scl_data)
    scl_range = scl_max - scl_min
    
    # Trend (slope)
    x = np.arange(len(scl_data))
    slope, _ = np.polyfit(x, scl_data, 1)
    
    return {
        'scl_mean': scl_mean,
        'scl_std': scl_std,
        'scl_min': scl_min,
        'scl_max': scl_max,
        'scl_range': scl_range,
        'scl_slope': slope
    }


def calculate_stress_score(hrv_features: Dict[str, float], 
                         scl_features: Optional[Dict[str, float]] = None,
                         subjective_stress: Optional[float] = None) -> float:
    """
    Calculate stress score from physiological features.
    
    Args:
        hrv_features: Dictionary of HRV features
        scl_features: Dictionary of SCL features (optional)
        subjective_stress: Self-reported stress level (0-10, optional)
        
    Returns:
        Stress score (0-10, higher = more stress)
    """
    # HRV-based stress components (normalized to 0-1 range)
    hrv_stress = 0.0
    
    # RMSSD: lower values indicate higher stress
    if hrv_features['rmssd'] > 0:
        rmssd_norm = np.clip(1 - (hrv_features['rmssd'] / 50), 0, 1)
        hrv_stress += rmssd_norm * 0.35
    
    # SDNN: lower values indicate higher stress
    if hrv_features['sdnn'] > 0:
        sdnn_norm = np.clip(1 - (hrv_features['sdnn'] / 100), 0, 1)
        hrv_stress += sdnn_norm * 0.25
    
    # LF/HF ratio: higher values indicate higher stress
    lf_hf_norm = np.clip((hrv_features['lf_hf_ratio'] - 1) / 3, 0, 1)
    hrv_stress += lf_hf_norm * 0.4
    
    # Calculate SCL-based stress if SCL features are available
    scl_stress = 0.0
    if scl_features:
        # SCL mean: higher values indicate higher stress
        scl_mean_norm = np.clip(scl_features['scl_mean'] / 20, 0, 1)
        
        # SCL range: higher values indicate higher stress
        scl_range_norm = np.clip(scl_features['scl_range'] / 10, 0, 1)
        
        # SCL slope: positive slope indicates increasing stress
        scl_slope_norm = np.clip((scl_features['scl_slope'] + 0.1) / 0.2, 0, 1)
        
        # Combine SCL stress components
        scl_stress = (scl_mean_norm * 0.4 + scl_range_norm * 0.3 + scl_slope_norm * 0.3)
    
    # Combine physiological stress indicators
    physio_stress = hrv_stress if scl_features is None else (hrv_stress * 0.7 + scl_stress * 0.3)
    
    # Include subjective stress if available
    if subjective_stress is not None:
        subjective_norm = subjective_stress / 10.0
        final_stress = physio_stress * 0.7 + subjective_norm * 0.3
    else:
        final_stress = physio_stress
    
    # Scale to 0-10 range
    return final_stress * 10


class StressProcessor:
    """
    Class for processing stress and physiological data for migraine prediction.
    Stress is a common migraine trigger.
    """
    
    def __init__(self):
        """
        Initialize stress processor.
        """
        pass
    
    def process_hrv_data(self, hrv_data: Dict) -> Dict[str, float]:
        """
        Process HRV data to extract stress-related features.
        
        Args:
            hrv_data: Raw HRV data
            
        Returns:
            Dictionary of processed HRV features
        """
        features = {}
        
        # Extract RR intervals
        if 'rr_intervals' in hrv_data:
            rr_intervals = np.array(hrv_data['rr_intervals'])
        elif 'rr' in hrv_data:
            rr_intervals = np.array(hrv_data['rr'])
        else:
            # No RR intervals found
            return {
                'timestamp': pd.NaT,
                'rmssd': 0.0,
                'sdnn': 0.0,
                'pnn50': 0.0,
                'lf_power': 0.0,
                'hf_power': 0.0,
                'lf_hf_ratio': 1.0,
                'stress_score': 5.0  # Default middle value
            }
        
        # Calculate HRV features
        hrv_features = calculate_hrv_features(rr_intervals)
        features.update(hrv_features)
        
        # Extract SCL data if available
        scl_features = None
        if 'scl' in hrv_data and len(hrv_data['scl']) > 0:
            scl_data = np.array(hrv_data['scl'])
            scl_features = extract_scl_features(scl_data)
            features.update(scl_features)
        
        # Extract subjective stress if available
        subjective_stress = None
        if 'subjective_stress' in hrv_data:
            subjective_stress = hrv_data['subjective_stress']
            features['subjective_stress'] = subjective_stress
        
        # Calculate overall stress score
        features['stress_score'] = calculate_stress_score(hrv_features, scl_features, subjective_stress)
        
        # Ensure timestamp is included
        if 'timestamp' not in features:
             if 'timestamp' in hrv_data and isinstance(hrv_data['timestamp'], pd.Timestamp):
                  features['timestamp'] = hrv_data['timestamp']
             else:
                  # Attempt to parse if it exists but wasn't a timestamp
                  try:
                       features['timestamp'] = pd.Timestamp(hrv_data.get('timestamp'))
                  except Exception:
                       print("Warning: Could not determine timestamp for HRV record, using NaT.")
                       features['timestamp'] = pd.NaT

        return features
    
    def process_hrv_dataset(self, hrv_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataset of HRV/stress records provided as a DataFrame.

        Args:
            hrv_data_df: DataFrame containing HRV/stress data (e.g., loaded from CSV).

        Returns:
            DataFrame with processed stress features.
        """
        all_features = []
        if hrv_data_df.empty:
            return pd.DataFrame(all_features)

        print(f"Processing {len(hrv_data_df)} HRV/stress records from DataFrame...")
        hrv_records_list = hrv_data_df.to_dict('records')

        for record_dict in hrv_records_list:
            original_timestamp = None # Store the original timestamp
            try:
                # --- Check and store timestamp BEFORE processing --- 
                if 'timestamp' not in record_dict or pd.isna(record_dict['timestamp']):
                    print(f"Warning: Skipping HRV record due to missing or invalid timestamp. Record: {record_dict}")
                    continue
                
                # Attempt to parse timestamp robustly if it's a string
                if isinstance(record_dict['timestamp'], str):
                    try:
                        original_timestamp = pd.to_datetime(record_dict['timestamp'])
                    except Exception as ts_err:
                        print(f"Warning: Could not parse timestamp string '{record_dict['timestamp']}'. Skipping record. Error: {ts_err}")
                        continue
                elif isinstance(record_dict['timestamp'], (pd.Timestamp, datetime)):
                    original_timestamp = pd.Timestamp(record_dict['timestamp']) # Ensure it's a pandas Timestamp
                else:
                     print(f"Warning: Skipping HRV record due to unexpected timestamp type: {type(record_dict['timestamp'])}. Record: {record_dict}")
                     continue
                
                # Check if timestamp parsing resulted in NaT
                if pd.isna(original_timestamp):
                    print(f"Warning: Timestamp parsed as NaT. Skipping record. Original value: {record_dict['timestamp']}")
                    continue
                # ----------------------------------------------------

                # RR intervals might be stored as strings in CSV, need to convert
                # Assuming they are stored as space-separated numbers or similar
                # This part is highly dependent on the actual CSV format
                if 'rr_intervals' in record_dict and isinstance(record_dict['rr_intervals'], str):
                     try:
                          # Example: Convert space-separated string to list of floats
                          rr_list = [float(x) for x in record_dict['rr_intervals'].split()] 
                          record_dict['rr_intervals'] = rr_list
                     except ValueError:
                          print(f"Warning: Could not parse rr_intervals string: {record_dict['rr_intervals']}. Skipping record.")
                          continue 
                # Add similar parsing for 'scl' if needed

                processed_features = self.process_hrv_data(record_dict)
                
                # --- Ensure timestamp is in the processed features --- 
                if processed_features is not None:
                    processed_features['timestamp'] = original_timestamp # Use the validated timestamp
                    all_features.append(processed_features)
                # ----------------------------------------------------

            except Exception as e:
                print(f"Warning: Skipping HRV record due to processing error: {e}. Record: {record_dict}")
                continue

        # Convert list of feature dictionaries to DataFrame
        if not all_features:
            return pd.DataFrame(all_features)

        stress_df = pd.DataFrame(all_features)

        # Convert timestamp column to datetime and set as index
        if 'timestamp' in stress_df.columns:
             # We should have only valid timestamps now, but keep the check just in case
             initial_len = len(stress_df)
             # Use errors='coerce' for final safety, though NaNs shouldn't occur now
             stress_df['timestamp'] = pd.to_datetime(stress_df['timestamp'], errors='coerce') 
             stress_df = stress_df.dropna(subset=['timestamp']) # Drop if coercion failed
             if len(stress_df) < initial_len:
                  dropped_count = initial_len - len(stress_df)
                  print(f"Warning: Dropped {dropped_count} rows from stress data due to timestamp conversion failure AFTER processing.")
             
             # No need for further try-except if dropna handles coercion errors
             stress_df = stress_df.set_index('timestamp')
             stress_df = stress_df.sort_index() # Ensure index is sorted
             
        else:
             print("Warning: 'timestamp' column missing after stress processing.")

        # --- Also drop non-numeric columns (like patient_id, source if they exist) --- 
        cols_to_drop = []
        for col in stress_df.columns:
            if col == stress_df.index.name: # Skip index
                continue
            try:
                # Attempt conversion, no need to store result, just check type
                pd.to_numeric(stress_df[col]) 
            except (ValueError, TypeError):
                # Identify non-numeric columns to drop
                # We will explicitly drop known string cols later if they exist
                if col not in ['patient_id', 'source']: # Avoid double warning
                    print(f"Warning: Column '{col}' is non-numeric in StressProcessor. Adding to drop list.")
                cols_to_drop.append(col)
        
        # Explicitly add known non-numeric columns if they haven't been added already
        if 'patient_id' in stress_df.columns and 'patient_id' not in cols_to_drop:
            cols_to_drop.append('patient_id')
        if 'source' in stress_df.columns and 'source' not in cols_to_drop:
            cols_to_drop.append('source')
            
        # Drop all identified non-numeric columns
        if cols_to_drop:
            # Ensure we don't try to drop columns that don't exist
            cols_to_drop = [col for col in cols_to_drop if col in stress_df.columns]
            if cols_to_drop:
                 stress_df = stress_df.drop(columns=cols_to_drop)
                 print(f"Dropped non-numeric columns from Stress data: {cols_to_drop}")
        # --- End check --- 

        return stress_df
    
    def get_migraine_risk_from_stress(self, stress_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate migraine risk score based on stress patterns.
        
        Args:
            stress_data: DataFrame with processed stress data
            
        Returns:
            DataFrame with original data plus migraine risk scores
        """
        if stress_data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        risk_df = stress_data.copy()
        
        # Calculate risk score components
        
        # 1. Current stress level
        current_stress_risk = np.clip(risk_df['stress_score'] / 10, 0, 1) * 3
        
        # 2. Stress volatility (changes in stress level)
        if 'stress_change' in risk_df.columns:
            stress_volatility_risk = np.clip(np.abs(risk_df['stress_change']) / 3, 0, 1) * 2
        else:
            stress_volatility_risk = 0
        
        # 3. Sustained high stress
        if 'sustained_high_stress' in risk_df.columns:
            sustained_stress_risk = risk_df['sustained_high_stress'] * 3
        else:
            sustained_stress_risk = np.clip((risk_df['stress_score'] - 7) / 3, 0, 1) * 3
        
        # 4. HRV-specific indicators
        hrv_risk = np.clip(1 - (risk_df['rmssd'] / 50), 0, 1) * 2
        
        # Combine into overall stress-based migraine risk score (0-10 scale)
        risk_df['stress_migraine_risk'] = np.clip(
            current_stress_risk + stress_volatility_risk + sustained_stress_risk + hrv_risk,
            0, 10
        )
        
        return risk_df
    
    def align_stress_with_migraine_events(self, 
                                         stress_data: pd.DataFrame,
                                         migraine_events: List[Dict]) -> pd.DataFrame:
        """
        Align stress data features with migraine events.
        
        Args:
            stress_data: DataFrame with processed stress data
            migraine_events: List of dictionaries with migraine events
                             (must contain 'start_time' and 'severity' keys)
            
        Returns:
            DataFrame with stress data aligned to migraine events
        """
        if stress_data.empty or not migraine_events:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        aligned_df = stress_data.copy()
        
        # Add a 'hours_to_next_migraine' column
        aligned_df['hours_to_next_migraine'] = np.inf
        aligned_df['next_migraine_severity'] = None

        # Ensure stress_data index is datetime
        if not isinstance(aligned_df.index, pd.DatetimeIndex):
             # Try converting the index if it's not already datetime
             try:
                  aligned_df.index = pd.to_datetime(aligned_df.index)
                  if not isinstance(aligned_df.index, pd.DatetimeIndex):
                       raise ValueError("Index conversion failed") # Re-raise if conversion didn't work
             except Exception as e:
                  print(f"Warning: Stress data index is not DatetimeIndex and conversion failed: {e}. Cannot align.")
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

        # Iterate through stress data
        for stress_idx_time in aligned_df.index:
            # Find the next migraine event *after* this stress record time
            next_events = [e for e in valid_events if e['start_time'] > stress_idx_time]

            if next_events:
                next_event = next_events[0] # The first event after the stress record
                time_diff = next_event['start_time'] - stress_idx_time
                hours_diff = time_diff.total_seconds() / 3600

                # Update the row in stress_data
                aligned_df.loc[stress_idx_time, 'hours_to_next_migraine'] = hours_diff
                aligned_df.loc[stress_idx_time, 'next_migraine_severity'] = next_event.get('severity')

        # Create binary label for prediction (migraine within next 24 hours)
        aligned_df['migraine_within_24h'] = aligned_df['hours_to_next_migraine'] <= 24
        
        return aligned_df 