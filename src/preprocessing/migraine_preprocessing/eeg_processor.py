import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from scipy import signal
from scipy.stats import entropy


def extract_band_power(eeg_data: np.ndarray, sampling_rate: int, 
                      low_freq: float, high_freq: float) -> np.ndarray:
    """
    Extract power in specific frequency band from EEG signal.
    
    Args:
        eeg_data: Raw EEG signal data
        sampling_rate: Sampling rate of EEG in Hz
        low_freq: Lower bound of frequency band in Hz
        high_freq: Upper bound of frequency band in Hz
        
    Returns:
        Band power values
    """
    # Define window length (2 seconds)
    win_len = 2 * sampling_rate
    
    # Apply Welch's method to estimate power spectral density
    freqs, psd = signal.welch(eeg_data, fs=sampling_rate, nperseg=win_len)
    
    # Find indices of frequencies within the band
    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    
    # Calculate band power
    band_power = np.sum(psd[idx_band])
    
    return band_power


def extract_eeg_features(eeg_data: np.ndarray, sampling_rate: int = 256,
                        window_size: int = 30) -> Dict[str, float]:
    """
    Process raw EEG data and extract features for migraine prediction.
    
    Args:
        eeg_data: Raw EEG signal data
        sampling_rate: Sampling rate of EEG in Hz
        window_size: Window size in seconds
        
    Returns:
        Dictionary of EEG features
    """
    features = {}
    
    # Extract frequency domain features (power bands)
    features['delta_power'] = extract_band_power(eeg_data, sampling_rate, 0.5, 4)
    features['theta_power'] = extract_band_power(eeg_data, sampling_rate, 4, 8)
    features['alpha_power'] = extract_band_power(eeg_data, sampling_rate, 8, 13)
    features['beta_power'] = extract_band_power(eeg_data, sampling_rate, 13, 30)
    features['gamma_power'] = extract_band_power(eeg_data, sampling_rate, 30, 100)
    
    # Calculate alpha/delta ratio (commonly used in migraine studies)
    if features['delta_power'] != 0:
        features['alpha_delta_ratio'] = features['alpha_power'] / features['delta_power']
    else:
        features['alpha_delta_ratio'] = 0
    
    # Extract time domain features
    features['signal_mean'] = np.mean(eeg_data)
    features['signal_std'] = np.std(eeg_data)
    
    # Calculate complexity measures
    features['signal_entropy'] = calculate_sample_entropy(eeg_data)
    
    # Calculate peak frequency
    freqs, psd = signal.welch(eeg_data, fs=sampling_rate, nperseg=window_size*sampling_rate)
    features['peak_frequency'] = freqs[np.argmax(psd)]
    
    return features


def calculate_sample_entropy(signal_data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate sample entropy for the given signal.
    
    Args:
        signal_data: Input signal
        m: Embedding dimension
        r: Tolerance
        
    Returns:
        Sample entropy value
    """
    # Normalize the signal
    signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    # If signal is too short, return a default value
    if len(signal_data) < 100:
        return 0.0
    
    # Calculate sample entropy using scipy's entropy function as an approximation
    # For a more accurate implementation, consider using specialized libraries
    hist, _ = np.histogram(signal_data, bins=20)
    prob = hist / np.sum(hist)
    return entropy(prob)


def process_eeg_data(eeg_files: List[str], sampling_rate: int = 256) -> pd.DataFrame:
    """
    Process multiple EEG files and extract features for migraine prediction.
    
    Args:
        eeg_files: List of file paths to EEG data
        sampling_rate: Sampling rate of EEG in Hz
        
    Returns:
        DataFrame with extracted EEG features
    """
    all_features = []
    
    for file in eeg_files:
        # Load EEG data
        # Note: The actual loading will depend on your file format
        try:
            eeg_data = np.load(file)  # Assuming numpy format
            # Process each channel separately
            for channel_idx, channel_data in enumerate(eeg_data.T):
                # Extract features
                features = extract_eeg_features(channel_data, sampling_rate)
                
                # Add metadata
                features['file'] = file
                features['channel'] = channel_idx
                features['timestamp'] = extract_timestamp_from_filename(file)
                
                all_features.append(features)
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    
    # Convert to DataFrame
    eeg_features_df = pd.DataFrame(all_features)
    
    return eeg_features_df


def extract_timestamp_from_filename(filename: str) -> pd.Timestamp:
    """
    Extract timestamp from EEG filename.
    
    Args:
        filename: EEG file name
        
    Returns:
        Timestamp object
    """
    # This function should be adapted to your filename format
    # Example: "eeg_2023-05-15_14-30-00.npy" -> 2023-05-15 14:30:00
    try:
        # Extract date and time from filename
        date_str = filename.split('_')[1]
        time_str = filename.split('_')[2].split('.')[0].replace('-', ':')
        timestamp_str = f"{date_str} {time_str}"
        
        # Convert to timestamp
        return pd.Timestamp(timestamp_str)
    except:
        # Return current time if parsing fails
        return pd.Timestamp.now()


class EEGProcessor:
    """
    Class for processing EEG data for migraine prediction.
    """
    
    def __init__(self, sampling_rate: int = 256, window_size: int = 30):
        """
        Initialize EEG processor.
        
        Args:
            sampling_rate: Sampling rate of EEG in Hz
            window_size: Window size in seconds
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
    
    def process_dataset(self, eeg_data_input: Union[List[str], pd.DataFrame]) -> pd.DataFrame:
        """
        Process a dataset of EEG data, accepting either file paths or a pre-loaded DataFrame.

        Args:
            eeg_data_input: Either a list of file paths to EEG data (.npy assumed)
                              or a pandas DataFrame containing the combined EEG data (e.g., from CSV).

        Returns:
            DataFrame with processed EEG features.
        """
        if isinstance(eeg_data_input, pd.DataFrame):
            # Input is already a DataFrame (e.g., loaded from all_eeg_data.csv)
            print("Processing EEG data from DataFrame...")
            eeg_df = eeg_data_input.copy()

            # Ensure timestamp column is datetime objects
            if 'timestamp' in eeg_df.columns:
                try:
                    eeg_df['timestamp'] = pd.to_datetime(eeg_df['timestamp'])
                    # Set timestamp as index if not already
                    if not isinstance(eeg_df.index, pd.DatetimeIndex):
                         eeg_df = eeg_df.set_index('timestamp')
                except Exception as e:
                    print(f"Warning: Could not parse EEG timestamps: {e}")
            else:
                 print("Warning: 'timestamp' column not found in EEG DataFrame.")

            # Data generation script already calculates features (alpha, beta, etc.)
            # So, the main processing here is ensuring format and selecting columns.
            # Define expected feature columns based on generation script
            expected_features = ['alpha', 'beta', 'theta', 'delta', 'gamma', 'frontal_asymmetry']
            if 'patient_id' in eeg_df.columns:
                 expected_features.insert(0, 'patient_id') # Keep patient ID if present

            # Check if expected columns exist
            missing_cols = [col for col in expected_features if col not in eeg_df.columns and col != 'patient_id'] # Don't require patient_id
            if missing_cols:
                 print(f"Warning: Missing expected EEG columns: {missing_cols}")

            # Select relevant columns (if they exist)
            available_cols = [col for col in expected_features if col in eeg_df.columns]
            if not available_cols:
                 print("Error: No relevant EEG feature columns found in DataFrame.")
                 return pd.DataFrame()

            processed_df = eeg_df[available_cols]

            # Ensure numeric types and drop non-numeric identifiers like patient_id
            cols_to_drop = []
            for col in processed_df.columns:
                 if col == 'patient_id': # Check for patient_id
                     cols_to_drop.append(col)
                 else:
                     # Attempt conversion for other columns
                     processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                     # Check if conversion failed (column is still object type)
                     if processed_df[col].dtype == 'object':
                          print(f"Warning: Column '{col}' could not be converted to numeric in EEGProcessor. Dropping.")
                          cols_to_drop.append(col)
            
            # Drop identified non-numeric columns
            if cols_to_drop:
                processed_df = processed_df.drop(columns=cols_to_drop)
                print(f"Dropped non-numeric columns from EEG data: {cols_to_drop}")

            print(f"EEG DataFrame processing complete. Shape: {processed_df.shape}")
            return processed_df

        elif isinstance(eeg_data_input, list):
            # Original logic: process list of file paths (assuming .npy)
            print("Processing EEG data from list of files...")
            return process_eeg_data(eeg_data_input, self.sampling_rate)
        else:
            raise TypeError("Input must be a list of file paths or a pandas DataFrame")
    
    def process_continuous_eeg(self, eeg_data: np.ndarray, 
                              window_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process continuous EEG data with sliding windows.
        
        Args:
            eeg_data: Continuous EEG data
            window_size: Window size in seconds (default: self.window_size)
            
        Returns:
            DataFrame with extracted EEG features for each window
        """
        if window_size is None:
            window_size = self.window_size
            
        window_samples = window_size * self.sampling_rate
        step_size = window_samples // 2  # 50% overlap
        
        all_features = []
        
        # Process each channel separately
        for channel_idx, channel_data in enumerate(eeg_data.T):
            # Sliding window processing
            for start in range(0, len(channel_data) - window_samples, step_size):
                window = channel_data[start:start + window_samples]
                
                # Extract features
                features = extract_eeg_features(window, self.sampling_rate, window_size)
                
                # Add metadata
                features['channel'] = channel_idx
                features['window_start'] = start
                features['window_end'] = start + window_samples
                
                all_features.append(features)
        
        # Convert to DataFrame
        eeg_features_df = pd.DataFrame(all_features)
        
        return eeg_features_df 