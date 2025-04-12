#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert the EEG CSV data to .npy format expected by the pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert EEG CSV data to NPY")
    
    parser.add_argument('--input_dir', type=str, default='../data/migraine/eeg',
                      help='Input directory with EEG CSV files')
    parser.add_argument('--output_dir', type=str, default='../data/migraine/eeg',
                      help='Output directory for NPY files')
    
    return parser.parse_args()

def convert_csv_to_npy(input_dir, output_dir):
    """Convert the EEG CSV data to .npy format."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('_eeg.csv')]
    
    if not csv_files:
        print(f"No EEG CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} EEG CSV files")
    
    # Process each file
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(os.path.join(input_dir, csv_file))
            print(f"Processing {csv_file} with {len(df)} records")
            
            # Process each timestamp
            for _, group in df.groupby('timestamp'):
                timestamp = pd.to_datetime(group['timestamp'].iloc[0])
                timestamp_str = timestamp.strftime('%Y-%m-%d_%H-%M-%S')
                
                # Create feature array
                feature_array = np.array([
                    group['alpha'].values,
                    group['beta'].values,
                    group['theta'].values,
                    group['delta'].values,
                    group['gamma'].values,
                    group['frontal_asymmetry'].values
                ])
                
                # Save as NPY file
                npy_filename = f"eeg_{timestamp_str}.npy"
                npy_path = os.path.join(output_dir, npy_filename)
                np.save(npy_path, feature_array)
                print(f"Saved {npy_path}")
        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    print(f"EEG data conversion complete.")

def main():
    """Main function."""
    args = parse_args()
    convert_csv_to_npy(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 