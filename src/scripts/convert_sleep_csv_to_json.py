#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert the sleep CSV data to JSON format expected by the pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert sleep CSV data to JSON")
    
    parser.add_argument('--input_file', type=str, default='../data/migraine/sleep/all_sleep_data.csv',
                      help='Input CSV file with sleep data')
    parser.add_argument('--output_file', type=str, default='../data/migraine/sleep_data.json',
                      help='Output JSON file for sleep data')
    
    return parser.parse_args()

def convert_csv_to_json(input_file, output_file):
    """Convert the sleep CSV data to JSON format."""
    # Read the CSV file
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded CSV with columns: {df.columns.tolist()}")
    
    # Convert to the expected format
    sleep_records = []
    for _, row in df.iterrows():
        record = {
            'patient_id': row['patient_id'],
            'date': row['date'],
            'sleep_start': row['sleep_start'],
            'sleep_end': row['sleep_end'],
            'duration_hours': row['duration'],  # Using 'duration' instead of 'duration_hours'
            'quality': row['efficiency'],  # Using 'efficiency' as quality metric
            'deep_sleep_percentage': row['deep_sleep_percentage'],
            'rem_sleep_percentage': row['rem_sleep_percentage'],
            'light_sleep_percentage': row['light_sleep_percentage'],
            'awake_percentage': 100 - (row['deep_sleep_percentage'] + row['rem_sleep_percentage'] + row['light_sleep_percentage']),  # Calculate awake percentage
            'interruptions': row['awakenings']  # Using 'awakenings' instead of 'interruptions'
        }
        sleep_records.append(record)
    
    # Write to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(sleep_records, f, indent=2)
    
    print(f"Converted {len(sleep_records)} sleep records from CSV to JSON")
    print(f"Output saved to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    convert_csv_to_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 