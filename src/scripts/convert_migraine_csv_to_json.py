#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert the migraine CSV data to JSON format expected by the pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert migraine CSV data to JSON")
    
    parser.add_argument('--input_file', type=str, default='../data/migraine/migraines/all_migraine_data.csv',
                      help='Input CSV file with migraine data')
    parser.add_argument('--output_file', type=str, default='../data/migraine/migraine_events.json',
                      help='Output JSON file for migraine events')
    
    return parser.parse_args()

def convert_csv_to_json(input_file, output_file):
    """Convert the migraine CSV data to JSON format."""
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded CSV with columns: {df.columns.tolist()}")
    
    # Convert to the expected format
    events = []
    for _, row in df.iterrows():
        event = {
            'patient_id': row['patient_id'],
            'date': row['date'],  # Include the date field
            'timestamp': row['start_time'],
            'severity': row['severity'],
            'duration_hours': row['duration_hours'],
            'aura': row['aura'],
            'nausea': row['nausea'],
            'photophobia': row['photophobia'],
            'phonophobia': row['phonophobia'],
            'triggers': row['triggers'].split(',') if isinstance(row['triggers'], str) else []
        }
        events.append(event)
    
    # Write to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"Converted {len(events)} migraine events from CSV to JSON")
    print(f"Output saved to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    convert_csv_to_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 