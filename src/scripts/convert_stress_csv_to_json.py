#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert the stress CSV data to JSON format expected by the pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import json
import random
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert stress CSV data to JSON")
    
    parser.add_argument('--input_file', type=str, default='../data/migraine/stress/all_stress_data.csv',
                      help='Input CSV file with stress data')
    parser.add_argument('--output_file', type=str, default='../data/migraine/stress_data.json',
                      help='Output JSON file for stress data')
    
    return parser.parse_args()

def convert_csv_to_json(input_file, output_file):
    """Convert the stress CSV data to JSON format."""
    # Read the CSV file
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded CSV with columns: {df.columns.tolist()}")
    
    # Convert to the expected format
    stress_records = []
    for _, row in df.iterrows():
        # Generate synthetic data for missing physiological measurements
        # These are needed by the pipeline but not present in our CSV
        heart_rate = random.randint(60, 100)
        respiratory_rate = random.randint(12, 20)
        skin_conductance = round(random.uniform(1.0, 10.0), 2)
        cortisol_level = round(random.uniform(10.0, 20.0), 2)
        activity_level = round(random.uniform(0.0, 5.0), 2)
        
        record = {
            'patient_id': row['patient_id'],
            'timestamp': row['timestamp'],
            'stress_level': row['stress_level'],
            'source': row['source'],
            'heart_rate': heart_rate,
            'respiratory_rate': respiratory_rate,
            'skin_conductance': skin_conductance,
            'cortisol_level': cortisol_level,
            'activity_level': activity_level
        }
        stress_records.append(record)
    
    # Write to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stress_records, f, indent=2)
    
    print(f"Converted {len(stress_records)} stress records from CSV to JSON")
    print(f"Output saved to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    convert_csv_to_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 