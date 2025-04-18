#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate synthetic migraine data for testing the FuseMOE migraine prediction pipeline.

This script creates synthetic data for:
1. EEG measurements
2. Weather data
3. Sleep tracking data
4. Stress level data
5. Migraine occurrence data

The data is structured to be compatible with the FuseMOE migraine prediction pipeline.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic migraine data")
    
    parser.add_argument('--output_dir', type=str, default='./data/migraine',
                      help='Directory to save the generated data')
    parser.add_argument('--num_patients', type=int, default=50,
                      help='Number of patients to generate')
    parser.add_argument('--days', type=int, default=30,
                      help='Number of days of data per patient')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--avg_migraine_freq', type=float, default=0.15,
                      help='Average daily probability of a migraine event (controls positive rate)')
    
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create the necessary directory structure for the dataset."""
    directories = [
        os.path.join(output_dir, 'eeg'),
        os.path.join(output_dir, 'weather'),
        os.path.join(output_dir, 'sleep'),
        os.path.join(output_dir, 'stress'),
        os.path.join(output_dir, 'migraines')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

def generate_patient_ids(num_patients):
    """Generate unique patient IDs."""
    return [f"P{str(i).zfill(3)}" for i in range(1, num_patients + 1)]

def generate_date_range(start_date, days):
    """Generate a range of dates."""
    return [start_date + timedelta(days=i) for i in range(days)]

def generate_eeg_data(patient_ids, date_range, output_dir, migraine_dates):
    """Generate synthetic EEG data with unique timestamps per patient."""
    eeg_dir = os.path.join(output_dir, 'eeg')
    all_patient_data = []
    
    # Define base frequencies for EEG bands
    base_freqs = {'alpha': 10, 'beta': 20, 'theta': 6, 'delta': 2, 'gamma': 40}
    
    for p_idx, patient_id in enumerate(patient_ids):
        patient_data = []
        
        # Patient-specific baseline to simulate individual differences
        patient_variation = np.random.uniform(0.8, 1.2, size=len(base_freqs))
        
        for date in date_range:
            # Check if date is close to a migraine (prodromal phase)
            date_str = date.strftime('%Y-%m-%d')
            is_migraine_day = False
            is_prodromal = False
            
            for migraine_date in migraine_dates.get(patient_id, []):
                migraine_date_obj = datetime.strptime(migraine_date, '%Y-%m-%d')
                days_before = (migraine_date_obj - date).days
                if days_before == 0:
                    is_migraine_day = True
                    break
                elif 1 <= days_before <= 2:  # 1-2 days before migraine
                    is_prodromal = True
                    break
            
            # Generate multiple EEG readings per day (e.g., every 4 hours)
            for hour in range(0, 24, 4): # Example: every 4 hours
                # --- Add unique timestamp offset per patient ---
                # Base timestamp
                base_timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
                # Add a small, *random* offset (e.g., milliseconds) to interleave patients
                offset_ms = np.random.randint(0, 1000) # Random offset 0-999 ms
                timestamp = base_timestamp + timedelta(milliseconds=offset_ms)
                # --- ---------------------------------------- ---
                
                # Simulate EEG band power variations
                eeg_bands = {}
                i = 0
                for band, base_freq in base_freqs.items():
                    # Apply variations: patient-specific, daily cycle, random noise
                    daily_cycle = np.sin(np.pi * hour / 24) * 0.1  # Small daily variation
                    random_noise = np.random.normal(0, 0.05)  # Random noise
                    
                    # Modify amplitude based on migraine proximity
                    migraine_effect = 1.0
                    if is_migraine_day:
                        migraine_effect = np.random.uniform(1.1, 1.4) if band in ['theta', 'delta'] else np.random.uniform(0.7, 0.9)
                    elif is_prodromal:
                        migraine_effect = np.random.uniform(1.05, 1.2) if band in ['theta', 'delta'] else np.random.uniform(0.8, 0.95)
                    
                    band_power = base_freq * patient_variation[i] * (1 + daily_cycle + random_noise) * migraine_effect
                    eeg_bands[band] = max(0, band_power) # Ensure non-negative
                    i += 1
                
                # Simulate frontal asymmetry (simple random walk)
                frontal_asymmetry = np.random.normal(0, 0.2) # Baseline around 0
                if is_migraine_day or is_prodromal:
                     frontal_asymmetry += np.random.normal(0.3, 0.1) # Shift during migraine phase
                
                patient_data.append({
                    'patient_id': patient_id,
                    'timestamp': timestamp, # Use the unique timestamp
                    **eeg_bands,
                    'frontal_asymmetry': frontal_asymmetry
                })
                
        # Create dataframe and save to CSV
        df = pd.DataFrame(patient_data)
        df.to_csv(os.path.join(eeg_dir, f"{patient_id}_eeg.csv"), index=False)
        all_patient_data.extend(patient_data)
    
    # Create a combined CSV with all patients
    all_df = pd.DataFrame(all_patient_data)
    all_df.to_csv(os.path.join(eeg_dir, "all_eeg_data.csv"), index=False)
    
    print(f"Generated EEG data: {len(all_patient_data)} records")
    return all_df

def generate_weather_data(date_range, output_dir, location=(40.7128, -74.0060)):
    """Generate synthetic weather data for a given location."""
    weather_dir = os.path.join(output_dir, 'weather')
    weather_data = []
    
    # Weather patterns that may trigger migraines
    # Baseline weather with some day-to-day variations
    baseline_temp = np.random.uniform(60, 75)  # Baseline temperature in F
    baseline_humidity = np.random.uniform(40, 60)  # Baseline humidity percentage
    baseline_pressure = np.random.uniform(1010, 1015)  # Baseline pressure in hPa
    
    # Create a small weather "front" that passes through
    front_start = len(date_range) // 3
    front_end = front_start + 5  # 5-day weather front
    
    for i, date in enumerate(date_range):
        date_str = date.strftime('%Y-%m-%d')
        
        # Generate hourly data
        for hour in range(24):
            timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            
            # Calculate daily and hourly variations
            daily_variation = np.sin(np.pi * i / len(date_range)) * 10  # Temperature variation over the month
            hourly_variation = np.sin(np.pi * hour / 12) * 5  # Temperature variation during the day
            
            # Weather front effects
            front_effect = 0
            if front_start <= i < front_end:
                front_progress = (i - front_start) / (front_end - front_start)
                front_effect = np.sin(np.pi * front_progress) * 15  # Stronger effect in the middle of the front
                pressure_drop = -np.sin(np.pi * front_progress) * 10  # Pressure drops during the front
            else:
                pressure_drop = 0
            
            # Calculate weather parameters with some randomness
            temperature = baseline_temp + daily_variation + hourly_variation + front_effect + np.random.uniform(-2, 2)
            humidity = baseline_humidity + (front_effect / 2) + np.random.uniform(-5, 5)
            pressure = baseline_pressure + pressure_drop + np.random.uniform(-1, 1)
            
            # Random precipitation (more likely during the front)
            precipitation = 0
            if front_start <= i < front_end:
                precipitation = max(0, np.random.uniform(-0.1, 0.3))
            elif np.random.random() < 0.1:  # 10% chance of rain on other days
                precipitation = max(0, np.random.uniform(-0.05, 0.2))
            
            # Wind speed tends to be higher during weather fronts
            if front_start <= i < front_end:
                wind_speed = np.random.uniform(5, 15)
            else:
                wind_speed = np.random.uniform(0, 10)
            
            weather_data.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'precipitation': precipitation,
                'wind_speed': wind_speed,
                'latitude': location[0],
                'longitude': location[1]
            })
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(weather_data)
    df.to_csv(os.path.join(weather_dir, "weather_data.csv"), index=False)
    
    print(f"Generated weather data: {len(weather_data)} records")
    return df

def generate_sleep_data(patient_ids, date_range, output_dir, migraine_dates):
    """Generate synthetic sleep tracking data."""
    sleep_dir = os.path.join(output_dir, 'sleep')
    all_patient_data = []
    
    for patient_id in patient_ids:
        patient_data = []
        
        # Patient-specific baseline to simulate individual differences
        baseline_duration = np.random.uniform(6.5, 8.5)  # Baseline sleep duration in hours
        baseline_efficiency = np.random.uniform(0.85, 0.95)  # Baseline sleep efficiency
        baseline_deep_pct = np.random.uniform(0.15, 0.25)  # Baseline deep sleep percentage
        baseline_rem_pct = np.random.uniform(0.20, 0.30)  # Baseline REM sleep percentage
        
        for date in date_range:
            # Check if date is close to a migraine (prodromal phase)
            date_str = date.strftime('%Y-%m-%d')
            is_prodromal = False
            for migraine_date in migraine_dates.get(patient_id, []):
                migraine_date_obj = datetime.strptime(migraine_date, '%Y-%m-%d')
                days_before = (migraine_date_obj - date).days
                if 0 <= days_before <= 2:  # 0-2 days before migraine
                    is_prodromal = True
                    break
            
            # Sleep start time (typically evening of the previous day)
            sleep_start = datetime.combine(date - timedelta(days=1), datetime.min.time()) + timedelta(hours=np.random.uniform(21, 24))
            
            # Sleep metrics with variations
            if is_prodromal:
                # Changes that might indicate upcoming migraine
                duration = baseline_duration * np.random.uniform(0.7, 0.9)  # Shorter sleep
                efficiency = baseline_efficiency * np.random.uniform(0.8, 0.9)  # Lower efficiency
                deep_pct = baseline_deep_pct * np.random.uniform(0.7, 0.9)  # Less deep sleep
                rem_pct = baseline_rem_pct * np.random.uniform(0.8, 1.0)  # Slightly less REM
                awakenings = int(np.random.uniform(2, 6))  # More awakenings
            else:
                # Normal variation
                duration = baseline_duration * np.random.uniform(0.9, 1.1)
                efficiency = baseline_efficiency * np.random.uniform(0.95, 1.05)
                deep_pct = baseline_deep_pct * np.random.uniform(0.9, 1.1)
                rem_pct = baseline_rem_pct * np.random.uniform(0.9, 1.1)
                awakenings = int(np.random.uniform(0, 3))
            
            # Calculate end time and actual sleep time
            sleep_end = sleep_start + timedelta(hours=duration)
            actual_sleep = duration * efficiency
            
            patient_data.append({
                'patient_id': patient_id,
                'date': date,
                'sleep_start': sleep_start,
                'sleep_end': sleep_end,
                'duration': duration,
                'actual_sleep': actual_sleep,
                'efficiency': efficiency,
                'deep_sleep_percentage': deep_pct,
                'rem_sleep_percentage': rem_pct,
                'light_sleep_percentage': 1 - deep_pct - rem_pct,
                'awakenings': awakenings
            })
        
        # Create dataframe and save to CSV
        df = pd.DataFrame(patient_data)
        df.to_csv(os.path.join(sleep_dir, f"{patient_id}_sleep.csv"), index=False)
        all_patient_data.extend(patient_data)
    
    # Create a combined CSV with all patients
    all_df = pd.DataFrame(all_patient_data)
    all_df.to_csv(os.path.join(sleep_dir, "all_sleep_data.csv"), index=False)
    
    print(f"Generated sleep data: {len(all_patient_data)} records")
    return all_df

def generate_stress_data(patient_ids, date_range, output_dir, migraine_dates):
    """Generate synthetic stress level data."""
    stress_dir = os.path.join(output_dir, 'stress')
    all_patient_data = []
    
    for patient_id in patient_ids:
        patient_data = []
        
        # Patient-specific baseline to simulate individual differences
        baseline_stress = np.random.uniform(3, 5)  # Baseline stress on 1-10 scale
        stress_volatility = np.random.uniform(0.5, 2.0)  # How much stress varies for this patient
        
        for date in date_range:
            # Check if date is close to a migraine (prodromal phase)
            date_str = date.strftime('%Y-%m-%d')
            is_migraine_day = False
            is_prodromal = False
            
            for migraine_date in migraine_dates.get(patient_id, []):
                migraine_date_obj = datetime.strptime(migraine_date, '%Y-%m-%d')
                days_before = (migraine_date_obj - date).days
                if days_before == 0:
                    is_migraine_day = True
                    break
                elif 1 <= days_before <= 2:  # 1-2 days before migraine
                    is_prodromal = True
                    break
            
            # Generate multiple stress readings per day
            for hour in range(8, 22, 2):  # Every 2 hours from 8 AM to 8 PM
                timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
                
                # Base stress level with random variation and time-of-day effect
                time_effect = np.sin(np.pi * (hour - 8) / 14) * 1.5  # Peaks in the middle of the day
                
                if is_migraine_day:
                    # Higher stress during migraine
                    stress_level = baseline_stress * np.random.uniform(1.3, 1.8) + time_effect
                elif is_prodromal:
                    # Increasing stress before migraine
                    stress_level = baseline_stress * np.random.uniform(1.1, 1.4) + time_effect
                else:
                    # Normal daily variation
                    stress_level = baseline_stress + np.random.uniform(-1, 1) * stress_volatility + time_effect
                
                # Ensure stress is in valid range (1-10)
                stress_level = max(1, min(10, stress_level))
                
                patient_data.append({
                    'patient_id': patient_id,
                    'timestamp': timestamp,
                    'stress_level': stress_level,
                    'source': np.random.choice(['self_report', 'wearable', 'app'], p=[0.3, 0.4, 0.3])
                })
        
        # Create dataframe and save to CSV
        df = pd.DataFrame(patient_data)
        df.to_csv(os.path.join(stress_dir, f"{patient_id}_stress.csv"), index=False)
        all_patient_data.extend(patient_data)
    
    # Create a combined CSV with all patients
    all_df = pd.DataFrame(all_patient_data)
    all_df.to_csv(os.path.join(stress_dir, "all_stress_data.csv"), index=False)
    
    print(f"Generated stress data: {len(all_patient_data)} records")
    return all_df

def generate_migraine_events(patient_ids, date_range, output_dir, avg_migraine_frequency=0.15):
    """Generate synthetic migraine event data with controllable positive rate."""
    migraine_dir = os.path.join(output_dir, 'migraines')
    all_patient_data = []
    migraine_dates = {}
    
    # Set target migraine rate (days with migraine / total days across all patients)
    # Changed from 0.5-2.0 migraines per month to a more balanced distribution
    # For 500 patients over 60 days, we want about 20-40% migraine days
    target_positive_rate = avg_migraine_frequency
    print(f"Target migraine positive rate: {target_positive_rate:.2%}")
    
    total_days = len(patient_ids) * len(date_range)
    target_migraine_count = int(total_days * target_positive_rate)
    
    # Calculate average migraines per patient to achieve target distribution
    avg_migraines_per_patient = target_migraine_count / len(patient_ids)
    print(f"Aiming for approximately {avg_migraines_per_patient:.1f} migraine events per patient")
    
    # MODIFIED: Limit the variability between patients to prevent skewed distributions
    patient_migraine_counts = {}
    remaining_migraines = target_migraine_count
    
    # Assign a base number of migraines to each patient first (more evenly distributed)
    min_migraines = max(1, int(avg_migraines_per_patient * 0.5))  # At least 1, or half the average
    total_min_migraines = min_migraines * len(patient_ids)
    
    # Ensure we don't assign more than target
    if total_min_migraines > target_migraine_count:
        min_migraines = max(1, int(target_migraine_count / len(patient_ids)))
        total_min_migraines = min_migraines * len(patient_ids)
    
    # First pass: assign minimum migraines to each patient
    for patient_id in patient_ids:
        patient_migraine_counts[patient_id] = min_migraines
    remaining_migraines -= total_min_migraines
    
    # Second pass: distribute remaining migraines with controlled randomness
    if remaining_migraines > 0:
        # Calculate maximum additional migraines per patient to prevent extremes
        max_additional = min(
            int(avg_migraines_per_patient * 1.5) - min_migraines,
            int(remaining_migraines / (len(patient_ids) * 0.3))  # Distribute to ~30% of patients
        )
        max_additional = max(1, max_additional)
        
        # Shuffle patients to randomize distribution
        shuffled_patients = patient_ids.copy()
        np.random.shuffle(shuffled_patients)
        
        # Distribute remaining migraines with decreasing probability
        for patient_id in shuffled_patients:
            if remaining_migraines <= 0:
                break
            
            # Decide how many additional migraines for this patient
            additional = min(
                np.random.randint(0, max_additional + 1),
                remaining_migraines
            )
            
            if additional > 0:
                patient_migraine_counts[patient_id] += additional
                remaining_migraines -= additional
    
    for patient_id in patient_ids:
        patient_data = []
        migraine_dates[patient_id] = []
        
        # Get the assigned migraine count for this patient
        migraine_count = patient_migraine_counts[patient_id]
        
        # Ensure we don't exceed reasonable limits
        migraine_count = min(migraine_count, len(date_range) // 3)
        
        # Generate random migraine dates
        if migraine_count > 0:
            migraine_days_indices = []
            last_migraine_day_idx = -999 # Initialize far in the past
            min_gap_days = 3 # Minimum days between migraines for a patient
            
            # Shuffle possible days to avoid bias towards earlier days
            possible_days = list(range(2, len(date_range)))
            np.random.shuffle(possible_days)
            
            # Select days ensuring minimum gap
            for day_idx in possible_days:
                if len(migraine_days_indices) >= migraine_count: # Stop if we have enough
                    break
                if day_idx >= last_migraine_day_idx + min_gap_days:
                    migraine_days_indices.append(day_idx)
                    last_migraine_day_idx = day_idx
                    # Sort temporarily to find the most recent for the next check
                    migraine_days_indices.sort()
                    last_migraine_day_idx = migraine_days_indices[-1]
            
            # Final sort of the selected indices
            migraine_days_indices.sort()

            for day_idx in migraine_days_indices: # Iterate through selected indices
                migraine_date = date_range[day_idx]
                migraine_date_str = migraine_date.strftime('%Y-%m-%d')
                migraine_dates[patient_id].append(migraine_date_str)
                
                # Migraine details
                severity = np.random.randint(1, 11)  # 1-10 scale
                duration = np.random.uniform(3, 72)  # Hours
                
                # Migraine start time (more common in morning)
                hour_weights = np.array([0.04, 0.04, 0.04, 0.04, 0.08, 0.08, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04,
                                      0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
                # Ensure weights sum to 1
                hour_weights = hour_weights / np.sum(hour_weights)
                hour = np.random.choice(range(24), p=hour_weights)
                # Convert numpy.int64 to Python int
                hour = int(hour)
                start_time = datetime.combine(migraine_date, datetime.min.time()) + timedelta(hours=hour)
                
                # Symptoms
                aura = np.random.random() < 0.3  # 30% chance of aura
                nausea = np.random.random() < 0.7  # 70% chance of nausea
                photophobia = np.random.random() < 0.8  # 80% chance of photophobia
                phonophobia = np.random.random() < 0.7  # 70% chance of phonophobia
                
                # Triggers (based on the data we're generating)
                triggers = []
                if np.random.random() < 0.5:
                    triggers.append('stress')
                if np.random.random() < 0.4:
                    triggers.append('weather_change')
                if np.random.random() < 0.4:
                    triggers.append('sleep_disruption')
                
                patient_data.append({
                    'patient_id': patient_id,
                    'date': migraine_date_str,
                    'start_time': start_time,
                    'duration_hours': duration,
                    'severity': severity,
                    'aura': aura,
                    'nausea': nausea,
                    'photophobia': photophobia,
                    'phonophobia': phonophobia,
                    'triggers': ','.join(triggers)
                })
        
        # Create dataframe and save to CSV only if it's not empty
        df = pd.DataFrame(patient_data)
        if not df.empty:
            df.to_csv(os.path.join(migraine_dir, f"{patient_id}_migraines.csv"), index=False)
            all_patient_data.extend(patient_data)
    
    # Create a combined CSV with all patients (only includes patients with events)
    all_df = pd.DataFrame(all_patient_data)
    if not all_df.empty:
        all_df.to_csv(os.path.join(migraine_dir, "all_migraine_data.csv"), index=False)
    else:
        print("Warning: No migraine events were generated. Creating an empty file.")
        pd.DataFrame(columns=['patient_id', 'date', 'start_time', 'duration_hours', 'severity', 
                             'aura', 'nausea', 'photophobia', 'phonophobia', 'triggers']
                   ).to_csv(os.path.join(migraine_dir, "all_migraine_data.csv"), index=False)
    
    actual_rate = len(all_patient_data) / total_days
    print(f"Generated migraine data: {len(all_patient_data)} events (actual positive rate: {actual_rate:.2%})")
    return all_df, migraine_dates

def create_dataset_metadata(args, output_dir, patient_ids, date_range, data_summary):
    """Create metadata file with dataset information."""
    metadata = {
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_patients': len(patient_ids),
        'date_range': {
            'start': date_range[0].strftime('%Y-%m-%d'),
            'end': date_range[-1].strftime('%Y-%m-%d'),
            'num_days': len(date_range)
        },
        'data_summary': data_summary,
        'generation_parameters': vars(args)
    }
    
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset metadata saved to {os.path.join(output_dir, 'dataset_metadata.json')}")

def main():
    """Main function to generate synthetic migraine data."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create necessary directories
    print(f"Creating directory structure in {args.output_dir}...")
    directories = create_directory_structure(args.output_dir)
    
    # Generate patient IDs
    print(f"Generating data for {args.num_patients} patients over {args.days} days...")
    patient_ids = generate_patient_ids(args.num_patients)
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    date_range = generate_date_range(start_date, args.days)
    
    # First generate migraine events (to inform other modalities)
    print("Generating migraine events...")
    migraine_df, migraine_dates = generate_migraine_events(patient_ids, date_range, args.output_dir, args.avg_migraine_freq)
    
    # Generate data for each modality
    print("Generating EEG data...")
    eeg_df = generate_eeg_data(patient_ids, date_range, args.output_dir, migraine_dates)
    
    print("Generating weather data...")
    weather_df = generate_weather_data(date_range, args.output_dir)
    
    print("Generating sleep data...")
    sleep_df = generate_sleep_data(patient_ids, date_range, args.output_dir, migraine_dates)
    
    print("Generating stress data...")
    stress_df = generate_stress_data(patient_ids, date_range, args.output_dir, migraine_dates)
    
    # Create dataset metadata
    data_summary = {
        'eeg_records': len(eeg_df),
        'weather_records': len(weather_df),
        'sleep_records': len(sleep_df),
        'stress_records': len(stress_df),
        'migraine_events': len(migraine_df)
    }
    create_dataset_metadata(args, args.output_dir, patient_ids, date_range, data_summary)
    
    print("\nSynthetic migraine dataset generation complete!")
    print(f"Generated {data_summary['migraine_events']} migraine events across {args.num_patients} patients")
    print(f"Data saved to {args.output_dir}")

if __name__ == "__main__":
    main()