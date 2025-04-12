#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate synthetic weather data for the migraine prediction pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic weather data")
    
    parser.add_argument('--output_dir', type=str, default='../data/migraine/weather',
                      help='Output directory for weather data')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                      help='Start date for weather data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-01-10',
                      help='End date for weather data (YYYY-MM-DD)')
    parser.add_argument('--latitude', type=float, default=40.7128,
                      help='Latitude for weather location')
    parser.add_argument('--longitude', type=float, default=-74.0060,
                      help='Longitude for weather location')
    
    return parser.parse_args()

def generate_weather_data(output_dir, start_date, end_date, latitude, longitude):
    """Generate synthetic weather data for the given date range and location."""
    # Parse dates
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate date range
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    # Base weather values with some day-to-day consistency
    base_temp = 65 + np.random.uniform(-5, 5)  # Base temperature in F
    base_humidity = 50 + np.random.uniform(-10, 10)  # Base humidity percentage
    base_pressure = 1013 + np.random.uniform(-3, 3)  # Base pressure in hPa
    base_wind = 5 + np.random.uniform(-2, 2)  # Base wind speed in mph
    
    # Weather data for all days
    weather_data = []
    
    # Introduce a small "weather system" in the middle of the date range
    weather_system_start = len(date_range) // 3
    weather_system_end = weather_system_start + 3  # 3-day weather system
    
    for i, date in enumerate(date_range):
        # Daily variation from the base
        is_in_system = weather_system_start <= i < weather_system_end
        
        if is_in_system:
            # Weather system - more variation, higher chance of precipitation
            daily_temp_var = np.random.uniform(-15, 5)  # Cooler during system
            daily_humidity_var = np.random.uniform(10, 30)  # More humidity
            daily_pressure_var = np.random.uniform(-10, -5)  # Lower pressure
            daily_wind_var = np.random.uniform(5, 15)  # Higher winds
            precip_chance = 0.7  # 70% chance of precipitation
        else:
            # Normal conditions
            daily_temp_var = np.random.uniform(-5, 5)
            daily_humidity_var = np.random.uniform(-10, 10)
            daily_pressure_var = np.random.uniform(-3, 3)
            daily_wind_var = np.random.uniform(-2, 2)
            precip_chance = 0.1  # 10% chance of precipitation
        
        # Generate hourly data for the day
        for hour in range(24):
            # Time of day affects temperature (cooler at night, warmer during day)
            hour_of_day_effect = -10 + 20 * np.sin(np.pi * (hour - 6) / 12)  # Peak at 12-13, low at 0-1
            
            # Calculate weather parameters with some randomness
            temp = base_temp + daily_temp_var + hour_of_day_effect + np.random.uniform(-2, 2)
            humidity = min(100, max(0, base_humidity + daily_humidity_var + np.random.uniform(-5, 5)))
            pressure = base_pressure + daily_pressure_var + np.random.uniform(-1, 1)
            wind_speed = max(0, base_wind + daily_wind_var + np.random.uniform(-2, 2))
            
            # Precipitation (if chance is met)
            precipitation = 0
            if np.random.random() < precip_chance:
                precipitation = np.random.uniform(0.01, 0.3)  # Inches of rain
            
            # Create timestamp
            timestamp = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=hour)
            
            # Add weather record
            weather_data.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': round(temp, 2),  # in F
                'humidity': round(humidity, 2),  # percentage
                'pressure': round(pressure, 2),  # hPa
                'precipitation': round(precipitation, 3),  # inches
                'wind_speed': round(wind_speed, 2),  # mph
                'latitude': latitude,
                'longitude': longitude
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(weather_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'synthetic_weather.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved weather data to {csv_path}")
    
    # Save to JSON (for the API format)
    json_data = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly_data': weather_data
    }
    
    json_path = os.path.join(output_dir, 'weather_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved weather data to {json_path}")
    
    return df

def main():
    """Main function."""
    args = parse_args()
    generate_weather_data(
        args.output_dir,
        args.start_date,
        args.end_date,
        args.latitude,
        args.longitude
    )

if __name__ == "__main__":
    main() 