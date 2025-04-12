import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import os
import json
import time


class WeatherConnector:
    """
    Class for fetching and processing weather data for migraine prediction.
    Weather changes, particularly barometric pressure, are known migraine triggers.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./weather_cache"):
        """
        Initialize weather connector.
        
        Args:
            api_key: API key for weather data provider (e.g., OpenWeatherMap)
            cache_dir: Directory to cache weather data
        """
        self.api_key = api_key or os.environ.get("WEATHER_API_KEY")
        if not self.api_key:
            print("Warning: No API key provided for weather data. Some functions may not work.")
        
        # Create cache directory if it doesn't exist
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def fetch_historical_weather(self, location: Tuple[float, float], 
                                start_date: Union[str, datetime],
                                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch historical weather data for a specific location and time range.
        
        Args:
            location: Location coordinates (latitude, longitude)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with weather parameters
        """
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Generate cache filename
        lat, lon = location
        cache_filename = f"{self.cache_dir}/weather_{lat}_{lon}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        # Check if cached data exists
        if os.path.exists(cache_filename):
            weather_df = pd.read_csv(cache_filename, parse_dates=['timestamp'])
            if not weather_df.empty and 'timestamp' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
                weather_df = weather_df.set_index('timestamp').sort_index()
            return weather_df
            
        # Check if synthetic data exists in the parent data directory
        data_dir = os.path.dirname(os.path.dirname(self.cache_dir))
        weather_json_path = os.path.join(data_dir, "migraine", "weather", "weather_data.json")
        
        if os.path.exists(weather_json_path):
            print(f"Using locally generated synthetic weather data: {weather_json_path}")
            try:
                with open(weather_json_path, 'r') as f:
                    json_data = json.load(f)
                    
                all_weather_data = []
                for hour_data in json_data.get('hourly_data', []):
                    timestamp = datetime.strptime(hour_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if start_date <= timestamp <= end_date:
                        weather_point = {
                            'timestamp': timestamp,
                            'temperature': hour_data.get('temperature'),
                            'pressure': hour_data.get('pressure'),
                            'humidity': hour_data.get('humidity'),
                            'wind_speed': hour_data.get('wind_speed'),
                            'precipitation': hour_data.get('precipitation', 0)
                        }
                        all_weather_data.append(weather_point)
                
                if all_weather_data:
                    weather_df = pd.DataFrame(all_weather_data)
                    # Calculate pressure changes (migraine trigger)
                    weather_df['pressure_change'] = weather_df['pressure'].diff().fillna(0)
                    # Cache the data
                    weather_df.to_csv(cache_filename, index=False)
                    # Set index before returning
                    if 'timestamp' in weather_df.columns:
                        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
                        weather_df = weather_df.set_index('timestamp').sort_index()
                    return weather_df
            except Exception as e:
                print(f"Error loading synthetic weather data: {e}")
                # Continue to API call attempt
        
        # Initialize result container
        all_weather_data = []
        api_failed = False
        
        # OpenWeatherMap historical data API requires data to be fetched by day
        current_date = start_date
        while current_date <= end_date:
            # Build API endpoint for historical data
            timestamp = int(current_date.timestamp())
            endpoint = f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
            params = {
                "lat": lat,
                "lon": lon,
                "dt": timestamp,
                "appid": self.api_key,
                "units": "metric"  # Use metric units
            }
            
            try:
                response = requests.get(endpoint, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process hourly data if available
                    if 'data' in data:
                        for hour_data in data['data']:
                            weather_point = {
                                'timestamp': datetime.fromtimestamp(hour_data.get('dt', 0)),
                                'temperature': hour_data.get('temp', None),
                                'pressure': hour_data.get('pressure', None),
                                'humidity': hour_data.get('humidity', None),
                                'clouds': hour_data.get('clouds', None),
                                'wind_speed': hour_data.get('wind_speed', None),
                            }
                            all_weather_data.append(weather_point)
                else:
                    print(f"Error fetching weather data for {current_date}: {response.status_code}")
                    api_failed = True
                    break
            except Exception as e:
                print(f"Exception fetching weather data: {e}")
                api_failed = True
                break
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Respect API rate limits
            time.sleep(1)
        
        # If API failed and we don't have any data yet, generate synthetic data
        if api_failed and not all_weather_data:
            print("API call failed. Generating synthetic weather data...")
            
            # Generate synthetic data as a fallback
            from subprocess import run, PIPE
            try:
                # Get the correct script path
                script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "scripts"))
                script_path = os.path.join(script_dir, "generate_weather_data.py")
                
                # Convert to string format for command line
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # Use the data_dir from the cache_dir path
                data_dir = os.path.abspath(os.path.join(os.path.dirname(self.cache_dir), ".."))
                weather_dir = os.path.join(data_dir, "data/migraine/weather")
                
                # Run the script to generate data
                cmd = [
                    "python", script_path,
                    "--output_dir", weather_dir,
                    "--start_date", start_date_str,
                    "--end_date", end_date_str,
                    "--latitude", str(lat),
                    "--longitude", str(lon)
                ]
                print(f"Executing: {' '.join(cmd)}")
                result = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
                
                if result.returncode == 0:
                    print("Successfully generated synthetic weather data")
                    # Attempt to load the generated data
                    with open(os.path.join(weather_dir, "weather_data.json"), 'r') as f:
                        json_data = json.load(f)
                    
                    for hour_data in json_data.get('hourly_data', []):
                        timestamp = datetime.strptime(hour_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                        if start_date <= timestamp <= end_date:
                            weather_point = {
                                'timestamp': timestamp,
                                'temperature': hour_data.get('temperature'),
                                'pressure': hour_data.get('pressure'),
                                'humidity': hour_data.get('humidity'),
                                'wind_speed': hour_data.get('wind_speed'),
                                'precipitation': hour_data.get('precipitation', 0)
                            }
                            all_weather_data.append(weather_point)
                else:
                    print(f"Error generating synthetic weather data: {result.stderr}")
            except Exception as e:
                print(f"Exception generating synthetic weather data: {e}")
        
        # Convert to DataFrame
        if all_weather_data:
            weather_df = pd.DataFrame(all_weather_data)
            
            # Calculate pressure changes (migraine trigger)
            if 'pressure' in weather_df.columns:
                 weather_df['pressure_change'] = weather_df['pressure'].diff().fillna(0)
            else: # Ensure column exists even if pressure data is missing
                 weather_df['pressure_change'] = 0
            
            # Cache the data
            # Important: Cache *before* setting index if index=False is desired for CSV
            weather_df.to_csv(cache_filename, index=False)
            
            # Set index before returning
            if 'timestamp' in weather_df.columns:
                 weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
                 weather_df = weather_df.set_index('timestamp').sort_index()
                 
            return weather_df
        else:
            # Return empty DataFrame with expected columns and index type
            cols = ['timestamp', 'temperature', 'pressure', 
                    'humidity', 'clouds', 'wind_speed', 
                    'precipitation', 'pressure_change'] # Added precipitation based on synthetic
            empty_df = pd.DataFrame(columns=cols)
            empty_df['timestamp'] = pd.to_datetime(empty_df['timestamp'])
            empty_df = empty_df.set_index('timestamp')
            return empty_df
    
    def fetch_current_weather(self, location: Tuple[float, float]) -> Dict:
        """
        Fetch current weather data for a specific location.
        
        Args:
            location: Location coordinates (latitude, longitude)
            
        Returns:
            Dictionary with current weather parameters
        """
        lat, lon = location
        endpoint = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"  # Use metric units
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                
                weather = {
                    'timestamp': datetime.fromtimestamp(data.get('dt', 0)),
                    'temperature': data.get('main', {}).get('temp'),
                    'pressure': data.get('main', {}).get('pressure'),
                    'humidity': data.get('main', {}).get('humidity'),
                    'clouds': data.get('clouds', {}).get('all'),
                    'wind_speed': data.get('wind', {}).get('speed'),
                    'conditions': data.get('weather', [{}])[0].get('main')
                }
                
                return weather
            else:
                print(f"Error fetching current weather: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Exception fetching current weather: {e}")
            return {}
    
    def process_weather_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process weather data to extract migraine-relevant features.
        
        Args:
            weather_df: DataFrame with raw weather data
            
        Returns:
            DataFrame with processed features
        """
        if weather_df.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        processed_df = weather_df.copy()
        
        # Calculate additional features
        
        # 1. Rolling pressure changes (3-hour window)
        processed_df['pressure_change_3h'] = processed_df['pressure'].diff(3).fillna(0)
        
        # 2. Rapid pressure changes (boolean flag for changes > 5 hPa in 3 hours)
        processed_df['rapid_pressure_change'] = abs(processed_df['pressure_change_3h']) > 5
        
        # 3. Temperature variations
        processed_df['temp_change'] = processed_df['temperature'].diff().fillna(0)
        processed_df['temp_change_day'] = processed_df['temperature'].diff(24).fillna(0)
        
        # 4. Humidity variations
        processed_df['humidity_change'] = processed_df['humidity'].diff().fillna(0)
        
        # 5. Weather instability score (combined metric)
        # Higher scores indicate more unstable weather (potential migraine trigger)
        processed_df['weather_instability'] = (
            abs(processed_df['pressure_change']) / 10 +
            abs(processed_df['temp_change']) / 5 +
            abs(processed_df['humidity_change']) / 20
        )
        
        return processed_df
    
    def get_migraine_risk_from_weather(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate migraine risk score based on weather conditions.
        
        Args:
            weather_data: DataFrame with processed weather data
            
        Returns:
            DataFrame with original data plus migraine risk scores
        """
        if weather_data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        risk_df = weather_data.copy()
        
        # Calculate risk score components
        # These weights should be optimized based on patient data
        pressure_risk = abs(risk_df['pressure_change_3h']) * 0.5  # Pressure changes
        temp_risk = abs(risk_df['temp_change_day']) * 0.3  # Temperature changes
        humidity_risk = risk_df['humidity'] * 0.2  # High humidity
        
        # Combine into overall weather-based migraine risk score (0-10 scale)
        risk_df['weather_migraine_risk'] = (
            pressure_risk + temp_risk + humidity_risk
        ).clip(0, 10)
        
        return risk_df
    
    def align_weather_with_migraine_events(self, 
                                          weather_data: pd.DataFrame,
                                          migraine_events: List[Dict]) -> pd.DataFrame:
        """
        Align weather data with migraine events to identify patterns.
        
        Args:
            weather_data: DataFrame with processed weather data
            migraine_events: List of dictionaries with migraine events
                             (must contain 'timestamp' and 'severity' keys)
            
        Returns:
            DataFrame with weather data aligned to migraine events
        """
        if weather_data.empty or not migraine_events:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        aligned_df = weather_data.copy()
        
        # Create a column for days to next migraine
        aligned_df['days_to_next_migraine'] = np.inf
        aligned_df['hours_to_next_migraine'] = np.inf
        aligned_df['next_migraine_severity'] = None
        
        # Convert event timestamps to datetime if they're strings
        events = []
        for event in migraine_events:
            # Use start_time from the event dictionary
            event_time = event.get('start_time')
            if event_time is None:
                print(f"Warning: Skipping migraine event due to missing start_time: {event}")
                continue

            # Ensure event_time is a Timestamp object
            if isinstance(event_time, str):
                try:
                    event_time = pd.Timestamp(event_time)
                except ValueError:
                    print(f"Warning: Skipping migraine event due to invalid start_time format: {event}")
                    continue
            elif not isinstance(event_time, pd.Timestamp):
                 # If it's some other type (like datetime.datetime), try converting
                 try:
                      event_time = pd.Timestamp(event_time)
                 except Exception:
                      print(f"Warning: Skipping migraine event due to unparseable start_time: {event}")
                      continue

            events.append({
                'timestamp': event_time,
                'severity': event['severity']
            })
        
        # Sort events by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        # For each row in the weather data, find the next migraine event
        for idx, row in aligned_df.iterrows():
            # idx is the timestamp (the index)
            # row is the Series of data for that timestamp
            # weather_time = row['timestamp'] # <-- ERROR HERE
            weather_time = idx # Use the index directly
            
            # Find the next migraine event after this weather timestamp
            next_events = [e for e in events if e['timestamp'] > weather_time]
            
            if next_events:
                next_event = next_events[0]
                time_diff = next_event['timestamp'] - weather_time
                aligned_df.at[idx, 'days_to_next_migraine'] = time_diff.total_seconds() / (24 * 3600)
                aligned_df.at[idx, 'hours_to_next_migraine'] = time_diff.total_seconds() / 3600
                aligned_df.at[idx, 'next_migraine_severity'] = next_event['severity']
        
        # Create binary label for prediction (migraine within next 48 hours)
        aligned_df['migraine_within_48h'] = aligned_df['hours_to_next_migraine'] <= 48
        
        return aligned_df 