#!/usr/bin/env python
# Integration Test for Migraine Data Pipeline with FuseMOE

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Add the src directory to the Python path to simplify imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import FuseMOE components
from utils.config import MoEConfig
from core.pygmo_fusemoe import MigraineFuseMoE

# Import migraine pipeline modules directly 
sys.path.insert(0, os.path.join(project_root, 'preprocessing', 'migraine_preprocessing'))
from eeg_processor import EEGProcessor
from weather_connector import WeatherConnector  
from sleep_processor import SleepProcessor
from stress_processor import StressProcessor
from migraine_data_pipeline import MigraineDataPipeline

def test_integration():
    """
    Test the integration between our migraine data pipeline and FuseMOE.
    Uses small synthetic datasets for testing purposes.
    """
    print("Starting integration test...")
    
    # Create processors with sample data
    print("Initializing data processors...")
    eeg_processor = EEGProcessor(sample_rate=256, window_size=5)
    weather_connector = WeatherConnector(api_key="sample_key")
    sleep_processor = SleepProcessor()
    stress_processor = StressProcessor()
    
    # Create sample data
    print("Creating synthetic datasets...")
    
    # Sample EEG data: 10 samples, 4 channels, 1280 data points per sample (5 seconds at 256 Hz)
    eeg_data = np.random.randn(10, 4, 1280)
    
    # Sample weather data: 10 samples with pressure, humidity, temperature
    weather_data = {
        'timestamps': [f"2023-06-{i+1} 12:00:00" for i in range(10)],
        'pressure': np.random.uniform(990, 1030, 10),
        'humidity': np.random.uniform(30, 90, 10),
        'temperature': np.random.uniform(15, 35, 10)
    }
    
    # Sample sleep data
    sleep_data = {
        'date': [f"2023-06-{i+1}" for i in range(10)],
        'duration': np.random.uniform(5, 9, 10),
        'efficiency': np.random.uniform(70, 95, 10),
        'deep_sleep': np.random.uniform(0.5, 2.5, 10),
        'rem_sleep': np.random.uniform(1, 3, 10)
    }
    
    # Sample stress data (HRV)
    stress_data = {
        'timestamps': [f"2023-06-{i+1} 12:00:00" for i in range(10)],
        'hrv_sdnn': np.random.uniform(30, 100, 10),
        'resting_hr': np.random.uniform(60, 90, 10)
    }
    
    # Sample migraine events
    migraine_events = {
        'timestamps': [f"2023-06-{i+3} 18:00:00" for i in range(3)],  # 3 migraines
        'severity': np.random.randint(1, 11, 3),
        'duration': np.random.uniform(2, 72, 3)  # 2-72 hours
    }
    
    # Process data with individual processors
    print("Processing individual modalities...")
    try:
        eeg_features = eeg_processor.process(eeg_data)
        weather_features = weather_connector.process_data(weather_data)
        sleep_features = sleep_processor.process(sleep_data)
        stress_features = stress_processor.process(stress_data)
        
        print("EEG features shape:", eeg_features.shape if hasattr(eeg_features, 'shape') else "Not array")
        print("Weather features shape:", weather_features.shape if hasattr(weather_features, 'shape') else "Not array")
        print("Sleep features shape:", sleep_features.shape if hasattr(sleep_features, 'shape') else "Not array")
        print("Stress features shape:", stress_features.shape if hasattr(stress_features, 'shape') else "Not array")
    except Exception as e:
        print(f"Error processing individual modalities: {str(e)}")
        return
    
    # Initialize the pipeline
    print("Initializing MigraineDataPipeline...")
    try:
        pipeline = MigraineDataPipeline(
            window_size=24,  # 24-hour window
            step_size=6,     # 6-hour step
            prediction_horizon=48  # Predict 48 hours ahead
        )
    except Exception as e:
        print(f"Error initializing pipeline: {str(e)}")
        return
    
    # Add data to pipeline
    print("Adding data to pipeline...")
    try:
        pipeline.add_eeg_data(eeg_data)
        pipeline.add_weather_data(weather_data)
        pipeline.add_sleep_data(sleep_data)
        pipeline.add_stress_data(stress_data)
        pipeline.add_migraine_events(migraine_events)
    except Exception as e:
        print(f"Error adding data to pipeline: {str(e)}")
        return
    
    # Process the combined data
    print("Processing combined data in pipeline...")
    try:
        processed_data = pipeline.process()
        X_train, y_train = pipeline.prepare_model_data(test_split=0.0)
        
        print(f"Processed data shapes:")
        print(f"X_train: {type(X_train)}")
        if isinstance(X_train, dict):
            for modality, data in X_train.items():
                print(f"  - {modality}: {data.shape}")
        else:
            print(f"  - Shape: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
    except Exception as e:
        print(f"Error processing combined data: {str(e)}")
        return
    
    # Initialize FuseMOE model for migraine prediction
    print("Initializing MigraineFuseMoE model...")
    
    # Get input dimensions from processed data
    modality_dims = {}
    if isinstance(X_train, dict):
        for modality, data in X_train.items():
            modality_dims[modality] = data.shape[1]
    else:
        modality_dims = {'combined': X_train.shape[1]}
    
    try:
        # Create model configuration
        config = MoEConfig(
            num_experts=8,
            moe_input_size=sum(modality_dims.values()),
            moe_hidden_size=64,
            moe_output_size=2,  # Binary classification (migraine or not)
            router_type='joint',
            dropout=0.1,
            hidden_act='gelu',
            noisy_gating=True,
            top_k=4
        )
        
        # Create model
        model = MigraineFuseMoE(
            config=config,
            input_sizes=modality_dims,
            hidden_size=64,
            output_size=2,
            num_experts=8,
            modality_experts={'eeg': 3, 'weather': 2, 'sleep': 2, 'stress': 1} 
            if 'eeg' in modality_dims else None,
            use_pso_gating=False,  # Skip optimization for this test
            use_evo_experts=False,
            patient_adaptation=False
        )
        
        print("Model initialized successfully!")
        
        # Try a forward pass
        print("Testing forward pass...")
        
        # Convert numpy to torch tensors
        if isinstance(X_train, dict):
            x_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_train.items()}
        else:
            x_torch = torch.tensor(X_train, dtype=torch.float32)
            
        y_torch = torch.tensor(y_train, dtype=torch.long)
        
        # Forward pass
        output = model(x_torch)
        print(f"Model output shape: {output.shape}")
        
        print("Integration test successful! Migraine pipeline data can be fed into FuseMOE model.")
        
    except Exception as e:
        print(f"Error initializing or using model: {str(e)}")
        return

if __name__ == "__main__":
    test_integration() 