#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run migraine prediction using FuseMOE with multimodal data.

This script demonstrates how to:
1. Process multimodal data for migraine prediction
2. Prepare data for the FuseMOE model
3. Train and evaluate the model
4. Visualize results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
import json
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging # Make sure logging is imported

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from preprocessing.migraine_preprocessing import MigraineDataPipeline, EEGProcessor, WeatherConnector, SleepProcessor, StressProcessor
from core.pygmo_fusemoe import PyGMOFuseMoE, MigraineFuseMoE
from utils.config import MoEConfig

# Define missing visualization functions
def plot_expert_usage(expert_usage, expert_names, title="Expert Usage in Migraine Prediction", output_path=None):
    """Plots the usage distribution of experts in the model."""
    plt.figure(figsize=(12, 6))
    plt.bar(expert_names, expert_usage, color='skyblue')
    plt.xlabel('Experts')
    plt.ylabel('Usage')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return plt.gcf()

def plot_modality_importance(modality_importance, modality_names, title="Modality Importance", output_path=None):
    """Plots the importance of each modality for predictions."""
    plt.figure(figsize=(10, 6))
    plt.bar(modality_names, modality_importance, color='lightgreen')
    plt.xlabel('Modalities')
    plt.ylabel('Importance')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return plt.gcf()

def plot_roc_curve(y_true, y_pred, title="ROC Curve", output_path=None):
    """Plots ROC curve for binary classification."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return plt.gcf()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run migraine data preprocessing pipeline")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data/migraine',
                      help='Directory containing migraine data')
    parser.add_argument('--cache_dir', type=str, default='./migraine_data_cache',
                      help='Directory to cache processed data')
    parser.add_argument('--weather_api_key', type=str, default=None,
                      help='API key for weather data service')
    
    # Location and date range
    parser.add_argument('--latitude', type=float, default=40.7128,
                      help='Latitude for weather data')
    parser.add_argument('--longitude', type=float, default=-74.0060,
                      help='Longitude for weather data')
    parser.add_argument('--start_date', type=str, default=None,
                      help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                      help='End date for data (YYYY-MM-DD)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                      help='Hidden size for the model')
    parser.add_argument('--num_experts', type=int, default=8,
                      help='Number of experts in the model')
    parser.add_argument('--modality_experts', type=str, default=None,
                      help='Allocation of experts to modalities, format: "modality1:num1,modality2:num2"')
    parser.add_argument('--top_k', type=int, default=2,
                      help='Number of experts to route to')
    parser.add_argument('--cross_method', type=str, default='moe',
                      help='Cross-modal fusion method')
    parser.add_argument('--gating_function', type=str, default='laplace',
                      help='Gating function type (softmax, laplace, gaussian)')
    parser.add_argument('--router_type', type=str, default='joint',
                      help='Router type for the model')
    parser.add_argument('--cpu', action='store_true',
                      help='Run on CPU even if GPU is available')
    
    # Data preparation parameters
    parser.add_argument('--prediction_horizon', type=int, default=24,
                      help='How many hours ahead to predict migraine events')
    parser.add_argument('--window_size', type=int, default=24,
                      help='Size of the input window in hours for feature preparation')
    
    # PyGMO optimization parameters
    parser.add_argument('--use_pygmo', action='store_true',
                      help='Use PyGMO optimization for experts and gating')
    parser.add_argument('--expert_algorithm', type=str, default='de',
                      help='Algorithm for expert optimization (de, sade, pso)')
    parser.add_argument('--gating_algorithm', type=str, default='pso',
                      help='Algorithm for gating optimization (pso, abc, sade)')
    parser.add_argument('--expert_population_size', type=int, default=10,
                      help='Population size for expert optimization')
    parser.add_argument('--gating_population_size', type=int, default=10,
                      help='Population size for gating optimization')
    parser.add_argument('--expert_generations', type=int, default=5,
                      help='Number of generations for expert optimization')
    parser.add_argument('--gating_generations', type=int, default=5,
                      help='Number of generations for gating optimization')
    
    # Patient-specific adaptation
    parser.add_argument('--patient_adaptation', action='store_true',
                      help='Enable patient-specific adaptation')
    parser.add_argument('--patient_id', type=str, default=None,
                      help='ID of the patient for patient-specific adaptation')
    parser.add_argument('--patient_iterations', type=int, default=3,
                      help='Number of iterations for patient adaptation')
    parser.add_argument('--load_base_model', type=str, default=None,
                      help='Path to base model for patient adaptation')
    
    # Cache Control
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable the use of cached data during preprocessing.')

    # Pipeline Config Path
    parser.add_argument('--config-path', type=str, 
                        default='src/preprocessing/migraine_preprocessing/config.yaml',
                        help='Path to the data pipeline configuration YAML file.')
                        
    # >>> ADDED IMPUTATION ARGUMENTS <<<
    parser.add_argument('--imputation_method', type=str, default='knn',
                      choices=['knn', 'iterative', 'none'], # Use strings only
                      help='Method to use for imputing missing values ("knn", "iterative", "none"). Use "none" to skip imputation.')
    parser.add_argument('--imputer_config', type=str, default=None,
                      help="""JSON string with configuration for the imputer (e.g., '{"n_neighbors": 3}').""")
    # >>> END IMPUTATION ARGUMENTS <<<
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    return args


def get_modality_experts_config(fusemoe_data):
    """
    Define modality-specific experts configuration based on data characteristics.
    
    Args:
        fusemoe_data: Data prepared for FuseMOE
        
    Returns:
        Dictionary of modality experts configuration
    """
    modalities = fusemoe_data['modalities']
    features = fusemoe_data['features_per_modality']
    
    # Default allocation: Equal experts per modality
    total_experts = 8
    experts_per_modality = {}
    
    # If we have all four modalities 
    if len(modalities) == 4:
        # Allocate experts based on feature importance for migraine
        experts_per_modality = {
            'eeg': 3,       # Highest priority: EEG data
            'weather': 2,   # Weather changes are strong triggers
            'sleep': 2,     # Sleep disruption is a key factor
            'stress': 1     # Stress is also important
        }
    # If we have three modalities
    elif len(modalities) == 3:
        if 'eeg' in modalities:
            experts_per_modality['eeg'] = 3
            remaining = [m for m in modalities if m != 'eeg']
            experts_per_modality[remaining[0]] = 3
            experts_per_modality[remaining[1]] = 2
        else:
            for m in modalities:
                experts_per_modality[m] = total_experts // len(modalities)
    # If we have two modalities
    elif len(modalities) == 2:
        experts_per_modality[modalities[0]] = total_experts // 2
        experts_per_modality[modalities[1]] = total_experts // 2
    # If we have one modality
    elif len(modalities) == 1:
        experts_per_modality[modalities[0]] = total_experts
    
    # Return only modalities that exist in the data
    return {m: experts_per_modality.get(m, 0) for m in modalities}


def evaluate_model(model, X_test, y_test, device):
    """
    Evaluate the trained FuseMOE model on test data.
    
    Args:
        model: Trained MigraineFuseMoE model.
        X_test: Dictionary of test data tensors for each modality.
        y_test: Test target tensor.
        device: Device to run evaluation on.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()
    model.to(device)
    
    # Move test data to the correct device
    X_test_dev = {mod: tensor.to(device) for mod, tensor in X_test.items()}
    y_test_dev = y_test.to(device)

    with torch.no_grad():
        outputs, gates = model(X_test_dev)
        probs = torch.sigmoid(outputs).cpu().numpy() # Probabilities needed for AUC
        predicted = (probs > 0.5).astype(int)
        y_true = y_test_dev.cpu().numpy().astype(int)

    accuracy = accuracy_score(y_true, predicted)
    precision = precision_score(y_true, predicted, zero_division=0)
    recall = recall_score(y_true, predicted, zero_division=0)
    f1 = f1_score(y_true, predicted, zero_division=0)
    
    # Calculate AUC
    try:
        auc = roc_auc_score(y_true, probs) 
    except ValueError as e:
        # Handle cases where AUC cannot be computed (e.g., only one class present)
        print(f"Warning: Could not compute AUC: {e}")
        auc = float('nan') # Assign NaN if AUC calculation fails

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc # Add AUC to metrics
    }
    
    # Optional: Get expert usage if needed for analysis
    expert_usage = model.get_expert_usage()
    if expert_usage is not None:
        metrics['expert_usage'] = expert_usage
        
    return metrics, probs, y_true # Return probs and y_true for ROC curve


def visualize_results(results_dir, metrics, y_true, probs, optimization_history, model=None, X_test=None):
    """
    Generate visualizations of model evaluation results.
    
    Args:
        results_dir: Directory to save plots.
        metrics: Dictionary of evaluation metrics (including AUC).
        y_true: True labels from test set.
        probs: Predicted probabilities from test set.
        optimization_history: Dictionary containing optimization history.
        model: Optional trained model for trigger analysis.
        X_test: Optional test data for trigger analysis.
    """
    print("\nGenerating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- ROC Curve --- 
    plt.figure(figsize=(10, 6))
    if not np.isnan(metrics.get('auc', float('nan'))):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = metrics['auc']
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
         print("Skipping ROC curve plot as AUC is NaN.")
         # Optionally plot a default line or message
         plt.text(0.5, 0.5, 'AUC could not be calculated', horizontalalignment='center', verticalalignment='center')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # ... rest of the visualization code ...


def parse_modality_experts(modality_experts_str):
    """
    Parse modality experts string into a dictionary.
    
    Args:
        modality_experts_str: String in format "modality1:num1,modality2:num2"
        
    Returns:
        Dictionary mapping modalities to number of experts
    """
    if not modality_experts_str:
        return None
        
    experts_dict = {}
    pairs = modality_experts_str.split(',')
    for pair in pairs:
        if ':' in pair:
            modality, num_experts = pair.split(':')
            experts_dict[modality.strip()] = int(num_experts.strip())
    
    return experts_dict


def adapt_to_patient(model, patient_data, iterations=3):
    """
    Adapt model to specific patient's data.
    
    Args:
        model: Trained FuseMOE model
        patient_data: Dictionary of patient-specific data
        iterations: Number of adaptation iterations
        
    Returns:
        Adapted model
    """
    print(f"\nAdapting model to patient-specific patterns ({iterations} iterations)...")
    
    # Extract patient data
    patient_X = patient_data['X']
    patient_y = patient_data['y']
    
    # Convert to PyTorch tensors
    X_tensors = {}
    for i, x_dict in enumerate(patient_X):
        for modality, features in x_dict.items():
            if modality not in X_tensors:
                X_tensors[modality] = []
            X_tensors[modality].append(torch.tensor(features, dtype=torch.float32))
    
    y_tensor = torch.tensor(patient_y, dtype=torch.float32)
    
    # Perform adaptation iterations
    for i in range(iterations):
        print(f"  Adaptation iteration {i+1}/{iterations}")
        
        # Fine-tune on patient data
        if hasattr(model, 'adapt_to_patient'):
            model.adapt_to_patient(X_tensors, y_tensor)
        else:
            # Fallback implementation if adapt_to_patient isn't implemented
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.BCEWithLogitsLoss()
            
            # Mini training loop on patient data
            model.train()
            for _ in range(10):  # 10 mini-iterations
                for j in range(len(patient_X)):
                    x_batch = {}
                    for modality, tensors in X_tensors.items():
                        x_batch[modality] = tensors[j].unsqueeze(0)
                    
                    # Forward pass
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_tensor[j].unsqueeze(0).unsqueeze(1))
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
    print("Patient adaptation complete.")
    return model


def main():
    """Main function to run migraine data pipeline."""
    # Parse command line arguments
    args = parse_args()

    # --- Create Output Directory --- 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup Logging --- 
    log_file = os.path.join(args.output_dir, 'prediction.log')
    # Remove existing handlers to avoid duplicate logs if run multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file), # Log to file
            logging.StreamHandler(sys.stdout) # Log to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    # ------------------------

    # --- Define Device Early --- 
    if torch.cuda.is_available() and not args.cpu:
        device = 'cuda'
        print("Using GPU")
    else:
        device = 'cpu'
        print("Using CPU")
    # ---------------------------

    # Create MigraineDataPipeline instance
    pipeline = MigraineDataPipeline(data_dir=args.data_dir, 
                                    cache_dir=args.cache_dir,
                                    weather_api_key=args.weather_api_key) # Pass API key if available

    # Process data
    logger.info("Processing data using the pipeline...")
    # Revert back to using run_full_pipeline with necessary arguments
    fusemoe_data = pipeline.run_full_pipeline(
        location=(args.latitude, args.longitude),
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_horizon=args.prediction_horizon,
        window_size=args.window_size,
        imputation_method=args.imputation_method if args.imputation_method != 'none' else None,
        imputer_config=args.imputer_config
    )

    # Check if data processing returned valid data
    if fusemoe_data is None or \
       not isinstance(fusemoe_data, dict) or \
       'X' not in fusemoe_data or \
       'y' not in fusemoe_data or \
       fusemoe_data['y'] is None or \
       len(fusemoe_data['X']) == 0 or \
       len(fusemoe_data['y']) == 0:
        logger.error("Failed to process data or data is empty/invalid format after pipeline execution.")
        sys.exit(1)

    logger.info("Data processing complete.")
    logger.info(f"Number of feature samples (X): {len(fusemoe_data['X'])}")
    logger.info(f"Number of target samples (y): {len(fusemoe_data['y'])}")

    # Check for sufficient data using len()
    if len(fusemoe_data['X']) < 2 or len(np.unique(fusemoe_data['y'])) < 2:
        print("Error: Insufficient data for training.")
        return

    # Get the number of features dynamically
    # input_features = fusemoe_data['X'].shape[1]

    # TODO: Load or define MoE model configuration properly
    # Configuration is set later using num_features_total
    # config = MoEConfig(

    # Data Splitting
    X_train, X_temp, y_train, y_temp = train_test_split(
        fusemoe_data['X'], 
        fusemoe_data['y'], 
        test_size=0.2, 
        random_state=42,
        stratify=fusemoe_data['y'] 
    )
    print(f"Train set targets: {np.unique(y_train, return_counts=True)}")
    print(f"Test set targets: {np.unique(y_temp, return_counts=True)}")

    # Print summary of processed data
    print("\nData Processing Summary:")
    print(f"Total samples: {len(fusemoe_data['X'])}")
    print(f"Available modalities: {fusemoe_data['modalities']}")
    
    for modality, feature_count in fusemoe_data['features_per_modality'].items():
        if feature_count > 0:
            print(f"  - {modality}: {feature_count} features")
    
    # Calculate positive rate
    positive_rate = np.mean(fusemoe_data['y'])
    print(f"Positive rate (migraine events): {positive_rate:.2%}")
    
    # Save dataset summary to output directory
    summary_file = os.path.join(args.output_dir, 'dataset_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Migraine Prediction Dataset Summary\n")
        f.write(f"===================================\n\n")
        f.write(f"Data range: {args.start_date} to {args.end_date}\n")
        f.write(f"Total samples: {len(fusemoe_data['X'])}\n")
        f.write(f"Available modalities: {', '.join(fusemoe_data['modalities'])}\n\n")
        
        f.write(f"Features per modality:\n")
        for modality, feature_count in fusemoe_data['features_per_modality'].items():
            if feature_count > 0:
                f.write(f"  - {modality}: {feature_count} features\n")
        
        f.write(f"\nPositive rate (migraine events): {positive_rate:.2%}\n")
    
    print(f"\nDataset summary saved to {summary_file}")
    print("Data processing complete.")
    
    # Split data into training and testing sets (stratified if possible)
    try:
        train_X, test_X, train_y, test_y = train_test_split(
            fusemoe_data['X'], 
            fusemoe_data['y'], 
            test_size=0.2, 
            random_state=42,
            stratify=fusemoe_data['y'] 
        )
        print(f"Train set targets: {np.unique(train_y, return_counts=True)}")
        print(f"Test set targets: {np.unique(test_y, return_counts=True)}")
    except ValueError as e:
        print(f"Warning: Stratified split failed: {e}. Performing regular split.")
        train_X, test_X, train_y, test_y = train_test_split(
            fusemoe_data['X'], fusemoe_data['y'], test_size=0.2, random_state=42
        )
        
    print(f"Training samples: {len(train_X)}")
    print(f"Testing samples: {len(test_X)}")

    # --- Prepare data dictionary for PyGMO (and potentially evaluation) --- 
    # Input to PyGMO should be: dict {modality: tensor[Batch, Window, Features]}
    train_data_dict = {mod: [] for mod in fusemoe_data['modalities']}
    for sample_dict in train_X:
        for mod in fusemoe_data['modalities']:
            if mod in sample_dict:
                train_data_dict[mod].append(torch.tensor(sample_dict[mod], dtype=torch.float32))
            else:
                # Handle missing modality for a sample (e.g., pad with zeros)
                num_features = fusemoe_data['features_per_modality'].get(mod, 0)
                train_data_dict[mod].append(torch.zeros((config.window_size, num_features), dtype=torch.float32))
                
    # Stack tensors for each modality to create [Batch, Window, Features]
    for mod in list(train_data_dict.keys()): # Use list to allow deletion
        if train_data_dict[mod]:
            train_data_dict[mod] = torch.stack(train_data_dict[mod]).to(device)
        else:
            # Remove modality if no data was collected
            print(f"Warning: No training data found for modality {mod}. Removing.")
            del train_data_dict[mod]
            
    # Convert train_y to tensor [Batch, 1]
    y_train_tensor = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1).to(device)

    # --- Prepare test data dictionary similarly --- 
    test_data_dict = {mod: [] for mod in fusemoe_data['modalities']}
    for sample_dict in test_X:
        for mod in fusemoe_data['modalities']:
            if mod in sample_dict:
                test_data_dict[mod].append(torch.tensor(sample_dict[mod], dtype=torch.float32))
            else:
                num_features = fusemoe_data['features_per_modality'].get(mod, 0)
                test_data_dict[mod].append(torch.zeros((config.window_size, num_features), dtype=torch.float32))
                
    for mod in list(test_data_dict.keys()):
        if test_data_dict[mod]:
            test_data_dict[mod] = torch.stack(test_data_dict[mod]).to(device)
        else:
            print(f"Warning: No test data found for modality {mod}. Removing.")
            del test_data_dict[mod]
            
    # Convert test_y to tensor [Batch, 1]
    y_test_tensor = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1).to(device)
    # --- End Data Preparation --- 
    
    # Define modality experts configuration
    # If provided via command line, use that instead of the auto-configuration
    if args.modality_experts:
        experts_config = parse_modality_experts(args.modality_experts)
        print("\nUsing provided expert allocation per modality:")
    else:
        experts_config = get_modality_experts_config(fusemoe_data)
        print("\nAuto-configured expert allocation per modality:")
    
    for modality, num_experts in experts_config.items():
        print(f"  - {modality}: {num_experts} experts")
    
    # Calculate total number of features before creating config
    num_features_total = sum(fusemoe_data['features_per_modality'].values())

    # Configure MoE model
    config = MoEConfig(
        # MoE specific args
        num_experts=args.num_experts,
        # Input size should be FLATTENED features per sample for PyGMO
        moe_input_size=num_features_total * args.window_size, # e.g., 27 features * 24 window = 648
        moe_hidden_size=args.hidden_size, # Use args.hidden_size for MoE hidden layer
        moe_output_size=1, # Binary classification output
        router_type=args.router_type,
        window_size=args.window_size, # <-- Pass window_size from args
        gating=args.gating_function,
        top_k=args.top_k,
        num_modalities=len(fusemoe_data['modalities']),
        
        # General Transformer args (using MoE args where appropriate or defaults)
        hidden_dim=args.hidden_size, # General hidden dim for embeddings/transformer layers
        num_layers=3, # Example value, maybe add as arg?
        n_heads=4, # Example value, maybe add as arg?
        dropout=0.1, # Example value, maybe add as arg?
        # Other args using defaults from MoEConfig definition
    )

    # Create the PyGMO-enhanced FuseMoE model
    print("Creating MigraineFuseMoE model...")
    migraine_fusemoe = MigraineFuseMoE(
        config=config,
        input_sizes=fusemoe_data['features_per_modality'],
        hidden_size=args.hidden_size,
        output_size=1,
        num_experts=args.num_experts,
        modality_experts=experts_config,
        dropout=config.dropout,
        use_pso_gating=args.use_pygmo,
        use_evo_experts=args.use_pygmo,
        patient_adaptation=args.patient_adaptation
    ).to(device) # Move model to device

    # --- MODEL TRAINING / OPTIMIZATION ---
    if args.use_pygmo:
        print("\nOptimizing model architecture and routing using PyGMO...")

        print("Prepared data for PyGMO optimization:")
        for mod, tensor in train_data_dict.items():
            print(f"  - {mod}: {tensor.shape}")
        print(f"  - targets: {y_train_tensor.shape}")

        print("\nStarting PyGMO-enhanced FuseMOE optimization...")
        try:
            # Capture the returned history along with the model
            optimized_model, optimization_history = migraine_fusemoe.optimize_model(
                train_data=(train_data_dict, y_train_tensor), # Pass data as a tuple to the 'train_data' argument
                expert_algo=args.expert_algorithm, # Corrected keyword
                gating_algo=args.gating_algorithm, # Corrected keyword
                expert_pop_size=args.expert_population_size,
                gating_pop_size=args.gating_population_size,
                seed=42, # Use a consistent seed
                device=device
            )
            migraine_fusemoe = optimized_model # Update the model variable
            print("PyGMO Model optimization complete!")
            
            # Save the optimization history
            history_file = os.path.join(args.output_dir, 'optimization_history.json')
            try:
                # Convert tensors/numpy arrays in history to lists for JSON serialization
                serializable_history = {}
                for stage, stage_data in optimization_history.items(): # stage_data is e.g., {'algorithm': 'sade', 'history': [...]} 
                    if isinstance(stage_data, dict):
                        algo = stage_data.get('algorithm', 'Unknown')
                        history_list = stage_data.get('history', []) # Get the list of records
                        
                        processed_records = []
                        if isinstance(history_list, list):
                            for record in history_list: # Iterate over the actual list
                                if isinstance(record, dict):
                                    new_record = {}
                                    for key, value in record.items():
                                        if isinstance(value, torch.Tensor):
                                            new_record[key] = value.item() if value.numel() == 1 else value.tolist()
                                        elif isinstance(value, np.ndarray):
                                            new_record[key] = value.item() if value.size == 1 else value.tolist()
                                        elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                                            new_record[key] = value # Keep JSON serializable types
                                        else:
                                            new_record[key] = str(value) # Fallback: convert to string
                                    processed_records.append(new_record)
                                else:
                                    print(f"Warning: Skipping non-dict record in history for stage '{stage}': {record}")
                        else:
                            print(f"Warning: 'history' data for stage '{stage}' is not a list: {type(history_list)}")

                        # Store the algorithm and the processed list for this stage
                        serializable_history[stage] = {
                            'algorithm': algo,
                            'history': processed_records
                        }
                    else:
                        # Handle unexpected format for a stage
                         print(f"Warning: Skipping unexpected data format for stage '{stage}': {type(stage_data)}")

                with open(history_file, 'w') as f:
                    json.dump(serializable_history, f, indent=4)
                print(f"Optimization history saved to {history_file}")
            except Exception as e:
                print(f"Error saving optimization history: {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback

        except Exception as e:
            print(f"Error during PyGMO model optimization: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping PyGMO optimization due to error.")
            # Fallback might be needed here, e.g., standard training or exiting

    else:
        # Standard PyTorch training loop
        print("\nStarting Standard PyTorch Training...")
        optimizer = torch.optim.Adam(migraine_fusemoe.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss() # Handles sigmoid internally
        num_epochs = 10 # Example: Train for a few epochs
        batch_size = 16 # Example batch size
        
        migraine_fusemoe.train() # Set model to training mode
        for epoch in range(num_epochs):
            epoch_loss = 0
            # Simple batching (consider DataLoader for larger datasets)
            for i in range(0, len(train_data_dict), batch_size):
                batch_X_list = list(train_data_dict.values())[i:i+batch_size]
                batch_y = torch.tensor(train_y[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(device)
                
                # Prepare batch input dictionary
                batch_input = {mod: batch_X_list[i] for i, mod in enumerate(fusemoe_data['modalities'])}
                
                optimizer.zero_grad()
                
                # Forward pass - Expects two outputs (predictions, gates), only need predictions for loss
                model_predictions, _ = migraine_fusemoe(batch_input) 
                
                # Calculate loss
                loss = criterion(model_predictions, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_epoch_loss = epoch_loss / len(train_data_dict) if len(train_data_dict) > 0 else 0
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        print("Standard Training complete.")
    # --- END TRAINING / OPTIMIZATION --- 

    # Use the potentially trained/optimized model going forward
    model = migraine_fusemoe 

    # Patient-specific adaptation if requested
    if args.patient_adaptation:
        if args.patient_id:
            print(f"Performing patient-specific adaptation for patient ID: {args.patient_id}")
            # In practice, you would load patient-specific data here
            # For now, just use a subset of the data to simulate patient data
            patient_indices = indices[:20]  # Just use the first 20 samples as "patient data"
            patient_data = {
                'X': [fusemoe_data['X'][i] for i in patient_indices],
                'y': [fusemoe_data['y'][i] for i in patient_indices],
                'modalities': fusemoe_data['modalities'],
                'features_per_modality': fusemoe_data['features_per_modality']
            }
            
            # Adapt model to patient data
            model = adapt_to_patient(model, patient_data, iterations=args.patient_iterations)
        else:
            print("Patient adaptation requested but no patient ID provided. Skipping adaptation.")
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    results, probs, y_true = evaluate_model(model, test_data_dict, y_test_tensor, device)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(args.output_dir, results, y_true, probs, optimization_history=None, model=model, X_test=test_data_dict)
    
    # Save the final model
    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), model_path) # Save state_dict instead of whole model
    print(f"Model state_dict saved to {model_path}")
    
    # If using PyGMO, identify and visualize triggers
    if args.use_pygmo:
        print("\nIdentifying potential migraine triggers...")
        # Choose a sample input for trigger identification - Use the test_data_dict
        sample_input = {}
        if len(y_test_tensor) > 0:
             # Find a positive sample index
            positive_idx = -1
            for i in range(len(y_test_tensor)):
                if y_test_tensor[i].item() == 1:
                    positive_idx = i
                    break
            
            target_idx = positive_idx if positive_idx != -1 else 0 # Use first sample if no positive found
            
            for modality, tensor_batch in test_data_dict.items():
                sample_input[modality] = tensor_batch[target_idx].unsqueeze(0) # Get sample and add batch dim
        else:
            print("Warning: Cannot identify triggers - test data is empty.")
            
        # Identify triggers if sample_input is populated
        if sample_input:
            trigger_scores = model.identify_triggers(sample_input)
            # Visualize triggers
            plt.figure(figsize=(12, 8))
            for i, (modality, scores) in enumerate(trigger_scores.items()):
                scores = scores.cpu().numpy().flatten()
                plt.subplot(len(trigger_scores), 1, i+1)
                plt.bar(range(len(scores)), scores, alpha=0.7)
                plt.title(f"Trigger Importance: {modality}")
                plt.xlabel("Feature Index")
                plt.ylabel("Importance Score")
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            trigger_path = os.path.join(args.output_dir, "trigger_analysis.png")
            plt.savefig(trigger_path)
            plt.close()
            print(f"Trigger analysis saved to {trigger_path}")
        else:
             print("Skipping trigger identification.")
    
    print("\nMigraine prediction completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 