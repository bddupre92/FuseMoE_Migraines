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


def evaluate_model(model, test_data):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained FuseMOE model
        test_data: Test data dictionary
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Extract test data
    X_test = test_data['X']
    y_test = test_data['y']
    
    # Make predictions
    all_predictions = []
    all_true_labels = []
    modality_contributions = []
    expert_usages = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            # Convert to model input format
            x_dict = X_test[i]
            x_batch = {}
            
            for modality, features in x_dict.items():
                x_batch[modality] = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Get model prediction (output has shape [TimeSteps, 1])
            output, router_logits, expert_mask = model(x_batch)
            # Take the prediction from the last time step
            last_step_output = output[-1]
            prediction = torch.sigmoid(last_step_output).item()
            
            # Store results
            all_predictions.append(prediction)
            all_true_labels.append(y_test[i])
            
            # Store modality contributions (if available)
            if hasattr(model, 'get_modality_importance'):
                modality_importance = model.get_modality_importance(x_batch)
                modality_contributions.append(modality_importance)
            
            # Store expert usage (if available)
            if expert_mask is not None:
                expert_usages.append(expert_mask.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Binary predictions (threshold 0.5)
    binary_predictions = (all_predictions >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(binary_predictions == all_true_labels)
    
    # Calculate precision, recall, f1 score
    true_positives = np.sum((binary_predictions == 1) & (all_true_labels == 1))
    false_positives = np.sum((binary_predictions == 1) & (all_true_labels == 0))
    false_negatives = np.sum((binary_predictions == 0) & (all_true_labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Return results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'modality_contributions': modality_contributions,
        'expert_usages': expert_usages
    }
    
    return results


def visualize_results(results, model, test_data, output_dir):
    """
    Visualize model results.
    
    Args:
        results: Evaluation results
        model: Trained model
        test_data: Test data
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot ROC curve
    plot_roc_curve(
        results['true_labels'],
        results['predictions'],
        title="Migraine Prediction ROC Curve",
        output_path=os.path.join(output_dir, 'roc_curve.png')
    )
    
    # 2. Plot expert usage
    if results['expert_usages']:
        expert_usage = np.mean(np.vstack(results['expert_usages']), axis=0)
        expert_names = [f"Expert {i+1}" for i in range(len(expert_usage))]
        plot_expert_usage(
            expert_usage,
            expert_names,
            title="Expert Usage in Migraine Prediction",
            output_path=os.path.join(output_dir, 'expert_usage.png')
        )
    
    # 3. Plot modality importance
    if results['modality_contributions']:
        modality_importance = np.mean(np.vstack(results['modality_contributions']), axis=0)
        modality_names = test_data['modalities']
        plot_modality_importance(
            modality_importance,
            modality_names,
            title="Modality Importance for Migraine Prediction",
            output_path=os.path.join(output_dir, 'modality_importance.png')
        )
    
    # 4. Plot prediction timeline
    plt.figure(figsize=(12, 6))
    plt.plot(results['true_labels'], label='Actual Migraines', marker='o', linestyle='--')
    plt.plot(results['predictions'], label='Predicted Risk', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Decision Threshold')
    plt.xlabel('Time')
    plt.ylabel('Migraine Risk / Occurrence')
    plt.title('Migraine Prediction Timeline')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_timeline.png'))
    plt.close()
    
    # 5. Save numerical results
    metrics = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score'])
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data pipeline
    print("Initializing migraine data pipeline...")
    pipeline = MigraineDataPipeline(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        weather_api_key=args.weather_api_key
    )

    # >>> PARSE IMPUTER CONFIG <<<
    imputer_config_dict = None
    if args.imputer_config:
        try:
            imputer_config_dict = json.loads(args.imputer_config)
            print(f"Using imputer config: {imputer_config_dict}")
        except json.JSONDecodeError as e:
            print(f"Error parsing imputer_config JSON: {e}. Using default imputer settings.")
            imputer_config_dict = None
    # >>> END PARSE <<<

    # Run data pipeline
    print("Processing migraine data...")
    fusemoe_data = pipeline.run_full_pipeline(
        location=(args.latitude, args.longitude),
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_horizon=24,  # 24-hour prediction window
        window_size=24,  # 24-hour data window
        imputation_method=args.imputation_method if args.imputation_method != 'none' else None, # Pass None if 'none' specified
        imputer_config=imputer_config_dict # Pass parsed dict
    )
    
    if not fusemoe_data['X']:
        print("Error: No data available for processing. Check data files and paths.")
        return
    
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
    
    # Split data into train/test sets
    print("\nSplitting data into train/test sets...")
    # Use 80% for training, 20% for testing
    num_samples = len(fusemoe_data['X'])
    train_size = int(0.8 * num_samples)
    
    # Create indices for train/test split
    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create train/test datasets
    train_data = {
        'X': [fusemoe_data['X'][i] for i in train_indices],
        'y': [fusemoe_data['y'][i] for i in train_indices],
        'modalities': fusemoe_data['modalities'],
        'features_per_modality': fusemoe_data['features_per_modality']
    }
    
    test_data = {
        'X': [fusemoe_data['X'][i] for i in test_indices],
        'y': [fusemoe_data['y'][i] for i in test_indices],
        'modalities': fusemoe_data['modalities'],
        'features_per_modality': fusemoe_data['features_per_modality']
    }
    
    print(f"Training samples: {len(train_data['X'])}")
    print(f"Testing samples: {len(test_data['X'])}")
    
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
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"\nUsing device: {device}")

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

        # --- Prepare input data as dictionary for optimize_model ---
        num_train_samples = len(train_data['X'])
        # Initialize lists for each modality based on available modalities in the dataset
        train_data_dict = {mod: [] for mod in fusemoe_data['modalities']}

        # Iterate through each training sample dictionary in train_data['X']
        for sample_dict in train_data['X']:
            for mod in fusemoe_data['modalities']:
                if mod in sample_dict:
                    # Append the features tensor for this modality and sample
                    train_data_dict[mod].append(torch.tensor(sample_dict[mod], dtype=torch.float32))
                else:
                    # Handle case where a modality might theoretically be missing for a sample
                    # Append zeros matching the expected feature size for that modality
                    num_features = fusemoe_data['features_per_modality'].get(mod, 0)
                    train_data_dict[mod].append(torch.zeros(num_features, dtype=torch.float32))

        # Stack tensors for each modality to get shape (num_samples, num_features)
        for mod in list(train_data_dict.keys()): # Iterate over keys copy
             if train_data_dict[mod]: # Ensure list is not empty
                try:
                    train_data_dict[mod] = torch.stack(train_data_dict[mod]).to(device)
                except RuntimeError as e:
                    print(f"Error stacking tensors for modality {mod}: {e}")
                    # Option: remove modality if stacking fails (e.g., inconsistent shapes)
                    del train_data_dict[mod]
             else:
                 # Remove modality if no data was collected (should not happen if fusemoe_data['modalities'] is accurate)
                 print(f"Warning: No data found for modality {mod} during PyGMO preparation. Removing.")
                 del train_data_dict[mod]

        # Correctly extract training labels from train_data['y']
        y_train_list = train_data['y']
        # Ensure target tensor shape is likely (num_samples, 1) for BCEWithLogitsLoss compatibility
        y_train_tensor = torch.tensor(y_train_list, dtype=torch.float32).unsqueeze(1).to(device)
        # --- End dictionary preparation ---

        print("Prepared data for PyGMO optimization:")
        for mod, tensor in train_data_dict.items():
            print(f"  - {mod}: {tensor.shape}")
        print(f"  - targets: {y_train_tensor.shape}")

        print("\nStarting PyGMO-enhanced FuseMOE optimization...")
        try:
            migraine_fusemoe.optimize_model(
                train_data=(train_data_dict, y_train_tensor), # Pass data as a tuple to the 'train_data' argument
                expert_algo=args.expert_algorithm,
                gating_algo=args.gating_algorithm,
                expert_pop_size=args.expert_population_size,
                gating_pop_size=args.gating_population_size,
                seed=42, # Use a consistent seed
                device=device
            )
            print("PyGMO Model optimization complete!")
        except Exception as e:
            print(f"Error during PyGMO model optimization: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping PyGMO optimization due to error.")
            # Fallback might be needed here, e.g., standard training or exiting

    else:
        # Standard PyTorch training loop
        print("\nStarting Standard PyTorch Training...")
        num_epochs = 10 # Example: Train for 10 epochs, make this configurable?
        learning_rate = 0.001 # Example learning rate
        batch_size = 16 # Example batch size

        optimizer = optim.Adam(migraine_fusemoe.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification

        # Prepare DataLoader (requires converting list of dicts to tensors)
        # We need to handle the dictionary structure of X
        # Option 1: Custom Dataset class (more robust)
        # Option 2: Simpler approach - iterate through samples directly (easier for now)
        
        migraine_fusemoe.train() # Set model to training mode
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            # Simple iteration without DataLoader for now
            permutation = torch.randperm(len(train_data['X']))
            
            for i in range(0, len(train_data['X']), batch_size):
                indices = permutation[i:i+batch_size]
                batch_X_list = [train_data['X'][idx] for idx in indices]
                batch_y_list = [train_data['y'][idx] for idx in indices]
                
                # Prepare batch input dictionary
                batch_input = {mod: [] for mod in fusemoe_data['modalities']}
                for sample_dict in batch_X_list:
                     for mod in fusemoe_data['modalities']:
                          batch_input[mod].append(torch.tensor(sample_dict[mod], dtype=torch.float32))
                
                # Stack tensors for each modality
                for mod in fusemoe_data['modalities']:
                     batch_input[mod] = torch.stack(batch_input[mod]).to(device)
                     
                batch_y = torch.tensor(batch_y_list, dtype=torch.float32).unsqueeze(1).to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                # Assuming model.forward takes the dictionary input
                outputs, _, _ = migraine_fusemoe(batch_input) 
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
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
    results = evaluate_model(model, test_data)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results, model, test_data, args.output_dir)
    
    # Save the final model
    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), model_path) # Save state_dict instead of whole model
    print(f"Model state_dict saved to {model_path}")
    
    # If using PyGMO, identify and visualize triggers
    if args.use_pygmo:
        print("\nIdentifying potential migraine triggers...")
        # Choose a sample input for trigger identification
        sample_input = {}
        for i, x_dict in enumerate(test_data['X']):
            if test_data['y'][i] == 1:  # Find a positive sample
                for modality, features in x_dict.items():
                    sample_input[modality] = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                break
        
        # If no positive sample found, use the first sample
        if not sample_input:
            for modality, features in test_data['X'][0].items():
                sample_input[modality] = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Identify triggers
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
    
    print("\nMigraine prediction completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 