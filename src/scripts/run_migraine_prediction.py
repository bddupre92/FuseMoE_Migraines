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
import seaborn as sns
import torch
import json
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, SubsetRandomSampler
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, average_precision_score, precision_recall_curve
)
import logging # Make sure logging is imported
import copy # Add copy for deep copying model state during training
import time # Import time for timing logs if needed
import subprocess # <<< Added import
import random # <<< Added import
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline # Alias to avoid clash
from tqdm.auto import tqdm # Import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Optional, Tuple

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

# >>> NEW UNCERTAINTY QUANTIFICATION FUNCTIONS <<<
def enable_dropout(model):
    """Enable dropout during inference by setting model to train mode but not tracking gradients."""
    def _enable_dropout(module):
        if type(module) == nn.Dropout:
            module.train()
    
    model.eval()
    model.apply(_enable_dropout)
    return model

def mc_dropout_predict(model, X_test, n_samples=20, device='cpu'):
    """
    Perform Monte Carlo Dropout predictions to estimate uncertainty.
    
    Args:
        model: The trained model
        X_test: Input data dictionary
        n_samples: Number of forward passes with dropout
        device: Device to run the model on
        
    Returns:
        mean_probs: Mean prediction probabilities
        std_probs: Standard deviation of probabilities (uncertainty measure)
        all_probs: All prediction samples
    """
    model = enable_dropout(model)
    model.to(device)
    
    # Move test data to the correct device
    X_test_dev = {mod: tensor.to(device) for mod, tensor in X_test.items()}
    
    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs, _ = model(X_test_dev)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
    
    # Stack results and compute statistics
    all_probs = np.stack(all_probs, axis=0)  # [n_samples, n_instances, 1]
    all_probs = all_probs.squeeze(axis=2) if all_probs.shape[2] == 1 else all_probs
    
    # Calculate mean and standard deviation across samples
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)
    
    return mean_probs, std_probs, all_probs

def calculate_prediction_intervals(all_probs, confidence=0.95):
    """
    Calculate prediction intervals from Monte Carlo Dropout samples.
    
    Args:
        all_probs: All prediction samples from MC Dropout [n_samples, n_instances]
        confidence: Confidence level for intervals (default: 0.95 for 95%)
        
    Returns:
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
    """
    # Calculate percentiles based on confidence level
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    # Calculate bounds for each instance
    lower_bound = np.percentile(all_probs, lower_percentile, axis=0)
    upper_bound = np.percentile(all_probs, upper_percentile, axis=0)
    
    return lower_bound, upper_bound

def calculate_entropy(probs):
    """
    Calculate predictive entropy as uncertainty measure.
    Higher entropy means higher uncertainty.
    
    Args:
        probs: Probabilities [n_instances]
        
    Returns:
        entropy: Entropy values for each instance
    """
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)
    
    # Binary classification entropy: -p*log(p) - (1-p)*log(1-p)
    entropy = -probs_clipped * np.log(probs_clipped) - (1 - probs_clipped) * np.log(1 - probs_clipped)
    
    return entropy

def plot_prediction_with_uncertainty(y_true, mean_probs, std_probs, indices=None, threshold=0.5,
                                    title="Prediction with Uncertainty", output_path=None):
    """
    Plot predictions with uncertainty bands.
    
    Args:
        y_true: Ground truth labels
        mean_probs: Mean prediction probabilities
        std_probs: Standard deviation of probabilities
        indices: Indices to plot (default: all)
        threshold: Decision threshold (default: 0.5)
        title: Plot title
        output_path: Path to save the plot
    """
    if indices is None:
        # If too many samples, select a reasonable number
        if len(mean_probs) > 50:
            indices = np.sort(np.random.choice(len(mean_probs), 50, replace=False))
        else:
            indices = np.arange(len(mean_probs))
    
    plt.figure(figsize=(15, 6))
    
    # Plot mean predictions with uncertainty bands
    plt.errorbar(
        x=indices,
        y=mean_probs[indices],
        yerr=2*std_probs[indices],  # 2σ for ~95% interval
        fmt='o',
        color='blue',
        ecolor='lightblue',
        elinewidth=3,
        capsize=0,
        alpha=0.7,
        label='Predicted probability with 2σ interval'
    )
    
    # Plot ground truth
    plt.scatter(indices, y_true[indices], color='red', marker='x', s=50, label='Ground truth')
    
    # Add threshold line
    plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, label=f'Decision threshold ({threshold})')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Probability of Migraine')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)  # Add some padding to y-axis
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_uncertainty_vs_error(y_true, mean_probs, uncertainty, threshold=0.5, 
                             title="Uncertainty vs Prediction Error", output_path=None):
    """
    Plot relationship between uncertainty and prediction error.
    
    Args:
        y_true: Ground truth labels
        mean_probs: Mean prediction probabilities
        uncertainty: Uncertainty measure (e.g., std_probs or entropy)
        threshold: Decision threshold
        title: Plot title
        output_path: Path to save the plot
    """
    # Calculate error (absolute difference between prediction and ground truth)
    predicted_class = (mean_probs > threshold).astype(int)
    error = np.abs(y_true - mean_probs)
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        uncertainty, 
        error,
        c=np.abs(predicted_class - y_true),  # Color by misclassification
        cmap='coolwarm',
        alpha=0.7,
        s=50
    )
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Misclassified (1) vs Correct (0)')
    
    # Add best fit line
    z = np.polyfit(uncertainty, error, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(uncertainty), p(np.sort(uncertainty)), "r--", alpha=0.8)
    
    # Calculate correlation
    corr = np.corrcoef(uncertainty, error)[0,1]
    
    plt.xlabel('Uncertainty (std dev / entropy)')
    plt.ylabel('Absolute Error')
    plt.title(f"{title}\nCorrelation: {corr:.3f}")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def identify_uncertain_predictions(mean_probs, uncertainty, threshold=0.5, uncertainty_threshold=None):
    """
    Identify uncertain predictions for further analysis.
    
    Args:
        mean_probs: Mean prediction probabilities
        uncertainty: Uncertainty measure (std_probs or entropy)
        threshold: Decision threshold for classification
        uncertainty_threshold: Threshold for high uncertainty (if None, use 75th percentile)
        
    Returns:
        Dictionary containing indices and information about uncertain predictions
    """
    if uncertainty_threshold is None:
        # Use the 75th percentile as default uncertainty threshold
        uncertainty_threshold = np.percentile(uncertainty, 75)
    
    # Find predictions close to decision boundary and with high uncertainty
    boundary_distance = np.abs(mean_probs - threshold)
    close_to_boundary = boundary_distance < 0.15
    high_uncertainty = uncertainty > uncertainty_threshold
    
    # Different categories of uncertain predictions
    uncertain_indices = {
        'high_uncertainty': np.where(high_uncertainty)[0],
        'close_to_boundary': np.where(close_to_boundary)[0],
        'critical': np.where(np.logical_and(high_uncertainty, close_to_boundary))[0]
    }
    
    # Sort by uncertainty for each category
    for category, indices in uncertain_indices.items():
        if len(indices) > 0:
            sorted_idx = indices[np.argsort(uncertainty[indices])[::-1]]  # Sort by decreasing uncertainty
            uncertain_indices[category] = sorted_idx
    
    return uncertain_indices
# >>> END UNCERTAINTY QUANTIFICATION FUNCTIONS <<<

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
    parser.add_argument('--window_size', type=int, default=12, # Updated default from 8 to 12 (range 12-24)
                      help='Lookback window size in hours for features.')
    parser.add_argument('--prediction_horizon', type=int, default=6,  # Updated default from 24
                      help='How many hours ahead to predict.')
    parser.add_argument('--step_size', type=int, default=1,
                      help='Step size for sliding window (default: 1 hour)')
    
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
                      choices=['knn', 'iterative', 'none', 'autoencoder'], # <<< Added 'autoencoder'
                      help='Method to use for imputing missing values ("knn", "iterative", "autoencoder", "none"). Use "none" to skip imputation.')
    parser.add_argument('--imputer_config', type=str, default=None,
                      help="""JSON string with configuration for the imputer (e.g., '{"n_neighbors": 3}').""")
    # >>> END IMPUTATION ARGUMENTS <<<
    
    # >>> ADDED CROSS-VALIDATION ARGUMENTS <<<
    parser.add_argument('--cv', type=int, default=5,
                      help='Number of folds for cross-validation. Use 1 for standard train/test split.')
    parser.add_argument('--cv_strategy', type=str, default='stratified',
                      choices=['stratified', 'kfold', 'time', 'groupkfold', 'stratifiedgroupkfold'], # <<< Added group options
                      help='Cross-validation strategy: stratified, kfold, time, groupkfold (keeps patients together), stratifiedgroupkfold.')
    parser.add_argument('--cv_shuffle', action='store_true',
                      help='Whether to shuffle data before non-temporal cross-validation splits.')
    # >>> END CROSS-VALIDATION ARGUMENTS <<<
    
    # >>> ADDED DATA BALANCING ARGUMENTS <<<
    parser.add_argument('--balance_method', type=str, default='none',
                      choices=['smote', 'random_over', 'random_under', 'none'],
                      help='Method to use for balancing the dataset ("smote", "random_over", "random_under", "none").')
    parser.add_argument('--sampling_ratio', type=float, default=0.5,
                      help='Target ratio of minority to majority class (for oversampling) or majority to minority (for undersampling).')
    # >>> END DATA BALANCING ARGUMENTS <<<
    
    # >>> DEVELOPMENT MODE PARAMETERS <<<
    parser.add_argument('--dev_mode', action='store_true',
                      help='Enable development mode for faster execution (overrides other parameters)')
    parser.add_argument('--dev_epochs', type=int, default=3,
                      help='Number of training epochs in development mode (default: 3)')
    parser.add_argument('--dev_batch_size', type=int, default=8,
                      help='Batch size in development mode (default: 8)')
    parser.add_argument('--skip_visualizations', action='store_true',
                      help='Skip generating visualizations to speed up execution')
    # >>> END DEVELOPMENT MODE PARAMETERS <<<
    
    # >>> ADDED UNCERTAINTY QUANTIFICATION ARGUMENTS <<<
    parser.add_argument('--uncertainty', action='store_true',
                      help='Enable uncertainty quantification with Monte Carlo Dropout')
    parser.add_argument('--dropout_rate', type=float, default=0.25,
                      help='Dropout rate to use for uncertainty quantification')
    parser.add_argument('--mc_samples', type=int, default=20,
                      help='Number of samples to generate with Monte Carlo Dropout')
    parser.add_argument('--uncertainty_threshold', type=float, default=None,
                      help='Custom threshold for high uncertainty (default: 75th percentile)')
    # >>> END UNCERTAINTY QUANTIFICATION ARGUMENTS <<<
    
    # >>> ADDED TRAINING PROCESS ARGUMENTS <<< 
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                      help='Initial learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training.')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                      help='Patience for early stopping based on validation loss.')
    parser.add_argument('--validation_split', type=float, default=0.2,
                      help='Fraction of training data to use for validation during training.')
    # >>> END TRAINING PROCESS ARGUMENTS <<< 
    
    # >>> ADDED THRESHOLD OPTIMIZATION ARGUMENTS <<<
    parser.add_argument('--threshold_search', action='store_true',
                      help='Enable automatic threshold optimization based on validation data')
    parser.add_argument('--optimize_metric', type=str, default='balanced_accuracy',
                      choices=['balanced_accuracy', 'f1', 'mcc', 'precision', 'recall'],
                      help='Metric to optimize when searching for best threshold')
    parser.add_argument('--class_weight', type=str, default=None,
                      choices=['balanced', 'none'],
                      help='Class weighting strategy for handling imbalanced data')
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility") # <<< Added seed argument
    # >>> ADDED LOGGING ARGUMENTS <<<                     
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log_file", type=str, default="prediction.log", help="Name of the log file in the output directory")
    # >>> END LOGGING ARGUMENTS <<<
    # >>> END THRESHOLD OPTIMIZATION ARGUMENTS <<<
    
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


def evaluate_model(model, X_test, y_test, device, threshold=0.5, optimize_threshold=False, optimize_metric='balanced_accuracy'):
    """
    Evaluate the trained FuseMoE model on test data.
    
    Args:
        model: Trained MigraineFuseMoE model.
        X_test: Dictionary of test data tensors for each modality.
        y_test: Test target tensor.
        device: Device to run evaluation on.
        threshold: Decision threshold for binary classification.
        optimize_threshold: Whether to search for optimal threshold.
        optimize_metric: Metric to optimize if optimize_threshold is True.
        
    Returns:
        Dictionary containing evaluation metrics, predicted probabilities, and true labels.
    """
    model.eval()
    model.to(device)
    
    # --- Convert input data to tensors and move to device --- #
    X_test_dev = {}
    for mod, data_array in X_test.items():
        if isinstance(data_array, np.ndarray):
            X_test_dev[mod] = torch.tensor(data_array, dtype=torch.float32).to(device)
        elif isinstance(data_array, torch.Tensor):
            X_test_dev[mod] = data_array.to(device)
        else:
            # Handle other potential types or raise error
            logging.warning(f"Unexpected data type for modality '{mod}': {type(data_array)}. Attempting conversion.")
            try:
                X_test_dev[mod] = torch.tensor(data_array, dtype=torch.float32).to(device)
            except Exception as e:
                logging.error(f"Could not convert modality '{mod}' data to tensor: {e}")
                # Decide how to handle this error, e.g., skip evaluation or raise
                raise TypeError(f"Invalid data type for modality '{mod}': {type(data_array)}")
                
    # Ensure y_test is a tensor and on the correct device
    if isinstance(y_test, np.ndarray):
        y_test_dev = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    elif isinstance(y_test, torch.Tensor):
        y_test_dev = y_test.to(device) # Assume it might need moving
        if y_test_dev.dim() == 1: # Ensure it has batch and feature dim
            y_test_dev = y_test_dev.unsqueeze(1)
    else:
        logging.error(f"Invalid type for y_test: {type(y_test)}")
        raise TypeError(f"Invalid type for y_test: {type(y_test)}")
    # --- End Data Conversion --- #

    with torch.no_grad():
        outputs, gates = model(X_test_dev)
        raw_outputs = outputs.cpu().numpy()  # Store raw outputs before sigmoid
        probs = torch.sigmoid(outputs).cpu().numpy() # Probabilities needed for AUC
        y_true = y_test_dev.cpu().numpy().astype(int).flatten() # Flatten y_true here
        
        # Find optimal threshold if requested
        if optimize_threshold and len(np.unique(y_true)) > 1:
            logging.debug(f"Finding optimal threshold by optimizing {optimize_metric}...")
            threshold = find_optimal_threshold(y_true, probs.flatten(), optimize_metric) # Pass flattened arrays
            logging.debug(f"Optimal threshold found: {threshold:.4f}")
        
        predicted = (probs > threshold).astype(int).flatten() # Flatten predictions
        
        # Log probabilities distribution statistics for debugging
        if len(probs) > 0:
            logging.debug(f"Probability stats - Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")
            if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):  # Both classes present
                pos_probs = probs[y_true == 1]
                neg_probs = probs[y_true == 0]
                logging.debug(f"Class 1 probabilities - Min: {pos_probs.min():.4f}, Max: {pos_probs.max():.4f}, Mean: {pos_probs.mean():.4f}")
                logging.debug(f"Class 0 probabilities - Min: {neg_probs.min():.4f}, Max: {neg_probs.max():.4f}, Mean: {neg_probs.mean():.4f}")
                # Calculate separation between classes
                logging.debug(f"Probability separation: {pos_probs.mean() - neg_probs.mean():.4f}")

    # Standard metrics
    accuracy = accuracy_score(y_true, predicted)
    precision = precision_score(y_true, predicted, zero_division=0)
    recall = recall_score(y_true, predicted, zero_division=0)
    f1 = f1_score(y_true, predicted, zero_division=0)
    
    # Calculate AUC with extra checks
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError as e:
        # Handle cases where AUC cannot be computed (e.g., only one class present)
        logging.warning(f"Could not compute AUC: {e}")
        auc = float('nan') # Assign NaN if AUC calculation fails
    
    # Additional metrics for imbalanced data
    try:
        # Balanced accuracy - average of sensitivity and specificity
        balanced_accuracy = balanced_accuracy_score(y_true, predicted)
        
        # Matthews correlation coefficient - handles imbalanced data better
        mcc = matthews_corrcoef(y_true, predicted)
        
        # Precision-Recall AUC - better for imbalanced datasets
        pr_auc = average_precision_score(y_true, probs)
    except Exception as e:
        logging.warning(f"Warning: Could not compute some advanced metrics: {e}")
        balanced_accuracy = float('nan')
        mcc = float('nan')
        pr_auc = float('nan')
    
    # Calculate confusion matrix for visualization
    confusion = confusion_matrix(y_true, predicted)
    
    # Calculate class distribution for reference
    class_counts = np.bincount(y_true.flatten().astype(int))
    class_distribution = {
        "total_samples": len(y_true),
        "class_0_count": class_counts[0] if len(class_counts) > 0 else 0,
        "class_1_count": class_counts[1] if len(class_counts) > 1 else 0,
    }
    if class_distribution["total_samples"] > 0:
        class_distribution["class_0_percent"] = class_distribution["class_0_count"] / class_distribution["total_samples"]
        class_distribution["class_1_percent"] = class_distribution["class_1_count"] / class_distribution["total_samples"]
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        # Additional metrics
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "confusion_matrix": confusion.tolist(),
        "class_distribution": class_distribution
    }
    
    # Optional: Get expert usage if needed for analysis
    expert_usage = model.get_expert_usage()
    if expert_usage is not None:
        metrics['expert_usage'] = expert_usage
        
    return metrics, probs, raw_outputs, y_true # Return probs, raw_outputs and y_true for further analysis

def find_optimal_threshold(y_true, probs, metric='balanced_accuracy'):
    """
    Find the optimal decision threshold for classification based on specified metric.
    
    Args:
        y_true: True labels
        probs: Predicted probabilities
        metric: Metric to optimize ('balanced_accuracy', 'f1', 'mcc', etc.)
        
    Returns:
        optimal_threshold: The threshold that maximizes the specified metric
    """
    thresholds = np.linspace(0.01, 0.99, 99)  # Test 99 threshold values
    best_score = 0
    optimal_threshold = 0.5  # Default
    
    # Map of metric names to scoring functions
    metric_functions = {
        'balanced_accuracy': balanced_accuracy_score,
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)
    }
    
    scoring_function = metric_functions.get(metric.lower(), balanced_accuracy_score)
    
    # Logging moved to evaluate_model where it's called
    
    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        score = scoring_function(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            optimal_threshold = threshold
    
    # Logging moved to evaluate_model where it's called
    return optimal_threshold

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
        model: Trained FuseMoE model
        patient_data: Dictionary of patient-specific data
        iterations: Number of adaptation iterations
        
    Returns:
        Adapted model
    """
    logging.info(f"Adapting model to patient-specific patterns ({iterations} iterations)...")
    
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
        logging.info(f"  Adaptation iteration {i+1}/{iterations}")
        
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
    
    logging.info("Patient adaptation complete.")
    return model


def calibrate_probabilities(y_train, probs_train, probs_test, calibration_method='isotonic'):
    """
    Calibrate probability outputs using sklearn's CalibratedClassifierCV.
    
    Args:
        y_train: Training labels (ground truth)
        probs_train: Training probabilities from model
        probs_test: Test probabilities to calibrate
        calibration_method: Method to use for calibration (isotonic or sigmoid/Platt)
        
    Returns:
        calibrated_probs: Calibrated probabilities
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    
    # Convert to numpy arrays if they're not already
    y_train_np = np.array(y_train).reshape(-1)
    probs_train_np = np.array(probs_train).reshape(-1)
    probs_test_np = np.array(probs_test).reshape(-1)
    
    # Print statistics before calibration
    logging.debug(f"Before calibration ({calibration_method}) - Train prob mean: {probs_train_np.mean():.4f}, Test prob mean: {probs_test_np.mean():.4f}")
    
    try:
        if calibration_method == 'isotonic':
            # Isotonic regression (non-parametric, piecewise constant)
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs_train_np, y_train_np)
            calibrated_probs = calibrator.transform(probs_test_np)
            
        elif calibration_method == 'platt' or calibration_method == 'sigmoid':
            # Platt scaling (logistic regression)
            # Reshape for sklearn's LogisticRegression
            calibrator = LogisticRegression(C=1.0, solver='lbfgs')
            calibrator.fit(probs_train_np.reshape(-1, 1), y_train_np)
            calibrated_probs = calibrator.predict_proba(probs_test_np.reshape(-1, 1))[:, 1]
            
        else:
            logging.warning(f"Unknown calibration method: {calibration_method}. Using original probabilities.")
            return probs_test_np
        
        # Print statistics after calibration
        logging.debug(f"After calibration ({calibration_method}) - Test prob mean: {calibrated_probs.mean():.4f}")
        
        return calibrated_probs
        
    except Exception as e:
        logging.warning(f"Warning: Probability calibration failed: {e}")
        logging.warning("Using original probabilities instead.")
        return probs_test_np


def weighted_bce_loss(outputs, targets, pos_weight=None):
    """
    Weighted binary cross entropy loss to handle class imbalance.
    
    Args:
        outputs: Model predictions (logits, before sigmoid)
        targets: Ground truth labels
        pos_weight: Weight for positive class. If None, will be calculated from targets.
        
    Returns:
        Weighted BCE loss
    """
    # If pos_weight not provided, calculate it based on class distribution
    if pos_weight is None:
        # Count positive and negative samples
        num_pos = torch.sum(targets)
        num_neg = targets.size(0) - num_pos
        
        if num_pos > 0 and num_neg > 0:
            # Weight based on inverse frequency (higher weight for minority class)
            pos_weight = num_neg / num_pos
            
            # Clamp weights to reasonable values
            pos_weight = torch.clamp(pos_weight, min=0.1, max=10.0)
        else:
            pos_weight = torch.tensor(1.0).to(targets.device)
    
    # Use PyTorch's BCEWithLogitsLoss with pos_weight
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(outputs, targets)


# --- Helper function for index-based balancing (Option 1) ---
def _balance_indices(X_dict, y_np, method, sampling_ratio, random_state=None):
    """Helper function for index-based random over/under sampling."""
    import numpy as np # Ensure numpy is imported locally if needed
    import logging     # Ensure logging is available
    
    unique_classes, counts = np.unique(y_np, return_counts=True)
    if len(unique_classes) <= 1:
        logging.warning("_balance_indices: Only one class present, skipping balancing.")
        return X_dict, y_np
        
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    
    pos_indices = np.where(y_np == minority_class)[0] # Assuming minority is positive (class 1), adjust if needed
    neg_indices = np.where(y_np == majority_class)[0] # Assuming majority is negative (class 0)
    
    # Ensure minority/majority assigned correctly if class 0 is minority
    if counts[0] < counts[1]:
        pos_indices, neg_indices = neg_indices, pos_indices # Swap if class 0 is minority
        
    logging.info(f"_balance_indices ({method}): Original counts - Minority ({minority_class}): {len(pos_indices)}, Majority ({majority_class}): {len(neg_indices)}")

    rng = np.random.default_rng(random_state) # Use random state for reproducibility
    
    if method == 'random_over':
        n_target = int(len(neg_indices) * sampling_ratio) # Target number of minority samples relative to majority
        if len(pos_indices) == 0: # Handle case with no minority samples
            logging.warning("Cannot oversample: No minority class samples found.")
            combined_indices = neg_indices
        elif len(pos_indices) < n_target:
            logging.debug(f"  Oversampling minority to {n_target}")
            pos_indices_oversampled = rng.choice(pos_indices, n_target, replace=True)
            combined_indices = np.concatenate([pos_indices_oversampled, neg_indices])
        else:
            # Minority count already meets or exceeds target ratio
            logging.debug("  Minority count meets target ratio, no oversampling needed.")
            combined_indices = np.concatenate([pos_indices, neg_indices]) # Keep all original samples
            
    elif method == 'random_under':
        n_target = int(len(pos_indices) / sampling_ratio) if sampling_ratio > 0 else 0 # Target number of majority samples relative to minority
        if len(neg_indices) == 0: # Handle case with no majority samples
             logging.warning("Cannot undersample: No majority class samples found.")
             combined_indices = pos_indices
        elif len(neg_indices) > n_target:
            logging.debug(f"  Undersampling majority to {n_target}")
            neg_indices_undersampled = rng.choice(neg_indices, n_target, replace=False)
            combined_indices = np.concatenate([pos_indices, neg_indices_undersampled])
        else:
            # Majority count already meets or exceeds target ratio
            logging.debug("  Majority count meets target ratio, no undersampling needed.")
            combined_indices = np.concatenate([pos_indices, neg_indices]) # Keep all original samples
            
    else: # Should not happen if called from balance_dataset, but handle defensively
        logging.warning(f"_balance_indices: Unknown method '{method}'. Returning original data.")
        return X_dict, y_np
        
    # Shuffle the combined indices
    rng.shuffle(combined_indices)
    
    # Create new X_dict and y using the selected indices
    # Check if X_dict is the expected dictionary structure
    if not isinstance(X_dict, dict):
        logging.error(f"_balance_indices: Expected X_dict to be a dictionary, but got {type(X_dict)}. Cannot balance.")
        # Decide how to handle - return original or raise error
        return X_dict, y_np # Returning original to avoid crashing

    try:
        # Ensure indices are valid before slicing
        max_index = max(combined_indices) if len(combined_indices) > 0 else -1
        valid_indices = True
        for modality, data_array in X_dict.items():
            if max_index >= data_array.shape[0]:
                 logging.error(f"_balance_indices: Index {max_index} out of bounds for modality '{modality}' (shape {data_array.shape}).")
                 valid_indices = False
                 break
        
        if valid_indices:
            X_balanced = {modality: data_array[combined_indices] for modality, data_array in X_dict.items()}
            y_balanced = y_np[combined_indices]
        else:
             logging.error("_balance_indices: Invalid indices detected. Returning original data.")
             return X_dict, y_np # Fallback

    except IndexError as e:
         logging.error(f"_balance_indices: IndexError during slicing: {e}. Indices: {combined_indices[:10]}..., Max Index: {max_index}")
         # Log shapes for debugging
         for mod, data in X_dict.items():
             logging.error(f"  Shape of {mod}: {data.shape}")
         logging.error("Returning original data due to slicing error.")
         return X_dict, y_np # Fallback
    except Exception as e:
         logging.error(f"_balance_indices: Unexpected error during slicing: {e}")
         # Log shapes for debugging
         for mod, data in X_dict.items():
             logging.error(f"  Shape of {mod}: {data.shape}")
         logging.error("Returning original data due to unexpected error.")
         return X_dict, y_np # Fallback

    # Report new class distribution
    new_counts = np.bincount(y_balanced.astype(int))
    logging.info(f"_balance_indices: New distribution after {method} - Class 0: {new_counts[0] if len(new_counts)>0 else 0}, Class 1: {new_counts[1] if len(new_counts)>1 else 0}")
    
    return X_balanced, y_balanced
# --- ------------------------------------------------------- ---

# --- SMOTE Implementation for Dictionary Inputs (Priority 1 Fix) ---
def smote_dictionary_data(X_dict, y_np, sampling_ratio=0.6, random_state=None):
    """SMOTE implementation for dictionary-based multimodal data."""
    import numpy as np
    from imblearn.over_sampling import SMOTE
    import logging
    
    logging.info(f"Applying SMOTE for dictionary input with ratio: {sampling_ratio}")
    
    # 1. Find minority class indices
    unique_classes, counts = np.unique(y_np, return_counts=True)
    if len(unique_classes) <= 1:
        logging.warning("SMOTE skipped: Only one class present.")
        return X_dict, y_np
        
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    minority_indices = np.where(y_np == minority_class)[0]
    majority_indices = np.where(y_np == majority_class)[0]
    
    logging.debug(f"SMOTE: Minority class {minority_class} ({len(minority_indices)} samples), Majority class {majority_class} ({len(majority_indices)} samples)")

    # 2. Concatenate all features for SMOTE
    all_features = []
    feature_dims = {}
    current_pos = 0
    ordered_modalities = sorted(X_dict.keys()) # Ensure consistent order
    
    # First pass: Calculate dimensions and check shapes
    num_samples = len(y_np)
    expected_sample_count = X_dict[ordered_modalities[0]].shape[0] if ordered_modalities else 0
    if num_samples != expected_sample_count:
         logging.error(f"SMOTE Error: Sample count mismatch! y has {num_samples}, but first modality has {expected_sample_count}. Aborting SMOTE.")
         return X_dict, y_np # Return original data
         
    for mod in ordered_modalities:
        data = X_dict[mod]
        if data.shape[0] != num_samples:
             logging.error(f"SMOTE Error: Sample count mismatch in modality '{mod}'. Expected {num_samples}, got {data.shape[0]}. Aborting SMOTE.")
             return X_dict, y_np # Return original data
             
        orig_shape = data.shape[1:]  # Window, features
        flat_size = np.prod(orig_shape)
        feature_dims[mod] = (current_pos, current_pos + flat_size, orig_shape)
        current_pos += flat_size
        
    total_flat_features = current_pos
    logging.debug(f"SMOTE: Total flattened features per sample: {total_flat_features}")
    
    # Second pass: Concatenate
    all_features = np.zeros((num_samples, total_flat_features), dtype=np.float32) # Pre-allocate
    for i in range(num_samples):
        current_col = 0
        for mod in ordered_modalities:
            start_idx, end_idx, _ = feature_dims[mod]
            all_features[i, start_idx:end_idx] = X_dict[mod][i].flatten()

    logging.debug(f"SMOTE: Concatenated feature array shape: {all_features.shape}")

    # 3. Apply SMOTE to concatenated features
    try:
        # Adjust sampling_strategy for imblearn > 0.10 compatibility if needed
        # For now, assume older style or float ratio works
        smote = SMOTE(sampling_strategy=sampling_ratio, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(all_features, y_np)
        logging.info(f"SMOTE: Resampled data shape: {X_resampled.shape}, Resampled labels shape: {y_resampled.shape}")
        # Log new class distribution
        new_counts = np.bincount(y_resampled.astype(int))
        logging.info(f"SMOTE: New distribution - Class 0: {new_counts[0] if len(new_counts)>0 else 0}, Class 1: {new_counts[1] if len(new_counts)>1 else 0}")

    except ValueError as e:
        logging.error(f"SMOTE Error during fit_resample: {e}")
        logging.error("This might happen if the minority class size is too small for the number of neighbors (k_neighbors). Try reducing sampling_ratio or checking data.")
        return X_dict, y_np # Return original data on SMOTE error
    except Exception as e:
        logging.error(f"Unexpected error during SMOTE: {e}")
        return X_dict, y_np

    # 4. Split back into dictionary structure
    num_new_samples = len(y_resampled)
    new_X_dict = {}
    
    # Reconstruct dictionary structure using calculated feature_dims
    for mod in ordered_modalities:
        start_idx, end_idx, orig_shape = feature_dims[mod]
        mod_features = X_resampled[:, start_idx:end_idx]
        # Reshape back to original dimensions (samples, window, features)
        try:
            new_X_dict[mod] = mod_features.reshape((num_new_samples,) + orig_shape)
        except ValueError as reshape_error:
             logging.error(f"SMOTE Error: Cannot reshape modality '{mod}' from {mod_features.shape} to {(num_new_samples,) + orig_shape}. Error: {reshape_error}")
             return X_dict, y_np # Return original if reshape fails
             
    logging.debug(f"SMOTE: Successfully reconstructed dictionary with {len(new_X_dict)} modalities.")
    
    return new_X_dict, y_resampled
# --- ------------------------------------------------------------ ---

def balance_dataset(X, y, method='smote', sampling_ratio=0.5, random_state=None):
    """
    Balance the dataset to address class imbalance using various methods.
    Handles dictionary-based multi-modal input (X).
    
    Args:
        X: Input features (dictionary where keys are modalities and values are numpy arrays
           [samples, window, features] or similar).
        y: Target labels (numpy array or list).
        method: Resampling method - 'smote', 'random_over', 'random_under', or 'none'.
        sampling_ratio: Target ratio for balancing.
        random_state: Random seed for reproducible results.
    
    Returns:
        X_resampled, y_resampled: Balanced features (dictionary) and labels (numpy array).
    """
    import numpy as np # Ensure numpy is available
    import logging     # Ensure logging is available
    
    # Ensure y is a numpy array
    y_np = np.array(y)
    
    # Check if X is the expected dictionary structure
    if not isinstance(X, dict):
        logging.error(f"balance_dataset: Expected X to be a dictionary, but got {type(X)}. Cannot balance.")
        # Decide how to handle - return original or raise error
        return X, y_np # Returning original to avoid crashing
        
    # Calculate original class distribution for logging
    unique_classes, counts = np.unique(y_np, return_counts=True)
    if len(unique_classes) <= 1 or (len(counts) > 1 and counts[0] == counts[1]):
        logging.info(f"Balancing skipped: Data already balanced or only one class present. Method requested: {method}")
        return X, y_np # Return original if balanced or single class

    logging.info(f"balance_dataset: Balancing requested with method '{method}'. Original counts: {dict(zip(unique_classes, counts))}")
    
    # --- Use the method passed as argument --- #
    requested_method = method.lower()
    # --------------------------------------- #
    
    if requested_method == 'none':
        logging.info("No balancing applied as method is 'none'.")
        return X, y_np
        
    elif requested_method in ['random_over', 'random_under']:
        logging.info(f"Applying index-based balancing: {requested_method}")
        if isinstance(X, dict):
             return _balance_indices(X, y_np, requested_method, sampling_ratio, random_state)
        else:
             logging.error("balance_dataset: X is not a dictionary, cannot apply index balancing.")
             return X, y_np
             
    elif requested_method == 'smote':
        # --- Use the new SMOTE function for dictionary inputs --- 
        logging.info("Applying SMOTE for dictionary input structure...")
        if isinstance(X, dict):
            return smote_dictionary_data(X, y_np, sampling_ratio, random_state)
        else:
            logging.error("balance_dataset: X is not a dictionary, cannot apply SMOTE.")
            return X, y_np
        # -----------------------------------------------------------
        
    else:
        logging.warning(f"Unknown balancing method '{method}'. Returning original data.")
        return X, y_np


def train_with_early_stopping(model, train_data_dict, y_train_tensor, val_data_dict, y_val_tensor, 
                         num_epochs, batch_size, learning_rate, pos_weight, device, patience=5):
    """
    Train a model with early stopping based on validation loss.
    
    Args:
        model: The model to train
        train_data_dict: Dictionary of training data by modality
        y_train_tensor: Training labels
        val_data_dict: Dictionary of validation data by modality
        y_val_tensor: Validation labels
        num_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer.
        pos_weight: Weight for positive class in loss function
        device: Device to train on (cuda or cpu)
        patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        model: Trained model
        training_history: Dictionary with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # <<< Use LR arg
    # --- Replace ReduceLROnPlateau with CosineAnnealingLR --- #
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
    #                                                        patience=patience//2, # Reduce LR if no improvement for half the early stopping patience
    #                                                        factor=0.1, 
    #                                                        verbose=False) # Quieten scheduler verbosity
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,      # Number of iterations for the first restart
        eta_min=learning_rate * 0.01 # Minimum learning rate
    )
    # --- --------------------------------------------- ---
    
    # Initialize best validation loss and patience counter
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None # Initialize best model state
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    model.to(device) # Ensure model is on the correct device
    
    # Move data to device ONCE before the loop
    train_data_dev = {mod: tensor.to(device) for mod, tensor in train_data_dict.items()}
    y_train_dev = y_train_tensor.to(device)
    val_data_dev = {mod: tensor.to(device) for mod, tensor in val_data_dict.items()}
    y_val_dev = y_val_tensor.to(device)

    # --- Wrap epoch loop with tqdm ---
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs (Early Stopping)", unit="epoch", leave=False)
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        epoch_loss = 0
        processed_batches = 0
        
        # --- Wrap batch loop with tqdm ---
        num_batches = (y_train_dev.shape[0] + batch_size - 1) // batch_size
        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs} Train", unit="batch", leave=False)
        for i in batch_pbar:
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = {mod: tensor[start_idx:end_idx] for mod, tensor in train_data_dev.items()}
            batch_y = y_train_dev[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = weighted_bce_loss(outputs, batch_y, pos_weight=pos_weight)
            loss.backward()
            # --- Add Gradient Clipping --- #
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # --- --------------------- --- #
            optimizer.step()
            
            epoch_loss += loss.item() * batch_y.shape[0]
            processed_batches += batch_y.shape[0]
            
            # Update batch progress bar postfix
            if processed_batches > 0:
                 batch_pbar.set_postfix({'Running Loss': f'{epoch_loss / processed_batches:.4f}'})
        
        avg_train_loss = epoch_loss / y_train_dev.shape[0]
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_outputs, _ = model(val_data_dev)
            val_loss = weighted_bce_loss(val_outputs, y_val_dev, pos_weight=pos_weight).item()
            history['val_loss'].append(val_loss)
        
        # --- Step the LR scheduler --- #
        scheduler.step(val_loss)
        # --------------------------- #
        
        # Update epoch progress bar postfix
        epoch_pbar.set_postfix({'Train Loss': f'{avg_train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state here if needed
            logging.debug(f"  -> New best validation loss: {best_val_loss:.4f}")
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            logging.debug(f"  -> Validation loss did not improve for {patience_counter} epoch(s).")
            if patience_counter >= patience and epoch > 2:  # Ensure we train for at least 3 epochs
                logging.info(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
                # Restore best model
                if best_model_state:
                    model.load_state_dict(best_model_state)
                epoch_pbar.close() # Close the epoch progress bar
                break # Exit epoch loop
    
    if not epoch_pbar.disable: # Ensure progress bar is closed if loop finished normally
        epoch_pbar.close()
        
    # Load best model state if early stopping happened and state was saved
    if best_model_state and patience_counter >= patience:
         model.load_state_dict(best_model_state)
         logging.info("Loaded best model state due to early stopping.")
    elif best_model_state: # If loop finished normally, ensure best state is loaded
         model.load_state_dict(best_model_state)
         logging.info("Loaded best model state from training.")

    return model, history


def train_model(model, train_loader, criterion, optimizer, args):
    """
    Train the FuseMoE model.
    
    Args:
        model: The FuseMoE model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        args: Command line arguments
    
    Returns:
        model: Trained model
        train_losses: List of training losses
    """
    model.train()
    device = next(model.parameters()).device
    train_losses = []
    
    # Enable early stopping
    best_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 5  # Number of epochs to wait for improvement
    best_model_state = None

    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_idx, (batch_data, targets) in enumerate(train_loader):
            # Move targets to device
            targets = targets.to(device)
            
            # Process each modality in the batch
            batch_data_dict = {}
            for modality, data in batch_data.items():
                batch_data_dict[modality] = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, expert_outputs = model(batch_data_dict)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            early_stop_counter = 0
            # Save best model state
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
        
        # Print progress
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Check for early stopping
        if early_stop_counter >= early_stop_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs during standard training.")
            break
    
    # Load best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses

def evaluate_model_with_uncertainty(model, test_loader, criterion, device, args=None):
    """
    Evaluate the FuseMoE model, potentially with uncertainty estimation.
    
    Args:
        model: The trained FuseMoE model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on
        args: Command line arguments containing uncertainty settings
    
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    # Default configuration if args is None
    if args is None:
        class DefaultArgs:
            def __init__(self):
                self.uncertainty = False
                self.mc_samples = 20
                self.threshold = 0.5
                
        args = DefaultArgs()
        
    # Set model to evaluation mode
    model.eval()
    
    # Without Monte Carlo Dropout - Standard evaluation
    if not args.uncertainty:
        test_loss = 0
        all_targets = []
        all_outputs = []
        all_expert_outputs = []
        
        with torch.no_grad():
            for batch_data, targets in test_loader:
                # Move targets to device
                targets = targets.to(device)
                
                # Process each modality in the batch
                batch_data_dict = {}
                for modality, data in batch_data.items():
                    batch_data_dict[modality] = data.to(device)
                
                # Forward pass
                outputs, expert_outputs = model(batch_data_dict)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # Store predictions and targets
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                all_expert_outputs.append([expert.cpu().numpy() for expert in expert_outputs])
                
        # Concatenate results
        y_true = np.concatenate(all_targets)
        y_pred_proba = np.concatenate(all_outputs)
        y_pred = (y_pred_proba > args.threshold).astype(int)
        
        # Calculate standard metrics
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Add loss to metrics
        metrics['loss'] = test_loss / len(test_loader)
        
        # Bundle outputs
        results = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        
        return results
    
    # With Monte Carlo Dropout - Uncertainty estimation
    else:
        # Collect all test data for batch processing with MC Dropout
        all_batch_data = {}
        all_targets = []
        
        for batch_data, targets in test_loader:
            # Collect targets
            all_targets.append(targets.cpu().numpy())
            
            # Collect data for each modality
            for modality, data in batch_data.items():
                if modality not in all_batch_data:
                    all_batch_data[modality] = []
                all_batch_data[modality].append(data)
        
        # Concatenate all data
        X_test = {modality: torch.cat(tensors, dim=0) for modality, tensors in all_batch_data.items()}
        y_true = np.concatenate(all_targets)
        
        # Get uncertainty predictions
        mean_probs, std_probs, all_probs = mc_dropout_predict(
            model, 
            X_test, 
            n_samples=args.mc_samples, 
            device=device
        )
        
        # Calculate prediction intervals
        lower_bound, upper_bound = calculate_prediction_intervals(all_probs, confidence=0.95)
        
        # Calculate entropy as alternative uncertainty measure
        entropy = calculate_entropy(mean_probs)
        
        # Calculate standard prediction
        y_pred = (mean_probs > args.threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, mean_probs)
        
        # Identify uncertain predictions
        uncertain_predictions = identify_uncertain_predictions(
            mean_probs, 
            std_probs, 
            threshold=args.threshold,
            uncertainty_threshold=args.uncertainty_threshold
        )
        
        # Bundle results
        results = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred_proba': mean_probs,
            'y_pred': y_pred,
            'uncertainty': {
                'std_probs': std_probs,
                'entropy': entropy,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'uncertain_indices': uncertain_predictions,
                'all_probs': all_probs
            }
        }
        
        return results

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else None,
        'average_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else None,
    }
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Calculate specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics

class ModalityDataset(Dataset):
    """Dataset class for handling multi-modal data dictionaries."""
    
    def __init__(self, X_dict, y):
        """
        Initialize dataset with modality dictionary and labels.
        
        Args:
            X_dict: Dictionary with modality name keys and numpy array values
                    with shape [samples, window, features]
            y: Labels array [samples]
        """
        self.X_dict = X_dict
        self.y = y
        self.modalities = list(X_dict.keys())
        self.num_samples = len(y)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return a dictionary of modality tensors for this sample
        X_sample = {}
        for modality in self.modalities:
            # Convert numpy to tensor if needed
            if isinstance(self.X_dict[modality][idx], np.ndarray):
                X_sample[modality] = torch.tensor(self.X_dict[modality][idx], dtype=torch.float32)
            else:
                X_sample[modality] = self.X_dict[modality][idx]
                
        # Convert label to tensor
        y_sample = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return X_sample, y_sample

def main():
    """Main function to run migraine data pipeline."""
    # Parse command line arguments
    args = parse_args()

    # --- Set Random Seeds for Reproducibility --- #
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) # for multi-GPU.
    logging.info(f"Using random seed: {args.seed}")
    # --- ------------------------------------ --- #

    # --- Apply development mode settings if enabled ---
    if args.dev_mode:
        logging.warning("=== RUNNING IN DEVELOPMENT MODE ===")
        logging.warning("Using simplified settings for faster execution")
        
        # Override normal parameters with dev-friendly values
        if not args.num_experts:
            args.num_experts = 4  # Reduce number of experts
        if not args.hidden_size:
            args.hidden_size = 32  # Smaller hidden dimension
        
        # Disable PyGMO optimization in dev mode to speed up execution
        args.use_pygmo = False
        logging.warning("- PyGMO optimization disabled in dev mode")
        
        # Use smaller window size if not explicitly specified
        if args.window_size > 12:
            args.window_size = 8
            logging.warning(f"- DEV MODE: Using smaller window size: {args.window_size}")
        
        # Reduce cross-validation folds if too many
        if args.cv > 3:
            args.cv = 2
            logging.warning(f"- DEV MODE: Reduced CV folds to {args.cv}")
        
        logging.warning(f"- DEV MODE: Using {args.dev_epochs} epochs")
        logging.warning(f"- DEV MODE: Using batch size {args.dev_batch_size}")
        
        if args.skip_visualizations:
            logging.warning("- DEV MODE: Visualizations disabled")
    # --- End development mode settings ---

    # --- Create Output Directory --- 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup Logging --- 
    log_file_path = os.path.join(args.output_dir, args.log_file)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
    
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    
    # Configure root logger
    logger = logging.getLogger() # Get root logger
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Starting Migraine Prediction Pipeline")
    logging.info(f"Full Arguments: {vars(args)}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Log file: {log_file_path}")
    # ------------------------

    # --- Define Device Early --- 
    if torch.cuda.is_available() and not args.cpu:
        device = 'cuda'
        logging.info("Using GPU")
    else:
        device = 'cpu'
        logging.info("Using CPU")
    # ---------------------------

    # Create MigraineDataPipeline instance
    pipeline = MigraineDataPipeline(data_dir=args.data_dir, 
                                    cache_dir=args.cache_dir,
                                    weather_api_key=args.weather_api_key) # Pass API key if available

    # Process data
    logging.info("Processing data using the pipeline...")
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
        logging.error("Failed to process data or data is empty/invalid format after pipeline execution.")
        sys.exit(1)

    logging.info("Data processing complete.")
    logging.info(f"Number of feature samples (X): {len(fusemoe_data['X'])}")
    logging.info(f"Number of target samples (y): {len(fusemoe_data['y'])}")
    logging.debug(f"Initial data shapes: {{mod: data.shape for mod, data in fusemoe_data['X'][0].items()}} (first sample)")
    if len(fusemoe_data['y']) > 0:
        logging.info(f"Initial target distribution: {np.bincount(fusemoe_data['y'])}")

    # Check for sufficient data
    if len(fusemoe_data['X']) < 2 or len(np.unique(fusemoe_data['y'])) < 2:
        logging.error("Insufficient data for training - need at least 2 samples and 2 classes.")
        sys.exit(1)

    # Print summary of processed data
    logging.info("Data Processing Summary:")
    logging.info(f"Total samples: {len(fusemoe_data['X'])}")
    logging.info(f"Available modalities: {fusemoe_data['modalities']}")
    
    for modality, feature_count in fusemoe_data['features_per_modality'].items():
        if feature_count > 0:
            logging.info(f"  - {modality}: {feature_count} features")
    
    # Calculate positive rate
    positive_rate = np.mean(fusemoe_data['y'])
    logging.info(f"Positive rate (migraine events): {positive_rate:.2%}")
    
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
        
        # Also write class distribution
        unique_classes, counts = np.unique(fusemoe_data['y'], return_counts=True)
        f.write(f"\nClass distribution:\n")
        for i, cls in enumerate(unique_classes):
            f.write(f"  - Class {cls}: {counts[i]} samples ({counts[i]/len(fusemoe_data['y']):.2%})\n")
    
    logging.info(f"\nDataset summary saved to {summary_file}")
    logging.info("Data processing complete.")
    
    # --- DEBUG: Check patient IDs before CV --- #
    groups_np = fusemoe_data.get('groups')
    if groups_np is not None and len(groups_np) > 0:
        unique_groups, group_counts = np.unique(groups_np, return_counts=True)
        logging.info(f"[DEBUG] Found {len(unique_groups)} unique patient IDs in prepared 'groups' array before CV.")
        # Log first 10 group counts for brevity, or all if fewer than 10
        group_counts_dict = dict(zip(unique_groups, group_counts))
        logging.debug(f"[DEBUG] Patient ID counts (prepared data): {dict(list(group_counts_dict.items())[:10])}{'...' if len(group_counts_dict) > 10 else ''}")
    elif groups_np is not None:
        logging.warning("[DEBUG] 'groups' array exists but is empty.")
    else:
        logging.warning("[DEBUG] 'groups' array not found in prepared data.")
    # --- END DEBUG --- #
    
    # Calculate total number of features before creating config
    num_features_total = sum(fusemoe_data['features_per_modality'].values())

    # Configure MoE model with development-appropriate parameters
    config = MoEConfig(
        # MoE specific args
        num_experts=args.num_experts,
        # Input size should be FLATTENED features per sample for PyGMO
        moe_input_size=num_features_total * args.window_size,
        moe_hidden_size=args.hidden_size,
        moe_output_size=1, # Binary classification output
        router_type=args.router_type,
        window_size=args.window_size,
        gating=args.gating_function,
        top_k=args.top_k,
        num_modalities=len(fusemoe_data['modalities']),
        
        # General Transformer args with smaller sizes in dev mode
        hidden_dim=args.hidden_size,
        num_layers=2 if args.dev_mode else 3,  # Fewer layers in dev mode
        n_heads=2 if args.dev_mode else 4,     # Fewer attention heads in dev mode
        dropout=args.dropout_rate, # <<< Use dropout_rate arg
    )

    # Define modality experts configuration
    if args.modality_experts:
        experts_config = parse_modality_experts(args.modality_experts)
        logging.info("Using provided expert allocation per modality:")
    else:
        experts_config = get_modality_experts_config(fusemoe_data)
        logging.info("Auto-configured expert allocation per modality:")
    
    for modality, num_experts in experts_config.items():
        logging.info(f"  - {modality}: {num_experts} experts")
        
    # --- Set up cross-validation ---
    logging.info(f"Setting up {args.cv}-fold cross-validation with {args.cv_strategy} strategy")
    
    # Convert data to numpy arrays for sklearn CV
    # Ensure X_np is handled correctly - it's a list of dicts, CV needs indexable array
    # For splitting purposes, we only need the indices, so use the length of y_np
    num_samples = len(fusemoe_data['y'])
    indices_np = np.arange(num_samples)
    y_np = np.array(fusemoe_data['y'])
    groups_np = fusemoe_data.get('groups') # Get the groups array

    # Check if groups are needed and available
    group_strategies = ['groupkfold', 'stratifiedgroupkfold']
    if args.cv_strategy in group_strategies and (groups_np is None or len(groups_np) != num_samples):
        logging.error(f"CV strategy '{args.cv_strategy}' requires patient groups, but they are missing or have incorrect length. Exiting.")
        sys.exit(1)
        
    # Initialize cross-validation splitter based on strategy
    if args.cv == 1:
        logging.info("Using standard train/test split (cv=1)")
        indices = np.arange(len(y_np))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42, 
            stratify=y_np if args.cv_strategy == 'stratified' else None
        )
        cv_splits = [(train_indices, test_indices)]
    else:
        cv_splitter = None # Initialize
        if args.cv_strategy == 'stratified':
            logging.info("Using StratifiedKFold cross-validation")
            cv_splitter = StratifiedKFold(n_splits=args.cv, shuffle=args.cv_shuffle, 
                                random_state=args.seed if args.cv_shuffle else None)
            cv_splits = list(cv_splitter.split(indices_np, y_np)) # Split indices

        elif args.cv_strategy == 'kfold':
            logging.info("Using standard KFold cross-validation")
            cv_splitter = KFold(n_splits=args.cv, shuffle=args.cv_shuffle, 
                       random_state=args.seed if args.cv_shuffle else None)
            cv_splits = list(cv_splitter.split(indices_np)) # Split indices

        elif args.cv_strategy == 'time':
            logging.info("Using TimeSeriesSplit cross-validation")
            cv_splitter = TimeSeriesSplit(n_splits=args.cv)
            cv_splits = list(cv_splitter.split(indices_np))

        elif args.cv_strategy == 'groupkfold':
            logging.info("Using GroupKFold cross-validation (patient-aware)")
            # GroupKFold does not shuffle
            cv_splitter = GroupKFold(n_splits=args.cv)
            cv_splits = list(cv_splitter.split(indices_np, y_np, groups=groups_np))

        elif args.cv_strategy == 'stratifiedgroupkfold':
            logging.info("Using StratifiedGroupKFold cross-validation (patient-aware, stratified)")
            # StratifiedGroupKFold requires y for stratification and groups
            cv_splitter = StratifiedGroupKFold(n_splits=args.cv, shuffle=args.cv_shuffle, 
                                               random_state=args.seed if args.cv_shuffle else None)
            cv_splits = list(cv_splitter.split(indices_np, y_np, groups=groups_np))

    # Lists to store results from each fold
    fold_metrics = []
    fold_models = []
    fold_histories = []

    # --- Initialize OOF arrays (only if CV > 1) ---
    oof_preds = None
    oof_true = None
    oof_raw = None
    if args.cv > 1:
        # Ensure num_samples is defined (it should be from CV setup)
        if 'num_samples' not in locals():
             logging.error("Error: num_samples not defined before OOF initialization. Cannot proceed with OOF.")
             # Handle error appropriately, maybe exit or disable OOF
             sys.exit(1)
        oof_preds = np.full(num_samples, np.nan, dtype=float)
        oof_true = np.full(num_samples, np.nan, dtype=float) # Store true labels corresponding to OOF preds
        oof_raw = np.full(num_samples, np.nan, dtype=float) # Store raw model outputs (logits)
        logging.info(f"Initialized OOF arrays for {num_samples} samples.")
    # --- -------------------------------------- ---

    # Loop through cross-validation folds
    # --- Wrap the CV loop with tqdm ---
    cv_pbar = tqdm(enumerate(cv_splits), total=len(cv_splits), desc="Cross-Validation Folds", leave=True)
    for fold_idx, (train_indices, test_indices) in cv_pbar:
    # --- --------------------------- ---
        logging.info(f"===== Starting Fold {fold_idx + 1}/{len(cv_splits)} =====")
        fold_start_time = time.time()
        
        # Update CV progress bar description
        cv_pbar.set_description(f"CV Fold {fold_idx+1}/{len(cv_splits)}")

        # --- Correctly Slice Data for the Fold --- #
        # Slice the target labels and groups directly
        train_y = fusemoe_data['y'][train_indices]
        test_y = fusemoe_data['y'][test_indices]

        if fusemoe_data['groups'] is not None:
            # Ensure indices are within bounds for groups_np
            if (len(train_indices) > 0 and np.max(train_indices) >= len(fusemoe_data['groups'])) or \
               (len(test_indices) > 0 and np.max(test_indices) >= len(fusemoe_data['groups'])):
                 logging.error(f"Fold {fold_idx+1}: Train/Test indices out of bounds for groups array (size {len(fusemoe_data['groups'])}). Max train: {np.max(train_indices) if len(train_indices) > 0 else 'N/A'}, Max test: {np.max(test_indices) if len(test_indices) > 0 else 'N/A'}")
                 # Handle error: maybe skip fold or raise
                 continue
            train_groups = fusemoe_data['groups'][train_indices]
            # test_groups needed if balancing uses group info, not currently used
            # test_groups = fusemoe_data['groups'][test_indices]
        else:
            train_groups = None
            # test_groups = None

        # Slice the feature data dictionary per modality
        train_X_dict = {}
        test_X_dict = {}
        if isinstance(fusemoe_data['X'], dict): # Check if it's a dictionary
            for modality, data_array in fusemoe_data['X'].items():
                try:
                    # Ensure data_array is a numpy array before slicing
                    if not isinstance(data_array, np.ndarray):
                         logging.error(f"  [Fold {fold_idx+1}] Data for modality '{modality}' is not a NumPy array (type: {type(data_array)}). Skipping.")
                         continue # Or raise error

                    # Ensure indices are within bounds for data_array
                    if (len(train_indices) > 0 and np.max(train_indices) >= data_array.shape[0]) or \
                       (len(test_indices) > 0 and np.max(test_indices) >= data_array.shape[0]):
                         logging.error(f"Fold {fold_idx+1}: Train/Test indices out of bounds for modality '{modality}' data array (shape {data_array.shape}). Max train: {np.max(train_indices) if len(train_indices) > 0 else 'N/A'}, Max test: {np.max(test_indices) if len(test_indices) > 0 else 'N/A'}")
                         # Handle error: maybe skip fold or raise
                         continue # Skip this modality for this fold

                    train_X_dict[modality] = data_array[train_indices]
                    test_X_dict[modality] = data_array[test_indices]
                    logging.debug(f"  [Fold {fold_idx+1}] Sliced {modality}: train_shape={train_X_dict[modality].shape}, test_shape={test_X_dict[modality].shape}")

                except IndexError as e:
                    logging.error(f"IndexError slicing modality '{modality}' in fold {fold_idx+1}: {e}")
                    logging.error(f"  data_array shape: {data_array.shape}, train_indices max: {np.max(train_indices) if len(train_indices) > 0 else 'N/A'}, test_indices max: {np.max(test_indices) if len(test_indices) > 0 else 'N/A'}")
                    raise e # Re-raise to stop execution and investigate
                except Exception as e:
                    logging.error(f"Unexpected error slicing modality '{modality}' in fold {fold_idx+1}: {e}")
                    raise e
        else:
            logging.error(f"fusemoe_data['X'] is not a dictionary (type: {type(fusemoe_data['X'])}). Cannot perform modality slicing.")
            # Handle error appropriately, maybe exit or raise
            raise TypeError("fusemoe_data['X'] must be a dictionary for modality slicing.")

        # Check if slicing was successful for all modalities
        if len(train_X_dict) != len(fusemoe_data['X']) or len(test_X_dict) != len(fusemoe_data['X']):
             logging.error(f"Fold {fold_idx+1}: Failed to slice data for all modalities. Skipping fold.")
             continue
        # --- End Corrected Data Slicing --- #

        logging.info(f"Fold {fold_idx+1} - Initial Split: Train samples={len(train_y)}, Test samples={len(test_y)}")
        if len(train_y) > 0: logging.info(f"  Train target distribution: {np.bincount(train_y)} (Positive rate: {np.mean(train_y):.2%})")
        if len(test_y) > 0: logging.info(f"  Test target distribution: {np.bincount(test_y)} (Positive rate: {np.mean(test_y):.2%})")
        if train_X_dict: logging.debug(f"  Train shapes (first sample): {{mod: data.shape for mod, data in train_X_dict.items()}}") # Use train_X_dict

        # --- Apply StandardScaler (Fit on Train, Transform Train & Test) ---
        scalers = {}
        X_train_fold_scaled = {}
        X_test_fold_scaled = {}
        logging.info(f"Fold {fold_idx+1} - Applying StandardScaler...")
        for mod, data in train_X_dict.items():
            scaler = StandardScaler()
            if data.ndim != 3 or data.shape[0] == 0:
                 logging.warning(f"  Skipping scaling for modality '{mod}': Invalid shape {data.shape}")
                 X_train_fold_scaled[mod] = data # Use original data if invalid
                 # Ensure corresponding test data is also handled
                 if mod in test_X_dict:
                     X_test_fold_scaled[mod] = test_X_dict[mod]
                 continue # Skip to next modality

            n_samples_train, n_timesteps_train, n_features_train = data.shape
            if n_features_train == 0:
                logging.warning(f"  Skipping scaling for modality '{mod}': 0 features.")
                X_train_fold_scaled[mod] = data
                if mod in test_X_dict:
                    X_test_fold_scaled[mod] = test_X_dict[mod]
                continue
            # Reshape to 2D for scaler: [samples * timesteps, features]
            data_reshaped_train = data.reshape(-1, n_features_train)
            # Fit scaler on the training data ONLY
            scaler.fit(data_reshaped_train)
            # Transform training data
            scaled_data_train = scaler.transform(data_reshaped_train)
            # Reshape back to 3D
            X_train_fold_scaled[mod] = scaled_data_train.reshape(n_samples_train, n_timesteps_train, n_features_train)
            scalers[mod] = scaler # Store the scaler for this modality
            logging.debug(f"  Scaled train data for modality '{mod}': shape={X_train_fold_scaled[mod].shape}")

            # Transform test data using the SAME scaler fitted on train data
            test_data = test_X_dict.get(mod)
            if test_data is not None:
                if test_data.ndim != 3 or test_data.shape[0] == 0:
                    logging.warning(f"  Skipping test scaling for modality '{mod}': Invalid shape {test_data.shape}")
                    X_test_fold_scaled[mod] = test_data # Use original test data
                    continue
                n_samples_test, n_timesteps_test, n_features_test = test_data.shape
                if n_features_test == 0:
                     logging.warning(f"  Skipping test scaling for modality '{mod}': 0 features.")
                     X_test_fold_scaled[mod] = test_data
                     continue
                if n_features_test != n_features_train:
                    logging.error(f"  Feature mismatch between train ({n_features_train}) and test ({n_features_test}) for modality '{mod}'. Skipping test scaling.")
                    X_test_fold_scaled[mod] = test_data # Use original if scaling fails
                    continue
                data_reshaped_test = test_data.reshape(-1, n_features_test)
                scaled_data_test = scaler.transform(data_reshaped_test)
                X_test_fold_scaled[mod] = scaled_data_test.reshape(n_samples_test, n_timesteps_test, n_features_test)
                logging.debug(f"  Scaled test data for modality '{mod}': shape={X_test_fold_scaled[mod].shape}")
            else:
                logging.warning(f"  Modality '{mod}' not found in test set for scaling.")
                # Handle case where modality exists in train but not test - X_test_fold_scaled[mod] remains unset
        # --- End StandardScaler ---

        # --- Data Balancing (Applied only to Scaled Training Data) --- #
        X_train_fold_balanced, y_train_fold_balanced = balance_dataset(
            X=X_train_fold_scaled, # <<< Use scaled training data
            y=train_y,
            method=args.balance_method,
            sampling_ratio=args.sampling_ratio,
            random_state=args.seed + fold_idx # Ensure reproducibility per fold
        )
        # Use X_train_fold_balanced and y_train_fold_balanced for dataset creation
        # Test data remains X_test_fold_scaled
        # --- End Data Balancing --- #

        # --- Prepare DataLoaders for this fold using SCALED and BALANCED data ---
        # Check if data is empty before creating dataset/loader
        if not X_train_fold_balanced or len(y_train_fold_balanced) == 0:
            logging.error(f"Fold {fold_idx+1}: Training data is empty after slicing/scaling/balancing. Skipping fold.")
            continue
        if not X_test_fold_scaled or len(test_y) == 0: # <<< Use scaled test data dict
            logging.error(f"Fold {fold_idx+1}: Test data is empty after slicing/scaling. Skipping fold.")
            continue

        # Create DataLoaders using the dictionaries
        try:
            # Use the BALANCED training data and SCALED test data
            train_dataset = ModalityDataset(X_train_fold_balanced, y_train_fold_balanced)
            test_dataset = ModalityDataset(X_test_fold_scaled, test_y) # <<< Use scaled test data
        except Exception as e:
            logging.error(f"Fold {fold_idx+1}: Error creating ModalityDataset: {e}")
            logging.error(f"  Train X keys: {list(X_train_fold_balanced.keys())}, Train y len: {len(y_train_fold_balanced)}")
            logging.error(f"  Test X keys: {list(X_test_fold_scaled.keys())}, Test y len: {len(test_y)}") # <<< Use scaled test keys
            continue # Skip fold if dataset creation fails

        # Use dev mode batch size if applicable
        current_batch_size = args.dev_batch_size if args.dev_mode else args.batch_size

        try:
            train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False)
        except Exception as e:
            logging.error(f"Fold {fold_idx+1}: Error creating DataLoader: {e}")
            continue # Skip fold if loader creation fails
        # --- End DataLoader Preparation ---

        # Create a fresh model for this fold
        logging.info(f"Fold {fold_idx+1} - Creating MigraineFuseMoE model...")
        migraine_fusemoe = MigraineFuseMoE(
            config=config,
            input_sizes=fusemoe_data['features_per_modality'],
            hidden_size=args.hidden_size,
            output_size=1,
            num_experts=args.num_experts,
            modality_experts=experts_config,
            dropout=config.dropout,
            use_pso_gating=args.use_pygmo and not args.dev_mode,  # Disable PSO in dev mode
            use_evo_experts=args.use_pygmo and not args.dev_mode, # Disable evo experts in dev mode
            patient_adaptation=args.patient_adaptation
        ).to(device) # Move model to device

        # --- MODEL TRAINING / OPTIMIZATION ---
        optimization_history = None  # Initialize history variable
        
        # --- Class Weight Logic ---
        if args.class_weight == 'balanced':
            unique, counts = np.unique(train_y, return_counts=True)
            if len(unique) > 1:
                # Calculate balanced weights
                class_weights = len(train_y) / (len(unique) * counts)
                # Note: Original script used class_weights[1] / class_weights[0] which corresponds to count(0)/count(1)
                # This is the standard definition for pos_weight in BCEWithLogitsLoss
                pos_weight_val = class_weights[1] / class_weights[0]
                pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device) # Ensure float32
                logging.info(f"Fold {fold_idx+1}: Using balanced class weights. Calculated pos_weight: {pos_weight.item():.4f}")
            else:
                logging.warning(f"Fold {fold_idx+1}: Only one class present in training data. Cannot balance. Using default pos_weight=1.0.")
                pos_weight = torch.tensor(1.0, dtype=torch.float32).to(device) # Default weight if only one class
        else:
            # No class weights requested
            logging.info(f"Fold {fold_idx+1}: Class weights disabled (args.class_weight != 'balanced'). Using default pos_weight=1.0.")
            pos_weight = torch.tensor(1.0, dtype=torch.float32).to(device) # Default weight if balancing is off

        # Note: We pass pos_weight directly to the training functions/loss calculations below
        # The 'criterion' variable might not be explicitly needed here anymore if training loops handle it.
        # However, we keep it defined for potential future use or clarity.
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        # --- End Class Weight Logic ---

        # Convert training data to tensors for PyGMO optimization
        train_data_dict = {}
        for mod in X_train_fold_balanced.keys(): # <<< Use balanced train keys
            train_data_dict[mod] = torch.tensor(X_train_fold_balanced[mod], dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_fold_balanced, dtype=torch.float32).unsqueeze(1).to(device) # <<< Use balanced train labels
        
        if args.use_pygmo:
            logging.info(f"Fold {fold_idx+1} - Optimizing model architecture and routing using PyGMO...")
            
            try:
                # For development mode, use very small population and generation sizes
                if args.dev_mode:
                    expert_pop_size = 5
                    gating_pop_size = 5
                    expert_gens = 2
                    gating_gens = 2
                else:
                    expert_pop_size = args.expert_population_size
                    gating_pop_size = args.gating_population_size
                    expert_gens = args.expert_generations
                    gating_gens = args.gating_generations
                
                # Capture the returned history along with the model
                optimized_model, optimization_history = migraine_fusemoe.optimize_model(
                    train_data=(train_data_dict, y_train_tensor), 
                    expert_algo=args.expert_algorithm,
                    gating_algo=args.gating_algorithm,
                    expert_pop_size=expert_pop_size,
                    gating_pop_size=gating_pop_size,
                    seed=42 + fold_idx, 
                    device=device
                )
                migraine_fusemoe = optimized_model # Update the model variable
                logging.info(f"Fold {fold_idx+1} - PyGMO Model optimization complete!")
                
                # Skip history saving in dev mode
                if not args.dev_mode or not args.skip_visualizations:
                    # Save the fold-specific optimization history
                    history_file = os.path.join(args.output_dir, f'fold{fold_idx+1}_optimization_history.json')
                    try:
                        # Convert to serializable format and save
                        serializable_history = {}
                        for stage, stage_data in optimization_history.items():
                            if isinstance(stage_data, dict):
                                algo = stage_data.get('algorithm', 'Unknown')
                                history_list = stage_data.get('history', [])
                                
                                processed_records = []
                                if isinstance(history_list, list):
                                    for record in history_list:
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

                                # Store processed records
                                serializable_history[stage] = {
                                    'algorithm': algo,
                                    'history': processed_records
                                }
                            
                        with open(history_file, 'w') as f:
                            json.dump(serializable_history, f, indent=4)
                            logging.info(f"Fold {fold_idx+1} - Optimization history saved to {history_file}")
                    except Exception as e:
                        logging.error(f"Error saving optimization history for fold {fold_idx+1}: {e}")

            except Exception as e:
                logging.error(f"Error during PyGMO model optimization for fold {fold_idx+1}: {str(e)}")
                logging.warning("Falling back to standard training for this fold.")
                
                # Fallback to standard training
                migraine_fusemoe = MigraineFuseMoE(
                    config=config,
                    input_sizes=fusemoe_data['features_per_modality'],
                    hidden_size=args.hidden_size,
                    output_size=1,
                    num_experts=args.num_experts,
                    modality_experts=experts_config,
                    dropout=config.dropout,
                    use_pso_gating=False,
                    use_evo_experts=False,
                    patient_adaptation=args.patient_adaptation
                ).to(device)
                
                # Standard training (modified for dev mode)
                optimizer = torch.optim.Adam(migraine_fusemoe.parameters(), lr=0.001)
                num_epochs = args.dev_epochs if args.dev_mode else 10
                batch_size = args.dev_batch_size if args.dev_mode else 16
                
                migraine_fusemoe.train()
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    for i in range(0, y_train_tensor.shape[0], batch_size):
                        batch_X = {mod: tensor[i:i+batch_size] for mod, tensor in train_data_dict.items()}
                        batch_y = y_train_tensor[i:i+batch_size]
                        
                        optimizer.zero_grad()
                        outputs, _ = migraine_fusemoe(batch_X)
                        # Use the calculated pos_weight (either balanced or 1.0)
                        loss = weighted_bce_loss(outputs, batch_y, pos_weight=pos_weight)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item() * batch_y.shape[0]
                    
                    avg_epoch_loss = epoch_loss / y_train_tensor.shape[0]
                    logging.info(f"Fold {fold_idx+1}, Fallback Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        else:
            # Standard PyTorch training loop with weighted loss
            logging.info(f"Fold {fold_idx+1} - Starting Standard PyTorch Training...")
            logging.info(f"  Using weighted BCE loss with positive class weight: {pos_weight.item():.4f}")
            
            # Use development mode settings if enabled
            num_epochs = args.dev_epochs if args.dev_mode else 10 # Keep dev epochs separate
            batch_size = args.dev_batch_size if args.dev_mode else 16
            patience = 2 if args.dev_mode else args.early_stopping_patience # Use arg, override for dev
            validation_split_size = args.validation_split # Use arg
            
            # Create a validation set for early stopping
            # Calculate validation size based on the length of the actual training data for the fold
            val_size = int(validation_split_size * len(y_train_fold_balanced)) # <<< Use length of balanced labels
            
            if val_size > 1 and len(y_train_fold_balanced) - val_size > 1: # <<< Use length of balanced labels
                # Create a stratified validation split from training data
                train_val_indices = np.arange(len(y_train_fold_balanced)) # <<< Use range of balanced labels
                train_indices_subset, val_indices_subset = train_test_split(
                    train_val_indices, test_size=val_size,
                    stratify=y_train_fold_balanced if len(np.unique(y_train_fold_balanced)) > 1 else None, # <<< Stratify by balanced labels
                    random_state=args.seed # <<< Use seed
                )

                # Create validation dict by slicing X_train_fold_balanced dictionary
                val_X_dict = {mod: data[val_indices_subset] for mod, data in X_train_fold_balanced.items()} # <<< Slice balanced data
                val_y = y_train_fold_balanced[val_indices_subset] # <<< Slice balanced labels

                # Create training subset dict similarly
                train_X_subset_dict = {mod: data[train_indices_subset] for mod, data in X_train_fold_balanced.items()} # <<< Slice balanced data
                train_y_subset = y_train_fold_balanced[train_indices_subset] # <<< Slice balanced labels

                # Prepare validation tensors
                val_data_dict = {mod: torch.tensor(data_array, dtype=torch.float32).to(device)
                                 for mod, data_array in val_X_dict.items()}
                val_y_tensor = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1).to(device)

                # Recreate train data dict with subset tensors
                train_subset_dict_tensors = {mod: torch.tensor(data_array, dtype=torch.float32).to(device)
                                           for mod, data_array in train_X_subset_dict.items()}
                train_y_subset_tensor = torch.tensor(train_y_subset, dtype=torch.float32).unsqueeze(1).to(device)

                # Use early stopping
                logging.info(f"  Training with early stopping (patience={patience}, max epochs={num_epochs}) using subset data")
                migraine_fusemoe, training_history = train_with_early_stopping(
                    model=migraine_fusemoe,
                    train_data_dict=train_subset_dict_tensors, # Pass tensor dictionary
                    y_train_tensor=train_y_subset_tensor,
                    val_data_dict=val_data_dict,
                    y_val_tensor=val_y_tensor,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=args.learning_rate, # <<< Pass LR arg
                    pos_weight=pos_weight, # <<< Pass the calculated pos_weight
                    device=device,
                    patience=patience # <<< Use patience arg
                )
                logging.info(f"Fold {fold_idx+1} - Training complete with early stopping.")
            else:
                # If dataset is too small for validation split, use standard training on full fold data
                logging.warning(f"Fold {fold_idx+1}: Dataset too small for validation split ({len(y_train_fold_balanced)} samples). Training on full fold data for {num_epochs} epochs.") # <<< Use length of balanced labels

                optimizer = torch.optim.Adam(migraine_fusemoe.parameters(), lr=args.learning_rate)
                migraine_fusemoe.train() # Set model to training mode
                # Use train_data_dict and y_train_tensor (already created tensors for PyGMO fallback - these ARE the balanced ones)
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    # Simple loop without dataloader needed here as data is already tensors
                    # Note: This uses the full fold balanced data, not a subset
                    for i in range(0, y_train_tensor.shape[0], batch_size):
                        batch_X = {mod: tensor[i:i+batch_size] for mod, tensor in train_data_dict.items()}
                        batch_y = y_train_tensor[i:i+batch_size]
                        
                        optimizer.zero_grad()
                        outputs, _ = migraine_fusemoe(batch_X)
                        loss = weighted_bce_loss(outputs, batch_y, pos_weight=pos_weight) 
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item() * batch_y.shape[0]
                    
                    avg_epoch_loss = epoch_loss / y_train_tensor.shape[0]
                    logging.info(f"Fold {fold_idx+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

            logging.info(f"Fold {fold_idx+1} - Standard Training complete.")
        # --- END TRAINING / OPTIMIZATION --- 

        # Use the trained model for this fold
        model = migraine_fusemoe 

        # Evaluate the model for this fold
        logging.info(f"Fold {fold_idx+1} - Evaluating model on test data...")
        # Check for threshold optimization flag
        optimize_threshold = args.threshold_search if hasattr(args, 'threshold_search') else False
        optimize_metric = args.optimize_metric if hasattr(args, 'optimize_metric') else 'balanced_accuracy'
        
        # Convert test_y to tensor
        y_test_tensor = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1).to(device)
        
        results, probs, raw_outputs, y_true = evaluate_model(
            model, 
            X_test_fold_scaled, # <<< Use SCALED test data for evaluation
            y_test_tensor,
            device,
            threshold=0.5,
            optimize_threshold=optimize_threshold,
            optimize_metric=optimize_metric
        )
        
        # Get probabilities for training data (needed for calibration)
        logging.debug(f"Fold {fold_idx+1} - Getting probabilities for training data (for calibration)...")
        model.eval()
        with torch.no_grad():
            # Need the BALANCED training data dictionary with tensors for calibration
            train_data_dict_eval = {mod: torch.tensor(data_array, dtype=torch.float32).to(device)
                                    for mod, data_array in X_train_fold_balanced.items()} # <<< Use balanced data
            # We also need the corresponding labels for the balanced data
            train_y_balanced_tensor = torch.tensor(y_train_fold_balanced, dtype=torch.float32).unsqueeze(1).to(device) # <<< Use balanced labels

            train_outputs, _ = model(train_data_dict_eval)
            train_probs = torch.sigmoid(train_outputs).cpu().numpy()
            train_y_true_balanced = train_y_balanced_tensor.cpu().numpy() # Use balanced labels

        # Calibrate probabilities if not in dev mode, or if specifically requested
        if not args.dev_mode or not args.skip_visualizations:
            logging.info(f"Fold {fold_idx+1} - Calibrating prediction probabilities...")
            # Use the BALANCED train labels and probs for calibration fitting
            calibrated_probs = calibrate_probabilities(train_y_true_balanced, train_probs, probs, calibration_method='isotonic')
            platt_probs = calibrate_probabilities(train_y_true_balanced, train_probs, probs, calibration_method='platt')

            # Recalculate metrics with calibrated probabilities (using the original TRUE test labels)
            calibrated_predicted = (calibrated_probs > 0.5).astype(int)
            try:
                calibrated_auc = roc_auc_score(y_true, calibrated_probs) # y_true are the ORIGINAL test labels
                calibrated_pr_auc = average_precision_score(y_true, calibrated_probs) # y_true are the ORIGINAL test labels

                # Store calibrated metrics
                results['calibrated_auc'] = calibrated_auc
                results['calibrated_pr_auc'] = calibrated_pr_auc
                results['platt_auc'] = roc_auc_score(y_true, platt_probs) if not np.isnan(platt_probs).any() else float('nan') # y_true are the ORIGINAL test labels
            except Exception as e:
                logging.warning(f"Fold {fold_idx+1}: Could not compute AUC for calibrated probabilities: {e}")
        
        # Print evaluation results for this fold
        logging.info(f"Fold {fold_idx+1} Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {key}: {value:.4f}")
            elif isinstance(value, list) and key == 'confusion_matrix': # Nicer print for confusion matrix
                logging.info(f"  {key}: {value}")
            # Skip printing complex items like expert_usage or class_distribution here, use debug if needed
        logging.debug(f"Fold {fold_idx+1} Full Metrics: {results}")
        
        # Store results for this fold
        fold_metrics.append(results)
        fold_models.append(model)
        fold_histories.append(optimization_history)
        
        # --- Store OOF predictions (only if CV > 1) ---
        if args.cv > 1 and oof_preds is not None and oof_true is not None and oof_raw is not None:
            if len(test_indices) != len(probs.flatten()):
                logging.warning(f"Fold {fold_idx+1}: Mismatch between test_indices ({len(test_indices)}) and predictions ({len(probs.flatten())}). Skipping OOF update for this fold.")
            else:
                oof_preds[test_indices] = probs.flatten()
                oof_true[test_indices] = y_true.flatten()
                oof_raw[test_indices] = raw_outputs.flatten()
                logging.debug(f"Fold {fold_idx+1}: Updated OOF predictions for {len(test_indices)} samples.")
        # --- ------------------------------------- ---
        
        # Save the fold-specific model (skip in dev mode to save time/space)
        if not args.dev_mode:
            model_path = os.path.join(args.output_dir, f"fold{fold_idx+1}_model.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model state_dict for fold {fold_idx+1} saved to {model_path}")
        
        # Skip visualizations in dev mode
        if args.skip_visualizations:
            logging.info(f"Fold {fold_idx+1}: Skipping visualizations in dev mode")
            continue
            
        # Generate visualizations for this fold
        logging.info(f"Fold {fold_idx+1}: Generating visualizations...")
        
        # ... rest of the existing code ...
    
    # --- End Cross-Validation Loop ---
    
    # Calculate and print average metrics across all folds
    if fold_metrics:
        logging.info("===== Cross-Validation Summary =====")
        logging.info(f"Cross-Validation Summary ({args.cv} folds)")
        logging.info("="*50)
        
        # Calculate average and std dev for each metric
        metrics_keys = fold_metrics[0].keys()
        avg_metrics = {}
        std_metrics = {}
        
        for key in metrics_keys:
            # Skip non-numeric metrics or complex data structures
            if key not in ['expert_usage', 'confusion_matrix', 'class_distribution']:
                try:
                    values = [fold[key] for fold in fold_metrics if key in fold and not isinstance(fold[key], dict)]
                    if values:  # Only calculate if we have values
                        avg_metrics[key] = float(np.nanmean(values))
                        std_metrics[key] = float(np.nanstd(values))
                except (TypeError, ValueError) as e:
                    logging.warning(f"Warning: Could not calculate statistics for metric '{key}': {e}")
                    avg_metrics[key] = "N/A"
                    std_metrics[key] = "N/A"
        
        # Print summary table
        logging.info(f"{'Metric':<20} {'Mean':<10} {'Std Dev':<10}")
        logging.info("-"*40)
        for key in sorted(avg_metrics.keys()):
            if isinstance(avg_metrics[key], (int, float)):
                logging.info(f"{key:<20} {avg_metrics[key]:.4f} {std_metrics[key]:.4f}")
            else:
                logging.info(f"{key:<20} {avg_metrics[key]} {std_metrics[key]}")
        
        # Save cross-validation summary to file
        summary_path = os.path.join(args.output_dir, 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'avg_metrics': avg_metrics,
                'std_metrics': std_metrics,
                'fold_metrics': [
                    {k: v for k, v in metrics.items() if k not in ['expert_usage', 'confusion_matrix', 'class_distribution']} 
                    for metrics in fold_metrics
                ]
            }, f, indent=4)
        
        logging.info(f"\nCross-validation summary saved to {summary_path}")
        
        # Visualize cross-validation results
        # Example: Box plot of key metrics across folds
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        plt.figure(figsize=(12, 8))
        
        data_to_plot = []
        for metric in metrics_to_plot:
            values = [fold.get(metric, float('nan')) for fold in fold_metrics]
            data_to_plot.append(values)
        
        plt.boxplot(data_to_plot, labels=metrics_to_plot)
        plt.title('Performance Metrics Across CV Folds')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(args.output_dir, 'cv_performance_boxplot.png'), dpi=300)
        plt.close()
        
        # If using multiple folds, select the best model based on some criterion (e.g., F1 score)
        if len(fold_models) > 1:
            best_idx = np.argmax([m['f1_score'] for m in fold_metrics])
            best_model = fold_models[best_idx]
            
            logging.info(f"\nSelected best model from fold {best_idx+1} with F1 Score: {fold_metrics[best_idx]['f1_score']:.4f}")
            
            # Save the best model as the final model
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(best_model.state_dict(), best_model_path)
            logging.info(f"Best model saved to {best_model_path}")
    
    # --- Aggregate and Evaluate Ensemble (OOF) Results ---
    if args.cv > 1 and oof_preds is not None and oof_true is not None:
        logging.info("===== Aggregating and Evaluating Out-of-Fold (OOF) Ensemble Results =====")

        # Check if OOF arrays were fully populated
        if np.isnan(oof_preds).any() or np.isnan(oof_true).any():
            logging.warning("Warning: OOF arrays contain NaN values. Some test samples might not have received a prediction. OOF evaluation might be inaccurate.")
            # Optionally, filter out NaNs before evaluation
            valid_oof_indices = ~np.isnan(oof_preds) & ~np.isnan(oof_true)
            oof_preds_valid = oof_preds[valid_oof_indices]
            oof_true_valid = oof_true[valid_oof_indices]
            oof_raw_valid = oof_raw[valid_oof_indices]
            logging.info(f"Evaluating OOF results on {len(oof_preds_valid)} non-NaN samples.")
        else:
            oof_preds_valid = oof_preds
            oof_true_valid = oof_true
            oof_raw_valid = oof_raw
            logging.info("OOF arrays fully populated. Evaluating on all samples.")

        if len(oof_preds_valid) > 0 and len(np.unique(oof_true_valid)) > 1:
            # Find optimal threshold on OOF predictions
            oof_optimize_metric = args.optimize_metric if hasattr(args, 'optimize_metric') else 'balanced_accuracy'
            logging.info(f"Finding optimal threshold for OOF predictions based on '{oof_optimize_metric}'...")
            oof_optimal_threshold = find_optimal_threshold(oof_true_valid, oof_preds_valid, metric=oof_optimize_metric)
            logging.info(f"Optimal OOF threshold: {oof_optimal_threshold:.4f}")

            # Calculate final ensemble metrics using the optimal OOF threshold
            oof_predicted_class = (oof_preds_valid >= oof_optimal_threshold).astype(int)
            ensemble_metrics = calculate_metrics(oof_true_valid, oof_predicted_class, oof_preds_valid)
            ensemble_metrics['optimal_threshold'] = oof_optimal_threshold # Add the threshold used

            logging.info("--- Final Ensemble (OOF) Performance ---")
            for key, value in ensemble_metrics.items():
                 if isinstance(value, (int, float)):
                     logging.info(f"  {key}: {value:.4f}")
                 elif isinstance(value, list) and key == 'confusion_matrix':
                     logging.info(f"  {key}: {value}")
            logging.info("----------------------------------------")

            # Save OOF predictions and metrics
            oof_output_path = os.path.join(args.output_dir, 'oof_predictions.npz')
            np.savez(oof_output_path,
                     oof_preds=oof_preds, # Save original possibly with NaNs
                     oof_true=oof_true,   # Save original possibly with NaNs
                     oof_raw=oof_raw,     # Save original possibly with NaNs
                     oof_optimal_threshold=oof_optimal_threshold)
            logging.info(f"OOF predictions saved to {oof_output_path}")

            ensemble_metrics_path = os.path.join(args.output_dir, 'ensemble_summary.json')
            # Convert confusion matrix ndarray to list for JSON
            if 'confusion_matrix' in ensemble_metrics and isinstance(ensemble_metrics['confusion_matrix'], np.ndarray):
                ensemble_metrics['confusion_matrix'] = ensemble_metrics['confusion_matrix'].tolist()
            with open(ensemble_metrics_path, 'w') as f:
                json.dump(ensemble_metrics, f, indent=4)
            logging.info(f"Ensemble metrics saved to {ensemble_metrics_path}")
        else:
            logging.warning("Could not evaluate OOF results: Not enough valid predictions or only one class present in OOF true labels.")
    # --- --------------------------------------------------- ---

    logging.info("\nMigraine prediction with cross-validation completed successfully!")
    logging.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()