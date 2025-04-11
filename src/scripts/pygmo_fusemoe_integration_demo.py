#!/usr/bin/env python
# PyGMO FuseMOE Integration Demo
# Demonstrates the integration of PyGMO evolutionary computing with FuseMOE

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
from typing import Dict, Tuple

# Add parent directory to path to access module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MoEConfig
from core.pygmo_fusemoe import PyGMOFuseMoE, MigraineFuseMoE, create_pygmo_fusemoe


def create_synthetic_data(num_samples=500, input_dim=20, num_classes=4, modalities=None, seed=42):
    """
    Create synthetic data for demonstration purposes.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Dimension of input features
        num_classes: Number of output classes
        modalities: Dict of modality names and their dimensions
        seed: Random seed
        
    Returns:
        Tuple of (inputs, targets)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if modalities:
        # Multi-modal data
        inputs = {}
        total_dim = 0
        
        for modality, dim in modalities.items():
            inputs[modality] = torch.randn(num_samples, dim)
            total_dim += dim
            
        # Create class-specific patterns in each modality
        for c in range(num_classes):
            class_indices = torch.where(torch.randint(0, num_classes, (num_samples,)) == c)[0]
            
            for modality, tensor in inputs.items():
                # Add class-specific pattern
                pattern = torch.randn(tensor.shape[1]) * 2
                for idx in class_indices:
                    inputs[modality][idx] += pattern
                    
        # Create targets
        x_combined = torch.cat([tensor for tensor in inputs.values()], dim=1)
        weights = torch.randn(total_dim, num_classes)
        logits = x_combined @ weights
        targets = torch.argmax(logits, dim=1)
        
        return inputs, targets
    else:
        # Single-modal data
        x = torch.randn(num_samples, input_dim)
        
        # Create class-specific patterns
        for c in range(num_classes):
            class_indices = torch.where(torch.randint(0, num_classes, (num_samples,)) == c)[0]
            pattern = torch.randn(input_dim) * 2
            
            for idx in class_indices:
                x[idx] += pattern
        
        # Create targets
        weights = torch.randn(input_dim, num_classes)
        logits = x @ weights
        targets = torch.argmax(logits, dim=1)
        
        return x, targets


def evaluate_model(model, inputs, targets):
    """
    Evaluate model performance.
    
    Args:
        model: The model to evaluate
        inputs: Input data
        targets: Target labels
        
    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        # Handle multi-modal inputs
        if isinstance(inputs, dict):
            outputs = model(inputs)
        else:
            outputs = model(inputs)
        
        # Calculate loss
        if outputs.shape[1] == 1:  # Binary classification
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(outputs.squeeze(), targets.float()).item()
            predictions = (outputs > 0).float().squeeze()
        else:  # Multi-class classification
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets).item()
            _, predictions = torch.max(outputs, 1)
        
        # Calculate accuracy
        accuracy = (predictions == targets).float().mean().item() * 100
        
        # Get expert usage if available
        expert_usage = None
        if hasattr(model, 'get_expert_usage') and callable(model.get_expert_usage):
            expert_usage = model.get_expert_usage()
            if expert_usage is not None:
                expert_usage = expert_usage.cpu().numpy()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'expert_usage': expert_usage
        }


def plot_expert_usage(original_usage, optimized_usage, num_experts, title="Expert Usage Comparison"):
    """
    Plot expert usage before and after optimization.
    
    Args:
        original_usage: Expert usage before optimization
        optimized_usage: Expert usage after optimization
        num_experts: Number of experts
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(num_experts)
    width = 0.35
    
    plt.bar(x - width/2, original_usage, width, label='Original')
    plt.bar(x + width/2, optimized_usage, width, label='Optimized')
    
    plt.xlabel('Expert')
    plt.ylabel('Usage')
    plt.title(title)
    plt.xticks(x, [f'E{i+1}' for i in range(num_experts)])
    plt.legend()
    plt.grid(axis='y')
    
    # Save the plot
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def demo_basic_integration():
    """
    Demonstrate basic PyGMO-FuseMOE integration with evolutionary experts and PSO gating.
    """
    print("\n=== Basic PyGMO-FuseMOE Integration Demo ===\n")
    
    # Create synthetic data
    num_features = 20
    num_classes = 4
    print(f"Creating synthetic dataset with {num_features} features and {num_classes} classes...")
    x_train, y_train = create_synthetic_data(num_samples=500, input_dim=num_features, num_classes=num_classes)
    x_test, y_test = create_synthetic_data(num_samples=100, input_dim=num_features, num_classes=num_classes)
    
    # Create MoE configuration
    config = MoEConfig(
        num_experts=8,
        moe_input_size=num_features,
        moe_hidden_size=64, 
        moe_output_size=num_classes,
        router_type='joint',
        dropout=0.1,
        hidden_act='gelu',
        noisy_gating=True,
        top_k=4
    )
    
    # Create baseline model (without evolutionary experts and PSO gating)
    print("\nCreating baseline model (standard MoE)...")
    baseline_model = PyGMOFuseMoE(
        config=config,
        input_size=num_features,
        hidden_size=64,
        output_size=num_classes,
        use_pso_gating=False,
        use_evo_experts=False
    )
    
    # Evaluate baseline model before training
    baseline_eval = evaluate_model(baseline_model, x_test, y_test)
    print(f"Baseline model before training:")
    print(f"  Loss: {baseline_eval['loss']:.6f}")
    print(f"  Accuracy: {baseline_eval['accuracy']:.2f}%")
    
    # Train baseline model with standard optimization
    print("\nTraining baseline model with standard optimization...")
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    baseline_model.train()
    for epoch in range(10):  # Just a few epochs for demonstration
        optimizer.zero_grad()
        outputs = baseline_model(x_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/10: Loss = {loss.item():.6f}")
    
    # Evaluate baseline model after training
    baseline_eval_after = evaluate_model(baseline_model, x_test, y_test)
    print(f"\nBaseline model after training:")
    print(f"  Loss: {baseline_eval_after['loss']:.6f}")
    print(f"  Accuracy: {baseline_eval_after['accuracy']:.2f}%")
    baseline_usage = baseline_eval_after['expert_usage']
    
    # Create and optimize PyGMO-enhanced model
    print("\nCreating and optimizing PyGMO-enhanced model...")
    pygmo_model = PyGMOFuseMoE(
        config=config,
        input_size=num_features,
        hidden_size=64,
        output_size=num_classes,
        use_pso_gating=True,
        use_evo_experts=True
    )
    
    # Optimize the model using PyGMO
    pygmo_model.optimize_model(
        train_data=(x_train, y_train),
        expert_algo='sade',
        gating_algo='pso',
        expert_pop_size=10,  # Small size for demo
        gating_pop_size=10   # Small size for demo
    )
    
    # Evaluate the optimized model
    pygmo_eval = evaluate_model(pygmo_model, x_test, y_test)
    print(f"\nPyGMO-enhanced model after optimization:")
    print(f"  Loss: {pygmo_eval['loss']:.6f}")
    print(f"  Accuracy: {pygmo_eval['accuracy']:.2f}%")
    
    # Compare results
    print("\nPerformance comparison:")
    print(f"  Baseline:  Accuracy = {baseline_eval_after['accuracy']:.2f}%")
    print(f"  PyGMO-FuseMoE: Accuracy = {pygmo_eval['accuracy']:.2f}%")
    print(f"  Improvement: {pygmo_eval['accuracy'] - baseline_eval_after['accuracy']:.2f}%")
    
    # Plot expert usage comparison
    if baseline_usage is not None and pygmo_eval['expert_usage'] is not None:
        plot_expert_usage(
            baseline_usage, 
            pygmo_eval['expert_usage'], 
            config.num_experts,
            title="Expert Usage: Baseline vs PyGMO-FuseMoE"
        )
        print("\nExpert usage comparison plot saved to expert_usage_baseline_vs_pygmo-fusemoe.png")


def demo_migraine_prediction():
    """
    Demonstrate migraine prediction using PyGMO-FuseMOE with multi-modal inputs.
    """
    print("\n=== Migraine Prediction with PyGMO-FuseMOE Demo ===\n")
    
    # Define modalities for migraine prediction
    modalities = {
        'weather': 5,   # Temperature, humidity, pressure, etc.
        'sleep': 3,     # Duration, quality, disturbances
        'diet': 8,      # Various food/drink intake features
        'stress': 4,    # Stress levels, anxiety, etc.
        'activity': 6,  # Exercise, screen time, etc.
    }
    total_features = sum(modalities.values())
    
    # Create synthetic migraine prediction data
    print(f"Creating synthetic multi-modal migraine data with {len(modalities)} modalities...")
    train_inputs, train_targets = create_synthetic_data(
        num_samples=300, 
        num_classes=2,  # Binary: migraine or not
        modalities=modalities
    )
    test_inputs, test_targets = create_synthetic_data(
        num_samples=100, 
        num_classes=2,
        modalities=modalities
    )
    
    # For simplicity, concatenate the modalities
    print("Converting multi-modal data to flat tensors for demonstration...")
    flat_train_inputs = torch.cat([tensor for tensor in train_inputs.values()], dim=1)
    flat_test_inputs = torch.cat([tensor for tensor in test_inputs.values()], dim=1)
    
    # Create MoE configuration for migraine prediction
    config = MoEConfig(
        num_experts=8,
        moe_input_size=total_features,
        moe_hidden_size=64, 
        moe_output_size=2,  # Binary classification
        router_type='joint',
        dropout=0.1,
        hidden_act='gelu',
        noisy_gating=True,
        top_k=2,
        num_modalities=len(modalities)
    )
    
    # Create standard PyGMO-enhanced model (not the migraine-specific one)
    print("\nCreating PyGMO-enhanced model for migraine prediction...")
    pygmo_model = PyGMOFuseMoE(
        config=config,
        input_size=total_features,
        hidden_size=64,
        output_size=2,
        num_experts=8,
        use_pso_gating=True,
        use_evo_experts=True
    )
    
    # Optimize the model
    print("\nOptimizing migraine prediction model...")
    pygmo_model.optimize_model(
        train_data=(flat_train_inputs, train_targets),
        expert_algo='sade',
        gating_algo='pso',
        expert_pop_size=10,  # Small size for demo
        gating_pop_size=10   # Small size for demo
    )
    
    # Evaluate the optimized model
    migraine_eval = evaluate_model(pygmo_model, flat_test_inputs, test_targets)
    print(f"\nPyGMO-enhanced model for migraine prediction:")
    print(f"  Loss: {migraine_eval['loss']:.6f}")
    print(f"  Accuracy: {migraine_eval['accuracy']:.2f}%")
    
    # For comparison, create and evaluate a traditional baseline model
    print("\nTraining traditional baseline model for comparison...")
    baseline = nn.Sequential(
        nn.Linear(total_features, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Train baseline model
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    baseline.train()
    for epoch in range(10):  # Just a few epochs for demonstration
        optimizer.zero_grad()
        outputs = baseline(flat_train_inputs)
        loss = loss_fn(outputs, train_targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10: Loss = {loss.item():.6f}")
    
    # Evaluate baseline
    baseline.eval()
    with torch.no_grad():
        baseline_outputs = baseline(flat_test_inputs)
        baseline_loss = loss_fn(baseline_outputs, test_targets).item()
        _, predicted = torch.max(baseline_outputs, 1)
        baseline_acc = (predicted == test_targets).float().mean().item() * 100
    
    print(f"\nBaseline model performance:")
    print(f"  Loss: {baseline_loss:.6f}")
    print(f"  Accuracy: {baseline_acc:.2f}%")
    
    # Compare results
    print("\nPerformance comparison:")
    print(f"  Baseline:          Accuracy = {baseline_acc:.2f}%")
    print(f"  PyGMO-FuseMoE:     Accuracy = {migraine_eval['accuracy']:.2f}%")
    print(f"  Improvement:       {migraine_eval['accuracy'] - baseline_acc:.2f}%")
    
    # Demonstrate expert usage analysis
    if migraine_eval['expert_usage'] is not None:
        usage = migraine_eval['expert_usage']
        print("\nExpert specialization in migraine prediction:")
        for i, u in enumerate(usage):
            print(f"  Expert {i+1}: {u:.4f} usage")
        
        # Plot expert usage
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(usage)), usage)
        plt.xlabel('Expert')
        plt.ylabel('Usage')
        plt.title('Expert Usage in Migraine Prediction')
        plt.xticks(range(len(usage)), [f'E{i+1}' for i in range(len(usage))])
        plt.grid(axis='y')
        plt.savefig("migraine_expert_usage.png")
        print("\nExpert usage plot saved to migraine_expert_usage.png")
        
    print("\nNote: For a complete migraine prediction system, additional components would be added:")
    print("  1. Multi-modal fusion with modality-specific feature extractors")
    print("  2. Early warning prediction (hours before onset)")
    print("  3. Patient-specific adaptation")
    print("  4. Trigger identification")


def main():
    parser = argparse.ArgumentParser(description="PyGMO FuseMOE Integration Demo")
    parser.add_argument("--demo", type=str, choices=["basic", "migraine", "all"], default="all",
                        help="Which demo to run (basic, migraine, or all)")
    args = parser.parse_args()
    
    if args.demo == "basic" or args.demo == "all":
        demo_basic_integration()
        
    if args.demo == "migraine" or args.demo == "all":
        demo_migraine_prediction()


if __name__ == "__main__":
    main() 