#!/usr/bin/env python
# Test script to demonstrate optimization of FuseMOE components with PyTorch

import sys
import os
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import copy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FuseMOE components
from utils.config import MoEConfig
from core.sparse_moe import MLP

# Define a simplified version of MoE for testing
class SimpleMoE(torch.nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size):
        super(SimpleMoE, self).__init__()
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            MLP(MoEConfig(
                num_experts=1,
                moe_input_size=input_size,
                moe_hidden_size=hidden_size,
                moe_output_size=output_size,
                router_type='joint'
            ), input_size, output_size, hidden_size)
            for _ in range(num_experts)
        ])
        
        # Simple gating network
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_experts),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_size]
        
        # Get gating weights
        gates = self.gate(x).unsqueeze(-1)  # [batch, num_experts, 1]
        
        # Combine outputs
        combined = (expert_outputs * gates).sum(dim=1)  # [batch, output_size]
        return combined


def create_synthetic_data(input_size, num_classes=4, samples_per_class=50):
    """Create a multi-class dataset with multiple clusters"""
    torch.manual_seed(42)
    all_data = []
    all_labels = []
    
    # Create 4 different clusters in different quadrants
    centers = [
        torch.tensor([3.0, 3.0] + [0.0] * (input_size - 2)),
        torch.tensor([-3.0, 3.0] + [0.0] * (input_size - 2)),
        torch.tensor([-3.0, -3.0] + [0.0] * (input_size - 2)),
        torch.tensor([3.0, -3.0] + [0.0] * (input_size - 2))
    ]
    
    # Add some dimension-specific features for each class
    for i in range(num_classes):
        # Create unique patterns in different dimensions for each class
        if i < len(centers):
            center = centers[i]
            
            # Add class-specific features in additional dimensions
            if input_size > 5:
                # Set a few dimensions to have strong signal
                strong_dims = [4 + i, 5 + i % 2, 6 + i % 3]
                for dim in strong_dims:
                    if dim < input_size:
                        center[dim] = 2.0 * ((-1) ** i)
            
            # Create samples around this center with noise
            samples = center.unsqueeze(0) + torch.randn(samples_per_class, input_size) * 0.5
            all_data.append(samples)
            all_labels.append(torch.ones(samples_per_class, 1) * i)
    
    # Combine all data
    X = torch.cat(all_data, dim=0)
    y_multiclass = torch.cat(all_labels, dim=0).long()
    
    # Create one-hot encoded labels
    y = torch.zeros(len(y_multiclass), num_classes)
    y.scatter_(1, y_multiclass, 1)
    
    return X, y, y_multiclass.squeeze()


def evaluate_model(model, X, y_multiclass, loss_fn, num_classes, prefix=""):
    """Evaluate model performance and expert usage"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = loss_fn(outputs, y_multiclass).item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_multiclass).float().mean().item() * 100
        
        # Look at expert usage
        gate_values = model.gate(X)
        expert_usage = gate_values.mean(dim=0)
        
        # Calculate per-class expert usage
        per_class_usage = []
        for class_idx in range(num_classes):
            class_mask = (y_multiclass == class_idx)
            if class_mask.sum() > 0:
                class_gate_values = gate_values[class_mask]
                class_expert_usage = class_gate_values.mean(dim=0)
                per_class_usage.append(class_expert_usage)
    
    # Print results
    print(f"\n{prefix} Model Evaluation:")
    print(f"Loss: {loss:.6f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    print(f"\n{prefix} Overall expert usage:")
    for i, usage in enumerate(expert_usage):
        print(f"  Expert {i+1}: {usage.item():.4f}")
    
    print(f"\n{prefix} Per-class expert usage:")
    for class_idx, usage in enumerate(per_class_usage):
        print(f"  Class {class_idx}:")
        for expert_idx, expert_usage in enumerate(usage):
            print(f"    Expert {expert_idx+1}: {expert_usage.item():.4f}")
    
    return loss, accuracy, expert_usage, per_class_usage


def plot_comparison(original_usage, optimized_usage, num_experts, num_classes, per_class_original=None, per_class_optimized=None):
    """Create comparison plots for expert usage before and after optimization"""
    # Plot overall expert usage comparison
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(num_experts)
    
    # Convert tensors to numpy for plotting
    # Handle both 0-d and multi-d tensors
    if isinstance(original_usage, torch.Tensor):
        if original_usage.dim() == 0:  # 0-d tensor (scalar)
            original_values = [original_usage.item()]
        else:
            original_values = [u.item() for u in original_usage]
    else:
        original_values = [u.item() if isinstance(u, torch.Tensor) else u for u in original_usage]
    
    if isinstance(optimized_usage, torch.Tensor):
        if optimized_usage.dim() == 0:  # 0-d tensor (scalar)
            optimized_values = [optimized_usage.item()]
        else:
            optimized_values = [u.item() for u in optimized_usage]
    else:
        optimized_values = [u.item() if isinstance(u, torch.Tensor) else u for u in optimized_usage]
    
    plt.bar(x - width/2, original_values, width, label='Original')
    plt.bar(x + width/2, optimized_values, width, label='Optimized')
    
    plt.xlabel('Expert')
    plt.ylabel('Usage')
    plt.title('Expert Usage Comparison: Original vs Optimized')
    plt.xticks(x, [f'E{i+1}' for i in range(num_experts)])
    plt.legend()
    plt.grid(axis='y')
    plt.savefig('expert_usage_comparison.png')
    
    # Plot per-class expert usage for original and optimized
    if per_class_original is not None and per_class_optimized is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Per-Class Expert Usage: Original vs Optimized', fontsize=16)
        
        for class_idx in range(num_classes):
            row, col = divmod(class_idx, 2)
            ax = axes[row, col]
            
            # Handle both 0-d and multi-d tensors for per-class usage
            if isinstance(per_class_original[class_idx], torch.Tensor):
                if per_class_original[class_idx].dim() == 0:
                    original = [per_class_original[class_idx].item()]
                else:
                    original = [u.item() for u in per_class_original[class_idx]]
            else:
                original = [u.item() if isinstance(u, torch.Tensor) else u for u in per_class_original[class_idx]]
            
            if isinstance(per_class_optimized[class_idx], torch.Tensor):
                if per_class_optimized[class_idx].dim() == 0:
                    optimized = [per_class_optimized[class_idx].item()]
                else:
                    optimized = [u.item() for u in per_class_optimized[class_idx]]
            else:
                optimized = [u.item() if isinstance(u, torch.Tensor) else u for u in per_class_optimized[class_idx]]
            
            ax.bar(x - width/2, original, width, label='Original')
            ax.bar(x + width/2, optimized, width, label='Optimized')
            
            ax.set_xlabel('Expert')
            ax.set_ylabel('Usage')
            ax.set_title(f'Class {class_idx}')
            ax.set_xticks(x)
            ax.set_xticklabels([f'E{i+1}' for i in range(num_experts)])
            ax.legend()
            ax.grid(axis='y')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('per_class_expert_usage_comparison.png')
        
    print("\nComparison charts saved to expert_usage_comparison.png and per_class_expert_usage_comparison.png")


def main():
    print("Testing PyTorch optimization with FuseMOE")
    
    # Define model dimensions
    input_size = 20
    num_classes = 4
    output_size = num_classes
    hidden_size = 32
    num_experts = 8
    
    # Create synthetic data
    X, y_onehot, y_multiclass = create_synthetic_data(
        input_size=input_size, 
        num_classes=num_classes, 
        samples_per_class=100
    )
    print(f"Created dataset with {X.shape[0]} samples, {input_size} features, {num_classes} classes")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create MoE model
    model = SimpleMoE(
        input_size=input_size,
        output_size=output_size,
        num_experts=num_experts,
        hidden_size=hidden_size
    )
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Evaluate model before optimization
    original_model = copy.deepcopy(model)
    original_loss, original_accuracy, original_usage, original_per_class = evaluate_model(
        original_model, X, y_multiclass, loss_fn, num_classes, prefix="Original (Unoptimized)"
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Training loop
    print("\nStarting optimization...")
    start_time = time.time()
    
    num_epochs = 200
    losses = []
    
    for epoch in range(num_epochs):
        # Forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y_multiclass)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_multiclass).float().mean().item() * 100
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.6f}, Accuracy = {accuracy:.2f}%")
    
    end_time = time.time()
    
    # Evaluate model after optimization
    optimized_loss, optimized_accuracy, optimized_usage, optimized_per_class = evaluate_model(
        model, X, y_multiclass, loss_fn, num_classes, prefix="Optimized"
    )
    
    # Print optimization summary
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    print(f"Improvement:")
    print(f"  Loss:     {original_loss:.6f} -> {optimized_loss:.6f} (Δ {original_loss - optimized_loss:.6f})")
    print(f"  Accuracy: {original_accuracy:.2f}% -> {optimized_accuracy:.2f}% (Δ {optimized_accuracy - original_accuracy:.2f}%)")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('moe_training_loss.png')
    print("\nLoss curve saved to moe_training_loss.png")
    
    # Plot expert usage by class
    plt.figure(figsize=(12, 8))
    width = 0.2
    x = np.arange(num_experts)
    
    for i, usage in enumerate(optimized_per_class):
        plt.bar(x + i*width, usage.numpy(), width=width, label=f'Class {i}')
    
    plt.xlabel('Expert')
    plt.ylabel('Usage')
    plt.title('Expert Usage by Class (Optimized)')
    plt.xticks(x + width * (num_classes-1)/2, [f'E{i+1}' for i in range(num_experts)])
    plt.legend()
    plt.grid(axis='y')
    plt.savefig('expert_usage_by_class.png')
    
    # Create comparison plots
    plot_comparison(original_usage, optimized_usage, num_experts, num_classes, 
                   original_per_class, optimized_per_class)
    
    return 0


if __name__ == "__main__":
    main() 