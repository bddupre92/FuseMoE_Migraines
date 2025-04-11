#!/usr/bin/env python
# PyGMO-FuseMOE Visualization Utilities
# Comprehensive visualization tools for evolutionary experts and PSO gating

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union, Any
import pandas as pd
from matplotlib.animation import FuncAnimation
import io
from PIL import Image
import base64
from IPython.display import HTML
import pygmo as pg
import os

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def visualize_expert_architecture(expert_model, title="Expert Architecture", filename=None):
    """
    Visualize the neural network architecture of an expert.
    
    Args:
        expert_model: PyTorch neural network model
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    G = nx.DiGraph()
    
    # Extract layers from the model
    layers = []
    for name, module in expert_model.named_modules():
        if name == '':  # Skip the top module
            continue
        layers.append((name, type(module).__name__))
    
    # Create nodes for input, layers, and output
    G.add_node("Input", layer_type="input")
    prev_node = "Input"
    
    # Add nodes and edges for each layer
    for i, (name, layer_type) in enumerate(layers):
        node_name = f"{layer_type}_{i}"
        G.add_node(node_name, layer_type=layer_type)
        G.add_edge(prev_node, node_name)
        prev_node = node_name
    
    # Add output node if not already present
    if not any("output" in G.nodes[n].get("layer_type", "").lower() for n in G.nodes()):
        G.add_node("Output", layer_type="output")
        G.add_edge(prev_node, "Output")
    
    # Create a hierarchical layout
    pos = nx.spring_layout(G)
    
    # Plot the network
    plt.figure(figsize=(12, 8))
    
    # Color nodes by layer type
    node_colors = []
    for node in G.nodes():
        layer_type = G.nodes[node].get('layer_type', '').lower()
        if 'input' in layer_type:
            node_colors.append('green')
        elif 'linear' in layer_type:
            node_colors.append('lightblue')
        elif 'relu' in layer_type or 'gelu' in layer_type or 'activation' in layer_type:
            node_colors.append('orange')
        elif 'output' in layer_type:
            node_colors.append('red')
        else:
            node_colors.append('gray')
    
    # Draw the network
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=2000, font_size=10, font_weight="bold",
            edge_color="gray", width=2, arrowsize=20)
    
    plt.title(title, fontsize=16)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    return plt.gcf()

def visualize_evolution_progress(evolution_history, title="Evolutionary Progress", filename=None):
    """
    Visualize the progress of evolutionary optimization.
    
    Args:
        evolution_history: List of dicts with fitness values over generations
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    generations = list(range(len(evolution_history)))
    
    # Extract metrics
    fitness_values = [entry.get('fitness', 0) for entry in evolution_history]
    accuracy_values = [entry.get('accuracy', 0) * 100 for entry in evolution_history]
    specialization_values = [entry.get('specialization', 0) for entry in evolution_history]
    
    # Create plot with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot fitness
    color = 'tab:blue'
    ax1.set_xlabel('Generation', fontsize=14)
    ax1.set_ylabel('Fitness (lower is better)', color=color, fontsize=14)
    ax1.plot(generations, fitness_values, marker='o', color=color, label='Fitness')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=14)
    ax2.plot(generations, accuracy_values, marker='s', color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Create third y-axis for specialization
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    color = 'tab:green'
    ax3.set_ylabel('Specialization', color=color, fontsize=14)
    ax3.plot(generations, specialization_values, marker='^', color=color, label='Specialization')
    ax3.tick_params(axis='y', labelcolor=color)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='center right')
    
    plt.title(title, fontsize=16)
    plt.grid(True)
    fig.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_expert_selection(selection_probs, title="Expert Selection Probabilities", filename=None):
    """
    Visualize how inputs are routed to different experts.
    
    Args:
        selection_probs: Array of expert selection probabilities [batch_size, num_experts]
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    if isinstance(selection_probs, torch.Tensor):
        selection_probs = selection_probs.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(selection_probs, cmap='viridis', cbar_kws={'label': 'Selection Probability'})
    
    plt.xlabel('Expert', fontsize=14)
    plt.ylabel('Sample', fontsize=14)
    plt.title(title, fontsize=16)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def visualize_expert_usage(usage_stats, title="Expert Usage", filename=None):
    """
    Visualize how frequently each expert is used.
    
    Args:
        usage_stats: Array of expert usage statistics
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    if isinstance(usage_stats, torch.Tensor):
        usage_stats = usage_stats.detach().cpu().numpy()
        
    plt.figure(figsize=(12, 8))
    
    num_experts = len(usage_stats)
    x = np.arange(num_experts)
    
    plt.bar(x, usage_stats, color='skyblue', edgecolor='navy')
    
    plt.xlabel('Expert', fontsize=14)
    plt.ylabel('Usage Frequency', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x, [f'Expert {i+1}' for i in range(num_experts)])
    plt.grid(axis='y')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def compare_expert_usage(before_usage, after_usage, title="Expert Usage Comparison", filename=None):
    """
    Compare expert usage before and after optimization.
    
    Args:
        before_usage: Expert usage before optimization
        after_usage: Expert usage after optimization
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    if isinstance(before_usage, torch.Tensor):
        before_usage = before_usage.detach().cpu().numpy()
    if isinstance(after_usage, torch.Tensor):
        after_usage = after_usage.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    
    num_experts = len(before_usage)
    x = np.arange(num_experts)
    width = 0.35
    
    plt.bar(x - width/2, before_usage, width, label='Before Optimization', color='lightblue', edgecolor='navy')
    plt.bar(x + width/2, after_usage, width, label='After Optimization', color='lightgreen', edgecolor='darkgreen')
    
    plt.xlabel('Expert', fontsize=14)
    plt.ylabel('Usage Frequency', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x, [f'Expert {i+1}' for i in range(num_experts)])
    plt.legend()
    plt.grid(axis='y')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def visualize_optimizer_comparison(optimizer_results, title="Optimizer Performance Comparison", filename=None):
    """
    Compare performance of different optimization algorithms.
    
    Args:
        optimizer_results: Dict mapping optimizer names to performance metrics
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=18)
    
    # Metrics to plot
    metrics = ['fitness', 'accuracy', 'time', 'generations']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Extract data
        optimizers = list(optimizer_results.keys())
        values = [optimizer_results[opt].get(metric, 0) for opt in optimizers]
        
        # Create bar plot
        bars = ax.bar(optimizers, values, color='skyblue', edgecolor='navy')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        ax.set_title(f"{metric.capitalize()}")
        ax.grid(axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_training_progress(training_history, title="Training Progress", filename=None):
    """
    Visualize the training progress over epochs.
    
    Args:
        training_history: Dict with lists of metrics over epochs
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    epochs = list(range(1, len(training_history.get('train_loss', [])) + 1))
    
    # Create plot with multiple lines
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot training and validation loss
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    
    if 'train_loss' in training_history:
        ax1.plot(epochs, training_history['train_loss'], 'b-', marker='o', label='Train Loss')
    if 'val_loss' in training_history:
        ax1.plot(epochs, training_history['val_loss'], 'g-', marker='s', label='Validation Loss')
    
    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    
    if 'train_accuracy' in training_history:
        ax2.plot(epochs, training_history['train_accuracy'], 'r-', marker='^', label='Train Accuracy')
    if 'val_accuracy' in training_history:
        ax2.plot(epochs, training_history['val_accuracy'], 'm-', marker='d', label='Validation Accuracy')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title(title, fontsize=16)
    plt.grid(True)
    fig.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_activation_landscape(activation_func, x_range=(-10, 10), num_points=1000, title=None, filename=None):
    """
    Visualize the activation function landscape.
    
    Args:
        activation_func: Function to visualize
        x_range: Range of x values
        num_points: Number of points to sample
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    # Generate x values
    x = torch.linspace(x_range[0], x_range[1], num_points)
    
    # Compute activation values
    if isinstance(activation_func, torch.nn.Module):
        with torch.no_grad():
            y = activation_func(x).numpy()
    else:
        y = activation_func(x)
        
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    
    # Plot activation function
    plt.plot(x.numpy(), y, 'b-', linewidth=2)
    
    plt.xlabel('Input', fontsize=14)
    plt.ylabel('Activation', fontsize=14)
    plt.title(title or f"Activation Landscape: {activation_func.__class__.__name__}", fontsize=16)
    plt.grid(True)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def create_evolution_animation(evolution_history, title="Evolutionary Progress Animation", filename=None):
    """
    Create an animation of the evolutionary optimization process.
    
    Args:
        evolution_history: List of dicts with metrics for each generation
        title: Animation title
        filename: If provided, save to this file
        
    Returns:
        HTML animation for Jupyter notebooks or path to saved file
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    generations = list(range(len(evolution_history)))
    
    # Extract metrics
    fitness_values = [entry.get('fitness', 0) for entry in evolution_history]
    
    # Check if we have valid fitness values
    if not fitness_values or all(f == 0 for f in fitness_values):
        # If no valid fitness values, create a static plot instead
        plt.text(0.5, 0.5, "No fitness data available for animation", 
                 ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.title(title, fontsize=16)
        
        if filename:
            # Ensure we're saving as a supported format
            if not filename.endswith(('.png', '.pdf', '.jpg', '.svg')):
                filename = os.path.splitext(filename)[0] + '.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            return filename
        else:
            return fig
    
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    padding = (max_fitness - min_fitness) * 0.1 if max_fitness > min_fitness else 0.1
    
    # Set up the plot
    ax.set_xlim(0, max(1, len(generations)))
    ax.set_ylim(min_fitness - padding, max_fitness + padding)
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Fitness', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    
    # Instead of animation, create a static plot for multiple frames
    if filename:
        # Ensure we're saving as a supported format
        if not filename.endswith(('.png', '.pdf', '.jpg', '.svg')):
            filename = os.path.splitext(filename)[0] + '.png'
            
        # Create a sequence plot instead of an animation
        plt.plot(generations, fitness_values, 'bo-', markersize=8, label='Fitness')
        
        # Highlight best point
        best_idx = fitness_values.index(min(fitness_values))
        plt.plot(best_idx, fitness_values[best_idx], 'ro', markersize=12, 
                 label=f'Best (Gen {best_idx})')
        
        plt.legend()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        return filename
        
    # For Jupyter notebooks, we'll still try to create an animation
    line, = ax.plot([], [], 'bo-', markersize=8)
    point, = ax.plot([], [], 'ro', markersize=12)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def update(frame):
        line.set_data(generations[:frame+1], fitness_values[:frame+1])
        point.set_data(frame, fitness_values[frame])
        ax.set_title(f"{title} - Generation {frame}")
        return line, point
    
    anim = FuncAnimation(fig, update, frames=len(generations),
                          init_func=init, blit=True)
    
    # For Jupyter notebook display
    video = anim.to_html5_video()
    html = HTML(video)
    plt.close()
    return html

def compare_model_performance(models_results, metric='accuracy', title="Model Performance Comparison", filename=None):
    """
    Compare performance of different models.
    
    Args:
        models_results: Dict mapping model names to performance metrics
        metric: Metric to compare ('accuracy', 'loss', etc.)
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    models = list(models_results.keys())
    values = [models_results[model].get(metric, 0) for model in models]
    
    # Create horizontal bar plot for better readability with many models
    y_pos = np.arange(len(models))
    
    # Sort by performance
    sorted_indices = np.argsort(values)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Create bar plot
    bars = plt.barh(y_pos, sorted_values, color='skyblue', edgecolor='navy')
    
    # Add value labels
    for i, (value, bar) in enumerate(zip(sorted_values, bars)):
        plt.text(value + max(sorted_values) * 0.01, i, f'{value:.3f}', 
                 va='center', fontsize=10)
    
    plt.yticks(y_pos, sorted_models)
    plt.xlabel(f'{metric.capitalize()}', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(axis='x')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def visualize_fitness_landscape(problem, dimensions=(0, 1), resolution=100, title="Fitness Landscape", filename=None):
    """
    Visualize the fitness landscape of an optimization problem.
    
    Args:
        problem: PyGMO problem object
        dimensions: Tuple of dimensions to visualize
        resolution: Resolution of the visualization grid
        title: Plot title
        filename: If provided, save to this file
        
    Returns:
        matplotlib figure
    """
    # Get bounds
    bounds = problem.get_bounds()
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])
    
    # Create grid for the two selected dimensions
    d1, d2 = dimensions
    x = np.linspace(lower_bounds[d1], upper_bounds[d1], resolution)
    y = np.linspace(lower_bounds[d2], upper_bounds[d2], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate fitness on grid
    Z = np.zeros_like(X)
    default_x = (lower_bounds + upper_bounds) / 2
    
    for i in range(resolution):
        for j in range(resolution):
            x_val = default_x.copy()
            x_val[d1] = X[i, j]
            x_val[d2] = Y[i, j]
            Z[i, j] = problem.fitness(x_val)[0]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    # Add contour plot at bottom
    offset = np.min(Z) - (np.max(Z) - np.min(Z)) * 0.1
    cset = ax.contour(X, Y, Z, zdir='z', offset=offset, cmap='viridis')
    
    ax.set_xlabel(f'Dimension {d1}', fontsize=12)
    ax.set_ylabel(f'Dimension {d2}', fontsize=12)
    ax.set_zlabel('Fitness', fontsize=12)
    ax.set_title(title, fontsize=16)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

# Utility function to convert Python file to Jupyter notebook
def py_to_jupyter(py_file, ipynb_file=None):
    """
    Convert a Python file with markdown comments to a Jupyter notebook.
    
    Args:
        py_file: Path to Python file
        ipynb_file: Path to output Jupyter notebook (default: same name with .ipynb)
        
    Returns:
        Path to the created notebook or None if nbformat is not available
    """
    if ipynb_file is None:
        ipynb_file = os.path.splitext(py_file)[0] + '.ipynb'
    
    try:
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    except ImportError:
        print("nbformat required to convert to Jupyter. Install with pip install nbformat")
        return None
    
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            py_content = f.read()
        
        # Create a new notebook
        nb = new_notebook()
        
        # Process multiline docstrings as markdown
        import re
        docstring_pattern = r'"""(.*?)"""'
        parts = re.split(docstring_pattern, py_content, flags=re.DOTALL)
        
        if parts[0].strip():
            nb.cells.append(new_code_cell(parts[0]))
        
        for i in range(1, len(parts), 2):
            if i < len(parts):  # Markdown content
                markdown_content = parts[i].strip()
                nb.cells.append(new_markdown_cell(markdown_content))
            if i+1 < len(parts):  # Code content
                code_content = parts[i+1].strip()
                if code_content:
                    nb.cells.append(new_code_cell(code_content))
        
        # Save the notebook
        with open(ipynb_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        return ipynb_file
    except Exception as e:
        print(f"Error converting Python to Jupyter: {e}")
        return None

def create_dashboard_notebook(experiment_folder, title="PyGMO-FuseMOE Experiment Dashboard"):
    """
    Create a Jupyter notebook dashboard for visualizing experiment results.
    
    Args:
        experiment_folder: Folder containing experiment results
        title: Dashboard title
        
    Returns:
        Path to the created notebook or None if nbformat is not available
    """
    try:
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    except ImportError:
        print("nbformat module not found. To create Jupyter notebooks, install it with:")
        print("pip install nbformat")
        # Create a simple HTML dashboard instead
        html_path = os.path.join(experiment_folder, "dashboard.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"<html>\n<head>\n<title>{title}</title>\n</head>\n<body>\n")
            f.write(f"<h1>{title}</h1>\n")
            
            # Add image sections
            f.write("<h2>Visualizations</h2>\n")
            for img_type in ["expert_architecture", "evolution_progress", "evolution_animation", 
                             "optimizer_comparison", "expert_usage_comparison"]:
                img_file = os.path.join(experiment_folder, f"{img_type}.png")
                if os.path.exists(img_file):
                    f.write(f"<div>\n<h3>{img_type.replace('_', ' ').title()}</h3>\n")
                    f.write(f"<img src='{img_type}.png' style='max-width:800px;'>\n</div>\n")
            
            f.write("</body>\n</html>")
        
        print(f"Created HTML dashboard instead: {html_path}")
        return html_path
    
    # Create a new notebook
    nb = new_notebook()
    
    # Add title
    nb.cells.append(new_markdown_cell(f"# {title}"))
    
    # Add import cell
    imports_cell = """
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML

# Add visualization module to path
sys.path.append("../../src/utils")
import visualization as viz
    """
    nb.cells.append(new_code_cell(imports_cell))
    
    # Add experiment loading cell
    load_cell = f"""
# Load experiment data
experiment_folder = "{experiment_folder}"
results_file = os.path.join(experiment_folder, "results.pkl")

import pickle
with open(results_file, 'rb') as f:
    results = pickle.load(f)
    
print(f"Loaded experiment results from {results_file}")
    """
    nb.cells.append(new_code_cell(load_cell))
    
    # Add sections for different visualizations
    nb.cells.append(new_markdown_cell("## Model Performance Comparison"))
    model_compare_cell = """
# Compare baseline vs optimized models
model_results = {
    'Baseline': results.get('baseline', {}),
    'PyGMO-FuseMOE': results.get('optimized', {})
}

viz.compare_model_performance(model_results, metric='accuracy', 
                             title="Accuracy Comparison", 
                             filename=os.path.join(experiment_folder, "accuracy_comparison.png"))
"""
    nb.cells.append(new_code_cell(model_compare_cell))
    
    # Expert usage section
    nb.cells.append(new_markdown_cell("## Expert Usage Analysis"))
    expert_usage_cell = """
# Compare expert usage before and after optimization
before_usage = results.get('baseline', {}).get('expert_usage', None)
after_usage = results.get('optimized', {}).get('expert_usage', None)

if before_usage is not None and after_usage is not None:
    viz.compare_expert_usage(before_usage, after_usage, 
                           title="Expert Usage Before vs After Optimization",
                           filename=os.path.join(experiment_folder, "expert_usage_comparison.png"))
"""
    nb.cells.append(new_code_cell(expert_usage_cell))
    
    # Evolution progress section
    nb.cells.append(new_markdown_cell("## Evolution Progress"))
    evolution_cell = """
# Visualize evolution progress
evolution_history = results.get('evolution_history', [])

if evolution_history:
    viz.visualize_evolution_progress(evolution_history, 
                                   title="Evolutionary Optimization Progress",
                                   filename=os.path.join(experiment_folder, "evolution_progress.png"))
    
    # Create animation
    viz.create_evolution_animation(evolution_history, 
                                 title="Evolution Animation",
                                 filename=os.path.join(experiment_folder, "evolution_animation.png"))
"""
    nb.cells.append(new_code_cell(evolution_cell))
    
    # Optimizer comparison section
    nb.cells.append(new_markdown_cell("## Optimizer Comparison"))
    optimizer_cell = """
# Compare different optimizers
optimizer_results = results.get('optimizer_comparison', {})

if optimizer_results:
    viz.visualize_optimizer_comparison(optimizer_results, 
                                     title="Optimizer Performance Comparison",
                                     filename=os.path.join(experiment_folder, "optimizer_comparison.png"))
"""
    nb.cells.append(new_code_cell(optimizer_cell))
    
    # Training progress section
    nb.cells.append(new_markdown_cell("## Training Progress"))
    training_cell = """
# Visualize training progress
training_history = results.get('training_history', {})

if training_history:
    viz.visualize_training_progress(training_history, 
                                  title="Training Progress",
                                  filename=os.path.join(experiment_folder, "training_progress.png"))
"""
    nb.cells.append(new_code_cell(training_cell))
    
    # Expert architecture section
    nb.cells.append(new_markdown_cell("## Expert Architecture Visualization"))
    architecture_cell = """
# Visualize expert architectures
expert_models = results.get('expert_models', [])

if expert_models:
    for i, expert in enumerate(expert_models):
        if expert is not None:
            viz.visualize_expert_architecture(expert, 
                                           title=f"Expert {i+1} Architecture",
                                           filename=os.path.join(experiment_folder, f"expert_{i+1}_architecture.png"))
"""
    nb.cells.append(new_code_cell(architecture_cell))
    
    # Save the notebook
    notebook_path = os.path.join(experiment_folder, "dashboard.ipynb")
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        return notebook_path
    except Exception as e:
        print(f"Error creating notebook: {e}")
        return None 