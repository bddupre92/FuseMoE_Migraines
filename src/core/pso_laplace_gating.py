#!/usr/bin/env python
# PSO-Enhanced Laplace Gating for FuseMOE
# Implements Particle Swarm Optimization for gating weight optimization

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygmo as pg
import time
import random
from typing import List, Dict, Tuple, Optional, Union, Any
from utils.config import MoEConfig
from sklearn.model_selection import train_test_split

class LaplaceActivation(nn.Module):
    """
    Laplace activation function with trainable scale parameter.
    
    The Laplace activation provides sharper gating decisions compared to softmax,
    leading to more specialized expert selection.
    
    f(x) = exp(-|x - μ| / b) / (2b)
    
    where μ is the mean (location) and b is the scale parameter.
    """
    
    def __init__(self, scale=1.0, trainable=True):
        """
        Initialize Laplace activation.
        
        Args:
            scale: Initial scale parameter
            trainable: Whether scale parameter should be learnable
        """
        super(LaplaceActivation, self).__init__()
        if trainable:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        else:
            self.register_buffer('scale', torch.tensor(scale, dtype=torch.float))
            
    def forward(self, x, mean=0.0):
        """
        Apply Laplace activation.
        
        Args:
            x: Input tensor
            mean: Mean/location parameter
            
        Returns:
            Activated tensor
        """
        # Ensure scale is positive
        scale = F.softplus(self.scale)
        
        # Apply Laplace activation 
        # f(x) = exp(-|x - μ| / b) / (2*b)
        x_centered = torch.abs(x - mean)
        activated = torch.exp(-x_centered / scale) / (2 * scale)
        
        # Normalize to ensure values sum to 1 (like softmax)
        activated = activated / (activated.sum(dim=-1, keepdim=True) + 1e-12)
        
        return activated


class PSOLaplaceGating(nn.Module):
    """
    PSO-enhanced Laplace gating mechanism for MoE.
    
    This module replaces standard softmax gating with Laplace activation,
    which provides sharper expert selection. The parameters of the gating
    mechanism are optimized using Particle Swarm Optimization.
    """
    
    def __init__(self, 
                 config: MoEConfig,
                 input_size: int,
                 num_experts: int,
                 hidden_size: int = 64,
                 scale_init: float = 1.0,
                 dropout: float = 0.1):
        """
        Initialize PSO Laplace gating.
        
        Args:
            config: MoE configuration
            input_size: Size of input features
            num_experts: Number of experts to gate
            hidden_size: Hidden layer size for gating network
            scale_init: Initial scale parameter for Laplace activation
            dropout: Dropout probability
        """
        super(PSOLaplaceGating, self).__init__()
        
        self.config = config
        self.input_size = input_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.scale_init = scale_init
        
        # Define gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts),
        )
        
        # Laplace activation instead of softmax
        self.laplace = LaplaceActivation(scale=scale_init)
        
        # For tracking expert usage
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('batch_count', torch.tensor(0))
        
    def forward(self, x):
        """
        Forward pass to compute gating weights.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            gates: Gating weights [batch_size, num_experts]
        """
        # Get raw logits from gating network
        logits = self.gate_network(x)
        
        # Apply Laplace activation
        gates = self.laplace(logits)
        
        # Update expert usage statistics
        with torch.no_grad():
            batch_usage = gates.mean(dim=0)
            self.expert_usage = (self.expert_usage * self.batch_count + batch_usage) / (self.batch_count + 1)
            self.batch_count += 1
            
        return gates
    
    def get_expert_usage(self):
        """Return the expert usage statistics"""
        return self.expert_usage.clone()
    
    def reset_usage_stats(self):
        """Reset expert usage statistics"""
        self.expert_usage.zero_()
        self.batch_count.zero_()


class PSOGatingProblem:
    """
    PyGMO problem wrapper for PSO-enhanced gating optimization.
    
    This class optimizes the parameters of the gating network using
    Particle Swarm Optimization.
    """
    
    def __init__(self, 
                 moe_model, 
                 gating_model, 
                 input_data, # Encoded: [Batch*Window, EncodedDim]
                 original_input_dict, # Dict: {mod: [Batch, Window, Feat]}
                 target_data, # Expanded: [Batch*Window, 1]
                 original_target_data, # Original: [Batch, 1]
                 window_size, # Needed for index mapping
                 validation_split=0.2, 
                 population_size=20, 
                 load_balance_coef=0.1, 
                 device='cpu',
                 seed=42):
        """Initialize the PyGMO problem for gating optimization."""
        self.moe_model = moe_model
        self.gating_model = gating_model
        self.encoded_input_data = input_data.to(device)
        self.original_input_dict = {mod: tensor.to(device) for mod, tensor in original_input_dict.items()}
        self.expanded_target_data = target_data.to(device)
        self.original_target_data = original_target_data.to(device)
        self.window_size = window_size
        self.validation_split = validation_split
        self.population_size = population_size
        self.load_balance_coef = load_balance_coef
        self.device = device
        self.seed = seed
        
        # Extract parameters and dimensions from the gating_model
        self.params = [p.data.clone() for p in self.gating_model.parameters() if p.requires_grad]
        self.param_shapes = [p.shape for p in self.params]
        self.param_sizes = [p.numel() for p in self.params]
        self.dim = sum(self.param_sizes)
        
        # History tracking
        self.history = []
        self.best_fitness = float('inf')
        self.best_solution = None
        self.eval_count = 0
        
        # --- Train/Validation Split based on Original Batch Indices --- 
        batch_size = self.original_target_data.shape[0]
        indices = np.arange(batch_size)
        
        try:
            # Stratify based on ORIGINAL targets
            train_indices, val_indices = train_test_split(
                indices, 
                test_size=self.validation_split, 
                random_state=self.seed,
                stratify=self.original_target_data.cpu().numpy() 
            )
            
            # Create validation dictionary using batch indices [ValBatch, Window, Feat]
            self.val_inputs_dict = { 
                mod: data[val_indices] 
                for mod, data in self.original_input_dict.items()
            }
            # Store original validation targets [ValBatch, 1]
            self.val_targets = self.original_target_data[val_indices]
            
            # Create encoded validation inputs needed for gating model [ValBatch*Window, EncodedDim]
            # Map batch indices to expanded indices (Batch*Window)
            expanded_val_indices = []
            for idx in val_indices:
                expanded_val_indices.extend(range(idx * self.window_size, (idx + 1) * self.window_size))
            self.encoded_val_inputs = self.encoded_input_data[expanded_val_indices]
            
            # Store expanded validation targets [ValBatch*Window, 1]
            self.expanded_val_targets = self.expanded_target_data[expanded_val_indices]

            
            print(f"[PSOGatingProblem Init] Validation target counts (original): {np.unique(self.val_targets.cpu().numpy(), return_counts=True)}")
            # Example print - get first modality key dynamically
            first_mod_key = next(iter(self.val_inputs_dict)) if self.val_inputs_dict else None
            if first_mod_key:
                print(f"[PSOGatingProblem Init] Shape of val_inputs_dict['{first_mod_key}']: {self.val_inputs_dict[first_mod_key].shape}")
            print(f"[PSOGatingProblem Init] Shape of encoded_val_inputs: {self.encoded_val_inputs.shape}")
            print(f"[PSOGatingProblem Init] Shape of val_targets: {self.val_targets.shape}")
            print(f"[PSOGatingProblem Init] Shape of expanded_val_targets: {self.expanded_val_targets.shape}")
            
        except ValueError as e:
            print(f"Warning: Stratified split failed: {e}. Performing regular split.")
            # Fallback to non-stratified split on indices if necessary
            train_indices, val_indices = train_test_split(
                indices, test_size=self.validation_split, random_state=self.seed
            )
            # Recreate dicts/tensors as above
            self.val_inputs_dict = { 
                mod: data[val_indices] 
                for mod, data in self.original_input_dict.items()
            }
            self.val_targets = self.original_target_data[val_indices]
            expanded_val_indices = []
            for idx in val_indices:
                expanded_val_indices.extend(range(idx * self.window_size, (idx + 1) * self.window_size))
            self.encoded_val_inputs = self.encoded_input_data[expanded_val_indices]
            self.expanded_val_targets = self.expanded_target_data[expanded_val_indices]
        # --- End Split ---
        
    def get_bounds(self):
        """
        Get the bounds of the decision variables.
        
        Returns:
            Tuple of lower and upper bounds
        """
        # Using fairly wide bounds for network parameters
        lb = [-5.0] * self.dim
        ub = [5.0] * self.dim
        
        return (lb, ub)
    
    def get_nobj(self):
        """Number of objectives (single objective optimization)"""
        return 1
    
    def _vector_to_parameters(self, x):
        """
        Convert flat vector to model parameters.
        
        Args:
            x: Flat parameter vector
            
        Returns:
            List of parameter tensors
        """
        params = []
        idx = 0
        
        for shape, size in zip(self.param_shapes, self.param_sizes):
            param_data = x[idx:idx+size].reshape(shape)
            params.append(torch.tensor(param_data, dtype=torch.float, device=self.device))
            idx += size
            
        return params
    
    def _update_model_parameters(self, params):
        """
        Update model with the given parameters.
        
        Args:
            params: List of parameter tensors
        """
        idx = 0
        for param in self.gating_model.parameters():
            if param.requires_grad:
                param.data.copy_(params[idx])
                idx += 1
    
    def _calculate_load_balance(self, expert_usage):
        """
        Calculate load balancing score.
        Higher is better - experts should be used equally.
        
        Args:
            expert_usage: Vector of expert usage frequencies
            
        Returns:
            Load balancing score
        """
        # Ideal usage is uniform distribution across experts
        num_experts = len(expert_usage)
        ideal_usage = 1.0 / num_experts
        
        # Calculate squared deviation from ideal
        if isinstance(expert_usage, torch.Tensor):
            deviation = torch.mean((expert_usage - ideal_usage) ** 2).item()
        else:
            deviation = np.mean((np.array(expert_usage) - ideal_usage) ** 2)
            
        # Negative deviation (higher is better)
        return -deviation
    
    def fitness(self, x):
        """
        Evaluate the fitness of a solution.
        
        Args:
            x: Solution vector from PyGMO
            
        Returns:
            Tuple of fitness values (combined_loss,)
        """
        # Set parameters in the gating model
        params = self._vector_to_parameters(x)
        param_idx = 0
        for model_param in self.gating_model.parameters():
            if model_param.requires_grad:
                model_param.data = params[param_idx]
                param_idx += 1
        
        # Ensure model is in evaluation mode for consistent results
        self.moe_model.eval()
        self.gating_model.eval()
        
        loss_fn = nn.BCEWithLogitsLoss()
        total_val_loss = 0.0
        total_accuracy = 0.0
        total_load_balance = 0.0
        num_batches = 0
        
        with torch.no_grad():
            # Use validation data for fitness evaluation
            # Gating model expects encoded inputs [ValBatch*Window, EncodedDim]
            val_gates = self.gating_model(self.encoded_val_inputs)
            
            # Main MoE model expects dictionary input [ValBatch, Window, Feat]
            # It returns (final_output, gates)
            val_outputs, returned_gates = self.moe_model(self.val_inputs_dict, gates=val_gates)
            
            # Ensure targets are float and have same shape as output for BCEWithLogitsLoss
            # The moe_model output is [ValBatch, 1], so use self.val_targets
            val_loss = loss_fn(val_outputs, self.val_targets.float()).item()

            # Calculate accuracy
            # Use original targets (shape [ValBatch, 1])
            predicted = (torch.sigmoid(val_outputs) > 0.5).float() # Shape [ValBatch, 1]
            accuracy = (predicted == self.val_targets).float().mean().item()
            
            # Calculate load balancing score (variance of gate usage)
            # Lower variance is better for load balancing
            gate_usage_variance = torch.var(val_gates.mean(dim=0)).item()
            load_balance_score = gate_usage_variance 

            total_val_loss = val_loss
            total_accuracy = accuracy
            total_load_balance = load_balance_score
            
        # Combined fitness: minimize loss and load imbalance
        fitness = total_val_loss + self.load_balance_coef * total_load_balance # Lower is better
        
        # Track best solution
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = x
        
        # Add detailed history entry
        self.eval_count += 1
        self.history.append({
            'eval_count': self.eval_count,
            'fitness': fitness,
            'loss': total_val_loss,
            'accuracy': total_accuracy,
            'load_balance': total_load_balance
        })
            
        return [fitness]
    
    def optimize(self, algorithm_id='pso', seed=42, verbosity=1):
        """
        Run the PSO optimization process.
        
        Args:
            algorithm_id: Algorithm ID ('pso', 'abc', 'sade')
            seed: Random seed
            verbosity: PyGMO verbosity level
            
        Returns:
            Optimized gating model, full history, and algorithm ID
        """
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create PyGMO problem
        prob = pg.problem(self)
        
        # Select algorithm
        algo = None # Initialize
        generations = 10 # Define generations
        if algorithm_id == 'pso':
            algo = pg.algorithm(pg.pso(gen=generations, seed=seed))
        elif algorithm_id == 'sade':
            algo = pg.algorithm(pg.sade(gen=generations, seed=seed))
        elif algorithm_id == 'abc':
            algo = pg.algorithm(pg.bee_colony(gen=generations, limit=1, seed=seed))
        else:
            algo = pg.algorithm(pg.pso(gen=generations, seed=seed))
        
        # Set verbosity
        algo.set_verbosity(verbosity)
        
        # Create population
        pop = pg.population(prob, size=self.population_size, seed=seed)
        
        # --- History Logging (Generation-based) ---
        self.history = [] # Ensure history is clear before starting
        start_time = time.time()
        
        for gen in range(generations):
            pop = algo.evolve(pop)
            # Log population stats after each generation
            fitnesses = pop.get_f()
            best_idx = pop.best_idx()
            self.history.append({
                'generation': gen + 1,
                'best_fitness': fitnesses[best_idx][0],
                'avg_fitness': np.mean(fitnesses),
                'eval_count': pop.problem.get_fevals() # Use PyGMO's built-in method
                # Add other relevant metrics if available directly from population/problem
            })
            # Optional: Log more details if needed
        
        evolve_end = time.time()
        # --- End History Logging ---
        
        end_time = time.time()
        
        best_x = pop.champion_x
        best_f = pop.champion_f[0]
        
        # --- REMOVED old history append logic ---
        
        print(f"   PSO Exit condition -- {algo.get_extra_info()}")
        print(f"Gating optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best gating fitness: {best_f}")
        
        # Set the gating model parameters to the best found solution
        best_params = self._vector_to_parameters(best_x)
        param_idx = 0
        for model_param in self.gating_model.parameters():
            if model_param.requires_grad:
                model_param.data = best_params[param_idx]
                param_idx += 1
        
        # Return the optimized gating model, full history, and algorithm ID
        return self.gating_model, self.history, algorithm_id


class MoEWithPSOGating(nn.Module):
    """
    Mixture of Experts with PSO-enhanced Laplace gating.
    
    This class combines standard MoE with the PSO-optimized Laplace gating
    mechanism.
    """
    
    def __init__(self, 
                 config: MoEConfig,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_experts: int,
                 dropout: float = 0.1):
        """
        Initialize MoE with PSO gating.
        
        Args:
            config: MoE configuration
            input_size: Size of input features
            hidden_size: Size of hidden layers
            output_size: Size of output features
            num_experts: Number of experts
            dropout: Dropout probability
        """
        super(MoEWithPSOGating, self).__init__()
        
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size)
            )
            for _ in range(num_experts)
        ])
        
        # Create PSO-enhanced Laplace gating
        self.gating = PSOLaplaceGating(
            config=config,
            input_size=input_size,
            num_experts=num_experts,
            hidden_size=hidden_size // 2,  # Smaller than expert networks
            scale_init=1.0,
            dropout=dropout
        )
        
    def forward(self, x, gates=None):
        """
        Forward pass through the MoE model.
        
        Args:
            x: Input tensor [batch_size, input_size]
            gates: Pre-computed gates (optional)
            
        Returns:
            outputs: Model outputs [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Get gating weights if not provided
        if gates is None:
            gates = self.gating(x)  # [batch_size, num_experts]
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch_size, output_size]
            expert_outputs.append(expert_out)
            
        # Stack expert outputs
        stacked_experts = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_size]
        
        # Apply gates
        gates_expanded = gates.unsqueeze(-1)  # [batch_size, num_experts, 1]
        combined = (stacked_experts * gates_expanded).sum(dim=1)  # [batch_size, output_size]
        
        return combined
    
    def get_expert_usage(self):
        """Return expert usage statistics from gating network"""
        return self.gating.get_expert_usage()


# Utility functions for PSO gating optimization

def optimize_gating_with_pso(moe_model: nn.Module, 
                           gating_model: PSOLaplaceGating, 
                           data: Tuple[torch.Tensor, torch.Tensor],
                           algorithm: str = 'pso',
                           population_size: int = 20,
                           load_balance_coef: float = 0.1,
                           seed: int = 42):
    """
    Optimize gating mechanism using PSO.
    
    Args:
        moe_model: MoE model with PSO gating
        gating_model: The gating model to optimize
        data: Tuple of (inputs, targets) for training/evaluation
        algorithm: PSO variant to use ('pso', 'abc', 'sade')
        population_size: Size of population
        load_balance_coef: Coefficient for load balancing
        seed: Random seed
        
    Returns:
        Optimized model and full history
    """
    inputs, targets = data
    
    # Configure the PSO problem
    problem = PSOGatingProblem(
        moe_model=moe_model,
        gating_model=gating_model,
        input_data=inputs,
        original_input_dict={},
        target_data=targets,
        original_target_data=targets,
        window_size=1,
        validation_split=0.2,
        population_size=population_size,
        load_balance_coef=load_balance_coef,
        device='cpu',
        seed=seed
    )
    
    # Run optimization
    optimized_gating, history, algo_used = problem.optimize(
        algorithm_id=algorithm,
        seed=seed,
        verbosity=1
    )
    
    # Print final metrics if needed (or rely on history)
    print(f"\nPSO Gating Optimization - Final Best Fitness: {problem.best_fitness:.4f}")
    
    # Return the optimized gating model and the history
    return optimized_gating, history 