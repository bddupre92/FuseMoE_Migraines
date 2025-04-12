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
                 moe_model: nn.Module,
                 gating_model: PSOLaplaceGating,
                 input_data: torch.Tensor,
                 target_data: torch.Tensor,
                 validation_split: float = 0.2,
                 population_size: int = 20,
                 load_balance_coef: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the PSO gating optimization problem.
        
        Args:
            moe_model: The MoE model containing experts
            gating_model: The gating model to optimize
            input_data: Training input data
            target_data: Training target data
            validation_split: Fraction of data to use for validation
            population_size: Size of population for PSO
            load_balance_coef: Coefficient for load balancing objective
            device: Device to run computations on
        """
        self.moe_model = moe_model
        self.gating_model = gating_model
        self.population_size = population_size
        self.load_balance_coef = load_balance_coef
        self.device = device
        
        # Prepare data
        val_size = int(len(input_data) * validation_split)
        indices = torch.randperm(len(input_data))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        self.train_inputs = input_data[train_indices].to(device)
        self.train_targets = target_data[train_indices].to(device)
        self.val_inputs = input_data[val_indices].to(device)
        self.val_targets = target_data[val_indices].to(device)
        
        # Extract parameters from gating model for optimization
        self.param_shapes = []
        self.param_sizes = []
        total_params = 0
        
        for param in self.gating_model.parameters():
            if param.requires_grad:
                self.param_shapes.append(param.shape)
                size = param.numel()
                self.param_sizes.append(size)
                total_params += size
        
        self.dim = total_params
        self.best_fitness = float('inf')
        self.best_solution = None
        self.history = []
        
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
            Tuple of fitness values (loss,)
        """
        # Convert solution vector to model parameters
        params = self._vector_to_parameters(x)
        
        # Update model with these parameters
        self._update_model_parameters(params)
        
        # Evaluate model 
        self.moe_model.eval()
        self.gating_model.eval()
        
        with torch.no_grad():
            # Reset expert usage statistics
            self.gating_model.reset_usage_stats()
            
            # Forward pass through gating and MoE model
            train_gates = self.gating_model(self.train_inputs)
            train_outputs = self.moe_model(self.train_inputs, gates=train_gates)
            
            val_gates = self.gating_model(self.val_inputs)
            val_outputs = self.moe_model(self.val_inputs, gates=val_gates)
            
            # Calculate training loss
            loss_fn = nn.BCEWithLogitsLoss()
            # Squeeze both outputs[0] (logits) and targets to match shape [N_train]
            train_loss = loss_fn(train_outputs[0].squeeze(), self.train_targets.squeeze().float()).item()
            
            # Validation phase
            with torch.no_grad():
                self.moe_model.eval()
                # Model returns tuple: (outputs, gates, gates)
                val_outputs_tuple = self.moe_model(self.val_inputs)
                val_outputs = val_outputs_tuple[0] # Extract actual outputs/logits
                
                # Squeeze both outputs[0] (logits) and targets to match shape [N_val]
                val_loss = loss_fn(val_outputs.squeeze(), self.val_targets.squeeze().float()).item()
                self.moe_model.train() # Set back to train mode
            
            # Get expert usage statistics
            expert_usage = self.gating_model.get_expert_usage()
            
            # Calculate load balancing score
            load_balance = self._calculate_load_balance(expert_usage)
        
        # Combined fitness score (lower is better)
        fitness = train_loss - self.load_balance_coef * load_balance
        
        # Track best solution
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = x.copy()
            
        # Add to history
        self.history.append({
            'fitness': fitness,
            'loss': train_loss,
            'accuracy': 0.0,  # Assuming accuracy is not available in the current implementation
            'load_balance': load_balance
        })
        
        return (fitness,)
    
    def optimize(self, algorithm_id='pso', seed=42, verbosity=1):
        """
        Run the optimization process.
        
        Args:
            algorithm_id: PyGMO algorithm to use
            seed: Random seed
            verbosity: Verbosity level
            
        Returns:
            Optimized gating model
        """
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create PyGMO problem
        prob = pg.problem(self)
        
        # Select algorithm
        if algorithm_id == 'pso':
            algo = pg.algorithm(pg.pso(gen=10, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5, seed=seed))
        elif algorithm_id == 'sade':
            algo = pg.algorithm(pg.sade(gen=10, variant=2, variant_adptv=1, ftol=1e-6, xtol=1e-6))
        elif algorithm_id == 'abc':
            algo = pg.algorithm(pg.bee_colony(gen=10, limit=20, seed=seed))
        else:
            algo = pg.algorithm(pg.pso(gen=10, seed=seed))
        
        # Set verbosity
        algo.set_verbosity(verbosity)
        
        # Create population
        pop = pg.population(prob, size=self.population_size, seed=seed)
        
        # Evolve population
        start_time = time.time()
        pop = algo.evolve(pop)
        end_time = time.time()
        
        # Extract best solution
        best_idx = pop.best_idx()
        best_x = pop.get_x()[best_idx]
        best_f = pop.get_f()[best_idx]
        
        # Update model with best solution
        best_params = self._vector_to_parameters(best_x)
        self._update_model_parameters(best_params)
        
        # Print summary
        print(f"PSO gating optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness: {best_f[0]}")
        
        return self.gating_model


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

def optimize_gating_with_pso(model: MoEWithPSOGating, 
                           data: Tuple[torch.Tensor, torch.Tensor],
                           algorithm: str = 'pso',
                           population_size: int = 20,
                           load_balance_coef: float = 0.1,
                           seed: int = 42):
    """
    Optimize gating mechanism using PSO.
    
    Args:
        model: MoE model with PSO gating
        data: Tuple of (inputs, targets) for training/evaluation
        algorithm: PSO variant to use ('pso', 'abc', 'sade')
        population_size: Size of population
        load_balance_coef: Coefficient for load balancing
        seed: Random seed
        
    Returns:
        Optimized model
    """
    inputs, targets = data
    
    # Configure the PSO problem
    problem = PSOGatingProblem(
        moe_model=model,
        gating_model=model.gating,
        input_data=inputs,
        target_data=targets,
        validation_split=0.2,
        population_size=population_size,
        load_balance_coef=load_balance_coef
    )
    
    # Run optimization
    optimized_gating = problem.optimize(
        algorithm_id=algorithm,
        seed=seed,
        verbosity=1
    )
    
    # Print summary of optimization
    expert_usage = model.get_expert_usage().cpu().numpy()
    print("\nExpert usage after PSO optimization:")
    for i, usage in enumerate(expert_usage):
        print(f"  Expert {i+1}: {usage:.4f}")
    
    return model 