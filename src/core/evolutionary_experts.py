#!/usr/bin/env python
# Evolutionary Computing integration for FuseMOE expert optimization
# Based on PyGMO (Parallel Global Multiobjective Optimizer)

import torch
import torch.nn as nn
import numpy as np
import pygmo as pg
import random
import time
from typing import List, Dict, Tuple, Optional, Union
from utils.config import MoEConfig
from core.sparse_moe import MLP, MoE

class ExpertEvolutionProblem:
    """
    PyGMO problem wrapper for expert architecture optimization.
    
    This class encodes expert network architectures and weights as a PyGMO problem
    for optimization using evolutionary computing algorithms.
    """
    
    def __init__(self, 
                 config: MoEConfig,
                 input_data: torch.Tensor,
                 target_data: torch.Tensor,
                 input_size: int,
                 output_size: int,
                 num_experts: int = 8,
                 hidden_size_range: Tuple[int, int] = (16, 256),
                 population_size: int = 20,
                 max_evaluations: int = 100,
                 validation_split: float = 0.2,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the evolution problem.
        
        Args:
            config: MoE configuration
            input_data: Training input data
            target_data: Training target data
            input_size: Size of input features
            output_size: Size of output features
            num_experts: Number of experts to optimize
            hidden_size_range: Range of hidden layer sizes
            population_size: Size of population for evolution
            max_evaluations: Maximum number of fitness evaluations
            validation_split: Fraction of data to use for validation
            device: Device to run computations on
        """
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.hidden_size_min, self.hidden_size_max = hidden_size_range
        self.population_size = population_size
        self.max_evaluations = max_evaluations
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
        
        # Set problem dimensions
        # Each expert is encoded as [hidden_size, activation_type]
        self.dim = num_experts * 2  
        
        # Track best expert configuration
        self.best_fitness = float('inf')
        self.best_solution = None
        self.best_model = None
        self.history = []
        
    def get_bounds(self):
        """
        Get the bounds of the decision variables.
        
        Returns:
            Tuple of lower and upper bounds
        """
        # Lower bounds: min hidden size (16) and activation function index (0)
        lb = [self.hidden_size_min, 0] * self.num_experts
        
        # Upper bounds: max hidden size (256) and activation function index (3)
        # Activation indices: 0=ReLU, 1=GELU, 2=Tanh, 3=Sigmoid
        ub = [self.hidden_size_max, 3] * self.num_experts
        
        return (lb, ub)
    
    def get_nobj(self):
        """Number of objectives (single objective optimization)"""
        return 1
    
    def _decode_solution(self, x):
        """
        Decode solution vector into expert configurations.
        
        Args:
            x: Solution vector from PyGMO
            
        Returns:
            List of expert configurations
        """
        expert_configs = []
        
        for i in range(0, len(x), 2):
            hidden_size = int(x[i])  # Hidden layer size
            activation_idx = int(x[i+1])  # Activation function type
            
            # Map activation index to actual activation function
            activations = [nn.ReLU(), nn.GELU(), nn.Tanh(), nn.Sigmoid()]
            activation = activations[min(activation_idx, len(activations)-1)]
            
            expert_configs.append({
                'hidden_size': hidden_size,
                'activation': activation
            })
            
        return expert_configs
    
    def _create_model(self, expert_configs):
        """
        Create a SimpleMoE model with the given expert configurations.
        
        Args:
            expert_configs: List of expert configurations
            
        Returns:
            Configured SimpleMoE model
        """
        # Create a modified MoE model with the specified expert configurations
        moe_config = self.config
        
        # Create a custom MoE with evolutionarily optimized experts
        model = EvolutionaryMoE(
            config=moe_config,
            input_size=self.input_size,
            output_size=self.output_size,
            expert_configs=expert_configs
        )
        
        return model.to(self.device)
    
    def fitness(self, x):
        """
        Evaluate the fitness of a solution.
        
        Args:
            x: Solution vector from PyGMO
            
        Returns:
            Tuple of fitness values (loss,)
        """
        # Decode the solution vector
        expert_configs = self._decode_solution(x)
        
        # Create the model with these configurations
        model = self._create_model(expert_configs)
        
        # Train the model with a few epochs to evaluate configuration fitness
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # Short training session to evaluate architecture quality
        model.train()
        for epoch in range(5):  # Just a few epochs to get a sense of performance
            optimizer.zero_grad()
            outputs = model(self.train_inputs)
            loss = loss_fn(outputs, self.train_targets)
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(self.val_inputs)
            val_loss = loss_fn(val_outputs, self.val_targets).item()
            
            # Calculate accuracy
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == self.val_targets).float().mean().item()
        
        # Multi-objective fitness: loss and negative specialization
        expert_usage = model.get_expert_usage()
        specialization_score = self._calculate_specialization(expert_usage)
        
        # Combined fitness score
        fitness = val_loss - 0.2 * specialization_score  # Lower is better
        
        # Track best solution
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = x
            self.best_model = model
            
        # Add to history
        self.history.append({
            'fitness': fitness,
            'loss': val_loss,
            'accuracy': accuracy,
            'specialization': specialization_score
        })
        
        return (fitness,)
    
    def _calculate_specialization(self, expert_usage):
        """
        Calculate expert specialization score.
        Higher is better - experts should specialize on different inputs.
        
        Args:
            expert_usage: Matrix of expert usage patterns
            
        Returns:
            Specialization score
        """
        # If usage is concentrated (high variance), specialization is high
        if isinstance(expert_usage, torch.Tensor):
            usage_var = torch.var(expert_usage).item()
        else:
            usage_var = np.var(expert_usage)
            
        return usage_var * 10  # Scale up for fitness function
    
    def optimize(self, algorithm_id='sade', seed=42):
        """
        Run the optimization process.
        
        Args:
            algorithm_id: PyGMO algorithm to use
            seed: Random seed
            
        Returns:
            Best solution found
        """
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create PyGMO problem
        prob = pg.problem(self)
        
        # Select algorithm
        if algorithm_id == 'sade':
            algo = pg.algorithm(pg.sade(gen=10, variant=2, variant_adptv=1, ftol=1e-6, xtol=1e-6))
        elif algorithm_id == 'pso':
            algo = pg.algorithm(pg.pso(gen=10, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5, seed=seed))
        elif algorithm_id == 'cmaes':
            algo = pg.algorithm(pg.cmaes(gen=10, sigma0=0.5, ftol=1e-6, xtol=1e-6, seed=seed))
        else:
            algo = pg.algorithm(pg.sade(gen=10))
        
        # Set verbosity
        algo.set_verbosity(1)
        
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
        
        # Print summary
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness: {best_f[0]}")
        
        # Create and return the best model
        expert_configs = self._decode_solution(best_x)
        best_model = self._create_model(expert_configs)
        
        return best_model, expert_configs


class EvolutionaryMLP(MLP):
    """Extension of MLP with configurable activation functions."""
    
    def __init__(self, config: MoEConfig, input_size: int, output_size: int, hidden_size: int, activation=None):
        super(EvolutionaryMLP, self).__init__(config, input_size, output_size, hidden_size)
        if activation is not None:
            self.activation = activation
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.log_soft(out)
        return out


class EvolutionaryMoE(nn.Module):
    """
    Mixture of Experts with evolutionary optimized experts.
    This is a simplified version for evolutionary optimization.
    """
    
    def __init__(self, config: MoEConfig, input_size: int, output_size: int, expert_configs: List[Dict]):
        super(EvolutionaryMoE, self).__init__()
        
        self.num_experts = len(expert_configs)
        self.output_size = output_size
        self.input_size = input_size
        
        # Create experts based on evolutionary configs
        self.experts = nn.ModuleList([
            EvolutionaryMLP(
                config=config,
                input_size=input_size,
                output_size=output_size,
                hidden_size=conf['hidden_size'],
                activation=conf['activation']
            )
            for conf in expert_configs
        ])
        
        # Simple gating network
        self.gate = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts),
            nn.Softmax(dim=1)
        )
        
        # For tracking expert usage
        self.expert_usage = None
    
    def forward(self, x):
        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_size]
        
        # Get gating weights
        gates = self.gate(x).unsqueeze(-1)  # [batch, num_experts, 1]
        
        # Store gates for usage analysis
        self.expert_usage = gates.mean(dim=0).squeeze().detach().cpu()
        
        # Combine outputs
        combined = (expert_outputs * gates).sum(dim=1)  # [batch, output_size]
        return combined
    
    def get_expert_usage(self):
        """Return the latest expert usage statistics"""
        return self.expert_usage


# Utility functions for evolutionary expert optimization

def create_evolutionary_moe(config: MoEConfig, input_size: int, output_size: int, 
                          train_data: Tuple[torch.Tensor, torch.Tensor], 
                          algorithm: str = 'sade'):
    """
    Create an evolutionarily optimized MoE model.
    
    Args:
        config: MoE configuration
        input_size: Size of input features
        output_size: Size of output features
        train_data: Tuple of (inputs, targets) for training
        algorithm: PyGMO algorithm to use ('sade', 'pso', 'cmaes')
        
    Returns:
        Optimized MoE model
    """
    inputs, targets = train_data
    
    # Configure the evolution problem
    problem = ExpertEvolutionProblem(
        config=config,
        input_data=inputs,
        target_data=targets,
        input_size=input_size,
        output_size=output_size,
        num_experts=config.num_experts,
        hidden_size_range=(16, 256),  # Configurable range for hidden sizes
        population_size=20,
        max_evaluations=100
    )
    
    # Run optimization
    best_model, expert_configs = problem.optimize(algorithm_id=algorithm)
    
    # Print expert configurations
    print("\nEvolutionarily optimized expert configurations:")
    for i, conf in enumerate(expert_configs):
        print(f"  Expert {i+1}: Hidden size={conf['hidden_size']}, Activation={conf['activation'].__class__.__name__}")
        
    return best_model 