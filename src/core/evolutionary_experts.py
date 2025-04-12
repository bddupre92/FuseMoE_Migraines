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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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
        
        # --- Use actual input data shape for internal model creation --- 
        self.actual_input_size = input_data.shape[1]
        # --- End --- 
        
        # Set problem dimensions
        # Each expert is encoded as [hidden_size, activation_type]
        self.dim = num_experts * 2  
        
        # Track best expert configuration
        self.best_fitness = float('inf')
        self.best_solution = None
        self.best_model = None
        self.history = []
        
        self.eval_count = 0
        self.best_x = None

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
        # moe_config = self.config # Don't necessarily use the global config's input size
        
        # Create a custom MoE with evolutionarily optimized experts
        # Use the *actual* input size from the data passed to the problem
        model = EvolutionaryMoE(
            config=self.config, # Pass config for other params
            input_size=self.actual_input_size, 
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
        self.eval_count += 1
        self.history.append({
            'eval_count': self.eval_count,
            'fitness': fitness,
            'loss': val_loss,
            'accuracy': accuracy,
            'specialization': specialization_score
        })
        
        return [fitness]
    
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
        algo = None # Initialize algo
        generations = 10 # Define generations here
        if algorithm_id == 'sade':
            algo = pg.algorithm(pg.sade(gen=generations, variant=2, variant_adptv=1, ftol=1e-6, xtol=1e-6))
        elif algorithm_id == 'pso':
            algo = pg.algorithm(pg.pso(gen=generations, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5, seed=seed))
        elif algorithm_id == 'cmaes':
            algo = pg.algorithm(pg.cmaes(gen=generations, sigma0=0.5, ftol=1e-6, xtol=1e-6, seed=seed))
        else:
            algo = pg.algorithm(pg.sade(gen=generations))
        
        # Set verbosity
        algo.set_verbosity(1)
        
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
                'avg_fitness': np.mean(fitnesses), # Add average fitness
                'eval_count': pop.problem.get_fevals() # Use PyGMO's built-in method
            })
            # Optional: Log more details if needed, like champion_x, etc.
            
            # Early stopping condition (example)
            # if algo.get_extra_info() contains stopping criteria, break
        
        evolve_end = time.time()
        # --- End History Logging ---
        
        end_time = time.time()
        
        best_solution_x = pop.champion_x
        best_fitness = pop.champion_f[0]
        expert_configs = self._decode_solution(best_solution_x)
        
        # --- REMOVED old history append/clearing logic --- 
        
        print(f"   Exit condition -- {algo.get_extra_info()}")
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness: {best_fitness}")
        
        # Create and return the best model
        best_model = self._create_model(expert_configs) # Keep this line if best_model is still needed
        
        # Return the full history, expert configs, and the algorithm used
        return self.history, expert_configs, algorithm_id


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
    A basic MoE layer using evolutionarily defined MLP experts.
    Assumes gating is handled externally or is a simple softmax gate.
    """
    def __init__(self, config: MoEConfig, input_size: int, output_size: int, expert_configs: List[Dict]):
        super().__init__()
        self.config = config
        self.num_experts = len(expert_configs)
        
        # Create experts based on the provided configurations
        self.experts = nn.ModuleList([
            EvolutionaryMLP(
                config=config, 
                input_size=input_size, # Use the input_size passed to this constructor
                output_size=output_size, 
                hidden_size=conf['hidden_size'], 
                activation=conf['activation']
            )
            for conf in expert_configs
        ])
        
        # Simple softmax gating layer (can be replaced by more complex gating)
        self.gate = nn.Linear(input_size, self.num_experts)
        
        # Initialize attribute to store usage stats
        self.expert_usage_stats = None
        
    def forward(self, x):
        # Get gating scores
        gates = self.gate(x)
        # Apply softmax to get probabilities
        gate_probs = F.softmax(gates, dim=1)
        
        # Store gate probabilities for usage analysis (mean across batch)
        self.expert_usage_stats = gate_probs.mean(dim=0).detach().cpu().numpy()
        
        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_size]
        
        # Combine outputs
        combined = (expert_outputs * gate_probs.unsqueeze(-1)).sum(dim=1)  # [batch, output_size]
        return combined
    
    def get_expert_usage(self):
        """Return expert usage statistics (e.g., mean gating probabilities)."""
        return self.expert_usage_stats


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
    history, expert_configs, algorithm_id = problem.optimize(algorithm_id=algorithm)
    
    # Print expert configurations
    print("\nEvolutionarily optimized expert configurations:")
    for i, conf in enumerate(expert_configs):
        print(f"  Expert {i+1}: Hidden size={conf['hidden_size']}, Activation={conf['activation'].__class__.__name__}")
        
    return history, expert_configs, algorithm_id 