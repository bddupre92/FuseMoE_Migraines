#!/usr/bin/env python
# PyGMO Enhanced FuseMOE Integration
# Combines evolutionary expert optimization and PSO-enhanced gating

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from utils.config import MoEConfig
from core.sparse_moe import MLP, MoE
from core.evolutionary_experts import ExpertEvolutionProblem, EvolutionaryMoE, create_evolutionary_moe
from core.pso_laplace_gating import PSOLaplaceGating, MoEWithPSOGating, optimize_gating_with_pso, PSOGatingProblem

class PyGMOFuseMoE(nn.Module):
    """
    PyGMO-enhanced FuseMoE model that combines evolutionary expert optimization
    and PSO-enhanced Laplace gating.
    
    This model implements:
    1. Evolutionary expert architecture optimization
    2. PSO-enhanced Laplace gating with adaptive parameters
    3. Integrated optimization pipeline
    """
    
    def __init__(self, 
                 config: MoEConfig,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_experts: int = 8,
                 dropout: float = 0.1,
                 use_pso_gating: bool = True,
                 use_evo_experts: bool = True):
        """
        Initialize the PyGMO-enhanced FuseMoE model.
        
        Args:
            config: MoE configuration
            input_size: Size of input features
            hidden_size: Size of hidden layers
            output_size: Size of output features
            num_experts: Number of experts
            dropout: Dropout probability
            use_pso_gating: Whether to use PSO-enhanced gating
            use_evo_experts: Whether to use evolutionary experts
        """
        super(PyGMOFuseMoE, self).__init__()
        
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.use_pso_gating = use_pso_gating
        self.use_evo_experts = use_evo_experts
        
        # Placeholder for experts - will be created during optimization
        self.experts = None
        
        # Initialize gating mechanism
        if use_pso_gating:
            self.gating = PSOLaplaceGating(
                config=config,
                input_size=input_size,
                num_experts=num_experts,
                hidden_size=hidden_size // 2,
                scale_init=1.0,
                dropout=dropout
            )
        else:
            # Simple standard gating
            self.gating = nn.Sequential(
                nn.Linear(input_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_experts),
                nn.Softmax(dim=1)
            )
        
        # If not using evolutionary experts, initialize standard ones
        if not use_evo_experts:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, output_size)
                )
                for _ in range(num_experts)
            ])
    
    def forward(self, x, gates=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, input_size]
            gates: Pre-computed gates (optional)
            
        Returns:
            outputs: Model outputs [batch_size, output_size]
        """
        if self.experts is None:
            raise RuntimeError("Model experts are not initialized. Call optimize_model() first.")
        
        batch_size = x.size(0)
        
        # Get gating weights if not provided
        if gates is None:
            if self.use_pso_gating:
                gates = self.gating(x)  # [batch_size, num_experts]
            else:
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
        if self.use_pso_gating:
            return self.gating.get_expert_usage()
        else:
            return None
    
    def optimize_model(self, 
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      expert_algo: str = 'sade',
                      gating_algo: str = 'pso',
                      expert_pop_size: int = 20,
                      gating_pop_size: int = 20,
                      load_balance_coef: float = 0.1,
                      seed: int = 42,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Run the full optimization process for both experts and gating.
        
        Args:
            train_data: Tuple of (inputs, targets) for training/evaluation
            expert_algo: Algorithm for expert optimization ('sade', 'pso', 'cmaes')
            gating_algo: Algorithm for gating optimization ('pso', 'abc', 'sade')
            expert_pop_size: Population size for expert evolution
            gating_pop_size: Population size for gating PSO
            load_balance_coef: Coefficient for load balancing in gating
            seed: Random seed
            device: Device to run computations on
            
        Returns:
            self: The optimized model
        """
        start_time = time.time()
        print("Starting PyGMO-enhanced FuseMoE optimization...")
        
        inputs, targets = train_data
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Step 1: Evolutionary expert optimization
        if self.use_evo_experts:
            print("\n--- Evolutionary Expert Optimization ---")
            
            # Configure the evolution problem
            problem = ExpertEvolutionProblem(
                config=self.config,
                input_data=inputs,
                target_data=targets,
                input_size=self.input_size,
                output_size=self.output_size,
                num_experts=self.num_experts,
                hidden_size_range=(16, self.hidden_size * 2),
                population_size=expert_pop_size,
                max_evaluations=100,
                device=device
            )
            
            # Run optimization
            _, expert_configs = problem.optimize(algorithm_id=expert_algo, seed=seed)
            
            # Create experts from optimized configurations
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_size, conf['hidden_size']),
                    conf['activation'],
                    nn.Dropout(self.config.dropout),
                    nn.Linear(conf['hidden_size'], self.output_size)
                )
                for conf in expert_configs
            ]).to(device)
            
            print("\nEvolutionary expert optimization complete.")
            print(f"Expert architectures:")
            for i, conf in enumerate(expert_configs):
                print(f"  Expert {i+1}: Hidden size={conf['hidden_size']}, Activation={conf['activation'].__class__.__name__}")
        
        # Step 2: PSO-enhanced gating optimization
        if self.use_pso_gating:
            print("\n--- PSO Gating Optimization ---")
            
            # Ensure experts are created if evolutionary optimization was skipped
            if self.experts is None:
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout),
                        nn.Linear(self.hidden_size, self.output_size)
                    )
                    for _ in range(self.num_experts)
                ]).to(device)
            
            # Configure the PSO problem
            problem = PSOGatingProblem(
                moe_model=self,
                gating_model=self.gating,
                input_data=inputs,
                target_data=targets,
                validation_split=0.2,
                population_size=gating_pop_size,
                load_balance_coef=load_balance_coef,
                device=device
            )
            
            # Run optimization
            self.gating = problem.optimize(
                algorithm_id=gating_algo,
                seed=seed,
                verbosity=1
            )
            
            print("\nPSO gating optimization complete.")
        
        end_time = time.time()
        print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds")
        
        return self


class MigraineFuseMoE(nn.Module):
    """
    Specialized PyGMO-enhanced FuseMoE for migraine prediction.
    
    This extension adds migraine-specific features:
    1. Multi-modal input handling
    2. Early warning prediction
    3. Patient-specific adaptation
    4. Trigger identification
    """
    
    def __init__(self,
                config: MoEConfig,
                input_sizes: Dict[str, int],
                hidden_size: int,
                output_size: int,
                num_experts: int = 8,
                modality_experts: Dict[str, int] = None,
                dropout: float = 0.1,
                use_pso_gating: bool = True,
                use_evo_experts: bool = True,
                patient_adaptation: bool = False):
        """
        Initialize migraine-specific PyGMO-enhanced FuseMoE.
        
        Args:
            config: MoE configuration
            input_sizes: Dict of input sizes for each modality
            hidden_size: Size of hidden layers
            output_size: Size of output features
            num_experts: Number of experts
            modality_experts: Dict specifying how many experts per modality
            dropout: Dropout probability
            use_pso_gating: Whether to use PSO-enhanced gating
            use_evo_experts: Whether to use evolutionary experts
            patient_adaptation: Whether to enable patient-specific adaptation
        """
        super(MigraineFuseMoE, self).__init__()
        
        self.config = config
        self.input_sizes = input_sizes
        self.modalities = list(input_sizes.keys())
        self.modality_experts = modality_experts or {}
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.use_pso_gating = use_pso_gating
        self.use_evo_experts = use_evo_experts
        self.patient_adaptation = patient_adaptation
        self.dropout = dropout
        
        # Calculate encoded dimension
        self.encoded_dim = hidden_size // len(input_sizes) * len(input_sizes)
        
        # Create modality-specific feature extractors
        self.modality_encoders = nn.ModuleDict()
        for modality, size in input_sizes.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // len(input_sizes))
            )
        
        # Initialize experts (placeholder if using evolutionary optimization)
        if not use_evo_experts:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.encoded_dim, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, output_size)
                )
                for _ in range(num_experts)
            ])
        else:
            self.experts = None
        
        # Create the gating network
        if use_pso_gating:
            self.gating = PSOLaplaceGating(
                config=config,
                input_size=self.encoded_dim,
                num_experts=num_experts,
                hidden_size=hidden_size // 2,
                scale_init=1.0,
                dropout=dropout
            )
        else:
            # Simple standard gating
            self.gating = nn.Sequential(
                nn.Linear(self.encoded_dim, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_experts),
                nn.Softmax(dim=1)
            )
        
        # Patient adaptation components
        if patient_adaptation:
            self.patient_embedding = nn.Embedding(100, hidden_size // 4)  # Up to 100 patients
            self.patient_gate = nn.Sequential(
                nn.Linear(hidden_size // 4, hidden_size // 8),
                nn.ReLU(),
                nn.Linear(hidden_size // 8, num_experts),
                nn.Softmax(dim=1)
            )
    
    def encode_modalities(self, inputs):
        """
        Encode multi-modal inputs.
        
        Args:
            inputs: Dict of input tensors for each modality
            
        Returns:
            Encoded features tensor
        """
        # Process each modality
        modality_features = []
        for modality in self.modalities:
            if modality in inputs:
                features = self.modality_encoders[modality](inputs[modality])
                modality_features.append(features)
        
        # Concatenate all modality features
        return torch.cat(modality_features, dim=1)
    
    def forward(self, inputs, patient_id=None, gates=None):
        """
        Forward pass with multi-modal inputs.
        
        Args:
            inputs: Dict of input tensors for each modality or tensor
            patient_id: Optional patient identifier for adaptation
            gates: Pre-computed gates (optional)
            
        Returns:
            outputs: Model outputs [batch_size, output_size]
        """
        if self.experts is None:
            raise RuntimeError("Model experts are not initialized. Call optimize_model() first.")
        
        # Handle the case when inputs is a tensor (for optimization compatibility)
        if isinstance(inputs, torch.Tensor):
            x = inputs
        else:
            # Encode multi-modal inputs
            x = self.encode_modalities(inputs)
        
        # If gates are provided, use them directly
        if gates is None:
            # Apply patient-specific gating if enabled
            if self.patient_adaptation and patient_id is not None:
                patient_emb = self.patient_embedding(patient_id)
                patient_gates = self.patient_gate(patient_emb)
                
                # Modify gates based on patient profile
                if self.use_pso_gating:
                    gates = self.gating(x)
                    # Blend standard gates with patient-specific gates
                    gates = 0.7 * gates + 0.3 * patient_gates
                else:
                    gates = 0.7 * self.gating(x) + 0.3 * patient_gates
            else:
                # Standard gating
                if self.use_pso_gating:
                    gates = self.gating(x)
                else:
                    gates = self.gating(x)
        
        # Forward through experts
        batch_size = x.size(0)
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
        if self.use_pso_gating:
            return self.gating.get_expert_usage()
        else:
            return None
    
    def predict_with_warning_time(self, inputs, threshold=0.7):
        """
        Make predictions with early warning time estimates.
        
        Args:
            inputs: Dict of input tensors containing temporal data
            threshold: Probability threshold for positive prediction
            
        Returns:
            Tuple of (predictions, warning_times)
        """
        outputs = self.forward(inputs)
        predictions = (outputs > threshold).float()
        
        # Warning time calculation would use temporal information
        # This is a placeholder - actual implementation would analyze 
        # prediction confidence trend over time
        warning_times = torch.zeros_like(predictions)
        confidence = outputs.detach()
        
        # Simple heuristic: higher confidence = earlier warning
        # In practice, this would be a more sophisticated analysis of temporal patterns
        warning_times = torch.where(
            predictions > 0.5,
            24.0 * (confidence - 0.5) / 0.5,  # Up to 24 hours warning 
            torch.zeros_like(predictions)
        )
        
        return predictions, warning_times
    
    def identify_triggers(self, inputs, attribution_method='integrated_gradients'):
        """
        Identify potential migraine triggers from inputs.
        
        Args:
            inputs: Dict of input tensors for each modality
            attribution_method: Method for feature attribution
            
        Returns:
            Dict of trigger scores for each modality's features
        """
        # Placeholder implementation - would use gradient-based attribution
        # or other explainability techniques in practice
        trigger_scores = {}
        
        # For now, just return random scores as placeholder
        for modality, tensor in inputs.items():
            # Would use actual attribution methods here
            trigger_scores[modality] = torch.rand_like(tensor)
            
        return trigger_scores
    
    def optimize_model(self, 
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      expert_algo: str = 'sade',
                      gating_algo: str = 'pso',
                      expert_pop_size: int = 20,
                      gating_pop_size: int = 20,
                      load_balance_coef: float = 0.1,
                      seed: int = 42,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Run the full optimization process for both experts and gating.
        
        Args:
            train_data: Tuple of (inputs, targets) for training/evaluation
            expert_algo: Algorithm for expert optimization ('sade', 'pso', 'cmaes')
            gating_algo: Algorithm for gating optimization ('pso', 'abc', 'sade')
            expert_pop_size: Population size for expert evolution
            gating_pop_size: Population size for gating PSO
            load_balance_coef: Coefficient for load balancing in gating
            seed: Random seed
            device: Device to run computations on
            
        Returns:
            self: The optimized model
        """
        start_time = time.time()
        print("Starting PyGMO-enhanced FuseMoE optimization...")
        
        inputs, targets = train_data
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Step 1: Evolutionary expert optimization
        if self.use_evo_experts:
            print("\n--- Evolutionary Expert Optimization ---")
            
            # Configure the evolution problem
            problem = ExpertEvolutionProblem(
                config=self.config,
                input_data=inputs,
                target_data=targets,
                input_size=self.encoded_dim,  # Use encoded dimension
                output_size=self.output_size,
                num_experts=self.num_experts,
                hidden_size_range=(16, self.hidden_size * 2),
                population_size=expert_pop_size,
                max_evaluations=100,
                device=device
            )
            
            # Run optimization
            _, expert_configs = problem.optimize(algorithm_id=expert_algo, seed=seed)
            
            # Create experts from optimized configurations
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.encoded_dim, conf['hidden_size']),
                    conf['activation'],
                    nn.Dropout(self.config.dropout),
                    nn.Linear(conf['hidden_size'], self.output_size)
                )
                for conf in expert_configs
            ]).to(device)
            
            print("\nEvolutionary expert optimization complete.")
            print(f"Expert architectures:")
            for i, conf in enumerate(expert_configs):
                print(f"  Expert {i+1}: Hidden size={conf['hidden_size']}, Activation={conf['activation'].__class__.__name__}")
        
        # Step 2: PSO-enhanced gating optimization
        if self.use_pso_gating:
            print("\n--- PSO Gating Optimization ---")
            
            # Ensure experts are created if evolutionary optimization was skipped
            if self.experts is None:
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.encoded_dim, self.hidden_size),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout),
                        nn.Linear(self.hidden_size, self.output_size)
                    )
                    for _ in range(self.num_experts)
                ]).to(device)
            
            # Configure the PSO problem
            problem = PSOGatingProblem(
                moe_model=self,
                gating_model=self.gating,
                input_data=inputs,
                target_data=targets,
                validation_split=0.2,
                population_size=gating_pop_size,
                load_balance_coef=load_balance_coef,
                device=device
            )
            
            # Run optimization
            self.gating = problem.optimize(
                algorithm_id=gating_algo,
                seed=seed,
                verbosity=1
            )
            
            print("\nPSO gating optimization complete.")
        
        end_time = time.time()
        print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds")
        
        return self


def create_pygmo_fusemoe(config: MoEConfig,
                       input_size: int,
                       hidden_size: int,
                       output_size: int,
                       train_data: Tuple[torch.Tensor, torch.Tensor],
                       num_experts: int = 8,
                       use_pso_gating: bool = True,
                       use_evo_experts: bool = True):
    """
    Create and optimize a PyGMO-enhanced FuseMoE model.
    
    Args:
        config: MoE configuration
        input_size: Size of input features
        hidden_size: Size of hidden layers
        output_size: Size of output features
        train_data: Tuple of (inputs, targets) for optimization
        num_experts: Number of experts
        use_pso_gating: Whether to use PSO-enhanced gating
        use_evo_experts: Whether to use evolutionary experts
        
    Returns:
        Optimized PyGMOFuseMoE model
    """
    # Create model
    model = PyGMOFuseMoE(
        config=config,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_experts=num_experts,
        use_pso_gating=use_pso_gating,
        use_evo_experts=use_evo_experts
    )
    
    # Optimize model
    model.optimize_model(
        train_data=train_data,
        expert_algo='sade',
        gating_algo='pso'
    )
    
    return model 