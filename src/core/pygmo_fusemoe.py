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
        
        # Return output, gating weights (router_logits), and gating weights again (expert_mask)
        return combined, gates, gates
    
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
        # Move each tensor within the inputs dictionary to the device
        inputs = {mod: tensor.to(device) for mod, tensor in inputs.items()}
        targets = targets.to(device)
        
        # Expand targets to match the Batch * Window dimension of encoded_inputs
        batch_size = -1
        window_size = -1
        if self.modalities and self.modalities[0] in inputs: # Get shape from original input before encoding
             batch_size, window_size, _ = inputs[self.modalities[0]].shape
        
        if batch_size != -1 and window_size != -1 and targets.shape[0] == batch_size:
            expanded_targets = targets.unsqueeze(1).repeat(1, window_size, 1).reshape(batch_size * window_size, 1)
            print(f"DEBUG: Expanded targets shape: {expanded_targets.shape}")
        else:
            # Fallback or error if shapes don't match expectation
            print(f"WARN: Could not expand targets. Target shape {targets.shape}, Batch {batch_size}, Window {window_size}")
            expanded_targets = targets # Use original targets, likely causing issues downstream

        # Step 1: Evolutionary expert optimization
        if self.use_evo_experts:
            print("\n--- Evolutionary Expert Optimization ---")
            
            # --- Encode training data first --- 
            encoded_inputs = self.encode_modalities(inputs) # Assuming inputs is dict of tensors
            print(f"DEBUG: Shape of encoded_inputs in optimize_model: {encoded_inputs.shape}") # DEBUG PRINT
            # --- End Encoding --- 
            
            # Configure the evolution problem using ENCODED dimension
            problem = ExpertEvolutionProblem(
                config=self.config,
                input_data=encoded_inputs.detach(), # Pass DETACHED encoded data
                target_data=expanded_targets.detach(),       # Pass DETACHED and EXPANDED targets
                input_size=self.encoded_dim, # Use encoded dimension
                output_size=self.output_size,
                num_experts=self.num_experts,
                hidden_size_range=(16, self.hidden_size * 2),
                population_size=expert_pop_size,
                max_evaluations=100,
                device=device
            )
            
            # Run optimization
            _, expert_configs = problem.optimize(algorithm_id=expert_algo, seed=seed)
            
            # Create experts from optimized configurations using ENCODED dimension
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
            
            # --- Encode training data if not already done --- 
            if not self.use_evo_experts:
                 # Experts were standard, need to encode inputs for gating problem
                 encoded_inputs = self.encode_modalities(inputs)
            # else: encoded_inputs already exists from expert opt step
            # --- End Encoding --- 
            
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
            
            # Configure the PSO problem using ENCODED data
            problem = PSOGatingProblem(
                moe_model=self, # Pass the main model instance
                gating_model=self.gating, # Pass the gating network
                input_data=encoded_inputs.detach(), # Pass DETACHED encoded data
                target_data=expanded_targets.detach(),       # Pass DETACHED and EXPANDED targets
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
                patient_adaptation: bool = False,
                device: str = 'cpu'):
        """
        Initialize the Migraine specific FuseMOE model.
        
        Args:
            config: MoE configuration
            input_sizes: Dictionary mapping modality name to its feature dimension.
            hidden_size: Size of hidden layers
            output_size: Size of output features
            num_experts: Number of experts
            modality_experts: Dict specifying how many experts per modality
            dropout: Dropout probability
            use_pso_gating: Whether to use PSO-enhanced gating
            use_evo_experts: Whether to use evolutionary experts
            patient_adaptation: Whether to enable patient-specific adaptation
            device: Device to place the model on.
        """
        super().__init__()
        
        self.config = config
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.modalities = list(input_sizes.keys())
        self.modality_experts = modality_experts or {mod: num_experts // len(self.modalities) for mod in self.modalities}
        self.use_pso_gating = use_pso_gating
        self.use_evo_experts = use_evo_experts
        self.patient_adaptation = patient_adaptation
        self.device = device

        # --- Restore Modality Encoders --- 
        encoder_output_dim_per_modality = hidden_size // len(self.modalities) # Example: divide hidden size equally
        self.modality_encoders = nn.ModuleDict()
        for modality, size in input_sizes.items():
             self.modality_encoders[modality] = nn.Sequential(
                 nn.Linear(size, hidden_size // 2), # Map raw features to intermediate
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(hidden_size // 2, encoder_output_dim_per_modality) # Map to final encoded dim per modality
             ).to(device)
        
        # Calculate total dimension AFTER encoding
        self.encoded_dim = encoder_output_dim_per_modality * len(self.modalities)
        # --- End Modality Encoders --- 

        # Placeholder for experts - potentially created by optimize_model
        self.experts = None 

        # Initialize Gating Mechanism using ENCODED dimension
        if use_pso_gating:
            print(f"Initializing PSO Laplace Gating with input_size={self.encoded_dim}")
            self.gating = PSOLaplaceGating(
                config=config,
                input_size=self.encoded_dim, # Use the ENCODED dimension
                num_experts=num_experts,
                hidden_size=hidden_size // 2, 
                scale_init=1.0, 
                dropout=dropout
            ).to(device)
        else:
            # Simple standard gating using ENCODED dimension
            print(f"Initializing Standard Softmax Gating with input_size={self.encoded_dim}")
            self.gating = nn.Sequential(
                nn.Linear(self.encoded_dim, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_experts),
                nn.Softmax(dim=1)
            ).to(device)
        
        # Initialize Standard Experts using ENCODED dimension
        if not use_evo_experts:
            print(f"Initializing Standard Experts with input_size={self.encoded_dim}")
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.encoded_dim, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, output_size)
                )
                for _ in range(num_experts)
            ]).to(device)
        
        # Placeholder for patient-specific layers if adaptation is enabled
        if patient_adaptation:
            self.patient_adapters = nn.ModuleDict()
    
    def encode_modalities(self, inputs):
        """
        Encode multi-modal inputs, handling (Batch, Window, Features) shape.
        """
        modality_features = []
        batch_size = -1
        window_size = -1

        # Determine batch_size and window_size from the first available modality
        for modality in self.modalities:
            if modality in inputs:
                batch_size, window_size, _ = inputs[modality].shape
                break
        
        if batch_size == -1: # Should not happen if inputs is valid
             return torch.empty(0) 

        for modality in self.modalities:
            if modality in inputs:
                tensor = inputs[modality] # Shape [Batch, Window, Features]
                _, _, feature_size = tensor.shape
                
                # Reshape to [Batch*Window, Features] for the encoder
                reshaped_tensor = tensor.reshape(batch_size * window_size, feature_size)
                
                # Pass through the specific modality encoder
                encoded_features = self.modality_encoders[modality](reshaped_tensor) 
                # Expected shape: [Batch*Window, encoder_output_dim_per_modality]
                modality_features.append(encoded_features)
            else:
                 # Handle potentially missing modality for a sample (e.g., add zeros)
                 # This part might need refinement depending on how missing modalities are handled upstream
                 encoder_output_dim = self.hidden_size // len(self.modalities) # Assuming equal split
                 zeros = torch.zeros(batch_size * window_size, encoder_output_dim, device=self.device)
                 modality_features.append(zeros)


        # Concatenate along the feature dimension (dim=1)
        # Input list shapes like: [[1056, 21], [1056, 21], [1056, 21]]
        # Expected output shape: [1056, 63] (where 1056 = 44*24, 63 = 21*3)
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

        # Return output, gating weights (router_logits), and gating weights again (expert_mask)
        return combined, gates, gates
    
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
        # Move each tensor within the inputs dictionary to the device
        inputs = {mod: tensor.to(device) for mod, tensor in inputs.items()}
        targets = targets.to(device)
        
        # Expand targets to match the Batch * Window dimension of encoded_inputs
        batch_size = -1
        window_size = -1
        if self.modalities and self.modalities[0] in inputs: # Get shape from original input before encoding
             batch_size, window_size, _ = inputs[self.modalities[0]].shape
        
        if batch_size != -1 and window_size != -1 and targets.shape[0] == batch_size:
            expanded_targets = targets.unsqueeze(1).repeat(1, window_size, 1).reshape(batch_size * window_size, 1)
            print(f"DEBUG: Expanded targets shape: {expanded_targets.shape}")
        else:
            # Fallback or error if shapes don't match expectation
            print(f"WARN: Could not expand targets. Target shape {targets.shape}, Batch {batch_size}, Window {window_size}")
            expanded_targets = targets # Use original targets, likely causing issues downstream

        # Step 1: Evolutionary expert optimization
        if self.use_evo_experts:
            print("\n--- Evolutionary Expert Optimization ---")
            
            # --- Encode training data first --- 
            encoded_inputs = self.encode_modalities(inputs) # Assuming inputs is dict of tensors
            print(f"DEBUG: Shape of encoded_inputs in optimize_model: {encoded_inputs.shape}") # DEBUG PRINT
            # --- End Encoding --- 
            
            # Configure the evolution problem using ENCODED dimension
            problem = ExpertEvolutionProblem(
                config=self.config,
                input_data=encoded_inputs.detach(), # Pass DETACHED encoded data
                target_data=expanded_targets.detach(),       # Pass DETACHED and EXPANDED targets
                input_size=self.encoded_dim, # Use encoded dimension
                output_size=self.output_size,
                num_experts=self.num_experts,
                hidden_size_range=(16, self.hidden_size * 2),
                population_size=expert_pop_size,
                max_evaluations=100,
                device=device
            )
            
            # Run optimization
            _, expert_configs = problem.optimize(algorithm_id=expert_algo, seed=seed)
            
            # Create experts from optimized configurations using ENCODED dimension
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
            
            # --- Encode training data if not already done --- 
            if not self.use_evo_experts:
                 # Experts were standard, need to encode inputs for gating problem
                 encoded_inputs = self.encode_modalities(inputs)
            # else: encoded_inputs already exists from expert opt step
            # --- End Encoding --- 
            
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
            
            # Configure the PSO problem using ENCODED data
            problem = PSOGatingProblem(
                moe_model=self, # Pass the main model instance
                gating_model=self.gating, # Pass the gating network
                input_data=encoded_inputs.detach(), # Pass DETACHED encoded data
                target_data=expanded_targets.detach(),       # Pass DETACHED and EXPANDED targets
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