# PyGMO-Enhanced FuseMOE for Migraine Prediction

This repository contains an implementation of PyGMO-enhanced FuseMOE for multimodal migraine prediction, integrating evolutionary computation and swarm intelligence algorithms with mixture-of-experts neural networks.

## Features

- **Evolutionary Expert Optimization**: Uses differential evolution (DE) to optimize expert architectures 
- **PSO-Enhanced Gating**: Implements particle swarm optimization for gating weights
- **Multimodal Integration**: Supports EEG, weather, sleep, and stress data for migraine prediction
- **Patient-Specific Adaptation**: Personalizes models for individual patients
- **Migraine Trigger Identification**: Analyzes and visualizes potential migraine triggers

## Getting Started

### Prerequisites

- Python 3.8+ 
- PyTorch 1.8+
- PyGMO 2.18+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd FuseMOE

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Migraine Prediction

```bash
cd src/scripts
./run_migraine.sh
```

For patient-specific adaptation:

```bash
./run_migraine.sh --patient-adaptation patient123
```

### Using PyGMO with MIMIC-IV

```bash
cd src/scripts
./run_pygmo_mimiciv.sh
```

### Using PyGMO with Standard FuseMOE

```bash
cd src/scripts
./run_pygmo.sh
```

## Key Components

### PyGMO Integration

The PyGMO integration is implemented in the following files:

- `src/core/pygmo_fusemoe.py`: Core implementation of PyGMO-enhanced FuseMOE
- `src/core/evolutionary_experts.py`: Implementation of evolutionary expert optimization
- `src/core/pso_laplace_gating.py`: Implementation of PSO-enhanced gating

### Migraine Data Processing

The migraine data pipeline is implemented in:

- `src/preprocessing/migraine_preprocessing/`: Modality-specific processors
- `src/scripts/run_migraine_prediction.py`: Main script for running migraine prediction
- `src/scripts/run_migraine.sh`: Shell script for running the full pipeline

### Command Line Arguments

The following arguments control the PyGMO optimization:

```
--use_pygmo                  Use PyGMO optimization for experts and gating
--expert_algorithm DE        Algorithm for expert optimization (de, sade, pso)
--gating_algorithm PSO       Algorithm for gating optimization (pso, abc, sade)
--expert_population_size 10  Population size for expert optimization
--gating_population_size 10  Population size for gating optimization
--expert_generations 5       Number of generations for expert optimization
--gating_generations 5       Number of generations for gating optimization
```

Patient-specific adaptation is controlled by:

```
--patient_adaptation         Enable patient-specific adaptation
--patient_id PATIENT_ID      ID of the patient for adaptation
--patient_iterations 3       Number of iterations for patient adaptation
--load_base_model MODEL      Path to base model for patient adaptation
```

Modality-specific experts can be configured with:

```
--modality_experts "eeg:3,weather:2,sleep:2,stress:1"
```

## Example Output

The migraine prediction script produces the following outputs:

1. **Dataset summary** (`dataset_summary.txt`)
2. **Evaluation metrics** (`metrics.json`)
3. **Visualizations**:
   - Expert usage (`expert_usage.png`)
   - Modality importance (`modality_importance.png`)
   - ROC curve (`roc_curve.png`)
   - Prediction timeline (`prediction_timeline.png`)
   - Trigger analysis (`trigger_analysis.png`)
4. **Trained model** (`model.pth`)

## Implementation Details

### Evolutionary Expert Optimization

The evolutionary expert optimization process:

1. Encodes expert architectures as PyGMO problem
2. Uses differential evolution to find optimal architectures
3. Optimizes hidden layer size and activation functions
4. Balances specialization and performance

### PSO-Enhanced Gating

The PSO-enhanced gating mechanism:

1. Implements Laplace-distributed gating with adaptive parameters
2. Uses particle swarm optimization for gating weights
3. Optimizes for load balancing and specialization
4. Dynamically adjusts to different modalities

### Patient-Specific Adaptation

The patient adaptation process:

1. Loads a pre-trained base model
2. Fine-tunes on patient-specific data
3. Adapts expert routing for the individual
4. Identifies personal migraine triggers

## License

This project is licensed under the MIT License - see the LICENSE file for details. 