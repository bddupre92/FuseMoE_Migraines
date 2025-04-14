# Development Plan: PyGMO EC/SI Integration with FuseMOE for Migraine Prediction

## Phase 1: Environment Setup & Analysis (Week 1)

### 1.1 Environment Configuration
- Install PyGMO alongside existing dependencies (`pip install pygmo`)
- Create a development branch for integration work
- Ensure compatibility between PyGMO (v2.18+) and existing PyTorch environment

### 1.2 Codebase Analysis
- Map expert network and gating mechanism implementations in FuseMOE (`src/core/sparse_moe.py`)
- Identify insertion points for evolutionary expert optimization (`src/core/evolutionary_experts.py`)
- Analyze Laplace gating implementation for PSO enhancement (`src/core/pso_laplace_gating.py`)
- Document key configurations in `MoEConfig` (`src/utils/config.py`) to extend

### 1.3 Data Assessment
- Review migraine dataset requirements and format (or synthetic data structure)
- Determine appropriate preprocessing for temporal migraine data (`src/preprocessing/migraine_preprocessing/*`)
- Identify relevant modalities for migraine prediction (e.g., EEG, weather, sleep, diet, stress) or other tasks (e.g., MIMIC-IV: vitals, labs, notes, CXR, ECG)
- **Critical**: Analyze class distribution in the dataset and assess prediction_horizon impact on target labeling

## Phase 2: Advanced Data Preprocessing & Imputation (Week 2)

### 2.1 Data Generation Configuration
- **Critical**: Configure balanced synthetic data generation in `create_migraine_dataset.py`
  - Set `avg_migraine_freq` to 0.15-0.2 (15-20% positive rate) instead of 0.4
  - Adjust `prediction_horizon` to 6 hours (down from 24) to create a more precise prediction task
  - Verify resulting class distribution is reasonably balanced (target: 15-20% positive rate)
  - Implement random seed setting for reproducibility (`--seed 42`)

### 2.2 Baseline Imputation Implementation
- Implement/verify standard baselines (Zero-fill, Forward-fill, Mean/Median fill) potentially within `src/preprocessing/migraine_preprocessing/migraine_data_pipeline.py` or a simple utility.
- *Notebook:* `notebooks/missing_data_handler/ec_si_missing_data_handler.ipynb` (Section 3)

### 2.3 Advanced Imputation Development
- Implement `KNNImputer` and `IterativeImputer` classes in `src/preprocessing/advanced_imputation.py`, inheriting from `BaseImputer`.
    - These wrappers will handle 3D time-series data reshaping (`(samples, seq_len, features)` -> `(samples * seq_len, features)`) for compatibility with `sklearn.impute.KNNImputer` and `sklearn.impute.IterativeImputer`.
    - Include optional `StandardScaler` within the wrappers, fitted only on observed data.
- *(Optional/Future)* Implement `PSOImputer` for PSO-optimized imputation (requires defining a PyGMO problem and fitness function considering temporal correlations/smoothness).
- *(Optional/Future)* Implement `AutoencoderImputer` using a suitable recurrent architecture (e.g., LSTM-AE in PyTorch) trained to reconstruct masked inputs.
- *Prototyping Notebook:* `notebooks/missing_data_handler/ec_si_missing_data_handler.ipynb` (Section 4) for initial exploration.
- *Note:* These implementations within the notebook serve as initial exploration and prototyping. Refactoring of promising methods into the main codebase (Phase 2.4) can proceed once their viability is understood, even if not all methods listed here are fully implemented in the notebook.

### 2.4 Imputation Evaluation Framework
- Develop quantitative evaluation (e.g., RMSE against masked ground truth in synthetic data) within the notebook.
- Implement qualitative evaluation (e.g., plotting distributions/time series of imputed vs. original observed data) within the notebook.
- *Evaluation Notebook:* `notebooks/missing_data_handler/ec_si_missing_data_handler.ipynb` (Section 6)

### 2.5 Refactor and Integrate Imputation Methods
- Refactor the chosen/implemented imputation methods (`KNNImputer`, `IterativeImputer`) into stable classes in `src/preprocessing/advanced_imputation.py`.
- Modify the data loading/preprocessing stage in `run_migraine_prediction.py` (potentially within the `MigraineDataPipeline.run_full_pipeline` method) to:
    - Accept `--imputation_method` ('knn', 'iterative', 'none') and `--imputer_config` arguments.
    - Instantiate the chosen imputer class from `advanced_imputation.py`.
    - Apply the imputer's `fit_transform` method to the prepared data tensors (X) and mask before they are used for training/evaluation.

### 2.6 Feature Engineering Enhancement
- Implement window_size parameter adjustment (increase from 8 to 12-24 hours) to capture longer-term patterns
- Add trend-based features for weather parameters known to trigger migraines
- Create sleep quality trend features spanning multiple days rather than just immediate window
- Develop stress pattern features (sustained stress or rapid changes)
- Ensure these enhanced features are properly integrated into the data pipeline

## Phase 3: Evolutionary Expert Optimization (Weeks 3-4)

### 3.1 Create Expert Evolution Framework
- Implement `ExpertEvolutionProblem` class in `src/core/evolutionary_experts.py` inheriting from `pygmo.problem`.
    - `get_bounds`: Define search space bounds for expert parameters (e.g., hidden size, activation type).
    - `fitness`: Decode solution vector `x`, create `EvolutionaryMoE` instance, train briefly, evaluate on validation set (loss, potentially accuracy/specialization), return fitness score(s).
- Implement `ExpertEvolutionProblem` class for encoding expert architectures
- Create configuration extensions for evolutionary experts
- **Add fixed random seed** for reproducible initialization, configurable via parameter

### 3.2 Expert Architecture Optimization
- Implement network architecture encoding/decoding for PyGMO
- Implement encoding in `ExpertEvolutionProblem._decode_solution`: Map genotype vector `x` to list of dicts `expert_configs` (e.g., `{'hidden_size': int(x[i]), 'activation': activation_map[int(x[i+1])]}`).
- Define fitness functions for expert specialization
- Define fitness in `ExpertEvolutionProblem.fitness`: Use validation loss as the primary objective. Optionally add terms for model complexity or expert specialization (e.g., based on variance of `expert_usage_stats` from `EvolutionaryMoE`).
- Implement `EvolutionaryMoE` class in `src/core/evolutionary_experts.py` that dynamically creates experts based on `expert_configs`.
- Implement knowledge sharing mechanism between experts
    - *(Optional/Future)* Explore techniques like weight sharing initialization or migration between individuals in the PyGMO population.

### 3.3 Expert Weight Optimization
- Implement expert weight optimization using EC algorithms (DE, SADE, PSO, CMA-ES)
- *(Note)* Current `ExpertEvolutionProblem.fitness` includes brief training, optimizing weights implicitly as part of architecture evaluation. A separate phase purely for weight optimization could be added if needed (e.g., evolving weights *after* finding a good architecture).
- Define fitness functions for expert weight adjustment
- Create mechanisms for specialized experts (e.g., modality-specific)
- Extend `MigraineFuseMoE` in `src/core/pygmo_fusemoe.py` to handle `modality_experts` configuration, potentially assigning specific evolved experts to modalities.
- **Update modality expert allocation** to ensure all modalities have at least 1 expert (including stress)
- **Increase hidden_size** to at least 64-128 for improved model capacity (even in dev mode)

### 3.4 MoE Integration for Experts
- Modify `src/core/pygmo_fusemoe.py` (`PyGMOFuseMoE`, `MigraineFuseMoE`) to support evolutionary expert optimization during `optimize_model`.
- In `MigraineFuseMoE.optimize_model`:
    - Instantiate `ExpertEvolutionProblem` with encoded training/validation data.
    - Call `problem.optimize(algorithm_id=args.expert_algorithm, ...)` using PyGMO algorithms (SADE, PSO, CMA-ES).
    - Retrieve the `best_expert_configs` from the optimization results.
    - Instantiate the `self.experts` (likely `nn.ModuleList`) within `MigraineFuseMoE` using the optimized configurations.
- Create expert population management systems within PyGMO context.
- Utilize `pygmo.population` and `pygmo.algorithm` for population management and evolution loop within `ExpertEvolutionProblem.optimize`.
- **Implement larger population sizes** (20-30) for more robust optimization

## Phase 4: PSO-Enhanced Laplace Gating (Weeks 5-6)

### 4.1 PSO Gating Framework
- Develop `src/core/pso_laplace_gating.py` for PSO-enhanced gating
- Implement `PSOLaplaceGating` class inheriting from `nn.Module`, containing learnable scale parameters and the Laplace activation logic.
- Implement the Laplace activation mechanism with adaptive parameters
- Implement `PSOGatingProblem` class (similar to `ExpertEvolutionProblem`) for PyGMO:
    - Decision vector `x` represents gating network parameters (e.g., weights, biases, potentially Laplace scales).
    - `fitness` function evaluates gating performance (e.g., based on main model's loss/accuracy, potentially including a load balancing term).
- **Add fixed random seed** for reproducible initialization, configurable via parameter

### 4.2 Swarm Intelligence Algorithms
- Implement Particle Swarm Optimization (PSO) for gating weight optimization
- Use `pygmo.pso` algorithm within `PSOGatingProblem.optimize`.
- Add Artificial Bee Colony (ABC) algorithm as alternative (Optional)
- *(Optional)* Use `pygmo.abc` if desired.
- Implement dynamic routing with adaptive thresholding (Optional)
- **Increase population size** to 20-30 for more robust optimization

### 4.3 Multi-objective Gating Optimization
- Implement NSGA-II for balancing expert utilization vs. prediction accuracy (Optional)
- *(Optional)* Modify `PSOGatingProblem` to return multiple objectives (`get_nobj` > 1) and use `pygmo.nsga2`.
- Create objective functions for load balancing and specialization
- Add a load balancing term to the `fitness` function in `PSOGatingProblem` (e.g., based on variance or entropy of `gating.get_expert_usage()`).
- Develop visualization tools for gating optimization analysis (`notebooks/pso_gating/gating_mechanism_experiments.ipynb`)

### 4.4 MoE Integration for Gating
- In `MigraineFuseMoE.__init__`: Instantiate `PSOLaplaceGating` if `use_pso_gating` is True.
- In `MigraineFuseMoE.optimize_model`:
    - If `use_pso_gating` is True, instantiate `PSOGatingProblem`.
    - Call `problem.optimize(algorithm_id=args.gating_algorithm, ...)` using PyGMO PSO.
    - Update the parameters of the `self.gating` instance with the optimized weights/parameters.
- Add relevant arguments (`--gating_algorithm`, `--gating_population_size`, `--load_balance_coef`) to `run_migraine_prediction.py`'s `parse_args`.
- **Modify parameter defaults** to use PSO in both dev and production mode

## Phase 5: Training Pipeline & Evolutionary Fitness (Week 7)

### 5.1 Combined Training Integration
- Modify training scripts (e.g., `run_migraine_prediction.py`, `main_mimiciv.py`) and `src/core/pygmo_fusemoe.py` (`optimize_model` method) to handle alternating or joint expert and gating optimization.
- Current `MigraineFuseMoE.optimize_model` structure suggests a sequential optimization (experts first, then potentially gating). Implement logic for alternating optimization if needed.
- Implement unified evolutionary fitness evaluation framework considering both prediction performance and potentially imputation quality.
- The `fitness` methods within `ExpertEvolutionProblem` and `PSOGatingProblem` define the evaluation. Ensure they use appropriate loss functions (e.g., `weighted_bce_loss`) and metrics.
- Add parallel evaluation capability for population-based methods (leveraging PyGMO's capabilities).
- PyGMO handles internal parallelization; ensure the `fitness` functions are thread-safe if necessary.

### 5.2 Training Process Enhancement
- **Implement lower learning rate** (0.0005 instead of 0.001) for more stable optimization
- **Add learning rate scheduling** (e.g., ReduceLROnPlateau) to adapt learning rate during training
- **Increase dropout rate** to 0.2-0.25 for better regularization
- **Increase early stopping patience** from 1 to 3-5 epochs to avoid premature stopping
- **Increase validation split** from 15% to 20% for more reliable validation metrics
- **Optimize for F1 score** rather than balanced accuracy for better precision-recall balance
- **Reduce SMOTE sampling_ratio** from 0.8 to 0.6 to avoid excessive synthetic sampling

### 5.3 Hyperparameter Management
- Create configuration files/arguments for different optimizer settings (expert/gating algorithms, population size, generations).
- Utilize `argparse` in `run_migraine_prediction.py` for algorithm selection (`--expert_algorithm`, `--gating_algorithm`), population sizes (`--expert_population_size`, `--gating_population_size`), etc.
- Implement automated hyperparameter sensitivity analysis (Optional, potentially using Optuna/Hyperopt).
- Develop adaptive parameter control during training (Optional).

### 5.4 Checkpointing & Visualization
- Implement population serialization (via PyGMO) for training resumption.
- *(Future Improvement)* Modify `optimize` methods in PyGMO problem wrappers to periodically save `pygmo.population` state.
- Add best-individual tracking across epochs/generations.
- The `problem.optimize` methods currently track and return the best solution found. History logging is implemented in `ExpertEvolutionProblem`.
- Create visualizations for evolution and swarm optimization progress.
- Enhance history logging in `ExpertEvolutionProblem.optimize` (and implement similarly for `PSOGatingProblem`) to capture generation-wise best/average fitness.
- Use logged history in notebooks to plot convergence.
- *Visualization Notebooks:* `notebooks/evolutionary_optimization/expert_evolution_experiments.ipynb`, `notebooks/pso_gating/gating_mechanism_experiments.ipynb`.

## Phase 6: Patient-Specific Adaptation (Weeks 8-9)

### 6.1 Patient Profile Framework
- Implement patient profile manager (if needed beyond simple data loading).
- Create adaptation mechanism for patient-specific optimization (fine-tuning gating/experts on patient data, potentially via `adapt_to_patient` method in `pygmo_fusemoe.py`).
- Implement the `adapt_to_patient` method within `MigraineFuseMoE`. This could involve:
    - Fine-tuning the entire model or specific layers (e.g., gating, specific experts) using the patient's data (`patient_data_dict`).
    - Potentially running a short PyGMO optimization focused on gating or specific expert weights for that patient.
- Add logic in `run_migraine_prediction.py` to call `adapt_to_patient` when `--patient_adaptation` flag is set, using `--patient_id` to load data and `--load_base_model` for the starting point.
- Develop transfer learning approaches using a pre-trained base model.

### 6.2 Domain-Specific Fitness Functions
- Implement early-warning focused fitness function for migraine.
- *(Future Improvement)* Modify `fitness` functions or evaluation metrics (`calculate_metrics`) to include domain-specific measures if needed beyond standard classification metrics.
- Add cost-weighted fitness for false positive/negative tradeoffs.
- Create time-to-onset prediction scoring.
- Develop patient-specific trigger identification metrics (`identify_triggers` method in `pygmo_fusemoe.py`).
- Implement `identify_triggers` potentially using model interpretability techniques (e.g., attention scores, gradient-based methods) on the adapted model.

### 6.3 Evaluation Metrics
- Implement migraine-specific evaluation metrics (e.g., early warning time).
- Create visualization for prediction timeline, confidence over time.
- *(Future Improvement)* Add specific plots for patient adaptation results in `notebooks/patient_adaptation/adaptation_experiments.ipynb`.
- Add patient-specific performance analysis.
- Develop adaptive threshold tuning based on patient feedback.
- Threshold tuning (`find_optimal_threshold`) is already implemented in `run_migraine_prediction.py`.
- *Visualization Notebooks:* `notebooks/patient_adaptation/adaptation_experiments.ipynb`, `notebooks/visualization/prediction_explainability.ipynb`.

## Phase 7: Testing and Evaluation (Weeks 10-11)

### 7.1 Cross-Validation Framework
- **Implement patient-aware stratification** in cross-validation to prevent data leakage
- **Increase CV folds** to 5-10 for more robust evaluation
- **Add fold analysis tools** to understand performance variance across folds
- **Implement consistent random seeding** across folds for reproducibility
- **Create ensemble methods** combining models from different folds

### 7.2 Benchmark Suite
- Utilize `run_migraine_prediction.py` with different argument combinations (e.g., `--use_pygmo` vs. no, different imputation methods, different EC/SI algorithms) to compare performance.
- Implement cross-validation protocol for migraine prediction.
- Cross-validation using `sklearn.model_selection` (StratifiedKFold, KFold, TimeSeriesSplit) is implemented in `run_migraine_prediction.py`.
- Add statistical significance testing.
- Perform statistical tests (e.g., t-tests) on collected cross-validation results (`cv_summary.json`) offline or in analysis notebooks.
- Benchmark individual components (imputation, experts, gating, patient adaptation).

### 7.3 Ablation Studies
- Test without advanced imputation (use baseline).
- Run with `--imputation_method none`.
- Test without evolutionary expert optimization.
- Run without `--use_pygmo` flag (or potentially add finer control to disable only expert evolution).
- Test without PSO-enhanced gating.
- Run without `--use_pygmo` or modify `MigraineFuseMoE` to allow disabling only gating optimization.
- Measure impact of patient-specific adaptations.
- Compare runs with and without the `--patient_adaptation` flag.
- Benchmark different EC/SI algorithms.
- Run with different `--expert_algorithm` and `--gating_algorithm` settings.
- **Test with different prediction_horizon values** (6, 12, 24 hours) to assess impact on performance

### 7.4 Performance Optimization
- Implement GPU acceleration for fitness evaluations (if feasible within PyGMO user-defined problems).
- Ensure PyTorch models created within `fitness` methods are moved to the specified `device`. PyGMO itself typically runs optimization logic on CPU, but the expensive fitness calls (model training/eval) should leverage GPU.
- Add batch processing for population evaluation.
- PyGMO handles population evaluation; ensure the `fitness` implementation is efficient.
- Optimize memory usage for large populations.
- Implement early stopping for evolutionary algorithms.
- Basic generation limit is used. Implement more sophisticated stopping criteria within `ExpertEvolutionProblem.optimize` if needed (e.g., based on fitness stagnation).

## Phase 8: Documentation and Deployment (Week 12)

### 8.1 Code Documentation
- Add detailed docstrings to all new components (including `advanced_imputation.py`).
- Review and enhance docstrings in `advanced_imputation.py`, `evolutionary_experts.py`, `pso_laplace_gating.py`, `pygmo_fusemoe.py`.
- Create architecture diagrams for integration points (including data flow through imputation).
- Document hyperparameter effects and recommendations.
- Create flowcharts for evolutionary and swarm processes.

### 8.2 Usage Examples
- Create/update Jupyter notebooks demonstrating the full pipeline including imputation (`notebooks/integration/end_to_end_workflow.ipynb`).
- Ensure notebooks reflect the use of `argparse` parameters and integrated components.
- Add step-by-step tutorial for migraine prediction setup.
- Document best practices for imputation, evolutionary expert optimization, and PSO-enhanced gating configuration.

### 8.3 Deployment Package
- Create containerized version with all dependencies.
- Develop `Dockerfile` and potentially `docker-compose.yml`.
- Add CI/CD pipeline for testing.
- Implement GitHub Actions or similar for automated testing.
- Create simplified API for clinical deployment (Optional).
- Develop monitoring tools for evolutionary progress (Optional).

## Deliverables

1.  **EC/SI-Enhanced FuseMOE Library**: Complete integrated package with:
    *   **Advanced Imputation Module (`src/preprocessing/advanced_imputation.py`)**
    *   Evolutionary optimized expert networks (`src/core/evolutionary_experts.py`)
    *   PSO-enhanced Laplace gating mechanism (`src/core/pso_laplace_gating.py`)
    *   Patient-specific adaptation framework (`src/core/pygmo_fusemoe.py`)
    *   Evolutionary fitness evaluation components

2.  **Configuration Templates**: Ready-to-use configs/args for migraine prediction
    *   Imputation method selection and parameters
    *   Expert evolution settings
    *   PSO gating parameters
    *   Patient adaptation configurations
    *   Modality-specific settings for EEG, weather, sleep, etc.

3.  **Visualization Tools & Notebooks**: Interactive tools/notebooks for:
    *   `notebooks/visualization/imputation_results.ipynb`: Visualization and comparison of integrated imputation methods (using code from `src/preprocessing/advanced_imputation.py`).
    *   Expert evolution tracking (`notebooks/evolutionary_optimization/expert_evolution_experiments.ipynb`)
    *   Gating weight visualization & optimization analysis (`notebooks/pso_gating/gating_mechanism_experiments.ipynb`)
    *   Expert specialization (`notebooks/visualization/expert_specialization_viz.ipynb`)
    *   Patient adaptation monitoring (`notebooks/patient_adaptation/adaptation_experiments.ipynb`)
    *   Prediction performance analysis & explainability (`notebooks/visualization/prediction_explainability.ipynb`)
    *   End-to-end workflow examples (`notebooks/integration/end_to_end_workflow.ipynb`)

4.  **Benchmark Results**: Comprehensive comparison of:
    *   Different imputation methods' impact on downstream tasks
    *   Standard vs. EC/SI-enhanced FuseMOE performance
    *   Different EC/SI algorithm combinations
    *   Patient-specific vs. general models

5.  **Documentation**: Complete usage guide and API reference (including imputation module).

6.  **Migraine Prediction Model**: Optimized model ready for deployment.

## Resource Requirements

- **Computing**: GPU workstation for training and population evaluation
- **Data**: Access to temporal migraine datasets with multiple modalities (EEG, weather, sleep, diet, stress) or well-defined synthetic data generation.
- **Tools**: PyGMO 2.18+, PyTorch 1.7+, Scikit-learn, visualization libraries
- **Personnel**: ML engineer with optimization experience, domain expert for migraine features


## NOTES: (Updated)

*   The enhanced FuseMOE architecture integrates multimodal data (Time series, Text, Imaging, Signal, Custom like EEG/Weather).
*   **Preprocessing now includes an advanced imputation step (`src/preprocessing/advanced_imputation.py`) to handle missing values before model input.**
*   Evolutionary computing (PyGMO) is used for optimizing Experts (`src/core/evolutionary_experts.py`) and the Gating network (`src/core/pso_laplace_gating.py`).
*   MIMIC-IV processing involves specific scripts (`src/preprocessing/mimiciv_preprocessing/*`, `src/scripts/create_*_task.ipynb`).
*   Migraine prediction uses custom processors (`src/preprocessing/migraine_preprocessing/*`) and data pipeline (`src/preprocessing/migraine_preprocessing/migraine_data_pipeline.py`).
*   Execution is handled via scripts like `run_migraine.sh` or `run_pygmo_mimiciv.sh`, calling main scripts (`run_migraine_prediction.py`, `main_mimiciv.py`) which configure and run the `PyGMOFuseMoE` or `MigraineFuseMoE` models (`src/core/pygmo_fusemoe.py`).
*   Comprehensive visualization notebooks are planned for analyzing imputation, optimization, and prediction stages.


## Phase 9: Advanced Architecture & Data Integration (Future Work)

### 9.1 Enhance Expert Optimization with Diverse Activations
- **Goal**: Potentially improve expert performance by broadening the activation function search space during evolutionary optimization.
- **Implementation**:
    - Modify `ExpertEvolutionProblem._decode_solution` in `src/core/evolutionary_experts.py`.
    - Instead of hardcoding `nn.ReLU`, `nn.GELU`, etc., map the evolved activation index to functions defined in the `ACT2FN` dictionary from `src/core/activations.py`.
    - Update the bounds in `ExpertEvolutionProblem.get_bounds` to match the number of available activations in `ACT2FN`.
- **Evaluation**: Compare performance (validation loss, specialization) of models optimized with the expanded activation set versus the current set.

### 9.2 Investigate Hierarchical MoE (HME) Integration
- **Goal**: Explore if a two-level hierarchical MoE structure improves performance by allowing for more complex routing and specialization, based on `src/core/hierarchical_moe.py`.
- **Implementation (Conceptual - Major Architectural Change)**:
    - Redesign `MigraineFuseMoE` in `src/core/pygmo_fusemoe.py` to incorporate two levels of gating (e.g., `Top2Gating`) and expert layers.
    - Adapt `ExpertEvolutionProblem` and `PSOGatingProblem` to optimize components at both hierarchical levels.
    - Modify the forward pass to handle the two-stage dispatch and combination logic.
- **Evaluation**: Benchmark HME performance against the current flat MoE architecture. Assess complexity vs. performance trade-offs.

### 9.3 Integrate Additional EHR Data Modalities
- **Goal**: Extend the model's predictive capabilities by incorporating richer clinical context, such as vitals or structured hospital records (admissions, diagnoses).
- **Implementation Steps**:
    1.  **Create New Preprocessor(s)**: Develop modules (e.g., in `src/preprocessing/ehr_preprocessing/`) to load, clean, and extract features from the new data source(s).
    2.  **Update Data Pipeline**: Modify `MigraineDataPipeline` (or a new `EHRPipeline`) to include the new processor(s) and merge the resulting features into the multimodal dataset under unique modality keys (e.g., `'vitals'`, `'records'`). Update `prepare_data_for_fusemoe` accordingly.
    3.  **Adapt Model (`MigraineFuseMoE`)**: The model should automatically adapt if the pipeline correctly adds the new modality to the `input_sizes` dictionary passed during initialization. Ensure the default encoder created in `__init__` is suitable, or customize it.
    4.  **Configure Expert Allocation**: Adjust `--modality_experts` or the logic in `get_modality_experts_config` to allocate experts to the new modality.
- **Evaluation**: Measure the impact of the new data modality on prediction accuracy, AUC, and other relevant metrics.

### 9.4 Advanced Ensemble Techniques for High Performance
- **Goal**: Achieve >95% accuracy and high AUC through sophisticated ensemble methods combining models from different folds and configurations.
- **Implementation Steps**:
    1. **Fold-based Ensemble**: Create a meta-ensemble combining predictions from models trained on different CV folds.
    2. **Heterogeneous Architecture Ensemble**: Combine models with different expert/gating configurations and hyperparameters.
    3. **Bagging with Data Perturbation**: Train multiple models on bootstrapped samples with varied class balancing approaches.
    4. **Time-aware Weighting**: Weight ensemble members differently based on their performance on time-adjacent predictions.
- **Evaluation**: Measure ensemble performance against target metrics (95% accuracy, high AUC) and compare to individual model performance to quantify ensemble gains.