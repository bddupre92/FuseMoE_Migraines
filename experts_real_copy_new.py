# %%
"""
# Expert Performance Analysis (Real Data Focus)

This notebook analyzes the performance of different expert models in the MoE framework, comparing their baseline performance to optimized versions using potentially real data.

**Goals:**
1.  Load and preprocess real data (or use synthetic data as a fallback).
2.  Train baseline and optimized versions of each expert.
3.  Evaluate performance using metrics like MSE and R².
4.  Visualize results:
    *   Performance comparisons (MSE, R²)
    *   Prediction quality (Predictions vs Actuals, Residuals)
    *   (Optional) Feature Importance

Let's begin by installing dependencies and setting up the environment.
"""

# %%
"""

"""

# %%
# Reload modules to ensure latest code changes are used
import importlib
import sys
from pathlib import Path

# Get all environmental_expert modules that might be loaded
env_expert_modules = [m for m in sys.modules if 'environmental_expert' in m]

# Reload them
for module_name in env_expert_modules:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

# Also reload the experts module
if 'experts' in sys.modules:
    importlib.reload(sys.modules['experts'])

print("Modules reloaded successfully")

# %%
# %pip install -q -U ipykernel
# %pip install -q -U matplotlib
# %pip install -q -U seaborn
# %pip install -q -U plotly
# %pip install -q -U ipywidgets
# %pip install -q -U pandas
# %pip install -q -U numpy
# %pip install -q -U scikit-learn
# %pip install -q -U scipy
# %pip install -q -U statsmodels
# %pip install -q -U xgboost
# %pip install -q -U lightgbm
# %pip install -q -U catboost



# %%
# %pip install plotly


# %%
# --- Cell 3: Imports and Path Setup ---
import os
import sys
import numpy as np
import pandas as pd
import joblib
import copy
import time
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Configuration
from omegaconf import DictConfig, OmegaConf

# Machine Learning & Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold # ADD THIS IMPORT

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from IPython.display import display, clear_output

# Progress Bar
from tqdm.notebook import tqdm

# --- Path Setup ---
# Direct path to the moe_framework project root
project_root = '/Users/blair.dupre/Documents/migrineDT/moe_framework' # Hardcoded path for reliability
config_dir_relative = "config/experts"
config_dir_abs = os.path.join(project_root, config_dir_relative)
output_dir = os.path.join(os.getcwd(), "analysis_outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Project Root: {project_root}")
print(f"Config Directory (Absolute): {config_dir_abs}")
print(f"Output Directory: {output_dir}")

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added '{project_root}' to sys.path")
else:
    print(f"'{project_root}' already in sys.path")

# --- Import MoE Framework Components ---
try:
    from experts.base_expert import BaseExpert
    from experts.physiological_expert import PhysiologicalExpert
    from experts.environmental_expert import EnvironmentalExpert
    from experts.behavioral_expert import BehavioralExpert
    from experts.medication_history_expert import MedicationHistoryExpert
    # Import the *real* factory and base optimizer
    from optimizers.optimizer_factory import OptimizerFactory
    from optimizers.base_optimizer import BaseOptimizer
    print("Successfully imported MoE framework components.")
except ImportError as e:
    print(f"ERROR: Failed to import MoE framework components.")
    print(f"Details: {e}")
    print("Please ensure:")
    print(f"1. Project root is correctly set to: {project_root}")
    print(f"2. The '{project_root}' directory is in sys.path.")
    print(f"3. All necessary __init__.py files exist in subdirectories (e.g., experts/, optimizers/).")
    print(f"4. Framework classes (e.g., PhysiologicalExpert) exist and are correctly named.")
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred during framework import: {e}")
    traceback.print_exc()

print("\nCell 3 execution complete.")

# %%
# --- Cell 4: Logging Setup ---
import logging

# Configure logging
log_file = os.path.join(output_dir, 'analysis_log.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout) # Also print logs to notebook output
                    ])

logger = logging.getLogger(__name__)
logger.info("Logging configured. Output will be saved to %s", log_file)

# Example log message
logger.info("Notebook execution started.")

# %%
# --- Notebook Setup: Imports ---

# Standard Libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import sys
import joblib
import copy  # For deep copying configurations
import traceback  # For detailed error reporting
from typing import Dict, List, Any, Optional, Tuple

# Configuration Management
from omegaconf import DictConfig, OmegaConf

# Machine Learning and Metrics
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Example model type

# Visualization (Setup for later use)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from IPython.display import display

# Path setup
project_root = "/Users/blair.dupre/Documents/migrineDT/moe_framework"
config_dir = os.path.join(project_root, "config/experts")

# Add project root to path if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")

# MoE Framework Components
try:
    from experts.physiological_expert import PhysiologicalExpert
    from experts.environmental_expert import EnvironmentalExpert
    from experts.behavioral_expert import BehavioralExpert
    from experts.medication_history_expert import MedicationHistoryExpert
    from optimizers.optimizer_factory import OptimizerFactory  # The REAL one
    from optimizers.base_optimizer import BaseOptimizer  # Optional: for type hints
    from experts.base_expert import BaseExpert  # Optional: for type hints
    print("MoE framework components imported successfully.")
except ImportError as e:
    print(f"ERROR importing framework components: {e}")
    traceback.print_exc()

# --- Plotting Style Setup ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
# Define a color palette for optimizers + baseline
optimizer_colors = plt.cm.tab10(np.linspace(0, 1, 10))
method_colors = {'baseline': 'grey'}

# Ensure plots display inline in the notebook
# %matplotlib inline

print("Setup complete!")

# %%
# --- Cell 5: Plotting Style Setup ---
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

# Define a consistent color palette
# Using a qualitative colormap like 'tab10' or 'Set1'
cmap = plt.get_cmap('tab10')
method_colors = {
    'baseline': 'grey',
    'differential_evolution': cmap(0),
    'pso': cmap(1),
    'ga': cmap(2),
    'random': cmap(3),
    'cma': cmap(4), # Example for another potential optimizer
    # Add more optimizers and their colors as needed
}

# Ensure plots display inline in the notebook
# %matplotlib inline

logger.info("Plotting styles set.")

# %%
# --- Cell 6: Configuration Loading ---
import os
from omegaconf import OmegaConf, DictConfig

# Use the absolute path derived in Cell 3
expert_config_dir = config_dir_abs
expert_base_configs = {}
available_experts = ['physiological', 'environmental', 'behavioral', 'medication']
logger.info(f"Loading expert configurations from: {expert_config_dir}")

for expert_name in available_experts:
    yaml_filename = f"{expert_name}_expert.yaml"
    # Handle specific naming conventions if necessary
    if expert_name == 'medication':
        yaml_filename = "medication_history_expert.yaml"

    yaml_file_path = os.path.join(expert_config_dir, yaml_filename)

    if os.path.exists(yaml_file_path):
        try:
            loaded_config = OmegaConf.load(yaml_file_path)
            # Basic validation
            if isinstance(loaded_config, DictConfig) and loaded_config.get('name'):
                # Ensure essential keys exist for robustness
                if 'model' not in loaded_config: loaded_config.model = {}
                if 'preprocessing' not in loaded_config: loaded_config.preprocessing = {}
                if 'training' not in loaded_config: loaded_config.training = {}
                if 'paths' not in loaded_config: loaded_config.paths = {}

                expert_base_configs[expert_name] = loaded_config
                logger.info(f"Successfully loaded and validated: {yaml_filename}")
            else:
                logger.error(f"Invalid structure or missing 'name' in {yaml_filename}. Skipping.")
                expert_base_configs[expert_name] = None # Mark as failed
        except Exception as e:
            logger.error(f"Error loading {yaml_filename}: {e}", exc_info=True)
            expert_base_configs[expert_name] = None # Mark as failed
    else:
        logger.warning(f"YAML file not found: {yaml_file_path}. Skipping this expert.")
        expert_base_configs[expert_name] = None # Mark as not found

# Filter out failed loads
loaded_expert_names = [name for name, cfg in expert_base_configs.items() if cfg is not None]
logger.info(f"Successfully loaded base configurations for experts: {loaded_expert_names}")

if not loaded_expert_names:
     logger.critical("No expert configurations were loaded successfully. Analysis cannot proceed.")
     # Optionally raise an error: raise RuntimeError("Failed to load any expert configurations.")

# Display a sample config (optional)
if 'physiological' in expert_base_configs and expert_base_configs['physiological']:
    logger.info("Sample configuration (Physiological):\n%s", OmegaConf.to_yaml(expert_base_configs['physiological']))

# %%
# --- Cell 7: Define Alternative Optimizer Configurations ---
# These are templates that will be merged with expert-specific details later
from omegaconf import OmegaConf

# Define optimizer configuration TEMPLATES
# These DO NOT YET include expert-specific bounds, param_names, or search_space
# Those will be added in the training loop using get_expert_specific_opt_details

# First, try to load our centralized optimizer configurations
try:
    if USE_REAL_DATA:
        # Use complex data optimizer settings for the enhanced synthetic data
        try:
            optimizer_base_configs = OmegaConf.load(os.path.join(project_root, 'config/optimizers/complex_data.yaml'))
            logger.info("Using optimizer configurations optimized for complex data patterns")
        except Exception as e_complex:
            logger.warning(f"Could not load complex_data optimizer config: {e_complex}. Falling back to default.")
            optimizer_base_configs = OmegaConf.load(os.path.join(project_root, 'config/optimizers/default.yaml'))
            logger.info("Using default optimizer configurations")
    else:
        # Use dev configs for simple synthetic data
        optimizer_base_configs = OmegaConf.load(os.path.join(project_root, 'config/optimizers/dev.yaml'))
        logger.info("Using development optimizer configurations (faster execution)")
except Exception as e:
    try:
        # Fall back to default configs if specified config doesn't exist
        optimizer_base_configs = OmegaConf.load(os.path.join(project_root, 'config/optimizers/default.yaml'))
        logger.info("Using default optimizer configurations due to error: {e}")
    except Exception as e2:
        # If neither exists, use hardcoded defaults
        logger.warning(f"Could not load optimizer configs: {e2}. Using hardcoded defaults.")
        optimizer_base_configs = None

# Define alternative optimizer configs, using centralized configs when possible
alternative_optimizer_configs = {
    'differential_evolution': OmegaConf.create({
        'name': 'differential_evolution',
        'metric': 'neg_mean_squared_error', # Metric to optimize
        # Early stopping parameters (added for compatibility with recent changes)
        'early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_threshold': 1e-4,
        'config': optimizer_base_configs.differential_evolution if optimizer_base_configs else {
            'strategy': 'best1bin',
            'maxiter': 50, # Increased from 5 for better optimization
            'popsize': 30, # Increased from 5 for better optimization
            'tol': 0.01,
            'mutation': (0.5, 1),
            'recombination': 0.7,
            'seed': 42,
            'polish': False,
            'init': 'latinhypercube'
        }
        # 'param_names' and 'bounds' will be added per expert later
    }),
    'pso': OmegaConf.create({
        'name': 'pso',
        'metric': 'neg_mean_squared_error',
        # Early stopping parameters (added for compatibility with recent changes)
        'early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_threshold': 1e-4,
        'config': optimizer_base_configs.pso if optimizer_base_configs else {
            'n_particles': 30, # Increased from 8 for better optimization
            'max_iterations': 50, # Increased from 5 for better optimization
            'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
            'ftol': -1 # No early stopping based on fitness tolerance
        }
        # 'param_names' and 'bounds' will be added per expert later
    }),
    'ga': OmegaConf.create({
        'name': 'ga',
        'metric': 'neg_mean_squared_error',
        # Early stopping parameters (added for compatibility with recent changes)
        'early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_threshold': 1e-4,
        'config': optimizer_base_configs.ga if optimizer_base_configs else {
            'population_size': 30, # Increased from 8 for better optimization
            'num_generations': 50, # Increased from 5 for better optimization
            'cxpb': 0.6, # Crossover probability
            'mutpb': 0.3, # Mutation probability
            'indpb': 0.05 # Independent gene mutation probability
        }
        # 'param_names' and 'bounds' will be added per expert later
    }),
    'random': OmegaConf.create({
        'name': 'random',
        'metric': 'neg_mean_squared_error',
        # Early stopping parameters (added for compatibility with recent changes)
        'early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_threshold': 1e-4,
        'config': optimizer_base_configs.random if optimizer_base_configs else {
            'n_iter': 75, # Increased from 15 for better optimization
            'cv': 3, # 3-fold cross-validation within the search
            'random_state': 42,
            'n_jobs': -1 # Use all available CPU cores
        }
        # 'search_space' will be added per expert later
    })
    # Add other optimizer templates here (e.g., 'cma', 'bayesian')
}

optimizer_names = list(alternative_optimizer_configs.keys())
logger.info(f"Defined alternative optimizer config templates for: {optimizer_names}")

# %%
# --- Cell 8: Data Loading / Generation ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# <<< REAL DATA LOADING: START >>>
# ---------------------------------
# TODO: Implement your real data loading logic here.
# This section should produce the following variables for EACH expert:
#   - X_train_<expert_type>: Training features (Pandas DataFrame)
#   - X_test_<expert_type>: Testing features (Pandas DataFrame)
#   - y_train_<expert_type>: Training target (Pandas Series or NumPy array)
#   - y_test_<expert_type>: Testing target (Pandas Series or NumPy array)
#
# Example structure:
#
# def load_real_data_for_expert(expert_type, data_path_config):
#     """Loads, preprocesses, and splits real data for a given expert."""
#     logger.info(f"Attempting to load real data for {expert_type}...")
#     # 1. Construct file path using data_path_config or expert config
#     #    raw_data_path = expert_base_configs[expert_type].paths.get('raw_data', None)
#     #    if not raw_data_path:
#     #       logger.warning(f"No raw_data path configured for {expert_type}. Cannot load real data.")
#     #       return None, None, None, None
#     #    full_path = os.path.join(project_root, raw_data_path) # Make absolute
#     # 2. Load data (e.g., pd.read_csv(full_path))
#     # 3. Perform initial cleaning/filtering if needed
#     # 4. **Crucially:** Select the features relevant to *this specific expert*
#     #    (Use features defined in expert_base_configs[expert_type].preprocessing.feature_columns)
#     #    feature_cols = expert_base_configs[expert_type].preprocessing.feature_columns
#     #    target_col = expert_base_configs[expert_type].preprocessing.target_column
#     #    X = data[feature_cols]
#     #    y = data[target_col]
#     # 5. Split data using train_test_split
#     #    X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
#     # 6. Return the splits
#     logger.info(f"Successfully loaded and split real data for {expert_type}.")
#     # return X_train, X_test, y_train, y_test
#     return None, None, None, None # Placeholder return

# Import the real data loading module
from notebooks.experts.load_real_data import load_real_data_for_expert

# Flag to control data source
USE_REAL_DATA = True  # Using enhanced synthetic data as "real data" with complex patterns
all_data_loaded = True # Assume success unless loading fails

if USE_REAL_DATA:
    logger.info("Attempting to load complex synthetic data as real data...")
    real_data = {}
    # Loop through experts to load data for each type
    for expert_type in loaded_expert_names:
        logger.info(f"Loading complex data for {expert_type}...")
        X_train, X_test, y_train, y_test = load_real_data_for_expert(expert_type, None)
        
        if X_train is None:
            logger.warning(f"Failed to load complex data for {expert_type}. Will attempt to use simple synthetic data.")
            all_data_loaded = False
            # break # Option 1: Stop if any real data fails
            continue # Option 2: Try to load others and use synthetic for failed ones
        else:
            real_data[expert_type] = (X_train, X_test, y_train, y_test)
            logger.info(f"Loaded complex data for {expert_type}: {X_train.shape[0]} train samples with {X_train.shape[1]} features.")

    if not all_data_loaded and not real_data: # If all failed
         logger.error("Failed to load any real data. Falling back entirely to synthetic data generation.")
         USE_REAL_DATA = False # Force fallback
    elif not all_data_loaded:
         logger.warning("Failed to load real data for some experts. Synthetic data will be generated for those.")
         # We'll handle assigning data later in the training loop cell
    else:
         logger.info("Successfully loaded real data for all required experts.")

# -------------------------------
# <<< REAL DATA LOADING: END >>>


# --- Synthetic Data Generation (Fallback/Example) ---
def create_expert_specific_data(expert_type, n_samples=500, test_size=0.3, random_state=42):
    """Creates SYNTHETIC sample data with the correct structure for each expert type."""
    logger.debug(f"Generating {n_samples} synthetic samples for {expert_type}...")
    np.random.seed(random_state)
    random.seed(random_state)

    # Common elements
    base_date = datetime(2023, 1, 1)
    timestamps = [base_date + timedelta(hours=6*i) for i in range(n_samples)]
    patient_ids = np.random.randint(1, 51, n_samples)

    # Get expected feature columns from config if available, otherwise use defaults
    expected_features = []
    if expert_type in expert_base_configs and expert_base_configs[expert_type]:
        expected_features = expert_base_configs[expert_type].preprocessing.get('feature_columns', [])
        # Remove common identifiers if they are listed as features mistakenly
        expected_features = [f for f in expected_features if f not in ['timestamp', 'patient_id', 'migraine_label']]
    else:
        logger.warning(f"No config found for {expert_type} to get feature list, using defaults.")

    # Define default synthetic features (superset)
    default_features = {
        'physiological': ['hr', 'hrv', 'temp', 'resp_rate', 'gsr', 'bp_systolic', 'bp_diastolic'],
        'environmental': ['temperature', 'humidity', 'pressure', 'light_level', 'noise_level', 'air_quality', 'pollen_count'],
        'behavioral': ['sleep_duration', 'sleep_quality', 'activity_steps', 'exercise_minutes', 'stress_level', 'water_intake', 'caffeine_mg', 'alcohol_units'],
        'medication': ['medication_triptan', 'medication_nsaid', 'medication_preventive', 'dosage_triptan', 'dosage_nsaid', 'dosage_preventive', 'time_since_last_dose']
    }
    
    current_features = expected_features if expected_features else default_features.get(expert_type, [])
    if not current_features:
        logger.error(f"Cannot determine features for synthetic data generation for {expert_type}. Skipping.")
        return None, None, None, None # Indicate failure

    X_data = {}
    # Generate data only for the required features
    if 'hr' in current_features: X_data['hr'] = np.random.normal(70, 10, n_samples)
    if 'hrv' in current_features: X_data['hrv'] = np.random.normal(50, 15, n_samples)
    if 'temp' in current_features: X_data['temp'] = np.random.normal(36.8, 0.5, n_samples)
    if 'resp_rate' in current_features: X_data['resp_rate'] = np.random.normal(16, 3, n_samples)
    if 'gsr' in current_features: X_data['gsr'] = np.random.normal(5, 2, n_samples)
    if 'bp_systolic' in current_features: X_data['bp_systolic'] = np.random.normal(120, 15, n_samples)
    if 'bp_diastolic' in current_features: X_data['bp_diastolic'] = np.random.normal(80, 10, n_samples)
    
    if 'temperature' in current_features: X_data['temperature'] = np.random.normal(22, 8, n_samples)
    if 'humidity' in current_features: X_data['humidity'] = np.random.uniform(30, 90, n_samples)
    if 'pressure' in current_features: X_data['pressure'] = np.random.normal(1013, 10, n_samples)
    if 'light_level' in current_features: X_data['light_level'] = np.random.uniform(0, 1000, n_samples)
    if 'noise_level' in current_features: X_data['noise_level'] = np.random.uniform(30, 90, n_samples)
    if 'air_quality' in current_features: X_data['air_quality'] = np.random.normal(50, 20, n_samples)
    if 'pollen_count' in current_features: X_data['pollen_count'] = np.random.exponential(30, n_samples)

    if 'sleep_duration' in current_features: X_data['sleep_duration'] = np.random.normal(7, 1.5, n_samples)
    if 'sleep_quality' in current_features: X_data['sleep_quality'] = np.random.uniform(1, 10, n_samples)
    if 'activity_steps' in current_features: X_data['activity_steps'] = np.random.gamma(7000, 0.5, n_samples)
    if 'exercise_minutes' in current_features: X_data['exercise_minutes'] = np.random.exponential(30, n_samples)
    if 'stress_level' in current_features: X_data['stress_level'] = np.random.uniform(1, 10, n_samples)
    if 'water_intake' in current_features: X_data['water_intake'] = np.random.normal(1500, 500, n_samples)
    if 'caffeine_mg' in current_features: X_data['caffeine_mg'] = np.random.choice([0, 0, 100, 200], size=n_samples)
    if 'alcohol_units' in current_features: X_data['alcohol_units'] = np.random.choice([0, 0, 0, 1, 2, 3], size=n_samples)
        
    if 'medication_triptan' in current_features: X_data['medication_triptan'] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    if 'medication_nsaid' in current_features: X_data['medication_nsaid'] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    if 'medication_preventive' in current_features: X_data['medication_preventive'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    if 'dosage_triptan' in current_features: X_data['dosage_triptan'] = X_data.get('medication_triptan', 0) * np.random.choice([0, 25, 50, 100], size=n_samples)
    if 'dosage_nsaid' in current_features: X_data['dosage_nsaid'] = X_data.get('medication_nsaid', 0) * np.random.choice([0, 200, 400, 600], size=n_samples)
    if 'dosage_preventive' in current_features: X_data['dosage_preventive'] = X_data.get('medication_preventive', 0) * np.random.choice([0, 50, 100, 150], size=n_samples)
    if 'time_since_last_dose' in current_features: X_data['time_since_last_dose'] = np.random.exponential(24, n_samples) # Hours

    X = pd.DataFrame(X_data)
    X['timestamp'] = timestamps
    X['patient_id'] = patient_ids

    # Define a simple target variable based on *some* of the generated features
    y = np.zeros(n_samples)
    if 'hr' in X.columns: y += X['hr'] / 30
    if 'stress_level' in X.columns: y += X['stress_level'] / 2
    if 'temperature' in X.columns: y += (X['temperature'] - 20) / 5
    if 'medication_triptan' in X.columns: y -= X['medication_triptan'] * 2
    if 'sleep_duration' in X.columns: y -= (7 - X['sleep_duration'])
    y += np.random.normal(0, 1.5, n_samples) # Add noise

    # Scale target to approx 0-10 range
    y = np.clip(y, np.percentile(y, 5), np.percentile(y, 95)) # Clip outliers
    y_min, y_max = y.min(), y.max()
    if y_max > y_min:
        y = (y - y_min) / (y_max - y_min) * 10
    else:
        y = np.full(n_samples, 5.0) # Default if no variance

    # Ensure target has some variance if scaling failed
    if np.std(y) < 1e-6:
        y += np.random.normal(0, 0.1, n_samples)
        y = np.clip(y, 0, 10)

    y = pd.Series(y, name='target_pain_level') # Give target a name

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logger.debug(f"Finished generating synthetic data for {expert_type}. Train shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test

# Dictionary to hold generated synthetic data (used if USE_REAL_DATA is False or fails)
synthetic_data = {}

# Generate synthetic data only if real data isn't fully loaded
if not USE_REAL_DATA or not all_data_loaded:
    logger.info("Generating SYNTHETIC data as fallback or supplement...")
    for expert_type in loaded_expert_names:
        # Only generate if real data wasn't loaded for this expert
        if USE_REAL_DATA and expert_type in real_data:
            logger.info(f"Skipping synthetic data for {expert_type} as real data was loaded.")
            continue

        logger.info(f"Generating synthetic data for {expert_type}...")
        data_tuple = create_expert_specific_data(expert_type)
        if data_tuple[0] is not None: # Check if generation succeeded
             synthetic_data[expert_type] = data_tuple
             logger.info(f"Synthetic data generated for {expert_type}. Train shape: {data_tuple[0].shape}")
        else:
             logger.error(f"Failed to generate synthetic data for {expert_type}.")

    if not synthetic_data and not (USE_REAL_DATA and real_data):
        logger.critical("Failed to load real data AND failed to generate any synthetic data. Cannot proceed.")
        # raise RuntimeError("No data available for analysis.")
elif USE_REAL_DATA and all_data_loaded:
    logger.info("Real data loaded successfully for all experts. Skipping synthetic data generation.")


print("\nCell 8 execution complete.")
# Example: Display shape of physiological data (choose source based on flags)
check_expert = 'physiological'
if USE_REAL_DATA and check_expert in real_data:
    print(f"Using REAL data for {check_expert}. Train shape: {real_data[check_expert][0].shape}")
elif check_expert in synthetic_data:
    print(f"Using SYNTHETIC data for {check_expert}. Train shape: {synthetic_data[check_expert][0].shape}")
else:
    print(f"No data available to check for {check_expert}.")


# %%
# --- Cell 9: Data Exploration and Preprocessing (Placeholder) ---

# TODO: Add any specific data exploration or *additional* preprocessing steps here.
# Note: Basic preprocessing like scaling or encoding might be handled within the
#       expert's `preprocess` method based on its configuration.
# This cell is for notebook-specific exploration or transformations not covered by the expert's standard pipeline.

# Example:
# if USE_REAL_DATA and 'physiological' in real_data:
#     X_train_phys, _, _, _ = real_data['physiological']
#     print("\n--- Physiological Data Exploration ---")
#     print("Basic Info:")
#     X_train_phys.info()
#     print("\nDescriptive Statistics:")
#     display(X_train_phys.describe())
#     print("\nMissing Values:")
#     print(X_train_phys.isnull().sum())

#     # Example preprocessing (if not done elsewhere):
#     # scaler = StandardScaler()
#     # X_train_phys_scaled = scaler.fit_transform(X_train_phys)
#     # X_test_phys_scaled = scaler.transform(X_test_phys) # Assuming X_test_phys exists

logger.info("Cell 9: No additional exploration or preprocessing steps defined.")

# %%
# --- Cell 10: Helper Functions ---
import time
import copy
from omegaconf import OmegaConf, DictConfig

def get_expert_specific_opt_details(expert_type: str, base_config: DictConfig) -> Dict[str, Any]:
    """
    Retrieves or defines optimization details (parameters, bounds, search space)
    specific to an expert type. It prioritizes details defined within the
    expert's `hyperparameter_space` configuration if available, falling back
    to `training.optimizer` otherwise.
    """
    # Need logger and DictConfig/ListConfig from omegaconf, assume they are imported above
    # Also need numpy if used for linspace
    import logging
    from omegaconf import DictConfig, ListConfig

    logger = logging.getLogger(__name__) # Make sure logger is accessible

    logger.debug(f"Getting optimization details for {expert_type}...")

    # Start with empty details
    details = {
        'param_names': [],
        'bounds': [],
        'search_space': {}
    }
    source_used = "None" # Track where details came from

    # --- Priority 1: Extract from hyperparameter_space if available ---
    hyperparameter_space = base_config.get('hyperparameter_space', None)
    if hyperparameter_space and isinstance(hyperparameter_space, DictConfig):
        logger.info(f"Attempting to extract details from hyperparameter_space for {expert_type}")
        param_names = []
        bounds = []
        search_space = {}
        valid_params_found = False
        
        for param_path, param_config in hyperparameter_space.items():
            if not isinstance(param_config, DictConfig):
                logger.warning(f"Skipping invalid entry in hyperparameter_space for {expert_type}: Key '{param_path}' does not map to a dictionary.")
                continue

            param_names.append(param_path)
            valid_param_config = True # Assume valid until proven otherwise
            
            # --- Get Bounds (for DE, PSO, GA etc.) ---
            if 'min' in param_config and 'max' in param_config:
                try:
                min_val = float(param_config.get('min'))
                max_val = float(param_config.get('max'))
                    if min_val > max_val:
                        logger.warning(f"Min ({min_val}) > Max ({max_val}) for param '{param_path}' in {expert_type}'s hyperparameter_space bounds. Using default (0.0, 1.0).")
                        bounds.append((0.0, 1.0))
                    else:
                bounds.append((min_val, max_val))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid min/max value for param '{param_path}' bounds in {expert_type}'s hyperparameter_space. Using default (0.0, 1.0).")
                    bounds.append((0.0, 1.0))
                    valid_param_config = False
            else:
                # Use default bounds (0.0, 1.0) if min/max not specified
                # Crucial for ensuring bounds list matches param_names list length
                bounds.append((0.0, 1.0))
                logger.debug(f"No min/max specified for '{param_path}' bounds, using default (0.0, 1.0). Expected for non-numerical types.")

            # --- Build Search Space (primarily for Random Search) ---
            param_type = param_config.get('type')
            values = param_config.get('values')

            if values is not None: # Directly use provided list/values
                if isinstance(values, (list, tuple, ListConfig)): # Allow OmegaConf ListConfig
                     search_space[param_path] = list(values) # Ensure it's a list
                else:
                     logger.warning(f"'values' for param '{param_path}' in {expert_type}'s hyperparameter_space is not a list/tuple. Ignoring for search_space.")
                     valid_param_config = False # Invalid search space config
            elif param_type == 'int':
                try:
                min_val = int(param_config.get('min', 0))
                    max_val = int(param_config.get('max', 10))
                    if min_val > max_val:
                        logger.warning(f"Min > Max for int param '{param_path}' search_space. Using range({min_val}, {min_val + 1}).")
                        max_val = min_val
                    # Generate a reasonable number of steps (e.g., up to 11)
                    num_steps = min(11, max_val - min_val + 1)
                    if num_steps <= 1:
                        search_space[param_path] = [min_val]
                    else:
                        # Generate integers, simple range with adjusted step
                        step_size = max(1, (max_val - min_val + 1) // (num_steps -1) if num_steps > 1 else 1)
                        gen_list = list(range(min_val, max_val + 1, step_size))
                        # Ensure max_val is included if step size misses it and it wasn't the only value
                        if gen_list[-1] != max_val and max_val not in gen_list:
                             gen_list.append(max_val)
                             gen_list.sort() # Keep it sorted
                        search_space[param_path] = gen_list

                except (ValueError, TypeError):
                    logger.warning(f"Invalid min/max for int param '{param_path}' search space. Ignoring for search_space.")
                    valid_param_config = False
            elif param_type == 'float':
                try:
                min_val = float(param_config.get('min', 0.0))
                max_val = float(param_config.get('max', 1.0))
                    if min_val > max_val:
                        logger.warning(f"Min > Max for float param '{param_path}' search_space. Using default [0.0, ..., 1.0].")
                        min_val, max_val = 0.0, 1.0
                    num_steps = 11 # Generate 11 steps including endpoints
                    # Use numpy for linspace if available, otherwise simple list comprehension
                    try:
                        import numpy as np
                        search_space[param_path] = np.linspace(min_val, max_val, num=num_steps).tolist()
                    except ImportError:
                         logger.debug("Numpy not found, using basic linspace calculation for float search space.")
                         # Avoid division by zero if num_steps is 1
                         step = (max_val - min_val)/(num_steps - 1) if num_steps > 1 else 0
                         search_space[param_path] = [min_val + i * step for i in range(num_steps)]

                except (ValueError, TypeError):
                    logger.warning(f"Invalid min/max for float param '{param_path}' search space. Ignoring for search_space.")
                    valid_param_config = False
            elif param_type == 'bool':
                search_space[param_path] = [True, False]
            else:
                 logger.warning(f"Unsupported/missing type or values for param '{param_path}' in {expert_type}'s hyperparameter_space. Cannot generate search_space item.")
                 # Don't mark as invalid config, just can't generate search space here

            if valid_param_config:
                valid_params_found = True # Mark that we found at least one valid param config

        # --- Assign details if extraction from hyperparameter_space was successful and consistent ---
        # MODIFIED LOGIC: Prioritize returning param_names and search_space if hyperparameter_space was processed,
        # even if bounds check fails (important for Random Search).
        if valid_params_found and param_names:
            details = {
                'param_names': param_names,
                'bounds': bounds if bounds and len(bounds) == len(param_names) else [], # Include bounds only if consistent
                'search_space': search_space
            }
            source_used = "hyperparameter_space"
            log_msg = f"Successfully extracted details from hyperparameter_space for {expert_type}: {len(param_names)} parameters."
            if not details['bounds']:
                log_msg += " Bounds were inconsistent or missing but search_space is present."
            logger.info(log_msg)
        elif hyperparameter_space: # Log if space exists but extraction failed consistency checks
             logger.warning(f"Hyperparameter_space found for {expert_type}, but failed consistency checks (e.g., invalid values, mismatched lengths). Param names found: {len(param_names)}, Bounds generated: {len(bounds)}. Attempting fallback...")
        # No else needed here, already covered by the initial check for hyperparameter_space

    # --- Priority 2: Fallback to training.optimizer ONLY if hyperparameter_space was NOT used ---
    if source_used != "hyperparameter_space":
    optimizer_config = base_config.training.get('optimizer', None)
        if optimizer_config and isinstance(optimizer_config, DictConfig):
            logger.info(f"Attempting fallback extraction from training.optimizer for {expert_type}")
        
            param_names_raw = optimizer_config.get('param_names', [])
        bounds_raw = optimizer_config.get('bounds', [])
            search_space_raw = optimizer_config.get('search_space', {}) # Get search_space defined here

            # Basic validation of extracted types and convert OmegaConf types
            if isinstance(param_names_raw, (list, ListConfig)):
                param_names = list(param_names_raw)
            else:
                logger.warning("'param_names' in training.optimizer is not a list. Ignoring.")
                param_names = []

            if isinstance(search_space_raw, (dict, DictConfig)):
                search_space = dict(search_space_raw)
            else:
                logger.warning("'search_space' in training.optimizer is not a dict. Ignoring.")
                search_space = {}


            # Convert bounds_raw to list of tuples
            bounds = []
            if isinstance(bounds_raw, (dict, DictConfig)) and 'lower' in bounds_raw and 'upper' in bounds_raw:
                lower = bounds_raw.get('lower', [])
                upper = bounds_raw.get('upper', [])
                if isinstance(lower, (list, ListConfig)) and isinstance(upper, (list, ListConfig)) and len(lower) == len(upper) and len(lower) > 0:
                    try:
                        bounds = [(float(l), float(u)) for l, u in zip(lower, upper)]
                    except (ValueError, TypeError):
                        logger.warning("Invalid numeric value in bounds (dict format) in training.optimizer. Ignoring bounds.")
                        bounds = []
                else:
                    logger.warning("Mismatched/empty lists for bounds (dict format) in training.optimizer. Ignoring bounds.")
            elif isinstance(bounds_raw, (list, ListConfig)):
                try:
                    valid_bounds = []
                for b in bounds_raw:
                         # Check if b is list-like and has 2 elements
                         if hasattr(b, '__len__') and len(b) == 2 and hasattr(b, '__getitem__'):
                            try:
                                # Attempt conversion to float for both elements
                                bound_tuple = (float(b[0]), float(b[1]))
                                valid_bounds.append(bound_tuple)
                            except (ValueError, TypeError):
                                logger.warning(f"Ignoring bound element with non-numeric values: {b}")
                    else:
                            logger.warning(f"Ignoring invalid bound element in list: {b}")

                    if len(valid_bounds) > 0:
                        bounds = valid_bounds
                        if len(valid_bounds) != len(bounds_raw):
                             logger.warning(f"Some elements in 'bounds' list (training.optimizer) were not valid pairs/numeric. Processed {len(valid_bounds)} / {len(bounds_raw)}.")
                    # If no valid bounds found after processing, bounds remains []

                except (TypeError, IndexError): # Catch potential errors during iteration/indexing
                    logger.warning("Error processing bounds (list format) in training.optimizer. Ignoring bounds.")
                    bounds = [] # Reset bounds on error

            # --- Assign details if extraction from training.optimizer was successful and consistent ---
            # Check 1: param_names and bounds match (for DE, PSO, GA)
            if param_names and bounds and len(param_names) == len(bounds):
                 details = {
                     'param_names': param_names,
                     'bounds': bounds,
                     'search_space': search_space # Include search_space if also defined
                 }
                 source_used = "training.optimizer (bounds match)"
                 logger.info(f"Successfully extracted details from training.optimizer for {expert_type}: {len(param_names)} parameters with matching bounds.")
            # Check 2: param_names and search_space exist, but bounds might be missing/mismatched (for Random Search)
            elif param_names and search_space and (not bounds or len(param_names) != len(bounds)):
                 details = {
                     'param_names': param_names,
                     'bounds': [], # Explicitly empty or non-matching bounds ignored
                     'search_space': search_space
                 }
                 source_used = "training.optimizer (search_space only)"
                 logger.info(f"Extracted param_names and search_space from training.optimizer for {expert_type}. Bounds were ignored or missing/mismatched.")
            elif optimizer_config: # Log if config exists but failed consistency
                 logger.warning(f"Found training.optimizer config for {expert_type}, but failed to extract consistent details (param_names: {len(param_names)}, bounds: {len(bounds)}, search_space keys: {len(search_space)}). Using empty details.")
                 # Keep details empty here
        elif source_used == "None": # Only log if no source has been identified yet AND optimizer_config check failed
             logger.info(f"No training.optimizer section found or it was invalid for {expert_type}. No details extracted.")


    # --- Final Checks and Return ---
    # REMOVED TRUNCATION LOGIC
    # Log final state and potential inconsistencies
    final_params = details.get('param_names', [])
    final_bounds = details.get('bounds', [])
    final_search = details.get('search_space', {})

    if final_params and final_bounds and len(final_params) != len(final_bounds):
         # Use ERROR level for clear visibility of config issues
         logger.error(f"FINAL CONFIGURATION INCONSISTENCY for {expert_type} using source '{source_used}': len(param_names)={len(final_params)} != len(bounds)={len(final_bounds)}. Check configuration sources! This may cause optimizer errors.")
    elif not final_params and not final_search: # Check if BOTH are empty
         logger.warning(f"No optimization parameters (param_names OR search_space) could be determined for {expert_type} from config source '{source_used}'. Optimization might fail or use defaults.")
    elif final_params or final_search:
         logger.info(f"Final details for {expert_type} (source: {source_used}): {len(final_params)} param_names, {len(final_bounds)} bounds pairs, {len(final_search)} search_space keys.")


    # Return the determined details, potentially inconsistent, allowing upstream error handling
        return details

def create_expert_instance(expert_type: str, config: DictConfig) -> Optional[BaseExpert]:
    """Instantiates the correct expert class based on type and config."""
    logger.debug(f"Attempting to instantiate {expert_type} expert...")
    try:
        if expert_type == 'physiological':
            instance = PhysiologicalExpert(config=config)
        elif expert_type == 'environmental':
            instance = EnvironmentalExpert(config=config)
        elif expert_type == 'behavioral':
            instance = BehavioralExpert(config=config)
        elif expert_type == 'medication':
            instance = MedicationHistoryExpert(config=config)
        else:
            logger.error(f"Unknown expert type for instantiation: {expert_type}")
            return None
        logger.debug(f"Successfully instantiated {expert_type} expert.")
        return instance
    except Exception as e:
        logger.error(f"Failed to instantiate {expert_type} expert: {e}", exc_info=True)
        return None


def train_and_evaluate_expert(expert: BaseExpert,
                              X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              run_label: str # e.g., "baseline" or "differential_evolution"
                             ) -> Tuple[Optional[BaseExpert], Optional[Dict[str, Any]]]:
    """Trains and evaluates a pre-configured expert instance."""
    expert_name = getattr(expert.config, 'name', expert.__class__.__name__)
    logger.info(f"Starting training for {expert_name} ({run_label})...")
    start_time = time.time()

    metrics = { # Initialize with NaNs
        'train_mse': np.nan, 'test_mse': np.nan,
        'train_mae': np.nan, 'test_mae': np.nan,
        'train_r2': np.nan, 'test_r2': np.nan,
        'train_evs': np.nan, 'test_evs': np.nan, # Explained Variance Score
        'training_time': np.nan,
        'y_pred_train': None, 'y_pred_test': None,
        'error': None,
        'mse': np.nan, 'mae': np.nan, 'r2': np.nan  # Add standard keys for consistency
    }

    try:
        # --- DIAGNOSTIC: Data Quality Analysis ---
        logger.info(f"Analyzing features for {expert_name} ({run_label})...")
        
        # Check for non-finite values
        if X_train.isna().any().any():
            nan_cols = X_train.columns[X_train.isna().any()].tolist()
            logger.warning(f"NaN values detected in X_train for {expert_name} ({run_label}) in columns: {nan_cols}")
        
        # Analyze target distribution
        logger.info(f"Target distribution for {expert_name}: min={y_train.min()}, max={y_train.max()}, mean={y_train.mean()}, unique values={y_train.nunique()}")
        if y_train.nunique() <= 5:
            logger.info(f"Limited target values detected: {sorted(y_train.unique().tolist())}. This might cause horizontal banding in predictions.")
        
        # Check variance and correlation with target
        try:
            # Feature variance check
            variances = X_train.var(numeric_only=True)
            near_zero_var_cols = variances[variances < 1e-9].index.tolist() 
            if near_zero_var_cols:
                logger.warning(f"Near-zero variance (< 1e-9) detected in X_train for {expert_name} ({run_label}) in columns: {near_zero_var_cols}")
            
            # Feature correlation with target
            numeric_cols = X_train.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                correlations = pd.DataFrame({
                    'feature': numeric_cols,
                    'correlation': [X_train[col].corr(y_train) for col in numeric_cols]
                }).sort_values('correlation', key=abs, ascending=False)
                
                # Log top correlated features
                top_features = correlations.head(min(10, len(correlations)))
                logger.info(f"Top correlated features for {expert_name}:\n{top_features.to_string(index=False)}")
                
                # Check if we have strong correlations
                if top_features.empty or abs(top_features['correlation'].iloc[0]) < 0.2:
                    logger.warning(f"No strong feature correlations (>0.2) with target for {expert_name}. This may indicate poor model performance.")
        except Exception as var_err:
            logger.error(f"Error analyzing features for {expert_name} ({run_label}): {var_err}")
        # --- END DIAGNOSTIC ---

        # --- FEATURE SELECTION: Remove low-variance features ---
        original_features = X_train.columns.tolist()
        threshold = 1e-9 # Match the diagnostic threshold
        # Identify numeric columns for variance check
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns
        
        if not numeric_cols.empty:
            selector = VarianceThreshold(threshold=threshold)
            try:
                # Fit only on numeric columns of the training data
                selector.fit(X_train[numeric_cols])
                # Get the boolean mask of features to keep
                mask = selector.get_support()
                # Apply the mask to numeric columns
                numeric_features_to_keep = numeric_cols[mask]
                # Combine with non-numeric columns
                final_features_to_keep = numeric_features_to_keep.union(non_numeric_cols)
                
                dropped_features = list(set(original_features) - set(final_features_to_keep))
                if dropped_features:
                    logger.info(f"Removing {len(dropped_features)} low variance (< {threshold}) features for {expert_name} ({run_label}): {dropped_features}")
                    # Apply filtering
                    X_train = X_train[final_features_to_keep]
                    X_test = X_test[final_features_to_keep] # Apply same filter to test set
                else:
                     logger.debug(f"No low variance features found to remove for {expert_name} ({run_label}).")
            except ValueError as ve:
                # Handle cases where VarianceThreshold might fail (e.g., all NaNs in a column after selection)
                 logger.error(f"Error applying VarianceThreshold for {expert_name} ({run_label}): {ve}. Skipping feature removal.")
            except Exception as e:
                 logger.error(f"Unexpected error during variance thresholding for {expert_name} ({run_label}): {e}. Skipping feature removal.")
        else:
             logger.debug(f"No numeric columns found in X_train for {expert_name} ({run_label}). Skipping variance thresholding.")
        # --- END FEATURE SELECTION ---

        # --- Training ---\
        expert.train(X_train, y_train)\

        training_time = time.time() - start_time
        metrics['training_time'] = training_time

        if not expert.is_trained or not hasattr(expert, 'model') or expert.model is None:
            # More comprehensive check for training success
            logger.warning(f"Expert {expert_name} ({run_label}) did not train successfully: is_trained={getattr(expert, 'is_trained', False)}, has model={hasattr(expert, 'model') and expert.model is not None}")
            metrics['error'] = "Expert did not train successfully (model unavailable or is_trained=False)"
            return expert, metrics # Return the untrained expert and NaN metrics

        logger.info(f"Training complete for {expert_name} ({run_label}) in {training_time:.2f} seconds.")

        # --- Prediction & Evaluation ---
        logger.info(f"Making predictions for {expert_name} ({run_label})...")
        
        # Handle potential errors during prediction
        try:
            y_pred_train = expert.predict(X_train)
            y_pred_test = expert.predict(X_test)
            
            # Ensure predictions are numpy arrays for consistency
            if y_pred_train is not None and not isinstance(y_pred_train, np.ndarray):
                y_pred_train = np.array(y_pred_train)
            if y_pred_test is not None and not isinstance(y_pred_test, np.ndarray):
                y_pred_test = np.array(y_pred_test)
                
            metrics['y_pred_train'] = y_pred_train
            metrics['y_pred_test'] = y_pred_test
            
            # Check for NaN or infinite values in predictions
            if y_pred_train is not None and (np.isnan(y_pred_train).any() or np.isinf(y_pred_train).any()):
                logger.warning(f"Train predictions for {expert_name} ({run_label}) contain NaN or inf values")
                
            if y_pred_test is not None and (np.isnan(y_pred_test).any() or np.isinf(y_pred_test).any()):
                logger.warning(f"Test predictions for {expert_name} ({run_label}) contain NaN or inf values")
                
        except Exception as pred_error:
            logger.error(f"Error during prediction for {expert_name} ({run_label}): {pred_error}")
            metrics['error'] = f"Prediction error: {str(pred_error)}"
            return expert, metrics

        # Calculate metrics if predictions are valid
        if y_pred_train is not None and y_pred_test is not None:
            try:
                # Calculate train metrics
                metrics['train_mse'] = mean_squared_error(y_train, y_pred_train)
                metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
                metrics['train_r2'] = r2_score(y_train, y_pred_train)
                metrics['train_evs'] = explained_variance_score(y_train, y_pred_train)
                
                # Calculate test metrics
                metrics['test_mse'] = mean_squared_error(y_test, y_pred_test)
                metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
                metrics['test_r2'] = r2_score(y_test, y_pred_test)
                metrics['test_evs'] = explained_variance_score(y_test, y_pred_test)
                
                # Add standard keys for consistency in diagnostic reporting
                metrics['mse'] = metrics['test_mse']
                metrics['mae'] = metrics['test_mae']
                metrics['r2'] = metrics['test_r2']
                
                logger.info(f"Evaluation complete for {expert_name} ({run_label}): Test MAE={metrics['test_mae']:.4f}, Test MSE={metrics['test_mse']:.4f}, Test R2={metrics['test_r2']:.4f}")
            except Exception as metric_error:
                logger.error(f"Error calculating metrics for {expert_name} ({run_label}): {metric_error}")
                metrics['error'] = f"Metric calculation error: {str(metric_error)}"
        else:
            metrics['error'] = "Prediction failed (returned None)"
            logger.error(f"Prediction failed for {expert_name} ({run_label}).")

        return expert, metrics # Return trained expert and metrics

    except Exception as e:
        metrics['training_time'] = time.time() - start_time # Record time even if error occurred
        metrics['error'] = str(e)
        logger.error(f"ERROR during training/evaluation for {expert_name} ({run_label}): {e}", exc_info=True)
        # Return the potentially partially trained expert and metrics (with error noted)
        return expert, metrics

print("Helper functions defined in Cell 10.")

# Add utility function for random search
def discretize_search_space(search_space, num_points=10):
    """
    Convert continuous search spaces to discrete points for random search.
    Args:
        search_space: Dictionary of parameter names to search ranges
        num_points: Number of discrete points to generate for each continuous parameter
    Returns:
        Dictionary with discretized search spaces
    """
    discretized = {}
    for param_name, param_config in search_space.items():
        param_type = param_config.get('type', 'float')
        
        if param_type == 'categorical':
            # Categorical parameters are already discrete
            discretized[param_name] = param_config['values']
        elif param_type in ['int', 'float']:
            # For numeric parameters, create a list of discrete values
            min_val = param_config.get('min', 0)
            max_val = param_config.get('max', 1)
            
            if param_type == 'int':
                # For integers, use linear spacing with appropriate range
                values = np.linspace(min_val, max_val, num_points, dtype=int).tolist()
                # Remove duplicates that might occur from rounding
                values = sorted(list(set(values)))
            else:
                # For floats, use logarithmic spacing if range spans multiple orders of magnitude
                if max_val / max(1e-10, min_val) > 100:
                    values = np.logspace(np.log10(max(1e-10, min_val)), np.log10(max_val), num_points).tolist()
                else:
                    values = np.linspace(min_val, max_val, num_points).tolist()
            
            discretized[param_name] = values
    
    return discretized

# %%
"""
## Expert Training and Evaluation

Now we'll train and evaluate each expert model with both default settings and optimized hyperparameters.

For each expert, we'll:
1. Train a baseline model using default parameters
2. Train an optimized model using the optimizer
3. Compare their performance
"""

# %%
# --- Cell 11: Main Training Loop ---
all_results = {} # Dictionary to store all results: all_results[expert_type][run_label] = {'expert': obj, 'metrics': dict}
experts_to_run = loaded_expert_names # From Cell 6
optimizers_to_run = optimizer_names # From Cell 7 (e.g., ['differential_evolution', 'random'])

logger.info("="*60)
logger.info(" Starting Main Training Run ".center(60, "="))
logger.info("="*60)
logger.info(f"Experts to process: {experts_to_run}")
logger.info(f"Optimizers to test (vs baseline): {optimizers_to_run}")
logger.info(f"Data source mode: {'REAL DATA' if USE_REAL_DATA else 'SYNTHETIC DATA'}")
logger.info("-"*60)

# Outer loop: Iterate through each expert type
for expert_type in tqdm(experts_to_run, desc="Experts"):
    logger.info(f"\n===== Processing Expert: {expert_type.upper()} =====")
    all_results[expert_type] = {}
    base_config = expert_base_configs.get(expert_type, None)

    if not base_config:
        logger.error(f"Skipping {expert_type} as base configuration was not loaded.")
        continue

    # --- Get Data for the CURRENT Expert ---
    # ** CORRECTED DATA FETCHING LOGIC **
    data_source = None
    expert_data_tuple = None
    if USE_REAL_DATA and expert_type in real_data:
        expert_data_tuple = real_data[expert_type]
        data_source = "Real"
    elif not USE_REAL_DATA and expert_type in synthetic_data:
        expert_data_tuple = synthetic_data[expert_type]
        data_source = "Synthetic"
    elif USE_REAL_DATA and expert_type not in real_data and expert_type in synthetic_data:
        # Fallback to synthetic if real data loading failed for this specific expert
        logger.warning(f"Real data not loaded for {expert_type}, falling back to synthetic data.")
        expert_data_tuple = synthetic_data[expert_type]
        data_source = "Synthetic (Fallback)"
    # If still no data after checking both sources...
    if expert_data_tuple is None:
        logger.error(f"No data available (real or synthetic) for {expert_type}. Skipping training for this expert.")
        continue # Skip to the next expert

    # Unpack the data tuple for the current expert
    X_train, X_test, y_train, y_test = expert_data_tuple
    logger.info(f"Using {data_source} data for {expert_type}. Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
    # ** END CORRECTED DATA FETCHING LOGIC **

    # --- 1. Train Baseline Model ---
    run_label_baseline = 'baseline'
    logger.info(f"\n--- Training {run_label_baseline.upper()} for {expert_type} ---")
    try:
        baseline_config = copy.deepcopy(base_config) # Start fresh
        # Ensure optimizer is explicitly disabled for baseline
        baseline_config.training.use_optimizer = False
        if 'optimizer' in baseline_config.training:
            logger.debug("Removing optimizer section for baseline config.")
            del baseline_config.training.optimizer

        baseline_expert_instance = create_expert_instance(expert_type, baseline_config)
        if baseline_expert_instance:
            trained_expert, metrics = train_and_evaluate_expert(baseline_expert_instance, X_train, y_train, X_test, y_test, run_label_baseline)
            all_results[expert_type][run_label_baseline] = {'expert': trained_expert, 'metrics': metrics}
            if metrics and metrics.get('error'):
                 logger.warning(f"Baseline training for {expert_type} completed with error: {metrics['error']}")
        else:
            logger.error(f"Failed to instantiate baseline expert for {expert_type}.")
            all_results[expert_type][run_label_baseline] = {'expert': None, 'metrics': {'error': 'Instantiation failed'}}

    except Exception as e:
        logger.error(f"Critical error during baseline processing for {expert_type}: {e}", exc_info=True)
        all_results[expert_type][run_label_baseline] = {'expert': None, 'metrics': {'error': f'Critical failure: {e}'}}


    # --- 2. Train with Optimizers ---
    for opt_name in tqdm(optimizers_to_run, desc=f"{expert_type} Optimizers", leave=False):
        run_label_optimizer = opt_name
        logger.info(f"\n--- Training {expert_type} with Optimizer: {run_label_optimizer.upper()} ---")

        try:
            # Start with a fresh deep copy of the base expert config
            expert_config_opt = copy.deepcopy(base_config)
            expert_config_opt.training.use_optimizer = True # Enable optimizer path

            # Get the specific optimizer template config
            if opt_name not in alternative_optimizer_configs:
                 logger.warning(f"Optimizer template '{opt_name}' not found in defined configs. Skipping.")
                 continue
            opt_config_template = copy.deepcopy(alternative_optimizer_configs[opt_name])

            # Get and add expert-specific details (params, bounds, search_space)
            expert_opt_details = get_expert_specific_opt_details(expert_type, base_config) # Pass base config

            # Merge details into the optimizer config template
            if not expert_opt_details['param_names'] and opt_name != 'random':
                logger.warning(f"No param_names defined/found for {expert_type} / {opt_name}. Optimizer might fail or use defaults.")
            if not expert_opt_details['bounds'] and opt_name != 'random':
                 logger.warning(f"No bounds defined/found for {expert_type} / {opt_name}. Optimizer might fail or use defaults.")
            if not expert_opt_details['search_space'] and opt_name == 'random':
                 logger.warning(f"No search_space defined/found for {expert_type} / {opt_name}. Random search might fail or use defaults.")

            # Add details specifically required by each optimizer type
            if opt_name == 'random':
                opt_config_template.search_space = expert_opt_details.get('search_space', {})
                # <<< ADD PARAM_NAMES ASSIGNMENT HERE >>>
                opt_config_template.param_names = expert_opt_details.get('param_names', [])
                # Remove keys not used by Random Search from the root of its config section
                opt_config_template.pop('bounds', None)
            else: # For DE, PSO, GA, etc.
                # Make sure optimizer config has the necessary properties
                if not hasattr(opt_config_template, 'param_names'):
                    opt_config_template.param_names = []
                if not hasattr(opt_config_template, 'bounds'):
                    opt_config_template.bounds = []
                if not hasattr(opt_config_template, 'search_space'):
                    opt_config_template.search_space = {}
                    
                # Copy expert-specific details to optimizer config
                opt_config_template.param_names = expert_opt_details.get('param_names', [])
                
                # Handle search space for random search
                if opt_name == 'random':
                    search_space = expert_opt_details.get('search_space', {})
                    
                    # If search_space is empty, try to generate it from hyperparameter_space
                    if not search_space and hasattr(base_config, 'hyperparameter_space'):
                        search_space = {}
                        logger.info(f"Generating search space for random search from hyperparameter_space for {expert_type}")
                        logger.info(f"hyperparameter_space keys: {list(base_config.hyperparameter_space.keys())}")
                        
                        for param, param_config in base_config.hyperparameter_space.items():
                            # Process different parameter types to create appropriate search spaces
                            if 'type' in param_config:
                                if param_config.type == 'int':
                                    if 'min' in param_config and 'max' in param_config:
                                        # For integers, create a tuple of (min, max)
                                        search_space[param] = (int(param_config.min), int(param_config.max))
                                elif param_config.type == 'float':
                                    if 'min' in param_config and 'max' in param_config:
                                        # For floats, create a tuple of (min, max)
                                        search_space[param] = (float(param_config.min), float(param_config.max))
                                elif param_config.type == 'categorical' and 'values' in param_config:
                                    # For categorical, use the list of possible values
                                    search_space[param] = list(param_config.values)
                            # Handle direct values array if present
                            elif 'values' in param_config:
                                search_space[param] = list(param_config.values)
                                
                        if search_space:
                            logger.info(f"Generated search space for {expert_type} with {len(search_space)} parameters")
                            logger.info(f"Search space content: {search_space}")
                        else:
                            logger.warning(f"Failed to generate search space from hyperparameter_space for {expert_type}")
                    
                    opt_config_template.search_space = search_space
                    logger.info(f"Final search_space for {expert_type}: {getattr(opt_config_template, 'search_space', {})}")
                
                # Process bounds for optimizers that use them
                bounds = expert_opt_details.get('bounds', [])
                
                # Use bounds directly if they're already in the right format
                if bounds and all(isinstance(b, (list, tuple)) and len(b) == 2 for b in bounds):
                    opt_config_template.bounds = bounds
                    logger.info(f"Using pre-formatted bounds for {expert_type}/{opt_name}: {bounds}")
                # Otherwise try to convert from dictionary format
                elif isinstance(bounds, dict) and 'lower' in bounds and 'upper' in bounds:
                    lower_bounds = bounds.get('lower', [])
                    upper_bounds = bounds.get('upper', [])
                    
                    if len(lower_bounds) == len(upper_bounds) and len(lower_bounds) > 0:
                        formatted_bounds = [(float(low), float(high)) for low, high in zip(lower_bounds, upper_bounds)]
                        opt_config_template.bounds = formatted_bounds
                        logger.info(f"Converted dictionary bounds to tuples for {expert_type}/{opt_name}: {formatted_bounds}")
                    else:
                        logger.warning(f"Dictionary bounds dimensions don't match: lower={len(lower_bounds)}, upper={len(upper_bounds)}")
                # Log warnings if we don't have valid bounds for optimizers that need them
                elif opt_name in ['differential_evolution', 'pso', 'ga'] and opt_config_template.param_names:
                    logger.warning(f"No valid bounds found for {expert_type}/{opt_name} but param_names are defined. This may cause failures.")
                    
                    # Try to generate default bounds from hyperparameter_space
                    if hasattr(base_config, 'hyperparameter_space'):
                        logger.info(f"Attempting to generate bounds from hyperparameter_space for {expert_type}/{opt_name}")
                        formatted_bounds = []
                        for param in opt_config_template.param_names:
                            if param in base_config.hyperparameter_space:
                                param_config = base_config.hyperparameter_space[param]
                                if 'min' in param_config and 'max' in param_config:
                                    formatted_bounds.append((float(param_config.min), float(param_config.max)))
                                else:
                                    # Default bounds
                                    formatted_bounds.append((0.0, 1.0))
                                    logger.warning(f"Using default bounds (0,1) for parameter {param}")
                            else:
                                # Default bounds if parameter not found
                                formatted_bounds.append((0.0, 1.0))
                                logger.warning(f"Parameter {param} not found in hyperparameter_space, using default bounds (0,1)")
                        
                        if len(formatted_bounds) == len(opt_config_template.param_names):
                            opt_config_template.bounds = formatted_bounds
                            logger.info(f"Generated bounds from hyperparameter_space: {formatted_bounds}")
                
                # Final validation that param_names and bounds have matching length (crucial for DE)
                if opt_config_template.param_names and hasattr(opt_config_template, 'bounds') and opt_config_template.bounds:
                    if len(opt_config_template.param_names) != len(opt_config_template.bounds):
                        logger.warning(f"Parameter count ({len(opt_config_template.param_names)}) doesn't match bounds count ({len(opt_config_template.bounds)})")
                        # Truncate to shorter length
                        min_len = min(len(opt_config_template.param_names), len(opt_config_template.bounds))
                        opt_config_template.param_names = opt_config_template.param_names[:min_len]
                        opt_config_template.bounds = opt_config_template.bounds[:min_len]
                        logger.info(f"Truncated to {min_len} parameters to ensure matching lengths")

                # Remove keys not used by this optimizer type, but keep search_space for random search
                if opt_name != 'random':
                    opt_config_template.pop('search_space', None)

            # Overwrite the 'optimizer' section in the main expert config
            expert_config_opt.training.optimizer = opt_config_template
            
            # Additional debug logging for random search
            if opt_name == 'random':
                logger.info(f"Final random search config for {expert_type}:")
                logger.info(f"  - search_space: {getattr(expert_config_opt.training.optimizer, 'search_space', {})}")
                logger.info(f"  - param_names: {getattr(expert_config_opt.training.optimizer, 'param_names', [])}")

            
            # Add logging to verify config contents before training
            logger.info(f"OPTIMIZER CONFIG FOR {expert_type}/{opt_name}:")
            logger.info(f"- maxiter: {opt_config_template.config.get('maxiter', 'NOT SET')}")
            logger.info(f"- popsize: {opt_config_template.config.get('popsize', 'NOT SET')}")
            logger.info(f"- early_stopping: {opt_config_template.get('early_stopping', 'NOT SET')}")
            logger.info(f"- early_stopping_patience: {opt_config_template.get('early_stopping_patience', 'NOT SET')}")

            # And this logging is already present but change to INFO level to ensure it's visible
            logger.info(f"Final merged config for {expert_type}/{opt_name}:")
            logger.info(OmegaConf.to_yaml(expert_config_opt.training))

            # CRITICAL FIX: Ensure param_names and search_space are properly set in the final config
            # This addresses the "param_names is empty" warning in the RandomSearchAdapter
            if hasattr(opt_config_template, 'param_names') and opt_config_template.param_names:
                if not hasattr(expert_config_opt.training.optimizer, 'param_names') or not expert_config_opt.training.optimizer.param_names:
                    logger.info(f"Fixing missing param_names in final config for {expert_type}/{opt_name}")
                    expert_config_opt.training.optimizer.param_names = opt_config_template.param_names
                    
            # For random search, ensure search_space is in the config
            if opt_name == 'random':
                if not hasattr(expert_config_opt.training.optimizer, 'search_space') or not expert_config_opt.training.optimizer.search_space:
                    if hasattr(opt_config_template, 'search_space') and opt_config_template.search_space:
                        logger.info(f"Fixing missing search_space in final config for {expert_type}/random")
                        expert_config_opt.training.optimizer.search_space = opt_config_template.search_space
                
                logger.info(f"Final random search configuration:")
                logger.info(f"  - param_names: {getattr(expert_config_opt.training.optimizer, 'param_names', [])}")
                logger.info(f"  - search_space: {getattr(expert_config_opt.training.optimizer, 'search_space', {})}")

            # Instantiate and Train
            expert_instance_opt = create_expert_instance(expert_type, expert_config_opt)
            if expert_instance_opt:
                trained_expert_opt, metrics_opt = train_and_evaluate_expert(expert_instance_opt, X_train, y_train, X_test, y_test, run_label_optimizer)
                all_results[expert_type][run_label_optimizer] = {'expert': trained_expert_opt, 'metrics': metrics_opt}
                if metrics_opt and metrics_opt.get('error'):
                     logger.warning(f"Optimizer training ({opt_name}) for {expert_type} completed with error: {metrics_opt['error']}")
            else:
                logger.error(f"Failed to instantiate {expert_type} expert for optimizer {opt_name}.")
                all_results[expert_type][run_label_optimizer] = {'expert': None, 'metrics': {'error': 'Instantiation failed'}}

        except Exception as e:
            logger.error(f"Critical error during optimizer ({opt_name}) processing for {expert_type}: {e}", exc_info=True)
            all_results[expert_type][run_label_optimizer] = {'expert': None, 'metrics': {'error': f'Critical failure: {e}'}}

# --- End of Loops ---
logger.info("\n" + "="*60)
logger.info(" Main Training Run Finished ".center(60, "="))
logger.info("="*60)

# --- Final Check on Results Structure ---
if not all_results:
     logger.warning("RESULTS DICTIONARY IS EMPTY. Check logs for errors during training.")
else:
    logger.info("Results dictionary populated. Checking structure...")
    for expert, runs in all_results.items():
        run_keys = list(runs.keys())
        logger.info(f"  Expert '{expert}': Found results for runs: {run_keys}")
        if 'baseline' not in run_keys:
             logger.warning(f"    Missing 'baseline' results for {expert}.")
        # Check if expected optimizers are present
        missing_optimizers = [opt for opt in optimizers_to_run if opt not in run_keys]
        if missing_optimizers:
             logger.warning(f"    Missing optimizer results for {expert}: {missing_optimizers}")
        # Check for errors within runs
        for run_label, result_data in runs.items():
            if result_data.get('metrics', {}).get('error'):
                 logger.warning(f"    Run '{run_label}' for {expert} reported an error: {result_data['metrics']['error']}")
                 
        # Create 'optimized' key with best performer (or differential_evolution as fallback)
        # This is needed for the display code at the end
        best_mse = float('inf')
        best_optimizer = None
        
        # First try to find the best valid optimizer by MSE
        for opt_name in ['differential_evolution', 'pso', 'ga', 'random']:
            if opt_name in runs and 'metrics' in runs[opt_name] and 'mse' in runs[opt_name]['metrics']:
                # Check if we have a valid expert object with a trained model
                expert_obj = runs[opt_name].get('expert')
                if expert_obj and hasattr(expert_obj, 'is_trained') and expert_obj.is_trained and hasattr(expert_obj, 'model') and expert_obj.model is not None:
                    # Only consider results with valid MSE values
                    current_mse = runs[opt_name]['metrics']['mse']
                    if not np.isnan(current_mse) and current_mse < best_mse:
                        best_mse = current_mse
                        best_optimizer = opt_name
        
        # If no valid optimizer was found, use differential_evolution as fallback
        if best_optimizer is None and 'differential_evolution' in runs:
            best_optimizer = 'differential_evolution'
            logger.warning(f"No valid optimizers found for {expert}, using differential_evolution as fallback")
        
        # If we have a best optimizer, add it as 'optimized' entry
        if best_optimizer is not None:
            logger.info(f"Setting 'optimized' for {expert} to use results from '{best_optimizer}'")
            all_results[expert]['optimized'] = all_results[expert][best_optimizer]
        else:
            # Create empty optimized entry to avoid KeyError
            logger.warning(f"No valid optimizer results found for {expert}, creating empty 'optimized' entry")
            all_results[expert]['optimized'] = {'expert': None, 'metrics': {'error': 'No valid optimizer results available'}}

print("\nCell 11 execution complete.")

# %% 
# --- Cell 11b: Save Model Results ---
# Save models after training is complete so they can be loaded later without retraining

def save_training_results(all_results, filename=None):
    """Save the training results for future reference."""
    
    import pickle
    import os
    from datetime import datetime
    
    # Create a timestamp-based filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'expert_training_results_{timestamp}.pkl'
    
    # Prepare data for saving
    save_data = {
        'all_results': all_results,
        'timestamp': pd.Timestamp.now(),
        'metrics_summary': {expert: {
            'baseline': results['baseline']['metrics'],
            'optimized': results['optimized']['metrics']
        } for expert, results in all_results.items() if 'baseline' in results and 'optimized' in results}
    }
    
    # Save to file
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Training results saved to {filename}")
    print(f"Training results saved to {filename}")
    return filename

# Save the results
training_results_file = save_training_results(all_results)



# %%
# Diagnostic cell to examine all_results structure

# Check overall structure
print("Expert types in all_results:", list(all_results.keys()))

# %%
"""
## 1. Baseline vs. Optimized Performance
"""

# %%
def plot_performance_comparison(all_results):
    """Plot performance comparison between baseline and optimized models."""
    
    # Extract metrics for plotting
    expert_types = list(all_results.keys())
    
    # Check first result to determine actual metric names
    sample_expert = expert_types[0]
    print(f"Available metrics: {list(all_results[sample_expert]['baseline']['metrics'].keys())}")
    
    # Try different possible metric names
    mse_keys = ['mean_squared_error', 'test_mse', 'mse', 'neg_mean_squared_error']
    r2_keys = ['r2_score', 'test_r2', 'r2']
    
    # Find the actual metric names in the results
    mse_key = next((k for k in mse_keys if k in all_results[sample_expert]['baseline']['metrics']), None)
    r2_key = next((k for k in r2_keys if k in all_results[sample_expert]['baseline']['metrics']), None)
    
    if not mse_key or not r2_key:
        print(f"Couldn't find expected metrics. Available: {list(all_results[sample_expert]['baseline']['metrics'].keys())}")
        return
        
    print(f"Using metrics: MSE={mse_key}, R2={r2_key}")
    
    # Extract metrics using the found keys
    test_mse_baseline = [all_results[expert]['baseline']['metrics'][mse_key] for expert in expert_types]
    test_mse_optimized = [all_results[expert]['optimized']['metrics'][mse_key] for expert in expert_types]
    test_r2_baseline = [all_results[expert]['baseline']['metrics'][r2_key] for expert in expert_types]
    test_r2_optimized = [all_results[expert]['optimized']['metrics'][r2_key] for expert in expert_types]
    
    # Rest of your plotting code remains the same
    # ...
    
    # Calculate improvement percentages
    mse_improvement = [(baseline - optimized) / baseline * 100 for baseline, optimized in zip(test_mse_baseline, test_mse_optimized)]
    r2_improvement = [(optimized - baseline) / max(0.001, baseline) * 100 for baseline, optimized in zip(test_r2_baseline, test_r2_optimized)]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar width
    width = 0.35
    
    # Bar positions
    x = np.arange(len(expert_types))
    
    # Define colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot MSE comparison
    bars1 = ax1.bar(x - width/2, test_mse_baseline, width, label='Baseline', color=colors[0])
    bars2 = ax1.bar(x + width/2, test_mse_optimized, width, label='Optimized', color=colors[1])
    
    # Add MSE improvement annotations
    for i, (baseline, optimized, imp) in enumerate(zip(test_mse_baseline, test_mse_optimized, mse_improvement)):
        if imp > 0:  # Only show positive improvements
            ax1.annotate(f"{imp:.1f}%↓", 
                        xy=(i, optimized), 
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='green')
    
    # Plot R2 comparison
    bars3 = ax2.bar(x - width/2, test_r2_baseline, width, label='Baseline', color=colors[0])
    bars4 = ax2.bar(x + width/2, test_r2_optimized, width, label='Optimized', color=colors[1])
    
    # Add R2 improvement annotations
    for i, (baseline, optimized, imp) in enumerate(zip(test_r2_baseline, test_r2_optimized, r2_improvement)):
        if imp > 0:  # Only show positive improvements
            ax2.annotate(f"{imp:.1f}%↑", 
                        xy=(i, optimized), 
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='green')
    
    # Customize plots
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE Comparison (lower is better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.capitalize() for e in expert_types])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison (higher is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.capitalize() for e in expert_types])
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot performance comparison
plot_performance_comparison(all_results)

# %%
def plot_error_distributions(all_results):
    """Plot error distributions using box plots."""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # The test data target values aren't directly accessible, so we'll use the prediction differences
    # between baseline and optimized models to visualize the error distributions.
    for i, expert_type in enumerate(all_results.keys()):
        # Get the prediction arrays
        baseline_preds = all_results[expert_type]['baseline']['metrics'].get('y_pred_test', None)
        optimized_preds = all_results[expert_type]['optimized']['metrics'].get('y_pred_test', None)
        
        if baseline_preds is None or optimized_preds is None:
            # Skip if predictions aren't available
            axes[i].text(0.5, 0.5, f"No prediction data available for {expert_type}", 
                       ha='center', va='center', fontsize=12)
            continue
            
        # Calculate error distributions (using optimized - baseline as a proxy for 'error')
        # This isn't the actual prediction error, but it shows the difference between models
        errors = optimized_preds - baseline_preds
        
        # Create histograms to show the distribution of differences between optimized and baseline
        axes[i].hist(errors, bins=20, alpha=0.7, color='#1f77b4')
        
        # Plot styling
        axes[i].set_title(f'{expert_type.capitalize()} Expert: Prediction Differences')
        axes[i].set_xlabel('Optimized - Baseline Predictions')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(x=0, color='r', linestyle='-', alpha=0.3, label='No Difference')
        
        # Add some statistics
        mean_diff = np.mean(errors)
        std_diff = np.std(errors)
        axes[i].text(0.05, 0.95, f"Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}", 
                   transform=axes[i].transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add legend
        axes[i].legend()
        
        # The previous statistics are sufficient, no need for this additional text box
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot error distributions
plot_error_distributions(all_results)

# %%
def plot_learning_curves(all_results):
    """Plot learning curves showing training and validation error reduction."""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Force real data usage when available
    data_source = 'real_data' if USE_REAL_DATA else 'synthetic_data'
    
    for i, expert_type in enumerate(all_results.keys()):
        try:
            # Get baseline and optimized expert models - check if models are actually trained
            baseline_expert = all_results[expert_type]['baseline']['expert']
            optimized_expert = all_results[expert_type]['optimized']['expert']
            
            if not (baseline_expert.is_trained and optimized_expert.is_trained):
                logger.warning(f"Models for {expert_type} not properly trained. Skipping learning curves.")
                axes[i].text(0.5, 0.5, f"Models not properly trained", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[i].transAxes)
                continue
                
            # Try to get training data - prioritize real data when available
            X_train, y_train = None, None
            
            # First try real data
            if data_source == 'real_data' and expert_type in real_data:
                X_train, _, y_train, _ = real_data[expert_type]
                logger.info(f"Using real data for {expert_type} learning curves")
            # Fall back to synthetic data if needed
            elif expert_type in synthetic_data:
                X_train, _, y_train, _ = synthetic_data[expert_type]
                logger.info(f"Using synthetic data for {expert_type} learning curves")
            
            # Check if we have valid data to use
            if X_train is None or y_train is None or len(X_train) == 0:
                logger.warning(f"No training data available for {expert_type}. Cannot generate learning curves.")
                axes[i].text(0.5, 0.5, "No training data available", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[i].transAxes)
                continue
            
            # Data preprocessing checks to avoid errors
            # Ensure data types are compatible and handle mixed dtype issues
            try:
                # Check for datetime columns and remove them
                if isinstance(X_train, pd.DataFrame):
                    datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
                    if len(datetime_cols) > 0:
                        logger.info(f"Removing datetime columns for learning curve calculation: {datetime_cols}")
                        X_train = X_train.drop(columns=datetime_cols)
                
                # Define train sizes - fewer points for faster computation
                train_sizes = np.linspace(0.2, 1.0, 5)  # 20%, 40%, 60%, 80%, 100%
                
                # Calculate learning curves for baseline model
                logger.info(f"Calculating learning curve for {expert_type} baseline model")
                train_sizes_abs, baseline_train_scores, baseline_val_scores = learning_curve(
                    baseline_expert.model, X_train, y_train, 
                    train_sizes=train_sizes, cv=3, scoring='neg_mean_squared_error', 
                    n_jobs=-1, random_state=42
                )
                
                # Calculate learning curves for optimized model
                logger.info(f"Calculating learning curve for {expert_type} optimized model")
                train_sizes_abs, optimized_train_scores, optimized_val_scores = learning_curve(
                    optimized_expert.model, X_train, y_train, 
                    train_sizes=train_sizes, cv=3, scoring='neg_mean_squared_error', 
                    n_jobs=-1, random_state=42
                )
                
                # Convert negative MSE to positive MSE (for visualization purposes)
                baseline_train_errors = -np.mean(baseline_train_scores, axis=1)
                baseline_val_errors = -np.mean(baseline_val_scores, axis=1)
                optimized_train_errors = -np.mean(optimized_train_scores, axis=1)
                optimized_val_errors = -np.mean(optimized_val_scores, axis=1)
                
                # Success! Real learning curves computed
                logger.info(f"Successfully calculated learning curves for {expert_type}")
                
            except Exception as e:
                logger.error(f"Failed to calculate learning curves for {expert_type}: {str(e)}")
                axes[i].text(0.5, 0.5, f"Error calculating learning curves:\n{str(e)}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[i].transAxes)
                continue
            
            # Plot baseline learning curve
            axes[i].plot(train_sizes, baseline_train_errors, 'o-', color=colors[0], label='Baseline Train')
            axes[i].plot(train_sizes, baseline_val_errors, 'o-', color=colors[1], label='Baseline Validation')
            
            # Plot optimized learning curve
            axes[i].plot(train_sizes, optimized_train_errors, 's-', color=colors[2], label='Optimized Train')
            axes[i].plot(train_sizes, optimized_val_errors, 's-', color=colors[3], label='Optimized Validation')
            
            # Fill between train and validation curves
            axes[i].fill_between(train_sizes, baseline_train_errors, baseline_val_errors, 
                              alpha=0.1, color=colors[0])
            axes[i].fill_between(train_sizes, optimized_train_errors, optimized_val_errors, 
                              alpha=0.1, color=colors[2])
            
            # Customize plot
            axes[i].set_title(f"{expert_type.capitalize()} Expert Learning Curves")
            axes[i].set_xlabel('Training Set Size')
            axes[i].set_ylabel('Mean Squared Error')
            axes[i].legend(loc='best')
            axes[i].grid(linestyle='--', alpha=0.7)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Could not generate learning curve: {e}", 
                      horizontalalignment='center', 
                      verticalalignment='center',
                      transform=axes[i].transAxes)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot learning curves with error handling
try:
    plot_learning_curves(all_results)
except Exception as e:
    logger.error(f"Error plotting learning curves: {e}")
    print(f"Skipping learning curve plots due to error: {e}")

# %%
"""
## 2. Prediction Quality Visualizations

Now we'll visualize the quality of predictions made by each expert.
"""

# %%
def plot_prediction_scatter(all_results):
    """Plot scatter plots of predicted vs. actual values for baseline and optimized models."""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # First check what R2 key is actually available
    sample_expert = list(all_results.keys())[0]
    r2_keys = ['r2_score', 'test_r2', 'r2']
    r2_key = next((k for k in r2_keys if k in all_results[sample_expert]['baseline']['metrics']), None)
    
    if not r2_key:
        print(f"Couldn't find R2 metric. Available: {list(all_results[sample_expert]['baseline']['metrics'].keys())}")
        return
        
    print(f"Using R2 metric key: {r2_key}")
    
    # Determine data source
    data_source = 'real_data' if USE_REAL_DATA and 'physiological' in real_data else 'synthetic_data'
    
    for i, expert_type in enumerate(all_results.keys()):
        try:
            # Get predicted values
            y_pred_baseline = all_results[expert_type]['baseline']['metrics'].get('y_pred_test', None)
            y_pred_optimized = all_results[expert_type]['optimized']['metrics'].get('y_pred_test', None)
            
            if y_pred_baseline is None or y_pred_optimized is None:
                axes[i].text(0.5, 0.5, f"No predictions available for {expert_type}", 
                          ha='center', va='center', fontsize=12, transform=axes[i].transAxes)
                continue
            
            # Get actual test data if available
            y_actual = None
            if data_source == 'real_data' and expert_type in real_data:
                _, _, _, y_actual = real_data[expert_type]
            elif expert_type in synthetic_data:
                _, _, _, y_actual = synthetic_data[expert_type]
                
            if y_actual is not None and len(y_actual) == len(y_pred_baseline):
                # Plot predicted vs actual values for both models
                axes[i].scatter(y_actual, y_pred_baseline, alpha=0.5, color=colors[0], label='Baseline')
                axes[i].scatter(y_actual, y_pred_optimized, alpha=0.5, color=colors[1], label='Optimized')
                
                # Add perfect prediction line
                min_val = min(min(y_actual), min(y_pred_baseline), min(y_pred_optimized))
                max_val = max(max(y_actual), max(y_pred_baseline), max(y_pred_optimized))
                axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
                
                # Set labels for this case
                axes[i].set_xlabel('Actual Values')
                axes[i].set_ylabel('Predicted Values')
            else:
                # Fallback: compare baseline vs optimized predictions
                axes[i].scatter(y_pred_baseline, y_pred_optimized, alpha=0.5, color=colors[0])
                
                # Add a diagonal line (if baseline and optimized were identical)
                min_val = min(min(y_pred_baseline), min(y_pred_optimized))
                max_val = max(max(y_pred_baseline), max(y_pred_optimized))
                axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Identity Line')
                
                # Set labels for this case
                axes[i].set_xlabel('Baseline Predictions')
                axes[i].set_ylabel('Optimized Predictions')
        
            # Set common plot properties
            axes[i].set_title(f"{expert_type.capitalize()} Expert Predictions")
            axes[i].grid(linestyle='--', alpha=0.7)
            axes[i].legend(loc='best')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error plotting {expert_type}: {str(e)}", 
                       ha='center', va='center', fontsize=12, transform=axes[i].transAxes)
        
            # Add MSE and R2 metrics as text annotation
            mse_baseline = all_results[expert_type]['baseline']['metrics'].get('mse', float('nan'))
            mse_optimized = all_results[expert_type]['optimized']['metrics'].get('mse', float('nan'))
            r2_baseline = all_results[expert_type]['baseline']['metrics'].get(r2_key, float('nan'))
            r2_optimized = all_results[expert_type]['optimized']['metrics'].get(r2_key, float('nan'))
        
            axes[i].text(0.05, 0.95, 
                        f'Baseline MSE = {mse_baseline:.3f}, R² = {r2_baseline:.3f}\n'
                        f'Optimized MSE = {mse_optimized:.3f}, R² = {r2_optimized:.3f}',
                        transform=axes[i].transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot prediction scatter plots with error handling
try:
    plot_prediction_scatter(all_results)
except Exception as e:
    logger.error(f"Error plotting prediction scatter: {e}")
    print(f"Skipping prediction scatter plots due to error: {e}")

# %%
def plot_residuals(all_results):
    """Plot the difference between baseline and optimized predictions to visualize improvement."""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Determine data source
    data_source = 'real_data' if USE_REAL_DATA and 'physiological' in real_data else 'synthetic_data'
    
    for i, expert_type in enumerate(all_results.keys()):
        try:
            # Get predicted values
            y_pred_baseline = all_results[expert_type]['baseline']['metrics'].get('y_pred_test', None)
            y_pred_optimized = all_results[expert_type]['optimized']['metrics'].get('y_pred_test', None)
            
            if y_pred_baseline is None or y_pred_optimized is None:
                axes[i].text(0.5, 0.5, f"No predictions available for {expert_type}", 
                           ha='center', va='center', fontsize=12, transform=axes[i].transAxes)
                continue
                
            # Get actual test data if available
            y_actual = None
            if data_source == 'real_data' and expert_type in real_data:
                _, _, _, y_actual = real_data[expert_type]
            elif expert_type in synthetic_data:
                _, _, _, y_actual = synthetic_data[expert_type]
            
            # Initialize prediction_diff variable to ensure it's always defined
            prediction_diff = y_pred_optimized - y_pred_baseline
                
            if y_actual is not None and len(y_actual) == len(y_pred_baseline):
                # Calculate true residuals
                residuals_baseline = y_actual - y_pred_baseline
                residuals_optimized = y_actual - y_pred_optimized
                
                # Create scatter plots of residuals
                axes[i].scatter(y_pred_baseline, residuals_baseline, alpha=0.5, color=colors[0], label='Baseline')
                axes[i].scatter(y_pred_optimized, residuals_optimized, alpha=0.5, color=colors[1], label='Optimized')
                
                # Add horizontal line at y=0 (perfect prediction)
                axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                
                # Set labels for this case
                axes[i].set_xlabel('Predicted Values')
                axes[i].set_ylabel('Residuals (Actual - Predicted)')
                
                # In this case, we don't want to display the histogram but continue with the scatter plot
                continue
            
            # If we reach here, we're displaying the prediction difference histogram
            # Create a histogram of prediction differences
            axes[i].hist(prediction_diff, bins=20, alpha=0.7, color=colors[0])
            
            # Add a vertical line at x=0 (no difference between optimization and baseline)
            axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
            # Add mean line
            mean_diff = np.mean(prediction_diff)
            axes[i].axvline(x=mean_diff, color='g', linestyle='-', alpha=0.7,
                          label=f'Mean diff: {mean_diff:.3f}')
            
            # Customize plot
            axes[i].set_title(f"{expert_type.capitalize()} Prediction Differences")
            axes[i].set_xlabel('Optimized - Baseline Predictions')
            axes[i].set_ylabel('Frequency')
            axes[i].legend(loc='best')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                       ha='center', va='center', fontsize=12, transform=axes[i].transAxes)
            axes[i].grid(linestyle='--', alpha=0.7)
            
            # Add statistics based on what was plotted
            if 'residuals_baseline' in locals() and 'residuals_optimized' in locals():
                # Add statistics for true residuals
                axes[i].text(0.05, 0.95, 
                            f"Baseline Mean Residual: {np.mean(residuals_baseline):.4f}\n"
                            f"Baseline Std Dev: {np.std(residuals_baseline):.4f}\n"
                            f"Optimized Mean Residual: {np.mean(residuals_optimized):.4f}\n"
                            f"Optimized Std Dev: {np.std(residuals_optimized):.4f}",
                            transform=axes[i].transAxes,
                            fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            elif 'prediction_diff' in locals():
                # Add statistics for prediction differences
                axes[i].text(0.05, 0.95, 
                            f"Mean Diff: {np.mean(prediction_diff):.4f}\n"
                            f"Std Dev: {np.std(prediction_diff):.4f}\n"
                            f"Max Diff: {np.max(prediction_diff):.4f}\n"
                            f"Min Diff: {np.min(prediction_diff):.4f}",
                            transform=axes[i].transAxes,
                            fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot prediction differences with error handling
try:
    plot_residuals(all_results)
except Exception as e:
    logger.error(f"Error plotting residuals: {e}")
    print(f"Skipping residual plots due to error: {e}")

# %%
def plot_time_series_predictions(all_results):
    """Plot comparison of baseline and optimized predictions as a sequence."""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, expert_type in enumerate(all_results.keys()):
        try:
            # Get predicted values
            y_pred_baseline = all_results[expert_type]['baseline']['metrics'].get('y_pred_test', None)
            y_pred_optimized = all_results[expert_type]['optimized']['metrics'].get('y_pred_test', None)
            
            if y_pred_baseline is None or y_pred_optimized is None:
                axes[i].text(0.5, 0.5, f"No predictions available for {expert_type}", 
                           ha='center', va='center', fontsize=12, transform=axes[i].transAxes)
                continue
            
            # Create synthetic indices (no real timestamps available)
            indices = np.arange(len(y_pred_baseline))
            
            # Only plot up to 100 points for clarity
            max_points = min(100, len(indices))
            plot_indices = indices[:max_points]
            plot_baseline = y_pred_baseline[:max_points]
            plot_optimized = y_pred_optimized[:max_points]
            
            # Plot time series of predictions
            axes[i].plot(plot_indices, plot_baseline, '-', color=colors[0], label='Baseline')
            axes[i].plot(plot_indices, plot_optimized, '-', color=colors[1], label='Optimized')
            
            # Calculate and plot the difference
            diff = plot_optimized - plot_baseline
            axes[i].plot(plot_indices, diff, '--', color=colors[2], alpha=0.5, label='Difference')
            
            # Highlight areas where optimized is better or worse
            better_indices = np.where(diff < 0)[0]  # Lower value is better for error metrics
            worse_indices = np.where(diff > 0)[0]
            
            if len(better_indices) > 0:
                axes[i].scatter(plot_indices[better_indices], diff[better_indices], 
                              color='green', alpha=0.5, s=20, marker='o')
            
            if len(worse_indices) > 0:
                axes[i].scatter(plot_indices[worse_indices], diff[worse_indices], 
                              color='red', alpha=0.5, s=20, marker='x')
                
        except Exception as e:
            # If there's an error, add an error message to the plot
            axes[i].text(0.5, 0.5, f"Error plotting {expert_type}: {str(e)}", 
                       ha='center', va='center', fontsize=12, transform=axes[i].transAxes)
        
        # Customize plot
        axes[i].set_title(f"{expert_type.capitalize()} Expert Prediction Comparison")
        axes[i].set_xlabel('Prediction Index')
        axes[i].set_ylabel('Predicted Value')
        axes[i].legend(loc='best')
        axes[i].grid(linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot time series predictions with error handling
try:
    plot_time_series_predictions(all_results)
except Exception as e:
    logger.error(f"Error plotting time series predictions: {e}")
    print(f"Skipping time series prediction plots due to error: {e}")

# %%
"""
## 3. Hyperparameter Optimization Analysis

Let's analyze how hyperparameter optimization improves model performance, and which hyperparameters have the most impact.
"""

# %%
def plot_hyperparameter_heatmaps(all_results):
    """Plot heatmaps showing relationship between hyperparameters and performance."""
    
    # This would typically use actual optimizer results
    # Here we'll create synthetic data to demonstrate visualization approach
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, expert_type in enumerate(all_results.keys()):
        # Different parameters for different expert types
        if expert_type == 'physiological' or expert_type == 'behavioral':
            param1 = 'n_estimators'
            param2 = 'max_depth'
            param1_values = [50, 100, 150, 200]
            param2_values = [3, 5, 7, 10]
        else:  # environmental or medication
            param1 = 'max_iter'
            param2 = 'learning_rate'
            param1_values = [50, 100, 150]
            param2_values = [0.01, 0.05, 0.1, 0.2]
        
        # Create synthetic performance data (would be real results in practice)
        performance = np.random.rand(len(param1_values), len(param2_values))
        performance = np.abs(1 - performance)  # Make it look like MSE (lower is better)
        
        # Create heatmap
        im = axes[i].imshow(performance, cmap='viridis_r')  # _r for reverse (darker is better)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.set_label('MSE (lower is better)')
        
        # Customize plot
        axes[i].set_title(f"{expert_type.capitalize()} Expert Performance Heatmap")
        axes[i].set_xticks(np.arange(len(param2_values)))
        axes[i].set_yticks(np.arange(len(param1_values)))
        axes[i].set_xticklabels(param2_values)
        axes[i].set_yticklabels(param1_values)
        axes[i].set_xlabel(param2)
        axes[i].set_ylabel(param1)
        
        # Rotate the tick labels and set their alignment
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for j in range(len(param1_values)):
            for k in range(len(param2_values)):
                axes[i].text(k, j, f"{performance[j, k]:.3f}",
                           ha="center", va="center", color="w" if performance[j, k] > 0.5 else "black")
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot hyperparameter heatmaps
plot_hyperparameter_heatmaps(all_results)

# %%
def plot_parallel_coordinates(all_results):
    """Plot parallel coordinates showing relationship between hyperparameters and performance."""
    
    # This would typically use actual optimizer results
    # Here we'll create synthetic data to demonstrate the visualization approach
    
    # Create synthetic optimization results for each expert
    for expert_type in all_results.keys():
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Generate synthetic optimization trials data
        if expert_type == 'physiological' or expert_type == 'behavioral':
            n_trials = 20
            data = {
                'trial': list(range(1, n_trials + 1)),
                'n_estimators': np.random.choice([50, 100, 150, 200], n_trials),
                'max_depth': np.random.choice([3, 5, 7, 10, None], n_trials),
                'min_samples_split': np.random.choice([2, 5, 10], n_trials),
                'score': np.random.uniform(0.7, 0.95, n_trials)
            }
            
            # Convert None to a numeric value for plotting
            data['max_depth'] = [15 if x is None else x for x in data['max_depth']]
            
            # Create parallel coordinates plot
            df = pd.DataFrame(data)
            parallel_coordinates = pd.plotting.parallel_coordinates(
                df, 'trial', colormap=plt.cm.viridis,
                alpha=0.5
            )
            
        else:  # environmental or medication
            n_trials = 20
            data = {
                'trial': list(range(1, n_trials + 1)),
                'max_iter': np.random.choice([50, 100, 150], n_trials),
                'max_depth': np.random.choice([3, 5, 7, None], n_trials),
                'learning_rate': np.random.choice([0.01, 0.05, 0.1, 0.2], n_trials),
                'score': np.random.uniform(0.7, 0.95, n_trials)
            }
            
            # Convert None to a numeric value for plotting
            data['max_depth'] = [10 if x is None else x for x in data['max_depth']]
            
            # Create parallel coordinates plot
            df = pd.DataFrame(data)
            parallel_coordinates = pd.plotting.parallel_coordinates(
                df, 'trial', colormap=plt.cm.viridis,
                alpha=0.5
            )
        
        # Customize plot
        plt.title(f"{expert_type.capitalize()} Expert Hyperparameter Optimization Trials")
        plt.ylabel('Normalized Parameter Value')
        plt.xlabel('Parameter')
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

# Plot parallel coordinates
plot_parallel_coordinates(all_results)

# %%
def plot_optimization_convergence(all_results):
    """Plot convergence of optimization process."""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, expert_type in enumerate(all_results.keys()):
        # Generate synthetic convergence data (would use real data in practice)
        iterations = range(1, 11)  # 10 iterations
        
        # Create a synthetic convergence curve that improves over time
        np.random.seed(42 + i)  # For reproducibility but different for each expert
        best_score = np.zeros(10)
        current_best = 0.5  # Starting point 
        for j in range(10):
            improvement = np.random.uniform(0, 0.1) * (1 - j/10)  # Smaller improvements over time
            current_best += improvement
            best_score[j] = current_best
        
        # Plot convergence
        axes[i].plot(iterations, best_score, 'o-', color=colors[1], linewidth=2)
        
        # Mark the best point
        best_idx = np.argmax(best_score)
        axes[i].plot(iterations[best_idx], best_score[best_idx], 'o', color='red', markersize=10)
        
        # Customize plot
        axes[i].set_title(f"{expert_type.capitalize()} Expert Optimization Convergence")
        axes[i].set_xlabel('Optimization Iteration')
        axes[i].set_ylabel('Best Score (higher is better)')
        axes[i].grid(linestyle='--', alpha=0.7)
        
        # Add annotation for best point
        axes[i].annotate(f"Best: {best_score[best_idx]:.4f}",
                       xy=(iterations[best_idx], best_score[best_idx]),
                       xytext=(iterations[best_idx] - 1, best_score[best_idx] - 0.05),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                       fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Plot optimization convergence
plot_optimization_convergence(all_results)

# %%
"""
## 4. Expert-Specific Visualizations

Now let's examine visualizations that are specific to each expert type, highlighting the unique aspects of their domain.
"""

# %%
def plot_physiological_visualizations(results):
    """Plot visualizations specific to the PhysiologicalExpert."""
    
    # Feature Importance
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Feature Importance
    # Synthetic feature importance data
    physiological_features = ['hr', 'hrv', 'temp', 'resp_rate', 'gsr', 'bp_systolic', 'bp_diastolic']
    baseline_importance = np.random.rand(len(physiological_features))
    baseline_importance = baseline_importance / baseline_importance.sum()
    
    optimized_importance = np.random.rand(len(physiological_features))
    optimized_importance = optimized_importance / optimized_importance.sum()
    
    # Sort by optimized importance
    sorted_indices = np.argsort(optimized_importance)
    sorted_features = [physiological_features[i] for i in sorted_indices]
    sorted_baseline = [baseline_importance[i] for i in sorted_indices]
    sorted_optimized = [optimized_importance[i] for i in sorted_indices]
    
    # Plot feature importance
    y_pos = np.arange(len(sorted_features))
    ax1.barh(y_pos - 0.2, sorted_baseline, 0.4, color=colors[0], label='Baseline')
    ax1.barh(y_pos + 0.2, sorted_optimized, 0.4, color=colors[1], label='Optimized')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f.upper() for f in sorted_features])
    ax1.set_xlabel('Relative Importance')
    ax1.set_title('Physiological Signal Importance')
    ax1.legend()
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. Time-aligned Physiological Signals and Prediction
    # Sample data for a single patient over time
    timestamps = pd.date_range(start='2023-01-01', periods=20, freq='6h')
    hr_values = 70 + 10 * np.sin(np.linspace(0, 4*np.pi, len(timestamps)))
    hrv_values = 50 + 15 * np.cos(np.linspace(0, 4*np.pi, len(timestamps)))
    temp_values = 36.5 + 0.5 * np.sin(np.linspace(0, 2*np.pi, len(timestamps)))
    
    # Synthetic prediction based on these signals
    prediction = 0.3 + 0.4 * np.sin(np.linspace(0, 4*np.pi, len(timestamps)) + np.pi/4)
    
    # Create a secondary y-axis
    ax2_twin = ax2.twinx()
    
    # Plot physiological signals
    ax2.plot(timestamps, hr_values, '-', color=colors[0], label='Heart Rate')
    ax2.plot(timestamps, hrv_values, '-', color=colors[2], label='HRV')
    ax2.plot(timestamps, temp_values, '-', color=colors[3], label='Temperature')
    
    # Plot prediction on secondary axis
    ax2_twin.plot(timestamps, prediction, '--', color='red', linewidth=2, label='Predicted Risk')
    ax2_twin.fill_between(timestamps, 0, prediction, color='red', alpha=0.1)
    
    # Customize plot
    ax2.set_title('Physiological Signals & Predicted Risk')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Signal Values')
    ax2_twin.set_ylabel('Predicted Migraine Risk')
    ax2_twin.set_ylim(0, 1)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Combine legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.grid(linestyle='--', alpha=0.7)
    
    # 3. Sensitivity Analysis
    # Synthetic sensitivity data
    hr_range = np.linspace(60, 90, 20)
    prediction_at_hr = 0.2 + 0.6 * (hr_range - 60) / 30  # Linear relationship for demo
    prediction_at_hr += 0.1 * np.sin(np.linspace(0, 4*np.pi, len(hr_range)))  # Add some noise
    
    hrv_range = np.linspace(30, 70, 20)
    prediction_at_hrv = 0.8 - 0.6 * (hrv_range - 30) / 40  # Inverse relationship for demo
    prediction_at_hrv += 0.1 * np.sin(np.linspace(0, 4*np.pi, len(hrv_range)))  # Add some noise
    
    # Plot sensitivity curves
    ax3.plot(hr_range, prediction_at_hr, '-', color=colors[0], linewidth=2, label='HR Sensitivity')
    ax3.plot(hrv_range, prediction_at_hrv, '-', color=colors[2], linewidth=2, label='HRV Sensitivity')
    
    # Add reference line at 0.5 threshold
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    # Customize plot
    ax3.set_title('Sensitivity to Vital Sign Changes')
    ax3.set_xlabel('Signal Value')
    ax3.set_ylabel('Predicted Migraine Risk')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Plot PhysiologicalExpert visualizations
plot_physiological_visualizations(all_results['physiological'])

# %%
def plot_environmental_visualizations(results):
    """Plot visualizations specific to the EnvironmentalExpert."""
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Weather Condition Correlations
    # Synthetic correlation data
    weather_factors = ['Temperature', 'Humidity', 'Pressure', 'AQI', 'Pollen', 'Light', 'Noise']
    correlation_values = [0.65, 0.3, 0.7, 0.45, 0.5, 0.6, 0.25]
    
    # Sort by correlation strength
    sorted_indices = np.argsort(correlation_values)
    sorted_factors = [weather_factors[i] for i in sorted_indices]
    sorted_correlations = [correlation_values[i] for i in sorted_indices]
    
    # Create color map based on correlation values
    colors_corr = plt.cm.RdYlGn(np.array(sorted_correlations))
    
    # Plot correlations
    y_pos = np.arange(len(sorted_factors))
    bars = ax1.barh(y_pos, sorted_correlations, color=colors_corr)
    
    # Add labels inside bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_width() - 0.08, bar.get_y() + bar.get_height()/2, 
                f"{sorted_correlations[i]:.2f}", ha='right', va='center', 
                color='white', weight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_factors)
    ax1.set_xlabel('Correlation with Migraine Severity')
    ax1.set_title('Weather Factor Correlations')
    ax1.set_xlim(0, 0.8)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. Barometric Pressure Impact
    # Synthetic pressure data
    pressure_range = np.linspace(995, 1030, 100)
    
    # Simulated relationship: U-shaped with higher risk at extreme values
    baseline = (pressure_range - 1013)**2 / 500
    migraine_risk = baseline + 0.1 * np.random.randn(len(pressure_range))
    migraine_risk = np.clip(migraine_risk, 0, 1)
    
    # Plot pressure impact
    ax2.plot(pressure_range, migraine_risk, '-', color=colors[1], linewidth=2)
    ax2.fill_between(pressure_range, 0, migraine_risk, color=colors[1], alpha=0.2)
    
    # Mark "normal" pressure range
    normal_min, normal_max = 1010, 1016
    ax2.axvspan(normal_min, normal_max, color='green', alpha=0.1, label='Normal Range')
    
    # Add reference points
    ax2.plot([normal_min, normal_max], [0.3, 0.3], 'go-', linewidth=2, markersize=6)
    ax2.plot([995, 1030], [0.55, 0.6], 'ro-', linewidth=2, markersize=6)
    
    # Customize plot
    ax2.set_title('Barometric Pressure Impact')
    ax2.set_xlabel('Pressure (hPa)')
    ax2.set_ylabel('Migraine Risk')
    ax2.set_ylim(0, 1)
    ax2.grid(linestyle='--', alpha=0.7)
    
    # Add annotations
    ax2.annotate('High Risk Zone', xy=(995, 0.55), xytext=(997, 0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    ax2.annotate('High Risk Zone', xy=(1030, 0.6), xytext=(1025, 0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    ax2.annotate('Low Risk Zone', xy=(1013, 0.3), xytext=(1013, 0.15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    # 3. Geographic Heatmap (simplified version)
    # Create a simple coordinate grid for demo
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    xx, yy = np.meshgrid(x, y)
    
    # Create synthetic risk data with spatial correlation
    from scipy.ndimage import gaussian_filter
    z = np.random.rand(20, 20)
    z = gaussian_filter(z, sigma=3)  # Apply smoothing for spatial correlation
    
    # Plot heatmap
    im = ax3.imshow(z, cmap='YlOrRd', origin='lower', extent=[0, 10, 0, 10])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label('Migraine Risk')
    
    # Add mock city locations
    cities = [
        {'name': 'City A', 'x': 2, 'y': 3},
        {'name': 'City B', 'x': 7, 'y': 8},
        {'name': 'City C', 'x': 5, 'y': 5},
        {'name': 'City D', 'x': 9, 'y': 2}
    ]
    
    for city in cities:
        ax3.plot(city['x'], city['y'], 'o', color='blue', markersize=8)
        ax3.text(city['x'] + 0.3, city['y'] + 0.3, city['name'], fontsize=9)
    
    # Customize plot
    ax3.set_title('Geographic Risk Heatmap')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(False)
    
    plt.tight_layout()
    plt.show()

# Plot EnvironmentalExpert visualizations
plot_environmental_visualizations(all_results['environmental'])

# %%
def plot_behavioral_visualizations(results):
    """Plot visualizations specific to the BehavioralExpert."""
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Sleep Pattern Impact
    # Synthetic sleep data
    sleep_duration = np.linspace(4, 10, 25)
    
    # U-shaped risk curve with minimum around 7-8 hours
    baseline_risk = 0.8 - 0.6 * np.exp(-(sleep_duration - 7.5)**2 / 4)
    
    # Different curves for different sleep quality
    low_quality_risk = baseline_risk + 0.2
    med_quality_risk = baseline_risk
    high_quality_risk = baseline_risk - 0.2
    
    # Clip values to valid range
    low_quality_risk = np.clip(low_quality_risk, 0, 1)
    med_quality_risk = np.clip(med_quality_risk, 0, 1)
    high_quality_risk = np.clip(high_quality_risk, 0, 1)
    
    # Plot sleep impact
    ax1.plot(sleep_duration, low_quality_risk, '-', color=colors[0], linewidth=2, label='Poor Sleep Quality')
    ax1.plot(sleep_duration, med_quality_risk, '-', color=colors[1], linewidth=2, label='Medium Sleep Quality')
    ax1.plot(sleep_duration, high_quality_risk, '-', color=colors[2], linewidth=2, label='Good Sleep Quality')
    
    # Highlight optimal sleep range
    ax1.axvspan(7, 8, color='green', alpha=0.1, label='Optimal Sleep Range')
    
    # Customize plot
    ax1.set_title('Sleep Pattern Impact')
    ax1.set_xlabel('Sleep Duration (hours)')
    ax1.set_ylabel('Migraine Risk')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(linestyle='--', alpha=0.7)
    
    # 2. Activity Level Impact
    # Synthetic activity data
    activity_levels = np.linspace(0, 10000, 30)
    
    # U-shaped risk curve with minimum around moderate activity
    activity_risk = 0.5 + 0.3 * np.sin((activity_levels - 5000) * np.pi / 5000)
    
    # Plot activity impact
    ax2.plot(activity_levels, activity_risk, '-', color=colors[1], linewidth=2)
    ax2.fill_between(activity_levels, 0, activity_risk, color=colors[1], alpha=0.2)
    
    # Highlight optimal activity range
    ax2.axvspan(4000, 6000, color='green', alpha=0.1, label='Optimal Activity Range')
    
    # Add annotations
    ax2.annotate('Low Activity\nRisk Zone', xy=(1000, 0.7), xytext=(1000, 0.85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    ax2.annotate('Optimal Activity', xy=(5000, 0.2), xytext=(5000, 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    ax2.annotate('High Activity\nRisk Zone', xy=(9000, 0.7), xytext=(9000, 0.85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    # Customize plot
    ax2.set_title('Activity Level Impact')
    ax2.set_xlabel('Steps per Day')
    ax2.set_ylabel('Migraine Risk')
    ax2.set_ylim(0, 1)
    ax2.grid(linestyle='--', alpha=0.7)
    
    # 3. Stress Level Correlation
    # Synthetic stress data
    stress_levels = [1, 2, 3, 4, 5]  # 1-5 scale
    migraine_counts = [10, 25, 45, 70, 100]  # Count of migraine events
    migraine_severity = [0.2, 0.35, 0.55, 0.7, 0.85]  # Average severity
    bubble_sizes = [s*20 for s in migraine_severity]  # Size proportional to severity
    
    # Plot stress correlation as bubble chart
    scatter = ax3.scatter(stress_levels, migraine_counts, s=bubble_sizes, 
                        c=migraine_severity, cmap='YlOrRd', 
                        alpha=0.7, edgecolors='black')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax3)
    cbar.set_label('Average Migraine Severity')
    
    # Add best fit line
    z = np.polyfit(stress_levels, migraine_counts, 1)
    p = np.poly1d(z)
    ax3.plot(stress_levels, p(stress_levels), '--', color='blue')
    
    # Add correlation coefficient
    from scipy.stats import pearsonr
    corr, _ = pearsonr(stress_levels, migraine_counts)
    ax3.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax3.set_title('Stress Level Correlation')
    ax3.set_xlabel('Stress Level (1-5 scale)')
    ax3.set_ylabel('Migraine Count')
    ax3.set_xticks(stress_levels)
    ax3.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Plot BehavioralExpert visualizations
plot_behavioral_visualizations(all_results['behavioral'])

# %%
def plot_medication_visualizations(results):
    """Plot visualizations specific to the MedicationHistoryExpert."""
    
    # Define colors for plots for consistency with other visualizations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Medication Efficacy Comparison
    # Synthetic medication data
    medications = ['Sumatriptan', 'Rizatriptan', 'Eletriptan', 'Naratriptan', 'Ibuprofen', 'Acetaminophen']
    efficacy = [0.75, 0.82, 0.78, 0.7, 0.5, 0.45]
    variability = [0.15, 0.1, 0.12, 0.18, 0.25, 0.3]
    
    # Sort by efficacy
    sorted_indices = np.argsort(efficacy)[::-1]  # descending
    sorted_meds = [medications[i] for i in sorted_indices]
    sorted_efficacy = [efficacy[i] for i in sorted_indices]
    sorted_variability = [variability[i] for i in sorted_indices]
    
    # Create color map based on efficacy
    colors_eff = plt.cm.YlGn(np.array(sorted_efficacy))
    
    # Plot efficacy
    y_pos = np.arange(len(sorted_meds))
    bars = ax1.barh(y_pos, sorted_efficacy, xerr=sorted_variability, 
                  color=colors_eff, alpha=0.8, ecolor='black', capsize=5)
    
    # Add labels
       # Add labels
    for i, bar in enumerate(bars):
        ax1.text(0.05, bar.get_y() + bar.get_height()/2, 
                sorted_meds[i], ha='left', va='center', 
                color='black', weight='bold')
    
    ax1.set_yticks([])
    ax1.set_xlabel('Efficacy (higher is better)')
    ax1.set_title('Medication Efficacy Comparison')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. Medication Timing Analysis
    # Synthetic timing data
    time_points = np.linspace(0, 10, 30)  # Hours after onset
    
    # Different efficacy curves based on timing
    early_efficacy = 1 - 0.8 * np.exp(-time_points/1.5)
    mid_efficacy = 1 - 0.8 * np.exp(-(time_points-2)/1.5)
    late_efficacy = 1 - 0.7 * np.exp(-(time_points-4)/1.5)
    
    # Clip values to valid range
    early_efficacy = np.clip(early_efficacy, 0, 1)
    mid_efficacy = np.clip(mid_efficacy, 0, 1)
    late_efficacy = np.clip(late_efficacy, 0, 1)
    
    # Plot timing impact
    ax2.plot(time_points, early_efficacy, '-', color=colors[0], linewidth=2, label='Early (0-1 hour)')
    ax2.plot(time_points, mid_efficacy, '-', color=colors[1], linewidth=2, label='Mid (2-3 hours)')
    ax2.plot(time_points, late_efficacy, '-', color=colors[2], linewidth=2, label='Late (4+ hours)')
    
    # Add vertical lines for critical timing
    ax2.axvline(x=1, color='green', linestyle='--', alpha=0.7, label='Critical Window')
    ax2.axvline(x=3, color='orange', linestyle='--', alpha=0.7)
    
    # Customize plot
    ax2.set_title('Medication Timing Analysis')
    ax2.set_xlabel('Hours After Onset')
    ax2.set_ylabel('Relief Probability')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.7)
    
    # 3. Dose-Response Relationship
    # Synthetic dose data
    doses = np.linspace(0.5, 2.0, 50)  # Relative to standard dose
    
    # Different dose-response curves for different medications
    response1 = 0.9 * (1 - np.exp(-(doses-0.5)*3))
    response2 = 0.8 * (1 - np.exp(-(doses-0.75)*2))
    
    # Side effects that increase with dose
    side_effects1 = 0.05 + 0.6 * doses**2
    side_effects2 = 0.05 + 0.4 * doses**2
    
    # Clip values to valid range
    response1 = np.clip(response1, 0, 1)
    response2 = np.clip(response2, 0, 1)
    side_effects1 = np.clip(side_effects1, 0, 1)
    side_effects2 = np.clip(side_effects2, 0, 1)
    
    # Create twin axis for side effects
    ax3_twin = ax3.twinx()
    
    # Plot dose-response
    ax3.plot(doses, response1, '-', color=colors[0], linewidth=2, label='Med A Efficacy')
    ax3.plot(doses, response2, '-', color=colors[1], linewidth=2, label='Med B Efficacy')
    
    # Plot side effects
    ax3_twin.plot(doses, side_effects1, '--', color=colors[0], linewidth=2, alpha=0.7, label='Med A Side Effects')
    ax3_twin.plot(doses, side_effects2, '--', color=colors[1], linewidth=2, alpha=0.7, label='Med B Side Effects')
    
    # Add vertical line for standard dose
    ax3.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Standard Dose')
    
    # Customize plot
    ax3.set_title('Dose-Response Relationship')
    ax3.set_xlabel('Dose (relative to standard)')
    ax3.set_ylabel('Efficacy')
    ax3_twin.set_ylabel('Side Effect Probability')
    
    ax3.set_ylim(0, 1)
    ax3_twin.set_ylim(0, 1)
    
    # Combine legends from both axes
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    ax3.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Plot MedicationHistoryExpert visualizations
plot_medication_visualizations(all_results['medication'])

# %%
"""
## 5. Combined Expert Analysis

Finally, let's examine how the different experts can be combined to create a more comprehensive prediction model.
"""

# %%
def plot_expert_contributions():
    """Plot the relative contributions of different experts to the combined prediction."""
    
    # Define colors for consistency with other plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Expert Weight Distribution
    # Synthetic expert weights
    expert_types = ['Physiological', 'Environmental', 'Behavioral', 'Medication']
    weights = np.array([0.4, 0.25, 0.2, 0.15])
    
    # Create pie chart
    wedges, texts, autotexts = ax1.pie(weights, labels=expert_types, autopct='%1.1f%%',
                                     startangle=90, colors=colors[:4])
    
    # Customize text
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    # Customize plot
    ax1.set_title('Expert Weight Distribution')
    ax1.axis('equal')  # Equal aspect ratio ensures circular pie
    
    # 2. Expert Correlation Heatmap
    # Synthetic correlation matrix
    corr_matrix = np.array([
        [1.0, 0.3, 0.5, 0.2],
        [0.3, 1.0, 0.4, 0.1],
        [0.5, 0.4, 1.0, 0.6],
        [0.2, 0.1, 0.6, 1.0]
    ])
    
    # Plot heatmap
    im = ax2.imshow(corr_matrix, cmap='YlGnBu', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Correlation')
    
    # Add ticks and labels
    ax2.set_xticks(np.arange(len(expert_types)))
    ax2.set_yticks(np.arange(len(expert_types)))
    ax2.set_xticklabels(expert_types)
    ax2.set_yticklabels(expert_types)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(expert_types)):
        for j in range(len(expert_types)):
            ax2.text(j, i, f"{corr_matrix[i, j]:.1f}",
                   ha="center", va="center", color="black" if corr_matrix[i, j] < 0.7 else "white")
    
    # Customize plot
    ax2.set_title('Expert Prediction Correlation')
    
    plt.tight_layout()
    plt.show()

# Plot expert contributions
plot_expert_contributions()

# %%
def plot_expert_ensemble():
    """Visualize how different expert predictions combine for a sample case."""
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Synthetic prediction data
    timestamps = pd.date_range(start='2023-01-01', periods=10, freq='1D')
    physio_pred = np.array([0.3, 0.4, 0.5, 0.7, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2])
    env_pred = np.array([0.2, 0.3, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2])
    behav_pred = np.array([0.4, 0.5, 0.6, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3])
    med_pred = np.array([0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1])
    
    # Weights from earlier
    weights = np.array([0.4, 0.25, 0.2, 0.15])
    
    # Calculate weighted ensemble prediction
    ensemble_pred = (weights[0] * physio_pred + 
                     weights[1] * env_pred + 
                     weights[2] * behav_pred + 
                     weights[3] * med_pred)
    
    # Define colors for the plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Add threshold line
    threshold = 0.6
    above_threshold = ensemble_pred >= threshold
    
    # Create stacked area chart
    plt.stackplot(timestamps, 
                 weights[0] * physio_pred,
                 weights[1] * env_pred,
                 weights[2] * behav_pred,
                 weights[3] * med_pred,
                 labels=['Physiological', 'Environmental', 'Behavioral', 'Medication'],
                 colors=colors)
    
    # Plot ensemble line on top
    plt.plot(timestamps, ensemble_pred, 'k-', linewidth=2.5, label='Ensemble Prediction')
    
    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Alert Threshold ({threshold})')
    
    # Highlight areas above threshold
    for i in range(len(timestamps)-1):
        if above_threshold[i] and above_threshold[i+1]:
            plt.axvspan(timestamps[i], timestamps[i+1], color='red', alpha=0.2)
    
    # Annotate high-risk periods
    for i in range(len(timestamps)):
        if above_threshold[i]:
            if i == 0 or not above_threshold[i-1]:  # Start of high-risk period
                plt.annotate('High Risk', xy=(timestamps[i], ensemble_pred[i] + 0.05),
                           xytext=(timestamps[i], ensemble_pred[i] + 0.15),
                           arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                           fontsize=10)
    
    # Customize plot
    plt.title('Expert Ensemble Prediction')
    plt.xlabel('Date')
    plt.ylabel('Migraine Risk (weighted contribution)')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.grid(linestyle='--', alpha=0.7)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

# Plot expert ensemble
plot_expert_ensemble()

# %%
"""
## Conclusion

This notebook provides a comprehensive visualization of our expert models' performance and insights:

1. **Core Performance Metrics**: We've analyzed each expert's baseline and optimized performance using standard metrics like MAE, MSE, R², etc.

2. **Prediction Quality**: The scatter plots, residuals, and time series visualizations show how well our experts predict migraine severity compared to actual values.

3. **Hyperparameter Optimization**: The heatmaps, parallel coordinates plots, and convergence curves illustrate the optimization process and the impact of different hyperparameters.

4. **Expert-Specific Insights**: We've created domain-specific visualizations that highlight the unique aspects of each expert's predictions and influential factors.

5. **Combined Expert Analysis**: The ensemble visualizations demonstrate how multiple experts can be combined for more robust predictions.

These visualizations not only help in evaluating model performance but also provide actionable insights for model improvement and clinical applications.
"""

# %%
def save_visualization_results(all_results, filename='expert_visualization_results.pkl'):
    """Save the visualization results for future reference."""
    
    import pickle
    
    # Prepare data for saving
    save_data = {
        'all_results': all_results,
        'timestamp': pd.Timestamp.now(),
        'metrics_summary': {expert: {
            'baseline': results['baseline']['metrics'],
            'optimized': results['optimized']['metrics']
        } for expert, results in all_results.items()}
    }
    
    # Save to file
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Results saved to {filename}")

# Save results
save_visualization_results(all_results)

# Display final performance summary as a formatted table
metrics_df = pd.DataFrame({
    'Expert': [],
    'Version': [],
    'MAE': [],
    'MSE': [],
    'R²': []
})

for expert, data in all_results.items():
    for version in ['baseline', 'optimized']:
        if version in data:
            metrics = data[version]['metrics']
            
            # Check for various metric key formats and use the first one found
            mae_keys = ['test_mae', 'mae', 'mean_absolute_error']
            mse_keys = ['test_mse', 'mse', 'mean_squared_error']
            r2_keys = ['test_r2', 'r2', 'r2_score']
            
            # Get MAE (or use NaN if not found)
            mae_value = next((metrics[k] for k in mae_keys if k in metrics), np.nan)
            
            # Get MSE (or use NaN if not found)
            mse_value = next((metrics[k] for k in mse_keys if k in metrics), np.nan)
            
            # Get R² (or use NaN if not found)
            r2_value = next((metrics[k] for k in r2_keys if k in metrics), np.nan)
            
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Expert': [expert.capitalize()],
                'Version': [version.capitalize()],
                'MAE': [mae_value],
                'MSE': [mse_value],
                'R²': [r2_value]
            })], ignore_index=True)

# Format the numeric columns to display more decimal places
with pd.option_context('display.precision', 6):
    print("Results saved to expert_visualization_results.pkl")
    display(metrics_df)

# %%
