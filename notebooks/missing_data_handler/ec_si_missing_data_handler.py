# %% [markdown]
# # EC/SI Missing Data Handler Development
#
# **Objective:** Develop and evaluate Evolutionary Computation (EC) and Swarm Intelligence (SI) based imputation strategies for missing time-series data in the MIMIC-IV / Migraine datasets, comparing them to existing baseline methods.

# %% [markdown]
# ## 1. Setup and Imports

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch # Added for synthetic data generation

# Add src directory to path - KEEP this part, it might be needed later
try:
    notebook_dir = os.path.dirname(__file__)
except NameError:
    notebook_dir = os.getcwd()
    # print(f"Warning: __file__ not defined. Assuming notebook dir: {notebook_dir}") # Optional warning

module_path = os.path.abspath(os.path.join(notebook_dir, '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")
else:
    print(f"{module_path} already in sys.path")

# --- Temporarily comment out imports causing errors ---
# try:
#     # from src.preprocessing.data_mimiciv import load_data as load_mimiciv_data, F_impute
#     # from src.preprocessing.preprocessing import Discretizer_multi
#     # # Potentially import PyGMO or other optimization libraries here
#     # # import pygmo as pg
#     # print("Successfully imported preprocessing modules.")
#     pass # Placeholder if all src imports are commented out
# except ImportError as e:
#     # print(f"Error importing modules: {e}")
#     # print("Please ensure the path to 'src' is correct and dependencies are installed.")
#     pass # Ignore import errors for now
# ----------------------------------------------------

print("Setup Complete")

# %% [markdown]
# ## 2. Generate Synthetic Data with Missingness
#
# Generate synthetic time-series data with controlled missing patterns for imputation testing.

# %%
def create_synthetic_data_with_missingness(
    num_samples=50, num_features=10, missing_fraction=0.3, seed=42
):
    """
    Creates synthetic irregular time series data with missing values.

    Args:
        num_samples (int): Number of time points.
        num_features (int): Number of features.
        missing_fraction (float): Approximate fraction of data points to mark as missing.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_true, X_irg, mask_irg, tt_irg)
               X_true (np.ndarray): Original data before masking.
               X_irg (np.ndarray): Feature matrix (num_samples, num_features). Values in missing spots are 0.
               mask_irg (np.ndarray): Binary mask (0=missing, 1=present) (num_samples, num_features).
               tt_irg (np.ndarray): Timestamps (num_samples,).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Generate smooth underlying signals (e.g., sine waves with noise)
    tt_irg = np.linspace(0, 48, num_samples) # Simulate 48 hours
    X_true = np.zeros((num_samples, num_features))
    for j in range(num_features):
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        offset = np.random.uniform(-0.5, 0.5)
        noise = np.random.normal(0, 0.1, num_samples)
        X_true[:, j] = amplitude * np.sin(2 * np.pi * freq * tt_irg / 48 + phase) + offset + noise

    # 2. Create missingness mask
    # Randomly mark points as missing
    mask_irg = np.random.rand(num_samples, num_features) > missing_fraction
    mask_irg = mask_irg.astype(float) # Convert boolean to float (0.0 or 1.0)

    # 3. Apply mask to create X_irg (set missing values to 0, as often done in preprocessing)
    X_irg = X_true * mask_irg

    print(f"Generated synthetic data:")
    print(f"  X_true shape: {X_true.shape}")
    print(f"  X_irg shape: {X_irg.shape}")
    print(f"  mask_irg shape: {mask_irg.shape}")
    print(f"  tt_irg shape: {tt_irg.shape}")
    actual_missing_frac = 1.0 - np.mean(mask_irg)
    print(f"  Target missing fraction: {missing_fraction:.2f}")
    print(f"  Actual missing fraction: {actual_missing_frac:.2f}")

    return X_true, X_irg, mask_irg, tt_irg

# Generate the data
# Now store X_true as well
X_true, X_irg, mask_irg, tt_irg = create_synthetic_data_with_missingness(
    num_samples=100,  # Number of time points
    num_features=8,   # Number of features/channels
    missing_fraction=0.25 # 25% missing data
)

# Ensure variables exist even if generation failed (though unlikely here)
if 'X_true' not in locals(): X_true = None
if 'X_irg' not in locals(): X_irg = None
if 'mask_irg' not in locals(): mask_irg = None
if 'tt_irg' not in locals(): tt_irg = None

# %% [markdown]
# ## 3. Baseline Imputation Methods
#
# Apply the existing imputation methods (Zero-fill, Forward-fill) as baselines.

# %%
X_zero_imputed = None
X_ffill_imputed = None

if X_irg is not None and mask_irg is not None:
    # Zero Imputation (values are already 0 where mask is 0 from extraction)
    X_zero_imputed = X_irg.copy()
    print(f"Zero imputation implicitly represented. Shape: {X_zero_imputed.shape}")

    # Forward Fill Imputation (using pandas for simplicity on the raw irregular data)
    try:
        # Assuming channel names aren't critical for ffill itself
        num_features = X_irg.shape[1]
        channel_names = [f'feature_{i}' for i in range(num_features)]

        df_irg = pd.DataFrame(X_irg, columns=channel_names)
        # Create a boolean mask for where data IS present
        df_present_mask = pd.DataFrame(mask_irg.astype(bool), columns=channel_names)

        # Set non-present values to NaN so ffill works
        df_irg_nan = df_irg.where(df_present_mask, np.nan)

        # Perform forward fill
        df_ffill = df_irg_nan.ffill()

        # Handle any remaining NaNs at the beginning (e.g., fill with zero)
        df_ffill = df_ffill.fillna(0)

        X_ffill_imputed = df_ffill.values
        print(f"Forward-fill imputed data shape: {X_ffill_imputed.shape}")
    except Exception as e:
        print(f"Error during baseline forward-fill imputation: {e}")
        X_ffill_imputed = None # Ensure it's None if failed
else:
    print("Skipping baseline imputation as sample data (X_irg, mask_irg) is not loaded.")

# %% [markdown]
# ## 4. EC/SI Imputation Strategy Development
#
# Implement and test advanced imputation methods.
#
# **Potential Strategies:**
# *   **PSO-Optimized Imputation:** Use Particle Swarm Optimization (or another EC algorithm like Differential Evolution from PyGMO) to find optimal imputation values based on minimizing reconstruction error or maximizing data likelihood, potentially considering correlations between channels.
# *   **Swarm-Based Correlation Modeling:** Model inter-channel correlations using SI principles and use these correlations to inform imputation.
# *   **Autoencoder-Based Imputation:** Train an autoencoder on the available data and use it to reconstruct missing values.
# *   **Generative Models (e.g., GANs, VAEs):** Train a generative model to learn the data distribution and sample plausible values for missing entries.

# %% [markdown]
# ### 4.1 Strategy 1: PSO-Optimized Imputation (Refined Fitness using PyGMO)

# %%
try:
    import pygmo as pg
    pygmo_available = True
except ImportError:
    print("PyGMO not found. Please install it (`pip install pygmo`) to run this section.")
    pygmo_available = False

X_pso_imputed = None
target_corr_matrix = None # Store target correlation matrix

if pygmo_available and X_irg is not None and mask_irg is not None:

    class ImputationProblemPyGMORefined: # Renamed class for clarity
        def __init__(self, data_with_missing, mask):
            self.original_data = data_with_missing.copy() # Work on a copy
            self.mask = mask.astype(bool) # Ensure boolean mask
            self.missing_indices_rows, self.missing_indices_cols = np.where(~self.mask)
            self.num_missing = len(self.missing_indices_rows)
            self.num_features = data_with_missing.shape[1]
            self.num_samples = data_with_missing.shape[0]

            if self.num_missing == 0:
                 print("Warning: No missing values found in the provided sample for PSO.")
                 self.num_missing = 0
                 self.observed_means = np.array([])
                 self.observed_stds = np.array([])
                 self.target_corr_matrix = np.identity(self.num_features) # Default identity
                 return

            # Precompute means and stds of observed data for bounds and fitness
            self.observed_means = np.zeros(self.num_features)
            self.observed_stds = np.ones(self.num_features) # Avoid division by zero

            # Calculate observed correlation matrix (handle pairs with insufficient data)
            observed_data_for_corr = self.original_data.copy()
            observed_data_for_corr[~self.mask] = np.nan # Set missing to NaN for corrcoef
            # Use nan-aware correlation calculation if available, fallback otherwise
            try:
                # Using pandas for robust NaN handling in correlation
                df_obs = pd.DataFrame(observed_data_for_corr)
                self.target_corr_matrix = df_obs.corr().fillna(0).values # Fill NaN correlations with 0
            except Exception:
                 print("Warning: Could not compute robust correlation matrix. Using np.corrcoef (may be less accurate with NaNs).")
                 # Fallback - may produce NaNs if columns have too few overlapping observations
                 self.target_corr_matrix = np.nan_to_num(np.corrcoef(observed_data_for_corr, rowvar=False))


            for j in range(self.num_features):
                observed_vals = self.original_data[self.mask[:, j], j]
                if len(observed_vals) > 1:
                    self.observed_means[j] = np.mean(observed_vals)
                    std_val = np.std(observed_vals)
                    self.observed_stds[j] = std_val if std_val > 1e-6 else 1.0 # Avoid zero std
                elif len(observed_vals) == 1:
                    self.observed_means[j] = observed_vals[0]

            print(f"ImputationProblemPyGMORefined initialized with {self.num_missing} missing values.")


        def fitness(self, x):
            if self.num_missing == 0: return [0.0] # No missing values, cost is 0

            # 1. Create the fully imputed dataset using candidate solution x
            imputed_data = self.original_data.copy()
            if len(x) != self.num_missing:
                 raise ValueError(f"Input vector x has length {len(x)}, expected {self.num_missing}")
            imputed_data[self.missing_indices_rows, self.missing_indices_cols] = x

            # --- Refined Objective Function ---
            # a) Penalize deviation of imputed values from observed mean/std
            deviation_cost = np.sum(((x - self.observed_means[self.missing_indices_cols]) / \
                                     self.observed_stds[self.missing_indices_cols])**2)

            # b) Penalize lack of temporal smoothness
            smoothness_cost = 0
            if imputed_data.shape[0] > 1:
                 diffs = np.diff(imputed_data, axis=0)
                 smoothness_cost = np.mean(diffs**2) # Mean squared difference

            # c) Penalize deviation from observed correlation structure
            correlation_cost = 0
            if self.num_features > 1:
                try:
                    current_corr_matrix = pd.DataFrame(imputed_data).corr().fillna(0).values
                    # Use Frobenius norm of the difference matrix
                    correlation_diff = self.target_corr_matrix - current_corr_matrix
                    correlation_cost = np.linalg.norm(correlation_diff, 'fro')**2
                except Exception as e:
                     # print(f"Warning: Could not calculate correlation cost: {e}")
                     correlation_cost = 0 # Assign no penalty if calculation fails


            # Combine costs (adjust weights as needed)
            # Give more weight to correlation structure perhaps
            w_dev = 0.2
            w_smooth = 0.3
            w_corr = 0.5
            total_cost = (w_dev * (deviation_cost / self.num_missing if self.num_missing > 0 else 0) +
                          w_smooth * smoothness_cost +
                          w_corr * correlation_cost)

            return [total_cost]

        def get_bounds(self): # Bounds remain the same
            if self.num_missing == 0: return ([], [])
            lower_bounds = []
            upper_bounds = []
            for i in range(self.num_missing):
                feature_idx = self.missing_indices_cols[i]
                mean = self.observed_means[feature_idx]
                std = self.observed_stds[feature_idx]
                lb = mean - 3 * std
                ub = mean + 3 * std
                if lb > ub: lb, ub = ub, lb
                lower_bounds.append(lb)
                upper_bounds.append(ub)
            return (lower_bounds, upper_bounds)

        def get_nobj(self): return 1
        def get_name(self): return "TimeSeriesImputationRefined"

    # --- Run PSO Optimization ---
    if mask_irg.size > 0 and np.sum(~mask_irg.astype(bool)) > 0:
        try:
            print("Setting up PyGMO PSO problem (Refined Fitness)...")
            problem_instance = ImputationProblemPyGMORefined(X_irg, mask_irg) # Use refined class
            if problem_instance.num_missing > 0:
                problem = pg.problem(problem_instance)
                print(problem)

                pop_size = 20
                generations = 30 # Might need more generations for complex fitness
                algo = pg.algorithm(pg.pso(gen=generations, omega=0.7, eta1=1.5, eta2=1.5)) # Adjusted PSO params slightly
                algo.set_verbosity(5)

                print(f"Running PSO (Refined) with {pop_size} particles for {generations} generations...")
                pop = pg.population(problem, size=pop_size)
                pop = algo.evolve(pop)

                print("PSO Optimization Finished.")
                print(f"Best fitness found: {pop.champion_f[0]}")

                best_imputed_values = pop.champion_x
                X_pso_imputed = X_irg.copy()
                X_pso_imputed[problem_instance.missing_indices_rows, problem_instance.missing_indices_cols] = best_imputed_values
                print(f"PSO Imputation Complete (Refined). Imputed data shape: {X_pso_imputed.shape}")
                # Store target correlation for potential analysis
                target_corr_matrix = problem_instance.target_corr_matrix
            else:
                 print("Skipping PSO optimization as no missing values were detected in the sample.")
                 X_pso_imputed = X_irg.copy()

        except Exception as e:
            print(f"Error during PyGMO PSO execution (Refined): {e}")
            import traceback
            traceback.print_exc()
            X_pso_imputed = None
    else:
         print("Skipping PSO optimization: Mask indicates no missing values or mask_irg is empty.")
         if X_irg is not None:
              X_pso_imputed = X_irg.copy()

elif X_irg is None or mask_irg is None:
    print("Skipping PSO development as sample data (X_irg or mask_irg) is not loaded.")
else:
    # Pygmo not available
    pass

# %% [markdown]
# ### 4.2 Other Standard Imputation Methods (KNN, Iterative)

# %%
from sklearn.experimental import enable_iterative_imputer # Enable experimental feature
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge # A common estimator for IterativeImputer

X_knn_imputed = None
X_iterative_imputed = None

if X_irg is not None and mask_irg is not None:
    # Sklearn imputers expect NaN for missing values
    X_nan = X_irg.copy()
    X_nan[mask_irg == 0] = np.nan # Set missing spots to NaN

    if np.any(np.isnan(X_nan)): # Check if there are any NaNs to impute
        # --- KNN Imputer ---
        try:
            print("\nRunning KNN Imputer...")
            knn_imputer = KNNImputer(n_neighbors=5, weights='uniform') # Configure KNN
            X_knn_imputed = knn_imputer.fit_transform(X_nan)
            print(f"KNN Imputation Complete. Shape: {X_knn_imputed.shape}")
        except Exception as e:
            print(f"Error during KNN Imputation: {e}")
            X_knn_imputed = None

        # --- Iterative Imputer ---
        try:
            print("\nRunning Iterative Imputer...")
            # Using BayesianRidge as the estimator, can be slow
            iterative_imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,  # Default is 10
                random_state=42,
                initial_strategy='mean',
                imputation_order='ascending'
            )
            X_iterative_imputed = iterative_imputer.fit_transform(X_nan)
            print(f"Iterative Imputation Complete. Shape: {X_iterative_imputed.shape}")
        except Exception as e:
            print(f"Error during Iterative Imputation: {e}")
            X_iterative_imputed = None
    else:
        print("Skipping KNN/Iterative imputation as no NaN values were found after masking.")
        # If no NaNs, the original data is complete w.r.t the mask
        X_knn_imputed = X_irg.copy()
        X_iterative_imputed = X_irg.copy()

else:
    print("Skipping KNN/Iterative imputation as sample data is not loaded.")



# %% [markdown]
# ### 4.2 Strategy 2: Autoencoder-Based Imputation (Conceptual Outline)

# %%
# This would involve:
# 1. Designing an Autoencoder (AE) architecture suitable for time series (e.g., LSTM-AE, Transformer-AE).
# 2. Training the AE on the *observed* parts of the data (masking out missing values during loss calculation).
# 3. Using the trained AE to reconstruct the full time series, including the missing parts.
# Requires PyTorch or TensorFlow.

print("Autoencoder-Based Imputation - Requires separate implementation using a deep learning framework.")

# %% [markdown]
# ## 5. Visualization and Comparison
#
# Visualize the results of different imputation methods on selected features.

# %%
def plot_imputation_comparison(feature_index, tt, X_original, mask, X_imputed_list, labels):
    """Helper function to plot imputation results for a single feature."""
    if X_original is None or mask is None or tt is None:
        print("Cannot plot: Original data, mask, or timesteps missing.")
        return

    if feature_index >= X_original.shape[1]:
         print(f"Cannot plot: feature_index {feature_index} out of bounds for data with shape {X_original.shape}")
         return

    plt.figure(figsize=(18, 6))

    # Plot original observed data
    observed_indices = np.where(mask[:, feature_index] == 1)[0]
    if len(observed_indices) > 0:
        # Ensure tt aligns with observed_indices
        tt_observed = tt[observed_indices] if len(tt) == len(mask) else tt[:len(observed_indices)] # Basic check
        plt.scatter(tt_observed, X_original[observed_indices, feature_index],
                    label='Observed Data', color='black', marker='x', s=60, zorder=len(X_imputed_list) + 2)
    else:
        print(f"Feature {feature_index} has no observed data in this sample.")

    # Plot original missing data points (where imputation happens)
    missing_indices = np.where(mask[:, feature_index] == 0)[0]
    if len(missing_indices) > 0:
        # Find a reasonable y-value to plot the missing markers
        try:
             plot_y = np.nanmin(X_original[:, feature_index]) if np.any(~np.isnan(X_original[:, feature_index])) else 0
        except ValueError:
             plot_y = 0 # Fallback if all are NaN

        # Ensure tt aligns with missing_indices
        tt_missing = tt[missing_indices] if len(tt) == len(mask) else tt[:len(missing_indices)] # Basic check
        plt.scatter(tt_missing, np.full_like(missing_indices, plot_y, dtype=float),
                    label='Missing Points', color='grey', marker='.', s=20, alpha=0.5, zorder=1)

    # Plot imputed data series
    # Use a different colormap perhaps, like 'tab10' for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(X_imputed_list))) # Changed colormap
    plotted_labels = set()

    for i, (X_imp, label) in enumerate(zip(X_imputed_list, labels)):
        if X_imp is not None:
             # Ensure tt aligns with imputed data length
             tt_plot = tt if len(tt) == len(X_imp) else tt[:len(X_imp)] # Basic check

             # Avoid plotting duplicate labels if multiple methods yield same result (e.g., failed PSO)
             plot_label = label
             loop_count = 0
             while plot_label in plotted_labels and loop_count < 10: # Safety break
                   plot_label = f"{label}_(duplicate_{loop_count+1})"
                   loop_count += 1

             plt.plot(tt_plot, X_imp[:, feature_index], label=plot_label, alpha=0.8, color=colors[i], zorder=i+2, linewidth=1.5)
             plotted_labels.add(plot_label)
        else:
            print(f"Skipping plot for '{label}' as imputed data is None.")

    plt.title(f'Imputation Comparison for Feature {feature_index}', fontsize=16)
    plt.xlabel('Time (Hours relative to start)', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- Create list of datasets to plot ---
imputed_datasets = []
imputation_labels = []

# Use isinstance checks for safety as variables might be None
if isinstance(X_zero_imputed, np.ndarray):
    imputed_datasets.append(X_zero_imputed)
    imputation_labels.append('Zero Imputed')
if isinstance(X_ffill_imputed, np.ndarray):
    imputed_datasets.append(X_ffill_imputed)
    imputation_labels.append('Forward Fill')
# <<< Add KNN and Iterative results >>>
if isinstance(X_knn_imputed, np.ndarray):
    imputed_datasets.append(X_knn_imputed)
    imputation_labels.append('KNN Imputed')
if isinstance(X_iterative_imputed, np.ndarray):
    imputed_datasets.append(X_iterative_imputed)
    imputation_labels.append('Iterative Imputed')
# <<< Add PSO last for potentially better color mapping >>>
if isinstance(X_pso_imputed, np.ndarray):
    imputed_datasets.append(X_pso_imputed)
    imputation_labels.append('PSO Imputed (Refined)')


# --- Plot Comparison ---
if X_irg is not None and tt_irg is not None and mask_irg is not None and len(imputed_datasets) > 0:
    num_features_to_plot = min(5, X_irg.shape[1]) # Plot first few features
    print(f"\nPlotting imputation comparisons for first {num_features_to_plot} features...")

    for feature_idx in range(num_features_to_plot):
        plot_imputation_comparison(feature_idx, tt_irg, X_irg, mask_irg,
                                   imputed_datasets, imputation_labels)
else:
    print("\nSkipping visualization comparison as prerequisite data (X_irg, tt_irg, mask_irg, or imputed results) is missing.")

# %% [markdown]
# ## 6. Evaluation Metrics & Distribution Comparison
#
# Define and calculate metrics to quantify the quality of imputation. Now using the ground truth `X_true` generated earlier.

# %%
def calculate_rmse(true_data, imputed_data, mask):
    """Calculate RMSE only on the originally missing values."""
    if true_data is None or imputed_data is None or mask is None:
         print("RMSE Calculation Error: Input data missing.")
         return np.nan
    missing_mask = (mask == 0) # Where data was originally missing
    if np.sum(missing_mask) == 0:
        # print("RMSE Calculation: No missing values to evaluate.")
        return 0.0 # No error if nothing was missing
    if true_data.shape != imputed_data.shape:
        print(f"RMSE Calculation Error: Shape mismatch - true {true_data.shape}, imputed {imputed_data.shape}")
        return np.nan

    # Ensure imputed values are numeric and handle potential NaNs from imputation methods
    imputed_vals_at_missing = np.nan_to_num(imputed_data[missing_mask])
    true_vals_at_missing = true_data[missing_mask]

    error = true_vals_at_missing - imputed_vals_at_missing
    rmse = np.sqrt(np.mean(error**2))
    return rmse

# --- Calculate RMSE for all methods ---
print("\n--- Quantitative Evaluation (RMSE on Missing Values) ---")
rmse_results = {}

if X_true is not None and mask_irg is not None:
    # Iterate through imputed datasets and calculate RMSE
    for data, label in zip(imputed_datasets, imputation_labels):
        if data is not None:
             rmse = calculate_rmse(X_true, data, mask_irg)
             rmse_results[label] = rmse
             print(f"RMSE for {label:<20}: {rmse:.4f}")
        else:
             rmse_results[label] = np.nan
             print(f"RMSE for {label:<20}: Not Available (Imputation Failed)")

    # Find best method based on RMSE
    if rmse_results:
        valid_rmse = {k: v for k, v in rmse_results.items() if not np.isnan(v)}
        if valid_rmse:
             best_method = min(valid_rmse, key=valid_rmse.get)
             print(f"\nBest method based on RMSE: {best_method} (RMSE: {valid_rmse[best_method]:.4f})")
        else:
             print("\nCould not determine best method (all failed or had NaN RMSE).")

else:
    print("Cannot calculate RMSE because ground truth (X_true) or mask (mask_irg) is missing.")


# --- Compare distributions of imputed vs observed values ---
print("\n--- Distribution Comparison ---")
if X_irg is not None and mask_irg is not None and len(imputed_datasets) > 0:
    num_features_to_plot = min(5, X_irg.shape[1])
    print(f"\nPlotting distribution comparisons for first {num_features_to_plot} features...")

    for feature_idx in range(num_features_to_plot):
        plt.figure(figsize=(12, 6))
        plot_title = f'Distribution Comparison for Feature {feature_idx}'
        has_data_to_plot = False

        # Observed data distribution
        observed_mask_feature = mask_irg[:, feature_idx] == 1
        if np.any(observed_mask_feature):
            observed_vals = X_irg[observed_mask_feature, feature_idx]
            observed_vals = observed_vals[~np.isnan(observed_vals)]
            if len(observed_vals) > 0:
                 sns.histplot(observed_vals, color='black', label='Observed', kde=True, stat='density', element='step', line_kws={'linewidth': 1.5})
                 has_data_to_plot = True

        # Imputed data distributions (only for originally missing points)
        missing_mask_feature = mask_irg[:, feature_idx] == 0
        # Use the same consistent color map as the time series plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(imputed_datasets)))

        if np.any(missing_mask_feature):
            for i, (X_imp, label) in enumerate(zip(imputed_datasets, imputation_labels)):
                 if X_imp is not None:
                      imputed_values = X_imp[missing_mask_feature, feature_idx]
                      imputed_values = imputed_values[~np.isnan(imputed_values)] # Handle potential NaNs
                      if len(imputed_values) > 0:
                           # Use the color corresponding to the method's index
                           sns.histplot(imputed_values, color=colors[i], label=f'{label} (Imputed)', kde=True, stat='density', element='step', line_kws={'linestyle': '--', 'linewidth': 1.5})
                           has_data_to_plot = True

        if not has_data_to_plot:
             plot_title += " (No data to plot)"

        plt.title(plot_title, fontsize=16)
        plt.xlabel('Feature Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        if has_data_to_plot:
            plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
else:
    print("\nSkipping distribution comparison as prerequisite data is missing.")

# %% [markdown]
# ## Discussion: Value of Encapsulation / Refactoring
#
# Now that we have experimented with several imputation methods directly within the notebook, let's consider why we would eventually move the most promising and refined method(s) into separate Python (`.py`) modules (e.g., `src/preprocessing/advanced_imputation.py`).
#
# **Why Refactor?**
#
# 1.  **Reusability:**
#     *   Instead of copying/pasting imputation code into different notebooks or scripts (like `run_migraine_prediction.py`), you can simply `import` the function/class from your module.
#     *   Ensures consistency: If you improve the imputation logic, you update it in one place, and all parts of your project benefit.
#
# 2.  **Testability:**
#     *   Code within `.py` modules is much easier to unit test using frameworks like `pytest`.
#     *   You can create specific test cases (e.g., data with all missing values, data with specific patterns) to ensure your imputation function handles edge cases correctly, which is hard to do reliably within a notebook.
#
# 3.  **Maintainability & Readability:**
#     *   Notebooks are great for exploration but can become long and difficult to navigate when they contain complex function definitions alongside the analysis.
#     *   Moving core logic to modules keeps notebooks focused on the specific workflow, analysis, or visualization, making them shorter and easier to understand.
#     *   Separation of Concerns: The module handles *how* imputation is done; the notebook/script handles *when* and *why* it's applied in a specific context.
#
# 4.  **Integration into Pipelines:**
#     *   Functions/classes in modules are easily integrated into larger data processing pipelines or machine learning workflows (like your main `run_migraine_prediction.py` script). Calling complex logic embedded within a notebook from another script is awkward and error-prone.
#
# 5.  **Collaboration:**
#     *   Standard Python modules are easier for multiple developers to work on simultaneously using version control systems (like Git) compared to merging changes in complex notebook JSON files.
#
# **Example Structure:**
#
# You might create `src/preprocessing/advanced_imputation.py` with content like:
#
# ```python
# # src/preprocessing/advanced_imputation.py
# import numpy as np
# import pygmo as pg
# from sklearn.impute import KNNImputer
#
# class BaseImputer:
#     def fit(self, X, mask):
#         raise NotImplementedError
#     def transform(self, X, mask):
#         raise NotImplementedError
#     def fit_transform(self, X, mask):
#         self.fit(X, mask)
#         return self.transform(X, mask)
#
# class PSOImputer(BaseImputer):
#     def __init__(self, pop_size=20, generations=30, pso_params=None, random_seed=42):
#         self.pop_size = pop_size
#         self.generations = generations
#         self.pso_params = pso_params if pso_params else {'omega': 0.7, 'eta1': 1.5, 'eta2': 1.5}
#         self.random_seed = random_seed
#         self.problem_instance_ = None # Store problem definition after fit
#         self.best_solution_ = None # Store best found values
#
#     # Inner class for PyGMO problem definition (adapted from notebook)
#     class _ImputationProblemPyGMO:
#         # ... (Include the refined PyGMO problem definition here) ...
#         pass
#
#     def fit(self, X, mask):
#         print("Fitting PSO Imputer (Defining Problem)...")
#         self.problem_instance_ = self._ImputationProblemPyGMO(X, mask)
#         # Fit doesn't run optimization here, just defines problem based on data stats
#         return self
#
#     def transform(self, X, mask): # Note: ignores X, mask - uses data from fit
#         if self.problem_instance_ is None:
#             raise RuntimeError("Imputer not fitted yet. Call fit() first.")
#         if self.problem_instance_.num_missing == 0:
#             print("PSO Transform: No missing values detected during fit. Returning original data.")
#             return self.problem_instance_.original_data.copy()
#
#         try:
#             pg.set_random_seed(self.random_seed)
#             problem = pg.problem(self.problem_instance_)
#             algo = pg.algorithm(pg.pso(gen=self.generations, **self.pso_params))
#             # algo.set_verbosity(1) # Optional verbosity
#             pop = pg.population(problem, size=self.pop_size, seed=self.random_seed)
#             print(f"Running PSO optimization for transform ({self.generations} gens)...")
#             pop = algo.evolve(pop)
#             self.best_solution_ = pop.champion_x
#             print(f"PSO optimization complete. Best fitness: {pop.champion_f[0]:.4f}")
#
#             imputed_data = self.problem_instance_.original_data.copy()
#             imputed_data[self.problem_instance_.missing_indices_rows, self.problem_instance_.missing_indices_cols] = self.best_solution_
#             return imputed_data
#         except Exception as e:
#             print(f"Error during PSO transform: {e}")
#             # Fallback: return original data with zeros? Or raise error?
#             return self.problem_instance_.original_data.copy() # Safer fallback
#
# # Wrapper for KNN
# class KNNImputerWrapper(BaseImputer):
#     def __init__(self, n_neighbors=5, **kwargs):
#         self.imputer = KNNImputer(n_neighbors=n_neighbors, **kwargs)
#
#     def fit(self, X, mask):
#         X_nan = X.copy()
#         X_nan[mask == 0] = np.nan
#         self.imputer.fit(X_nan) # KNN fit doesn't do much, mainly transform
#         return self
#
#     def transform(self, X, mask):
#         X_nan = X.copy()
#         X_nan[mask == 0] = np.nan
#         return self.imputer.transform(X_nan)
#
# # ... add IterativeImputerWrapper etc. ...
# ```
#
# In the notebook or script, you'd then use it like this:
# ```python
# # from src.preprocessing.advanced_imputation import PSOImputer, KNNImputerWrapper
# # pso_imp = PSOImputer(generations=50)
# # X_pso_final = pso_imp.fit_transform(X_irg, mask_irg)
# #
# # knn_imp = KNNImputerWrapper(n_neighbors=3)
# # X_knn_final = knn_imp.fit_transform(X_irg, mask_irg)
# ```
#
# This keeps the implementation details separate from the analysis workflow.

# %% [markdown]
# ## 7. Conclusion and Refactoring
#
# Summarize findings and identify promising imputation strategies. Plan for refactoring the best method(s) into reusable Python functions or classes within the `src/preprocessing` module.

# %%
# TODO: Summarize findings based on the experiments above.
# Which methods seem plausible? Which are computationally expensive?
# Did PSO converge to reasonable values? How sensitive is it to the objective function?
# How do the imputed distributions compare to the observed ones?

# TODO: Outline plan for refactoring successful methods into .py files.
# E.g., Create src/preprocessing/advanced_imputation.py
# Define functions like `pso_impute(X, mask, config)` or classes like `AutoencoderImputer`.
# Ensure refactored code is robust, handles edge cases (e.g., no missing data, all missing data),
# and includes documentation/type hints.

# %%
