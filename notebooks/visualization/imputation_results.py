# %% [markdown]
# Imputation Results Visualization & Testing
#
# Objective: Test the refactored imputation classes (`KNNImputer`, `IterativeImputer`)
# from `src.preprocessing.advanced_imputation` and visualize their behavior on synthetic data.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path to allow importing src modules
try:
    notebook_dir = os.path.dirname(__file__)
except NameError:
    notebook_dir = os.getcwd()

module_path = os.path.abspath(os.path.join(notebook_dir, '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")
else:
    print(f"{module_path} already in sys.path")

# Import imputation classes
try:
    from src.preprocessing.advanced_imputation import KNNImputer, IterativeImputer
    print("Successfully imported imputation classes.")
except ImportError as e:
    print(f"Error importing imputation classes: {e}")
    print("Ensure src/preprocessing/advanced_imputation.py exists and is correct.")
    KNNImputer = None
    IterativeImputer = None

print("Setup Complete")

# %% [markdown]
# ## 2. Generate Synthetic Data
#
# Create a sample dataset with missing values (NaNs) suitable for the imputers.
# The imputers expect data in the shape (n_samples, sequence_length, n_features).
# For this test, we'll use n_samples=1.

# %%
def create_synthetic_data_with_nans(
    sequence_length=50, num_features=4, missing_fraction=0.2, seed=42
):
    """
    Creates a simple synthetic time series with NaNs.

    Returns:
        tuple: (X_nan, X_true, mask)
               X_nan (np.ndarray): Data with NaNs (1, seq_len, features).
               X_true (np.ndarray): Original data before masking (1, seq_len, features).
               mask (np.ndarray): Boolean mask (True=observed) (1, seq_len, features).
    """
    np.random.seed(seed)
    tt = np.linspace(0, 20, sequence_length)
    X_true_2d = np.zeros((sequence_length, num_features))

    for j in range(num_features):
        freq = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, np.pi)
        amp = np.random.uniform(0.8, 1.2)
        offset = np.random.uniform(-0.2, 0.2)
        noise = np.random.normal(0, 0.05, sequence_length)
        X_true_2d[:, j] = amp * np.sin(2 * np.pi * freq * tt / 20 + phase) + offset + noise

    # Introduce NaNs
    mask_2d = np.random.rand(sequence_length, num_features) > missing_fraction
    X_nan_2d = X_true_2d.copy()
    X_nan_2d[~mask_2d] = np.nan

    # Reshape to 3D (1 sample)
    X_nan_3d = X_nan_2d.reshape(1, sequence_length, num_features)
    X_true_3d = X_true_2d.reshape(1, sequence_length, num_features)
    mask_3d = mask_2d.reshape(1, sequence_length, num_features)

    print(f"Generated synthetic data with NaNs:")
    print(f"  Shape: {X_nan_3d.shape}")
    print(f"  Total NaNs: {np.isnan(X_nan_3d).sum()}")
    print(f"  NaN Percentage: {np.isnan(X_nan_3d).mean() * 100:.2f}%")

    return X_nan_3d, X_true_3d, mask_3d

# Generate data
X_nan, X_true, mask = create_synthetic_data_with_nans()

# %% [markdown]
# ## 3. Test Imputation Methods
#
# Instantiate and run the imputation classes on the synthetic data.

# %%
imputed_results = {}

# --- KNN Imputation ---
if KNNImputer is not None:
    print("\n--- Testing KNNImputer ---")
    try:
        knn_imputer = KNNImputer(n_neighbors=5, scale=True)
        X_knn_imputed = knn_imputer.fit_transform(X_nan.copy(), mask.copy())
        imputed_results['KNN'] = X_knn_imputed
        print(f"KNN Imputation successful. Output shape: {X_knn_imputed.shape}")
        print(f"Remaining NaNs after KNN: {np.isnan(X_knn_imputed).sum()}")
    except Exception as e:
        print(f"ERROR during KNN Imputation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nSkipping KNNImputer test (class not imported).")

# --- Iterative Imputation ---
if IterativeImputer is not None:
    print("\n--- Testing IterativeImputer ---")
    try:
        # Using default BayesianRidge estimator
        iterative_imputer = IterativeImputer(max_iter=10, random_state=42, scale=True)
        X_iterative_imputed = iterative_imputer.fit_transform(X_nan.copy(), mask.copy())
        imputed_results['Iterative'] = X_iterative_imputed
        print(f"Iterative Imputation successful. Output shape: {X_iterative_imputed.shape}")
        print(f"Remaining NaNs after Iterative: {np.isnan(X_iterative_imputed).sum()}")
    except Exception as e:
        print(f"ERROR during Iterative Imputation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nSkipping IterativeImputer test (class not imported).")

# %% [markdown]
# ## 4. Analyze Results
#
# Check the imputed data (e.g., presence of NaNs) and calculate basic metrics like RMSE against the ground truth.

# %%
def calculate_rmse_3d(true_data, imputed_data, original_mask):
    """Calculate RMSE on originally missing values for 3D data (1 sample)."""
    if true_data is None or imputed_data is None or original_mask is None:
        return np.nan
    if true_data.shape != imputed_data.shape or true_data.shape != original_mask.shape:
        print("Shape mismatch for RMSE calculation.")
        return np.nan

    missing_mask_flat = ~original_mask.flatten()
    if not np.any(missing_mask_flat):
        return 0.0

    true_flat = true_data.flatten()
    imputed_flat = imputed_data.flatten()

    error = true_flat[missing_mask_flat] - imputed_flat[missing_mask_flat]
    rmse = np.sqrt(np.mean(error**2))
    return rmse

print("\n--- RMSE Evaluation (on originally missing values) ---")
for label, data_imputed in imputed_results.items():
    if data_imputed is not None:
        rmse = calculate_rmse_3d(X_true, data_imputed, mask)
        print(f"RMSE for {label}: {rmse:.4f}")
    else:
        print(f"RMSE for {label}: N/A (Imputation Failed)")

# Convert back to DataFrame for easier inspection (optional)
if imputed_results:
    print("\n--- Imputed DataFrames (first 5 rows, sample 0) ---")
    num_features = X_nan.shape[2]
    columns = [f'feature_{i}' for i in range(num_features)]

    df_original = pd.DataFrame(X_nan[0, :, :], columns=columns)
    print("Original Data (with NaNs):")
    print(df_original.head())
    print(f"NaN counts: \n{df_original.isnull().sum()}\n")

    for label, data_imputed in imputed_results.items():
        df_imputed = pd.DataFrame(data_imputed[0, :, :], columns=columns)
        print(f"Imputed Data ({label}):")
        print(df_imputed.head())
        print(f"NaN counts: \n{df_imputed.isnull().sum()}\n")

# %% [markdown]
# ## 5. Visualization (Optional)
#
# Plot the original vs imputed values for a selected feature.

# %%
def plot_feature_imputation(feature_index, X_nan_3d, results_dict, X_true_3d=None):
    """Plots original vs imputed for one feature."""
    seq_len = X_nan_3d.shape[1]
    if feature_index >= X_nan_3d.shape[2]:
        print("Invalid feature index.")
        return

    tt = np.arange(seq_len)
    original_nan = X_nan_3d[0, :, feature_index]
    observed_indices = np.where(~np.isnan(original_nan))[0]
    missing_indices = np.where(np.isnan(original_nan))[0]

    plt.figure(figsize=(15, 5))

    # Plot true data if available
    if X_true_3d is not None:
        plt.plot(tt, X_true_3d[0, :, feature_index], label='Ground Truth', color='grey', linestyle='--', alpha=0.7, zorder=1)

    # Plot original observed points
    plt.scatter(tt[observed_indices], original_nan[observed_indices], label='Observed', marker='x', color='black', zorder=len(results_dict)+2)
    # Indicate missing locations
    if X_true_3d is not None:
        plt.scatter(tt[missing_indices], X_true_3d[0, missing_indices, feature_index], label='Missing (True Value)', marker='.', color='red', alpha=0.5, s=50, zorder=2)
    else:
        # Plot missing markers at y=0 if no true value
         plt.scatter(tt[missing_indices], np.zeros_like(missing_indices), label='Missing Location', marker='.', color='red', alpha=0.3, s=50, zorder=2)


    # Plot imputed series
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))
    for i, (label, data_imputed) in enumerate(results_dict.items()):
        if data_imputed is not None:
            plt.plot(tt, data_imputed[0, :, feature_index], label=f'{label} Imputed', color=colors[i], alpha=0.8, zorder=i+3)

    plt.title(f'Imputation Comparison - Feature {feature_index}')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot for the first feature
if imputed_results:
    plot_feature_imputation(0, X_nan, imputed_results, X_true)

# %% [markdown]
# **Next Steps:**
# * Analyze the plots and RMSE values.
# * If the imputers work as expected here, the next step is to update the main scripts (e.g., `run_migraine_prediction.py`) to accept `--imputation_method` and `--imputer_config` arguments and pass them down to the `MigraineDataPipeline`. 