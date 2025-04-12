"""Advanced imputation techniques for time series data."""

import numpy as np
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union, Type

class BaseImputer:
    """Base class for imputation methods."""
    def fit(self, data: np.ndarray, mask: np.ndarray) -> 'BaseImputer':
        """
        Fit the imputer on the observed data.

        Args:
            data: The input data array with missing values (e.g., marked as NaN or 0).
                  Shape typically (n_samples, sequence_length, n_features).
            mask: A boolean or binary array indicating observed (True/1) and missing (False/0) values.
                  Same shape as data.

        Returns:
            The fitted imputer instance.
        """
        raise NotImplementedError

    def transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Impute missing values in the data.

        Args:
            data: The input data array with missing values.
            mask: The mask indicating observed/missing values.

        Returns:
            Data array with missing values imputed.
        """
        raise NotImplementedError

    def fit_transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Fit the imputer and then transform the data.

        Args:
            data: The input data array with missing values.
            mask: The mask indicating observed/missing values.

        Returns:
            Data array with missing values imputed.
        """
        self.fit(data, mask)
        return self.transform(data, mask)


class KNNImputer(BaseImputer):
    """
    Imputes missing values using K-Nearest Neighbors.

    This is a wrapper around scikit-learn's KNNImputer, handling the
    reshaping and scaling often required for time series data.
    Assumes data shape (n_samples, sequence_length, n_features).
    Imputation is done feature-wise across samples and time steps.
    """
    def __init__(self, n_neighbors: int = 5, scale: bool = True, **kwargs):
        """
        Initialize the KNN Imputer.

        Args:
            n_neighbors: Number of neighboring samples to use for imputation.
            scale: Whether to scale data before imputation using StandardScaler.
                   Recommended for KNN.
            **kwargs: Additional keyword arguments passed to scikit-learn's KNNImputer.
        """
        self.n_neighbors = n_neighbors
        self.scale = scale
        self.imputer = SklearnKNNImputer(n_neighbors=self.n_neighbors, **kwargs)
        self.scaler: Optional[StandardScaler] = None
        self._original_shape: Optional[tuple] = None
        self._n_features: Optional[int] = None

    def fit(self, data: np.ndarray, mask: np.ndarray) -> 'KNNImputer':
        """
        Fit the KNN imputer on the observed data.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Boolean mask (not directly used by KNNImputer but required by BaseImputer).

        Returns:
            Fitted KNNImputer instance.
        """
        if data.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (samples, seq_len, features)")
        self._original_shape = data.shape
        self._n_features = data.shape[2]

        # Reshape data so that features are columns for the imputer
        # Combine samples and sequence length dimensions
        # (n_samples * sequence_length, n_features)
        data_reshaped = data.reshape(-1, self._n_features)

        if self.scale:
            self.scaler = StandardScaler()
            # Fit scaler only on observed values to avoid bias from imputed values (like 0)
            observed_data = data_reshaped[~np.isnan(data_reshaped).any(axis=1)]
            if observed_data.shape[0] > 0:
                 self.scaler.fit(observed_data)
                 data_scaled = self.scaler.transform(data_reshaped)
                 # Replace NaNs after scaling, as scaler might turn NaNs into numbers
                 nan_mask_reshaped = np.isnan(data_reshaped)
                 data_scaled[nan_mask_reshaped] = np.nan
            else:
                 # Handle case with no fully observed rows for scaling
                 print("Warning: No fully observed samples/timesteps found for scaling. Skipping scaling.")
                 data_scaled = data_reshaped # Use original data if scaling fails
                 self.scale = False # Disable scaling for transform
        else:
            data_scaled = data_reshaped

        # Fit the KNN imputer
        # Note: Sklearn KNNImputer naturally handles NaNs
        self.imputer.fit(data_scaled)
        return self

    def transform(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Impute missing values using the fitted KNN imputer.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Optional boolean mask (not directly used here).

        Returns:
            Data array with missing values imputed.
        """
        if self._original_shape is None or self._n_features is None:
            raise RuntimeError("Imputer has not been fitted yet. Call fit() first.")
        if data.shape[2] != self._n_features:
             raise ValueError(f"Input data has {data.shape[2]} features, expected {self._n_features}")
        if data.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (samples, seq_len, features)")

        current_shape = data.shape
        data_reshaped = data.reshape(-1, self._n_features)

        if self.scale and self.scaler:
            data_scaled = self.scaler.transform(data_reshaped)
            # Reapply NaN mask after scaling
            nan_mask_reshaped = np.isnan(data_reshaped)
            data_scaled[nan_mask_reshaped] = np.nan
        else:
            data_scaled = data_reshaped

        # Perform imputation
        data_imputed_scaled = self.imputer.transform(data_scaled)

        # Inverse scaling if data was scaled
        if self.scale and self.scaler:
            # Check for NaNs introduced by imputer (unlikely with KNN but possible in edge cases)
            if np.isnan(data_imputed_scaled).any():
                 print("Warning: NaNs detected after imputation before inverse scaling. Check KNN parameters.")
                 # Handle remaining NaNs if necessary, e.g., replace with mean of scaled data
                 col_means = np.nanmean(data_imputed_scaled, axis=0)
                 nan_indices = np.where(np.isnan(data_imputed_scaled))
                 data_imputed_scaled[nan_indices] = np.take(col_means, nan_indices[1])

            data_imputed = self.scaler.inverse_transform(data_imputed_scaled)
        else:
            data_imputed = data_imputed_scaled

        # Reshape back to original dimensions (samples, seq_len, features)
        # Use the shape of the *input* data for transform, not necessarily the fitted shape
        return data_imputed.reshape(current_shape)

    def fit_transform(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the imputer and transform the data.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Optional boolean mask.

        Returns:
            Data array with missing values imputed.
        """
        self.fit(data, mask if mask is not None else ~np.isnan(data)) # Pass mask even if unused by fit impl.
        return self.transform(data, mask)

# Placeholder for IterativeImputer - we will refactor this next
class IterativeImputer(BaseImputer):
    """
    Imputes missing values using a multivariate imputation strategy.

    Models each feature with missing values as a function of other features,
    and uses an iterative approach wherein it predicts the missing feature
    values, updates the dataset, and repeats until convergence.

    This is a wrapper around scikit-learn's IterativeImputer, handling the
    reshaping and scaling often required for time series data.
    Assumes data shape (n_samples, sequence_length, n_features).
    Imputation is done feature-wise across samples and time steps.
    """
    def __init__(self, estimator=None, max_iter: int = 10, random_state: Optional[int] = None,
                 initial_strategy: str = 'mean', imputation_order: str = 'ascending',
                 scale: bool = True, verbose: int = 0, **kwargs):
        """
        Initialize the Iterative Imputer.

        Args:
            estimator: The estimator object to use at each step of the imputation.
                       If None, defaults to BayesianRidge().
            max_iter: Maximum number of imputation rounds to perform.
            random_state: Seed of the pseudo random number generator to use.
            initial_strategy: Strategy to use to initialize missing values ('mean', 'median', 'most_frequent', 'constant').
            imputation_order: The order in which features will be imputed ('ascending', 'descending', 'roman', 'arabic', 'random').
            scale: Whether to scale data before imputation using StandardScaler.
                   Recommended for many estimators.
            verbose: Verbosity level of the imputation process.
            **kwargs: Additional keyword arguments passed to scikit-learn's IterativeImputer.
        """
        # Need to import here because it's experimental
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer as SklearnIterativeImputer
            from sklearn.linear_model import BayesianRidge
        except ImportError:
            raise ImportError("scikit-learn>=0.21 is required for IterativeImputer. Run 'pip install -U scikit-learn'")

        self.estimator = estimator if estimator is not None else BayesianRidge()
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.scale = scale
        self.verbose = verbose
        self.kwargs = kwargs

        self.imputer = SklearnIterativeImputer(
            estimator=self.estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            initial_strategy=self.initial_strategy,
            imputation_order=self.imputation_order,
            verbose=self.verbose,
            **self.kwargs
        )
        self.scaler: Optional[StandardScaler] = None
        self._original_shape: Optional[tuple] = None
        self._n_features: Optional[int] = None

    def fit(self, data: np.ndarray, mask: np.ndarray) -> 'IterativeImputer':
        """
        Fit the Iterative imputer on the observed data.

        Note: For IterativeImputer, 'fit' primarily prepares the internal state
        but the main work happens during 'transform' or 'fit_transform'.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Boolean mask indicating observed (True/1) and missing (False/0).

        Returns:
            Fitted IterativeImputer instance.
        """
        if data.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (samples, seq_len, features)")
        self._original_shape = data.shape
        self._n_features = data.shape[2]

        # Reshape for sklearn imputer: (n_samples * sequence_length, n_features)
        data_reshaped = data.reshape(-1, self._n_features)
        # Convert 0-masked data to NaN-masked data if necessary
        data_nan = data_reshaped.copy()
        if not np.isnan(data_nan).any(): # Check if data already uses NaNs
            mask_reshaped = mask.reshape(-1, self._n_features)
            data_nan[mask_reshaped == 0] = np.nan # Use mask to set NaNs

        if self.scale:
            self.scaler = StandardScaler()
            # Fit scaler only on observed values to avoid bias
            # Need to handle potential all-NaN columns if scaling
            valid_data_for_scaling = data_nan[~np.isnan(data_nan).all(axis=1)]
            if valid_data_for_scaling.shape[0] > 0:
                self.scaler.fit(valid_data_for_scaling)
                # Note: Scaling happens *before* fit_transform in sklearn's pipeline usually,
                # but here we fit the scaler now and apply in transform.
            else:
                print("Warning: Insufficient valid data for scaling. Skipping scaling.")
                self.scale = False
        else:
             self.scaler = None

        # Unlike KNN, IterativeImputer's fit *does* learn from the data structure
        # We'll fit the scaler here, but fit the imputer during transform/fit_transform
        # as the actual imputation happens iteratively there.
        # However, calling fit here is good practice per sklearn API
        # We'll scale *before* fitting the imputer itself.
        data_to_fit = data_nan
        if self.scale and self.scaler:
            data_scaled_fit = self.scaler.transform(data_nan)
             # Reapply NaN mask after scaling
            nan_mask_reshaped = np.isnan(data_nan)
            data_scaled_fit[nan_mask_reshaped] = np.nan
            data_to_fit = data_scaled_fit

        if np.isnan(data_to_fit).any(): # Only fit if there are NaNs
            self.imputer.fit(data_to_fit) # Fit the imputer structure

        return self

    def transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Impute missing values using the fitted Iterative imputer.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with potential missing values.
            mask: Boolean mask indicating observed/missing.

        Returns:
            Data array with missing values imputed.
        """
        if self._original_shape is None or self._n_features is None:
            raise RuntimeError("Imputer has not been fitted yet. Call fit() first.")
        if data.shape[2] != self._n_features:
             raise ValueError(f"Input data has {data.shape[2]} features, expected {self._n_features}")
        if data.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (samples, seq_len, features)")

        current_shape = data.shape
        data_reshaped = data.reshape(-1, self._n_features)
        data_nan = data_reshaped.copy()
        mask_reshaped = mask.reshape(-1, self._n_features)
        data_nan[mask_reshaped == 0] = np.nan

        if not np.isnan(data_nan).any():
            print("Warning: No missing values (NaNs) found in data provided to transform. Returning original data.")
            return data # Return original if no NaNs

        data_to_transform = data_nan
        if self.scale and self.scaler:
            data_scaled = self.scaler.transform(data_nan)
            # Reapply NaN mask after scaling
            nan_mask_reshaped = np.isnan(data_nan)
            data_scaled[nan_mask_reshaped] = np.nan
            data_to_transform = data_scaled

        # Perform imputation using transform (fit should have been called)
        data_imputed_scaled = self.imputer.transform(data_to_transform)

        # Inverse scaling if data was scaled
        if self.scale and self.scaler:
            if np.isnan(data_imputed_scaled).any():
                 print("Warning: NaNs detected after imputation before inverse scaling. Check IterativeImputer parameters.")
                 # Handle remaining NaNs if necessary, e.g., replace with mean of scaled data
                 col_means = np.nanmean(data_imputed_scaled, axis=0)
                 nan_indices = np.where(np.isnan(data_imputed_scaled))
                 data_imputed_scaled[nan_indices] = np.take(col_means, nan_indices[1])

            data_imputed = self.scaler.inverse_transform(data_imputed_scaled)
        else:
            data_imputed = data_imputed_scaled

        # Reshape back to original dimensions
        return data_imputed.reshape(current_shape)

    def fit_transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Fit the imputer and transform the data in one step.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with potential missing values.
            mask: Boolean mask indicating observed/missing.

        Returns:
            Data array with missing values imputed.
        """
        if data.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (samples, seq_len, features)")
        self._original_shape = data.shape
        self._n_features = data.shape[2]

        current_shape = data.shape
        data_reshaped = data.reshape(-1, self._n_features)
        data_nan = data_reshaped.copy()
        mask_reshaped = mask.reshape(-1, self._n_features)
        data_nan[mask_reshaped == 0] = np.nan

        if not np.isnan(data_nan).any():
            print("Warning: No missing values (NaNs) found in data provided to fit_transform. Returning original data.")
            # Still fit the scaler if needed, even if no NaNs currently
            if self.scale:
                self.scaler = StandardScaler()
                valid_data_for_scaling = data_nan[~np.isnan(data_nan).all(axis=1)] # Should be all data here
                if valid_data_for_scaling.shape[0] > 0:
                    self.scaler.fit(valid_data_for_scaling)
                else:
                    self.scale = False # Disable scaling if fit fails
            return data # Return original if no NaNs

        data_to_impute = data_nan
        if self.scale:
            self.scaler = StandardScaler()
            # Fit scaler only on observed values to avoid bias
            valid_data_for_scaling = data_nan[~np.isnan(data_nan).all(axis=1)]
            if valid_data_for_scaling.shape[0] > 0:
                self.scaler.fit(valid_data_for_scaling)
                data_scaled = self.scaler.transform(data_nan)
                # Reapply NaN mask after scaling
                nan_mask_reshaped = np.isnan(data_nan)
                data_scaled[nan_mask_reshaped] = np.nan
                data_to_impute = data_scaled
            else:
                print("Warning: Insufficient valid data for scaling. Skipping scaling for fit_transform.")
                self.scale = False # Disable scaling

        # Perform imputation using fit_transform
        data_imputed_scaled = self.imputer.fit_transform(data_to_impute)

        # Inverse scaling if data was scaled
        if self.scale and self.scaler:
            if np.isnan(data_imputed_scaled).any():
                 print("Warning: NaNs detected after imputation before inverse scaling. Check IterativeImputer parameters.")
                 col_means = np.nanmean(data_imputed_scaled, axis=0)
                 nan_indices = np.where(np.isnan(data_imputed_scaled))
                 data_imputed_scaled[nan_indices] = np.take(col_means, nan_indices[1])

            data_imputed = self.scaler.inverse_transform(data_imputed_scaled)
        else:
            data_imputed = data_imputed_scaled

        # Reshape back to original dimensions
        return data_imputed.reshape(current_shape)


# Placeholder for AutoencoderImputer - might implement later if needed
class AutoencoderImputer(BaseImputer):
     def __init__(self, **kwargs):
          raise NotImplementedError("AutoencoderImputer not implemented yet.")

# Placeholder for PSOImputer - might implement later if needed
class PSOImputer(BaseImputer):
     def __init__(self, **kwargs):
          raise NotImplementedError("PSOImputer not implemented yet.") 