"""Advanced imputation techniques for time series data."""

import numpy as np
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union, Type
# --- Add PyTorch imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# --- ------------------- ---

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

# --- Renamed class ---
class IterativeImputerWrapper(BaseImputer):
    """
    Imputes missing values using a multivariate imputation strategy (sklearn's IterativeImputer).

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
        Initialize the Iterative Imputer Wrapper.

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
            from sklearn.experimental import enable_iterative_imputer # noqa
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

    def fit(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> 'IterativeImputerWrapper':
        """
        Fit the Iterative Imputer on the observed data.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Optional boolean mask (not directly used by IterativeImputer).

        Returns:
            Fitted IterativeImputerWrapper instance.
        """
        if data.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (samples, seq_len, features)")
        self._original_shape = data.shape
        self._n_features = data.shape[2]

        # Reshape data so that features are columns for the imputer
        data_reshaped = data.reshape(-1, self._n_features)

        if self.scale:
            self.scaler = StandardScaler()
            observed_data = data_reshaped[~np.isnan(data_reshaped).any(axis=1)]
            if observed_data.shape[0] > 0:
                 self.scaler.fit(observed_data)
                 data_scaled = self.scaler.transform(data_reshaped)
                 # Reapply NaN mask after scaling
                 nan_mask_reshaped = np.isnan(data_reshaped)
                 data_scaled[nan_mask_reshaped] = np.nan
            else:
                 print("Warning: No fully observed samples/timesteps found for scaling. Skipping scaling.")
                 data_scaled = data_reshaped
                 self.scale = False
        else:
            data_scaled = data_reshaped

        # Fit the IterativeImputer
        self.imputer.fit(data_scaled)
        return self

    def transform(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Impute missing values using the fitted Iterative Imputer.

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
            # Check for NaNs introduced by imputer (can happen)
            if np.isnan(data_imputed_scaled).any():
                 print("Warning: NaNs detected after imputation before inverse scaling. Check IterativeImputer parameters.")
                 # Handle remaining NaNs if necessary, e.g., replace with mean of scaled data
                 col_means = np.nanmean(data_imputed_scaled, axis=0) # Use nanmean
                 nan_indices = np.where(np.isnan(data_imputed_scaled))
                 # Handle case where col_means itself might have NaNs if a column was all NaN
                 col_means = np.nan_to_num(col_means)
                 data_imputed_scaled[nan_indices] = np.take(col_means, nan_indices[1])


            data_imputed = self.scaler.inverse_transform(data_imputed_scaled)
        else:
            data_imputed = data_imputed_scaled

        # Reshape back to original dimensions
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
        self.fit(data, mask if mask is not None else ~np.isnan(data))
        return self.transform(data, mask)


# --- LSTM Autoencoder Implementation ---
class _LSTMAutoencoder(nn.Module):
    """Simple LSTM Autoencoder model."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # The context is the final hidden state
        context = hidden

        # Decoder setup: Use last hidden state as initial state
        # Use a dummy input sequence of same length as input for teacher forcing (or zeros)
        decoder_input = torch.zeros_like(x) # Using zeros as input
        # OR Use the last encoded output as the first input:
        # decoder_input = torch.zeros(x.size(0), x.size(1), context.size(-1), device=x.device)

        # Pass context to decoder
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # Reconstruct original features
        reconstruction = self.fc(decoder_output)
        return reconstruction

class AutoencoderImputer(BaseImputer):
    """
    Imputes missing values using an LSTM Autoencoder.

    Trains an autoencoder on the observed parts of the time series data
    and uses the reconstruction to fill in missing values.
    Assumes data shape (n_samples, sequence_length, n_features).
    """
    def __init__(self, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1,
                 epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
                 scale: bool = True, device: Optional[str] = None, random_state: Optional[int] = None):
        """
        Initialize the Autoencoder Imputer.

        Args:
            hidden_dim: Hidden dimension size for LSTM layers.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate between LSTM layers (if num_layers > 1).
            epochs: Number of training epochs for the autoencoder.
            batch_size: Batch size for training.
            learning_rate: Learning rate for the Adam optimizer.
            scale: Whether to scale data before training/imputation using StandardScaler.
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect).
            random_state: Seed for reproducibility.
        """
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scale = scale
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"AutoencoderImputer using device: {self.device}")

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[_LSTMAutoencoder] = None
        self._original_shape: Optional[tuple] = None
        self._n_features: Optional[int] = None

    def fit(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> 'AutoencoderImputer':
        """
        Fit the LSTM Autoencoder on the observed data.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Boolean mask indicating observed (True/1) values. If None, assumes NaNs mark missing.

        Returns:
            Fitted AutoencoderImputer instance.
        """
        if data.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (samples, seq_len, features)")
        self._original_shape = data.shape
        self._n_features = data.shape[2]

        if mask is None:
            mask = ~np.isnan(data)

        # --- Prepare data for training ---
        # 1. Reshape to (n_total_samples, n_features) for scaling
        data_reshaped = data.reshape(-1, self._n_features)
        mask_reshaped = mask.reshape(-1, self._n_features)

        # 2. Scale if required
        if self.scale:
            self.scaler = StandardScaler()
            # Fit scaler only on observed values across all samples/timesteps
            observed_flat = data_reshaped[mask_reshaped] # Get only observed values (1D)
            if observed_flat.shape[0] > 0:
                 # Reshape for scaler (needs 2D, even if one feature)
                 self.scaler.fit(observed_flat.reshape(-1, 1) if self._n_features == 1 else data_reshaped[np.all(mask_reshaped, axis=1)]) # Fit on fully observed rows
                 data_scaled = self.scaler.transform(data_reshaped)
                 data_scaled[~mask_reshaped] = np.nan # Reapply NaNs after scaling
            else:
                 print("Warning: No observed data found for scaling. Skipping scaling.")
                 data_scaled = data_reshaped # Use original data if scaling fails
                 self.scale = False # Disable scaling for transform
        else:
            data_scaled = data_reshaped

        # 3. Reshape back to (n_samples, seq_len, n_features)
        data_scaled = data_scaled.reshape(self._original_shape)

        # 4. Create training dataset - Use only *fully observed* sequences for simplicity
        #    More advanced: train with masking loss on partially observed sequences.
        fully_observed_mask = np.all(mask, axis=(1, 2)) # Check if entire sequence is observed
        if not np.any(fully_observed_mask):
            print("Warning: No fully observed sequences found. Autoencoder might not train well.")
            # Fallback: Use any sequence with at least one observed value (might be noisy)
            observed_indices = np.where(np.any(mask, axis=(1, 2)))[0]
            if len(observed_indices) == 0:
                 raise ValueError("No observed data found at all. Cannot train Autoencoder.")
            train_data_sequences = data_scaled[observed_indices]
        else:
             train_data_sequences = data_scaled[fully_observed_mask]


        # Convert to PyTorch tensors
        train_tensor = torch.tensor(train_data_sequences, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(train_tensor, train_tensor) # Input = Target for AE
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # --- Initialize and Train Model ---
        self.model = _LSTMAutoencoder(
            input_dim=self._n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"Training Autoencoder for {self.epochs} epochs...")
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                reconstructions = self.model(inputs)
                loss = criterion(reconstructions, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            if (epoch + 1) % 10 == 0 or epoch == 0: # Print every 10 epochs
                 print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

        print("Autoencoder training finished.")
        return self

    def transform(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Impute missing values using the fitted Autoencoder.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Optional boolean mask indicating observed (True/1) values.

        Returns:
            Data array with missing values imputed.
        """
        if self.model is None:
            raise RuntimeError("Imputer has not been fitted yet. Call fit() first.")
        if data.ndim != 3 or data.shape[2] != self._n_features:
            raise ValueError("Input data must be 3D with the same number of features as fitted data.")

        if mask is None:
            mask = ~np.isnan(data)

        # --- Prepare data for imputation ---
        # 1. Reshape and Scale
        data_reshaped = data.reshape(-1, self._n_features)
        mask_reshaped = mask.reshape(-1, self._n_features)
        if self.scale and self.scaler:
            data_scaled = self.scaler.transform(data_reshaped)
            data_scaled[~mask_reshaped] = 0 # Replace NaNs with 0 for AE input (or use mean)
        else:
            data_scaled = np.nan_to_num(data_reshaped) # Replace NaNs if not scaling

        # 2. Reshape back and convert to tensor
        data_scaled = data_scaled.reshape(data.shape)
        input_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(self.device)

        # --- Impute using Autoencoder ---
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(input_tensor).cpu().numpy()

        # --- Combine original observed data with reconstructed missing data ---
        # Reshape reconstructions and original data/mask back to 2D for easier indexing
        reconstructions_flat = reconstructions.reshape(-1, self._n_features)
        data_imputed_scaled_flat = data_scaled.reshape(-1, self._n_features)

        # Fill only the originally missing values with reconstructions
        missing_mask_flat = ~mask_reshaped
        data_imputed_scaled_flat[missing_mask_flat] = reconstructions_flat[missing_mask_flat]

        # --- Inverse Scale ---
        if self.scale and self.scaler:
            data_imputed_flat = self.scaler.inverse_transform(data_imputed_scaled_flat)
        else:
            data_imputed_flat = data_imputed_scaled_flat

        # --- Reshape back to original 3D ---
        return data_imputed_flat.reshape(data.shape)

    def fit_transform(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the imputer and transform the data.

        Args:
            data: Input data array (n_samples, sequence_length, n_features) with NaNs for missing.
            mask: Optional boolean mask.

        Returns:
            Data array with missing values imputed.
        """
        self.fit(data, mask)
        return self.transform(data, mask)


# --- Placeholder for PSO Imputer ---
class PSOImputer(BaseImputer):
    """
    Imputes missing values using Particle Swarm Optimization. (Placeholder)

    TODO: Implement PSO-based imputation, likely requiring defining a PyGMO
    problem where particles represent missing values and the fitness function
    evaluates the quality of the imputation (e.g., smoothness, consistency).
    """
    def __init__(self, **kwargs):
        raise NotImplementedError("PSOImputer is not yet implemented.")

    def fit(self, data: np.ndarray, mask: np.ndarray) -> 'PSOImputer':
        raise NotImplementedError("PSOImputer is not yet implemented.")

    def transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PSOImputer is not yet implemented.") 