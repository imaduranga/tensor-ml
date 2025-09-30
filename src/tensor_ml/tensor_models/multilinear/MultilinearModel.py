from typing import Optional, Any, Union
import numpy as np
import torch
import pandas as pd
from tensor_ml.tensor_models.base import BaseTensorModel
from tensor_ml.enums import BackendType
from tensor_ml.utils import infer_backend
import logging

logger = logging.getLogger(__name__)

class MultilinearModel(BaseTensorModel):
    """
    Base class for multilinear tensor models. Inherit from this class for specific multilinear algorithms
    (e.g., TLARS, TNET, Kronecker OMP, etc.). Handles backend detection and input normalization.
    """
    def __init__(self, backend: Optional[Union[str, BackendType]] = None):
        if isinstance(backend, BackendType):
            self.backend = backend
        elif isinstance(backend, str):
            self.backend = BackendType(backend)
        else:
            self.backend = None
        super().__init__()

    def get_backend(self, X: Any = None) -> BackendType:
        """
        Returns the backend as a BackendType. If not set, infers from X and sets self.backend.
        :param X: Optional input to infer backend if not set.
        :return: BackendType (NUMPY, TORCH, or PANDAS)
        """
        if self.backend is not None:
            return self.backend
        if X is not None:
            # Use shared infer_backend for numpy/torch, else handle pandas
            try:
                self.backend = infer_backend(X)
            except ValueError:
                import pandas as pd
                if isinstance(X, pd.DataFrame):
                    self.backend = BackendType.PANDAS
                else:
                    raise ValueError("Input X must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame")
            return self.backend
        raise ValueError("Backend is not set and cannot be inferred without input X.")

    def normalize_input(self, X: Any, device: Optional[Union[str, torch.device]] = None) -> Any:
        """
        Converts input to the appropriate type based on backend.
        - If backend is numpy, converts pandas DataFrame to np.ndarray.
        - If backend is torch, converts pandas DataFrame or np.ndarray to torch.Tensor and moves to the specified device (default: keep on current device if torch.Tensor, else 'cuda' if available, else 'cpu').
        - If backend is pandas, returns as is.
        :param X: Input data (np.ndarray, torch.Tensor, or pd.DataFrame)
        :param device: Device for torch.Tensor ('cpu', 'cuda', or torch.device), only used if backend is torch.
        :return: Normalized input
        """
        backend = self.get_backend(X)
        if backend == BackendType.NUMPY:
            if isinstance(X, pd.DataFrame):
                return X.values
            if isinstance(X, np.ndarray):
                return X
            if isinstance(X, torch.Tensor):
                return X.cpu().numpy()
        elif backend == BackendType.TORCH:
            # Device selection
            import logging
            logger = logging.getLogger(__name__)
            if isinstance(X, torch.Tensor):
                # If device is None, keep on current device
                if device is None:
                    target_device = X.device
                else:
                    target_device = torch.device(device)
                    if target_device.type == 'cuda' and not torch.cuda.is_available():
                        logger.warning("Requested CUDA device, but CUDA is not available. Falling back to CPU.")
                        target_device = torch.device('cpu')
                return X.to(target_device)
            # If not a tensor, select device as before
            if device is None:
                target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                target_device = torch.device(device)
                if target_device.type == 'cuda' and not torch.cuda.is_available():
                    logger.warning("Requested CUDA device, but CUDA is not available. Falling back to CPU.")
                    target_device = torch.device('cpu')
            if isinstance(X, np.ndarray):
                return torch.from_numpy(X).to(target_device)
            if isinstance(X, pd.DataFrame):
                return torch.from_numpy(X.values).to(target_device)
        elif backend == BackendType.PANDAS:
            if isinstance(X, pd.DataFrame):
                return X
            if isinstance(X, np.ndarray):
                return pd.DataFrame(X)
            if isinstance(X, torch.Tensor):
                return pd.DataFrame(X.cpu().numpy())
        raise ValueError("Unsupported input type for normalization.")

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> 'MultilinearModel':
        """
        Fit the multilinear model to data. Must be implemented by subclasses.
        Handles backend detection and input normalization.
        :param X: np.ndarray, torch.Tensor, or pandas.DataFrame
        :param y: np.ndarray, torch.Tensor, or pandas.DataFrame, optional
        :return: self
        """
        raise NotImplementedError("fit method must be implemented by subclass.")

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        Predict using the multilinear model. Must be implemented by subclasses.
        :param X: np.ndarray, torch.Tensor, or pandas.DataFrame
        :return: predictions as np.ndarray, torch.Tensor, or pandas.DataFrame
        """
        raise NotImplementedError("predict method must be implemented by subclass.")
