"""Base multilinear tensor model with backend-agnostic ops."""

from typing import Optional, Any, Union
import numpy as np
from tensor_ml.tensor_models.base import BaseTensorModel
from tensor_ml.enums import BackendType
from tensor_ml.utils import infer_backend
from tensor_ml.tensor_ops import TensorOpsFactory, TensorProductsFactory
import logging

__all__ = ["MultilinearModel"]

logger = logging.getLogger(__name__)


class MultilinearModel(BaseTensorModel):
    """
    Base class for multilinear tensor models. Inherit from this class for specific multilinear algorithms
    (e.g., TLARS, TNET, Kronecker OMP, etc.). Handles backend detection and input normalization.
    """
    def __init__(self, backend: Optional[Union[str, BackendType]] = None, device: Optional[Union[str, Any]] = None):
        """
        Initialize the MultilinearModel.

        Parameters
        ----------
        backend : Optional[Union[str, BackendType]], default=None
            Backend to use (NUMPY or TORCH). If None, will be inferred from data at fit time.
        device : Optional[Union[str, Any]], default=None
            Device for torch tensors ('cuda', 'cpu', or torch.device). Defaults to None
            (auto-detected by TorchOps: CUDA if available, else CPU). Ignored for non-torch backends.
        """
        super().__init__()
        self._device_hint = device

        if backend is not None:
            if isinstance(backend, BackendType):
                self.backend = backend
            else:
                self.backend = BackendType(str(backend).lower())
            self._setup_ops()
        else:
            self.backend = None
            self.ops = None
            self.tp = None

    def _setup_ops(self) -> None:
        """Create the ops and tensor-products instances for the current backend."""
        if self.backend == BackendType.TORCH:
            self.ops = TensorOpsFactory.get(self.backend, self._device_hint)
            self.tp = TensorProductsFactory.get(self.backend, self._device_hint)
        else:
            self.ops = TensorOpsFactory.get(self.backend)
            self.tp = TensorProductsFactory.get(self.backend)

    def _resolve_backend(self, data: Any) -> None:
        """
        Resolve the backend from input data if not already set.
        Called at the start of fit() to enable lazy detection.
        """
        if self.backend is not None and self.ops is not None:
            return
        self.backend = infer_backend(data)
        self._setup_ops()

    def get_backend(self, X: Any = None) -> BackendType:
        """
        Returns the backend as a BackendType. If not set, infers from X and sets self.backend.
        :param X: Optional input to infer backend if not set.
        :return: BackendType (NUMPY or TORCH)
        """
        if self.backend is not None:
            return self.backend
        if X is not None:
            self._resolve_backend(X)
            return self.backend
        raise ValueError("Backend is not set and cannot be inferred without input X.")

    def normalize_input(self, X: Any) -> Any:
        """
        Converts input to the appropriate type based on backend.
        - If backend is numpy, converts pandas DataFrame or torch Tensor to np.ndarray.
        - If backend is torch, converts pandas DataFrame or np.ndarray to torch.Tensor and moves to device.
        :param X: Input data (np.ndarray or torch.Tensor)
        :return: Normalized input
        """
        backend = self.get_backend(X)
        if backend == BackendType.NUMPY:
            try:
                import pandas as pd
                if isinstance(X, pd.DataFrame):
                    return X.values
            except ImportError:
                pass
            if isinstance(X, np.ndarray):
                return X
            try:
                import torch
                if isinstance(X, torch.Tensor):
                    return X.cpu().numpy()
            except ImportError:
                pass
        elif backend == BackendType.TORCH:
            import torch
            if isinstance(X, torch.Tensor):
                return self.ops.to_device(X)
            if isinstance(X, np.ndarray):
                return self.ops.to_device(torch.from_numpy(X))
            try:
                import pandas as pd
                if isinstance(X, pd.DataFrame):
                    return self.ops.to_device(torch.from_numpy(X.values))
            except ImportError:
                pass
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

    def score(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs: Any,
    ) -> float:
        """Return the coefficient of determination R² of the prediction.

        Parameters
        ----------
        X : array-like
            Factor / dictionary matrices passed to :meth:`predict`.
        y : array-like, optional
            Ground-truth tensor.

        Returns
        -------
        float
        """
        Y_pred = self.predict(X)
        Y_true_flat = self.ops.flatten(self.normalize_input(y))
        Y_pred_flat = self.ops.flatten(Y_pred)
        u = self.ops.sum((Y_true_flat - Y_pred_flat) ** 2)
        v = self.ops.sum((Y_true_flat - self.ops.mean(Y_true_flat)) ** 2)
        if v == 0:
            return 0.0
        return float(1 - u / v)

    def mse(
        self,
        X: Any,
        y: Any,
    ) -> float:
        """Return the mean squared error (MSE) of the prediction.

        Parameters
        ----------
        X : array-like
            Factor / dictionary matrices passed to :meth:`predict`.
        y : array-like
            Ground-truth tensor.

        Returns
        -------
        float
        """
        Y_pred = self.predict(X)
        Y_true_flat = self.ops.flatten(self.normalize_input(y))
        Y_pred_flat = self.ops.flatten(Y_pred)
        mse = self.ops.mean((Y_true_flat - Y_pred_flat) ** 2)
        return float(mse)

    def mae(
        self,
        X: Any,
        y: Any,
    ) -> float:
        """Return the mean absolute error (MAE) of the prediction.

        Parameters
        ----------
        X : array-like
            Factor / dictionary matrices passed to :meth:`predict`.
        y : array-like
            Ground-truth tensor.

        Returns
        -------
        float
        """
        Y_pred = self.predict(X)
        Y_true_flat = self.ops.flatten(self.normalize_input(y))
        Y_pred_flat = self.ops.flatten(Y_pred)
        mae = self.ops.mean(self.ops.abs(Y_true_flat - Y_pred_flat))
        return float(mae)

