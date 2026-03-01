"""Backend-agnostic element-wise tensor operations API.

Provides:
- ``TensorOps``         – abstract base class (contract for backends)
- ``NumpyOps``          – NumPy implementation
- ``TorchOps``          – PyTorch implementation
- ``TensorOpsFactory``  – registry-based factory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np

from tensor_ml.enums import BackendType

__all__ = ["TensorOps", "NumpyOps", "TorchOps", "TensorOpsFactory"]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TensorOps(ABC):
    """Abstract base class for backend-specific element-wise tensor operations."""

    @abstractmethod
    def norm(self, x: Any) -> float:
        """Return the L2 norm of *x*."""
        ...

    @abstractmethod
    def normalize(self, D: Any) -> Any:
        """Column-normalise matrix *D*."""
        ...

    @abstractmethod
    def zeros(self, shape: int | tuple[int, ...]) -> Any:
        """Return a zero-filled array of *shape*."""
        ...

    @abstractmethod
    def ones(self, shape: int | tuple[int, ...]) -> Any:
        """Return a ones-filled array of *shape*."""
        ...

    @abstractmethod
    def abs(self, x: Any) -> Any:
        """Element-wise absolute value."""
        ...

    @abstractmethod
    def sign(self, x: Any) -> Any:
        """Element-wise sign."""
        ...

    @abstractmethod
    def argmax(self, x: Any) -> int:
        """Index of the maximum element."""
        ...

    @abstractmethod
    def argmin(self, x: Any) -> int:
        """Index of the minimum element."""
        ...

    @abstractmethod
    def concatenate(self, arrs: list[Any]) -> Any:
        """Concatenate a list of arrays along the first axis."""
        ...

    @property
    @abstractmethod
    def inf(self) -> float:
        """Positive infinity constant."""
        ...

    @abstractmethod
    def asarray(self, x: Any) -> Any:
        """Convert *x* to the backend's native array type."""
        ...

    @abstractmethod
    def flatten(self, x: Any) -> Any:
        """Flatten *x* in Fortran (column-major) order."""
        ...

    @abstractmethod
    def to_device(self, x: Any) -> Any:
        """Move *x* to the configured device (no-op for NumPy)."""
        ...

    @abstractmethod
    def nonzero(self, x: Any) -> Any:
        """Return indices of non-zero elements."""
        ...

    @abstractmethod
    def mean(self, x: Any) -> float:
        """Arithmetic mean of all elements."""
        ...

    @abstractmethod
    def sum(self, x: Any) -> Any:
        """Sum of all elements."""
        ...

    @abstractmethod
    def gramian(self, D: Any) -> Any:
        """Return the Gramian Dᵀ D."""
        ...

    @abstractmethod
    def copy(self, x: Any) -> Any:
        """Return a copy (clone) of *x*."""
        ...

    @abstractmethod
    def to_scalar(self, x: Any) -> float:
        """Extract a Python float from a scalar tensor/array."""
        ...

    @abstractmethod
    def has_nan(self, x: Any) -> bool:
        """Return ``True`` if *x* contains any NaN values."""
        ...

    @abstractmethod
    def pinv(self, x: Any) -> Any:
        """Moore-Penrose pseudo-inverse."""
        ...

    @abstractmethod
    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert *x* to a NumPy ``ndarray``."""
        ...

    @abstractmethod
    def find_index(self, arr: Any, val: Any) -> int:
        """Return the first index of *val* in *arr*."""
        ...

    @abstractmethod
    def numel(self, x: Any) -> int:
        """Number of elements along the first axis."""
        ...

    @abstractmethod
    def eye(self, n: int) -> Any:
        """Return an *n × n* identity matrix."""
        ...

    @abstractmethod
    def allclose(self, a: Any, b: Any) -> bool:
        """Return ``True`` if *a* and *b* are element-wise close."""
        ...

    @abstractmethod
    def max(self, x: Any) -> Any:
        """Maximum element."""
        ...


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

class NumpyOps(TensorOps):
    """NumPy implementation of element-wise tensor operations."""

    def norm(self, x: Any) -> float:
        return float(np.linalg.norm(x))

    def normalize(self, D: Any) -> np.ndarray:
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return D / norms

    def zeros(self, shape: int | tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape)

    def ones(self, shape: int | tuple[int, ...]) -> np.ndarray:
        return np.ones(shape)

    def abs(self, x: Any) -> np.ndarray:
        return np.abs(x)

    def sign(self, x: Any) -> np.ndarray:
        return np.sign(x)

    def argmax(self, x: Any) -> int:
        return int(np.argmax(x))

    def argmin(self, x: Any) -> int:
        return int(np.argmin(x))

    def concatenate(self, arrs: list[Any]) -> np.ndarray:
        return np.concatenate(arrs)

    @property
    def inf(self) -> float:
        return float(np.inf)

    def asarray(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def flatten(self, x: Any) -> np.ndarray:
        return x.flatten(order='F')

    def to_device(self, x: Any) -> Any:
        return x

    def nonzero(self, x: Any) -> np.ndarray:
        return np.nonzero(x)[0]

    def mean(self, x: Any) -> float:
        return float(np.mean(x))

    def sum(self, x: Any) -> Any:
        return np.sum(x)

    def gramian(self, D: Any) -> np.ndarray:
        return D.T @ D

    def copy(self, x: Any) -> np.ndarray:
        return x.copy()

    def to_scalar(self, x: Any) -> float:
        return float(x)

    def has_nan(self, x: Any) -> bool:
        return bool(np.any(np.isnan(x)))

    def pinv(self, x: Any) -> np.ndarray:
        return np.linalg.pinv(np.asarray(x))

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def find_index(self, arr: Any, val: Any) -> int:
        return int(np.where(arr == val)[0][0])

    def numel(self, x: Any) -> int:
        return len(x)

    def eye(self, n: int) -> np.ndarray:
        return np.eye(n)

    def allclose(self, a: Any, b: Any) -> bool:
        return bool(np.allclose(a, b))

    def max(self, x: Any) -> Any:
        return np.max(x)


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------

class TorchOps(TensorOps):
    """PyTorch implementation of element-wise tensor operations."""

    def __init__(self, device: Any = 'cuda') -> None:
        import torch
        import torch.nn.functional as F
        self.torch = torch
        self.F = F
        if device is None or (isinstance(device, str) and device == 'cuda'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            raise TypeError(f"device must be None, a string, or torch.device, got {type(device)}")

    def norm(self, x: Any) -> float:
        return float(self.torch.norm(x))

    def normalize(self, D: Any) -> Any:
        return self.F.normalize(D, dim=0)

    def zeros(self, shape: int | tuple[int, ...]) -> Any:
        return self.torch.zeros(shape, device=self.device)

    def ones(self, shape: int | tuple[int, ...]) -> Any:
        return self.torch.ones(shape, device=self.device)

    def abs(self, x: Any) -> Any:
        return self.torch.abs(x)

    def sign(self, x: Any) -> Any:
        return self.torch.sign(x)

    def argmax(self, x: Any) -> int:
        return int(self.torch.argmax(x))

    def argmin(self, x: Any) -> int:
        return int(self.torch.argmin(x))

    def concatenate(self, arrs: list[Any]) -> Any:
        return self.torch.cat(arrs)

    @property
    def inf(self) -> float:
        return float('inf')

    def asarray(self, x: Any) -> Any:
        return self.torch.as_tensor(x, device=self.device)

    def flatten(self, x: Any) -> Any:
        if x.ndim <= 1:
            return x.contiguous().flatten()
        return x.permute(*reversed(range(x.ndim))).contiguous().flatten()

    def to_device(self, x: Any) -> Any:
        return x.to(self.device)

    def nonzero(self, x: Any) -> Any:
        return self.torch.nonzero(x, as_tuple=True)[0]

    def mean(self, x: Any) -> float:
        return float(self.torch.mean(x))

    def sum(self, x: Any) -> Any:
        return self.torch.sum(x)

    def gramian(self, D: Any) -> Any:
        return D.t() @ D

    def copy(self, x: Any) -> Any:
        return x.clone()

    def to_scalar(self, x: Any) -> float:
        return float(x.item()) if hasattr(x, 'item') else float(x)

    def has_nan(self, x: Any) -> bool:
        return bool(x.isnan().any())

    def pinv(self, x: Any) -> Any:
        return self.torch.linalg.pinv(x)

    def to_numpy(self, x: Any) -> np.ndarray:
        return x.detach().cpu().numpy()

    def find_index(self, arr: Any, val: Any) -> int:
        return int((arr == val).nonzero(as_tuple=True)[0][0])

    def numel(self, x: Any) -> int:
        return int(x.size(0))

    def eye(self, n: int) -> Any:
        return self.torch.eye(n, device=self.device, dtype=self.torch.float64)

    def allclose(self, a: Any, b: Any) -> bool:
        return bool(self.torch.allclose(a.to(dtype=self.torch.float64), b.to(dtype=self.torch.float64)))

    def max(self, x: Any) -> Any:
        return x.max()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TensorOpsFactory:
    """Registry-based factory for tensor operations backends."""

    _registry: dict[BackendType, type[TensorOps]] = {}

    @classmethod
    def register(cls, backend_type: BackendType, ops_class: type[TensorOps]) -> None:
        """Register an ops class for a backend type."""
        cls._registry[backend_type] = ops_class

    @classmethod
    def get(cls, backend: Union[str, BackendType], device: Any = None) -> TensorOps:
        """Return a ``TensorOps`` instance for the given *backend*.

        Parameters
        ----------
        backend : str | BackendType
            The backend identifier.
        device : optional
            Device hint (only used by the PyTorch backend).

        Returns
        -------
        instance : TensorOps
        """
        if isinstance(backend, str):
            backend = BackendType(backend.lower())
        ops_class = cls._registry.get(backend)
        if ops_class is None:
            if backend == BackendType.TORCH:
                raise ImportError(
                    "PyTorch is required for the TORCH backend but is not installed. "
                    "Install it with: pip install torch"
                )
            raise ValueError(f"Unsupported backend: {backend}")
        if backend == BackendType.TORCH:
            return ops_class(device=device)
        return ops_class()


# Register built-in backends
TensorOpsFactory.register(BackendType.NUMPY, NumpyOps)
try:
    TensorOpsFactory.register(BackendType.TORCH, TorchOps)
except ImportError:
    pass
