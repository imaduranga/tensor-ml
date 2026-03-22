"""Utility helpers for backend inference."""

import numpy as np
from typing import Any, Optional
from tensor_ml.enums import BackendType
from tensor_ml.exceptions import BackendError

__all__ = ["infer_backend"]

def infer_backend(data: Any, backend: Optional[BackendType] = None) -> BackendType:
    """Infer the compute backend from input data.

    Parameters
    ----------
    data : array-like
        Input data (``np.ndarray``, ``torch.Tensor``, or a list of them).
    backend : BackendType, optional
        If provided, returned directly without inference.

    Returns
    -------
    BackendType
    """
    if backend is not None:
        return backend
    try:
        import torch
        if isinstance(data, torch.Tensor) or (isinstance(data, list) and len(data) > 0 and isinstance(data[0], torch.Tensor)):
            return BackendType.TORCH
    except ImportError:
        pass
    if isinstance(data, np.ndarray) or (isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray)):
        return BackendType.NUMPY
    raise BackendError(
        f"Cannot infer backend from data type {type(data).__name__!r}. "
        "Expected np.ndarray, torch.Tensor, or a list of them."
    )

