import numpy as np
from typing import Any, Optional
from tensor_ml.enums import BackendType

def infer_backend(data: Any, backend: Optional[BackendType] = None) -> BackendType:
    """
    Infer the backend (NUMPY or TORCH) from the input data or use the provided backend.
    :param data: Input data (tensor, ndarray, or list of them)
    :param backend: Optional explicit backend
    :return: BackendType
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
    raise ValueError("Cannot infer backend from data type.")

