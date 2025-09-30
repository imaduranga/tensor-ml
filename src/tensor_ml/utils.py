import torch
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
    if isinstance(data, torch.Tensor) or (isinstance(data, list) and isinstance(data[0], torch.Tensor)):
        return BackendType.TORCH
    elif isinstance(data, np.ndarray) or (isinstance(data, list) and isinstance(data[0], np.ndarray)):
        return BackendType.NUMPY
    else:
        raise ValueError("Cannot infer backend from data type.")

