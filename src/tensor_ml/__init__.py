"""tensor_ml – public API."""

from tensor_ml.enums import BackendType

from tensor_ml.tensorops import (
    # TensorOps
    TensorOps, NumpyOps, TensorOpsFactory,
    # TensorProducts
    TensorProductsBase, TensorProducts, TensorProductsFactory,
    NumpyTensorProducts,
)

from tensor_ml.tensor_models.base import BaseTensorModel
from tensor_ml.tensor_models.multilinear.multilinear_model import MultilinearModel
from tensor_ml.tensor_models.multilinear.tlars import TLARS

__all__ = [
    "BackendType",
    # TensorOps
    "TensorOps", "NumpyOps", "TensorOpsFactory",
    # TensorProducts
    "TensorProductsBase", "TensorProducts", "TensorProductsFactory",
    "NumpyTensorProducts",
    # Models
    "BaseTensorModel", "MultilinearModel", "TLARS",
]

try:
    from tensor_ml.tensorops import TorchOps, TorchTensorProducts
    __all__ += ["TorchOps", "TorchTensorProducts"]
except ImportError:
    pass
