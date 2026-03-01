"""tensor_ml – public API."""

from tensor_ml.enums import BackendType

from tensor_ml.tensor_ops import (
    # TensorOps
    TensorOps, NumpyOps, TensorOpsFactory,
    # TensorProducts
    TensorProductsBase, TensorProducts, TensorProductsFactory,
    NumpyTensorProducts,
)

from tensor_ml.tensor_models import BaseTensorModel, MultilinearModel, TLARS

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
    from tensor_ml.tensor_ops import TorchOps, TorchTensorProducts
    __all__ += ["TorchOps", "TorchTensorProducts"]
except ImportError:
    pass
