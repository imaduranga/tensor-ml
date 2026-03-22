"""tensor_ml – public API."""

__version__ = "0.2.0"

from tensor_ml.enums import BackendType

from tensor_ml.exceptions import (
    TensorMLError,
    NotFittedError,
    BackendError,
    ShapeMismatchError,
    ValidationError,
)

from tensor_ml.tensor_ops import (
    # TensorOps
    TensorOps, NumpyOps, TensorOpsFactory,
    # TensorProducts
    TensorProductsBase, TensorProducts, TensorProductsFactory,
    NumpyTensorProducts,
)

from tensor_ml.tensor_models import BaseTensorModel, MultilinearModel, TLARS, TLARSConfig

__all__ = [
    "__version__",
    "BackendType",
    # Exceptions
    "TensorMLError", "NotFittedError", "BackendError",
    "ShapeMismatchError", "ValidationError",
    # TensorOps
    "TensorOps", "NumpyOps", "TensorOpsFactory",
    # TensorProducts
    "TensorProductsBase", "TensorProducts", "TensorProductsFactory",
    "NumpyTensorProducts",
    # Models
    "BaseTensorModel", "MultilinearModel", "TLARS", "TLARSConfig",
]

try:
    from tensor_ml.tensor_ops import TorchOps, TorchTensorProducts
    __all__ += ["TorchOps", "TorchTensorProducts"]
except ImportError:
    pass
